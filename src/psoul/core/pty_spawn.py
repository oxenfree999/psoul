"""PTY-backed spawn and supervise for headless sessions. Unix only."""

import ctypes
import errno
import fcntl
import logging
import os
import selectors
import socket
import struct
import sys
import termios
import threading
import time
from contextlib import closing, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import psutil
from ptyprocess import PtyProcess, PtyProcessError

from psoul.core import control
from psoul.core.db import open_db
from psoul.core.events import EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.output import READ_CHUNK_SIZE
from psoul.core.resources import ResourceSampler
from psoul.core.session import Session, SessionState
from psoul.core.store import SessionStore

_logger = logging.getLogger(__name__)

_SAMPLE_INTERVAL = 2.0  # seconds between resource samples
_SUPERVISE_TICK = 0.1  # seconds per supervise-loop iteration
_RESPAWN_BACKOFFS: tuple[float, ...] = (1.0, 2.0, 4.0)  # sleep before each retry when a restart respawns the child
_DESCENDANT_GRACE_SECONDS = (
    5.0  # seconds the supervisor lingers reaping reparented descendants after the managed child exits
)
_PR_SET_CHILD_SUBREAPER = 36  # Linux prctl option, kernel 3.4+, hardcoded to avoid a one-constant ctypes binding dep
_DRAIN_POLL_INTERVAL = 0.05  # seconds between descendant-drain ticks
_DRAIN_STABILIZATION = (
    0.2  # seconds the drain lingers after detecting empty descendants, so observers see the empty state
)

_FRAME_DATA = 0x01
_FRAME_WINSIZE = 0x02
_FRAME_HELLO = 0x03
_FRAME_HEADER = struct.Struct("!BH")  # network-order: 1-byte type + 2-byte length
_HELLO_PAYLOAD = struct.Struct("!I")  # network-order 4-byte unsigned int: client pid
_WINSIZE_STRUCT = struct.Struct("HHHH")  # native-order rows, cols, xpixel, ypixel matching ``struct winsize``
_MAX_FRAME_PAYLOAD = 32768  # replay chunk cap, stays under the wire format's 65535-byte length ceiling
_REPLAY_EVENT_LIMIT = 1000  # most recent runtime.stdout events sent to a freshly attached client

_TAG_PTY_MAIN = "pty_main"  # selector data tag for the PTY main fd
_TAG_LISTEN = "listen"  # selector data tag for the supervisor's listen socket
_TAG_CLIENT = "client"  # selector data tag for an accepted attach-client socket


def _encode_frame(kind: int, payload: bytes) -> bytes:
    """Build a wire frame: 1 byte type + 2 bytes big-endian length + payload bytes."""
    return _FRAME_HEADER.pack(kind, len(payload)) + payload


def _decode_frames(buffer: bytearray) -> list[tuple[int, bytes]]:
    """Consume complete frames from *buffer* in place. Return ``[(kind, payload), ...]``.

    Mutates *buffer*: complete frames are removed from the front, and any
    trailing partial bytes (an incomplete header or an unfilled payload)
    remain so the next call resumes parsing where this one left off.
    """
    frames: list[tuple[int, bytes]] = []
    header_size = _FRAME_HEADER.size
    while len(buffer) >= header_size:
        kind, length = _FRAME_HEADER.unpack_from(buffer)
        if len(buffer) < header_size + length:
            break
        payload = bytes(buffer[header_size : header_size + length])
        frames.append((kind, payload))
        del buffer[: header_size + length]
    return frames


@dataclass(slots=True)
class ManagedChild:
    """Wrapper around a ``ptyprocess.PtyProcess`` that satisfies ``control.SupervisedProc``.

    Holds the PID, the PTY main fd, the underlying ``PtyProcess``, and a
    cached ``returncode``. ``.poll()`` is an exit-only WNOHANG observer
    matching ``subprocess.Popen.poll()`` semantics. Stop and continue
    transitions are observed separately by ``_poll_child_status``.
    """

    pid: int
    main_fd: int
    pty_process: PtyProcess
    returncode: int | None = None

    def poll(self) -> int | None:
        """Return the child's exit code if it has terminated, else ``None``.

        Uses ``os.waitpid(pid, WNOHANG)`` without ``WUNTRACED`` or
        ``WCONTINUED``, so stop and continue events are left for
        ``_poll_child_status`` to observe. Returns the cached
        ``returncode`` on subsequent calls after exit.
        """
        if self.returncode is not None:
            return self.returncode
        try:
            wait_pid, sts = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            return None
        if wait_pid == 0:
            return None
        if os.WIFEXITED(sts):
            self.returncode = os.WEXITSTATUS(sts)
        elif os.WIFSIGNALED(sts):
            self.returncode = -os.WTERMSIG(sts)
        return self.returncode


@dataclass(slots=True)
class _AttachClient:
    """Per-connection state for an attached client.

    ``decode_buffer`` holds partial frames between ``recv()`` calls.
    ``authenticated`` flips ``True`` once HELLO matches ``controller_pid``,
    at which point ``client_pid`` records the matched pid for race-safe release.
    """

    sock: socket.socket
    decode_buffer: bytearray = field(default_factory=bytearray)
    authenticated: bool = False
    client_pid: int | None = None


def _disconnect_attach_client(
    client: _AttachClient,
    selector: selectors.DefaultSelector,
    clients: dict[int, _AttachClient],
    store: SessionStore,
    event_store: EventStore,
    session_id: str,
    generation: int,
) -> None:
    """Release the controller row when *client* was authenticated, then close and drop it. Idempotent."""
    if client.authenticated:
        control.handle_controller_released(store, event_store, session_id, generation, controller_pid=client.client_pid)
    clients.pop(client.sock.fileno(), None)
    with suppress(KeyError, ValueError):
        selector.unregister(client.sock)
    client.sock.close()


def _fan_out_to_clients(
    chunk: bytes,
    selector: selectors.DefaultSelector,
    clients: dict[int, _AttachClient],
    store: SessionStore,
    event_store: EventStore,
    session_id: str,
    generation: int,
) -> None:
    """Send *chunk* as a DATA frame to each authenticated client. Drop clients whose send fails."""
    if not clients:
        return
    frame = _encode_frame(_FRAME_DATA, chunk)
    for client in list(clients.values()):
        if not client.authenticated:
            continue
        try:
            client.sock.sendall(frame)
        except OSError:
            _disconnect_attach_client(client, selector, clients, store, event_store, session_id, generation)


def _accept_attach_client(
    listen_socket: socket.socket,
    selector: selectors.DefaultSelector,
    clients: dict[int, _AttachClient],
) -> None:
    """Accept one pending connection on *listen_socket* and register it for reading."""
    try:
        sock, _addr = listen_socket.accept()
    except OSError:
        return
    sock.setblocking(False)
    fd = sock.fileno()
    clients[fd] = _AttachClient(sock=sock)
    selector.register(fd, selectors.EVENT_READ, _TAG_CLIENT)


def _apply_winsize(pty_main_fd: int, payload: bytes) -> bool:
    """Forward a WINSIZE payload to *pty_main_fd* via ``TIOCSWINSZ``. Returns ``False`` on bad payload length."""
    if len(payload) != _WINSIZE_STRUCT.size:
        return False
    with suppress(OSError):
        fcntl.ioctl(pty_main_fd, termios.TIOCSWINSZ, payload)
    return True


def _send_replay(client: _AttachClient, event_store: EventStore, session_id: str) -> bool:
    """Send recent ``runtime.stdout`` event payloads to *client* as DATA frames. Returns ``False`` on send failure."""
    events = event_store.list_recent(session_id, event_type=EVENT_RUNTIME_STDOUT, limit=_REPLAY_EVENT_LIMIT)
    text_parts: list[str] = []
    for ev in events:
        payload = ev["payload"]
        if not isinstance(payload, dict):
            continue
        text = cast("dict[str, object]", payload).get("text")
        if isinstance(text, str):
            text_parts.append(text)
    encoded = "".join(text_parts).encode("utf-8")
    for offset in range(0, len(encoded), _MAX_FRAME_PAYLOAD):
        try:
            client.sock.sendall(_encode_frame(_FRAME_DATA, encoded[offset : offset + _MAX_FRAME_PAYLOAD]))
        except OSError:
            return False
    return True


def _authenticate_attach_client(
    client: _AttachClient,
    kind: int,
    payload: bytes,
    store: SessionStore,
    session_id: str,
    event_store: EventStore,
) -> bool:
    """Validate the HELLO frame and send replay. Returns ``False`` to drop the client."""
    if kind != _FRAME_HELLO or len(payload) != _HELLO_PAYLOAD.size:
        return False
    (declared_pid,) = _HELLO_PAYLOAD.unpack(payload)
    session = store.get(session_id)
    if session is None or session.controller_pid != declared_pid:
        return False
    client.authenticated = True
    client.client_pid = declared_pid
    return _send_replay(client, event_store, session_id)


def _handle_attach_client_read(
    client: _AttachClient,
    pty_main_fd: int,
    store: SessionStore,
    session_id: str,
    event_store: EventStore,
) -> bool:
    """Drain readable bytes, decode frames, dispatch HELLO/DATA/WINSIZE. Returns ``False`` to drop the client."""
    try:
        data = client.sock.recv(READ_CHUNK_SIZE)
    except OSError:
        return False
    if not data:
        return False
    client.decode_buffer.extend(data)
    for kind, payload in _decode_frames(client.decode_buffer):
        if not client.authenticated:
            if not _authenticate_attach_client(client, kind, payload, store, session_id, event_store):
                return False
            continue
        if kind == _FRAME_DATA:
            try:
                os.write(pty_main_fd, payload)
            except OSError:
                return False
        elif kind == _FRAME_WINSIZE and not _apply_winsize(pty_main_fd, payload):
            return False
    return True


def _drain_pty_chunk_to_clients(
    fd: int,
    selector: selectors.DefaultSelector,
    event_store: EventStore,
    session_id: str,
    generation: int,
    clients: dict[int, _AttachClient],
    store: SessionStore,
) -> bool:
    """Read one PTY main chunk, emit it as ``runtime.stdout``, and fan out to *clients*. Returns ``False`` on EOF.

    Treats both ``OSError(EIO)`` (Linux) and an empty read (macOS/BSD) as EOF.
    """
    try:
        chunk = os.read(fd, READ_CHUNK_SIZE)
    except OSError as exc:
        if exc.errno != errno.EIO:
            raise
        chunk = b""
    if not chunk:
        return False
    event_store.append(
        session_id=session_id,
        event_type=EVENT_RUNTIME_STDOUT,
        payload={"text": chunk.decode("utf-8", errors="replace")},
        generation=generation,
        commit=False,
    )
    _fan_out_to_clients(chunk, selector, clients, store, event_store, session_id, generation)
    return True


def _drain_tick(
    selector: selectors.DefaultSelector,
    event_store: EventStore,
    session_id: str,
    generation: int,
    timeout: float,
    clients: dict[int, _AttachClient],
    store: SessionStore,
) -> bool:
    """Read one selector pass of PTY main output. Emit each chunk as ``runtime.stdout`` and fan out to *clients*.

    Acts only on selector keys tagged ``_TAG_PTY_MAIN``. Returns ``True`` when
    at least one chunk was read. Unregisters the main fd on EOF.
    """
    any_read = False
    for key, _mask in selector.select(timeout=timeout):
        if key.data != _TAG_PTY_MAIN:
            continue
        if _drain_pty_chunk_to_clients(key.fd, selector, event_store, session_id, generation, clients, store):
            any_read = True
        else:
            selector.unregister(key.fileobj)
    event_store.conn.commit()
    return any_read


def _poll_child_status(pty_child: ManagedChild) -> str | None:
    """Return the child's status change as ``"stopped"``, ``"continued"``, ``"exited"``, or ``None``.

    Observes stop and continue transitions via ``WUNTRACED | WCONTINUED``,
    which ``ManagedChild.poll()`` deliberately leaves alone. On
    ``"exited"``, sets ``pty_child.returncode`` to the child's exit code.
    """
    try:
        wait_pid, sts = os.waitpid(pty_child.pid, os.WNOHANG | os.WUNTRACED | os.WCONTINUED)
    except ChildProcessError:
        return None
    if wait_pid == 0:
        return None
    if os.WIFSTOPPED(sts):
        return "stopped"
    if os.WIFCONTINUED(sts):
        return "continued"
    if os.WIFEXITED(sts):
        pty_child.returncode = os.WEXITSTATUS(sts)
    elif os.WIFSIGNALED(sts):
        pty_child.returncode = -os.WTERMSIG(sts)
    else:
        return None
    return "exited"


def _set_subreaper() -> None:
    """Ask the kernel to reparent orphan descendants to this process. Linux 3.4+ only, no-op elsewhere."""
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError as exc:
        _logger.warning("libc.so.6 unavailable, subreaper not set: %s", exc)
        return
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        _logger.warning(
            "prctl(PR_SET_CHILD_SUBREAPER) failed errno=%d, orphan grandchildren may escape to PID 1",
            ctypes.get_errno(),
        )


def _reap_descendants(pty_child: ManagedChild) -> None:
    """Reap zombie descendants. Sync ``pty_child.returncode`` if the wildcard waitpid reaps the managed child."""
    while True:
        try:
            wait_pid, sts = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if wait_pid == 0:
            return
        if wait_pid == pty_child.pid and pty_child.returncode is None:
            if os.WIFEXITED(sts):
                pty_child.returncode = os.WEXITSTATUS(sts)
            elif os.WIFSIGNALED(sts):
                pty_child.returncode = -os.WTERMSIG(sts)


def _drain_descendants(pty_child: ManagedChild) -> None:
    """Linger past managed-child exit reaping reparented descendants, bounded by ``_DESCENDANT_GRACE_SECONDS``."""
    deadline = time.monotonic() + _DESCENDANT_GRACE_SECONDS
    ever_had_descendants = False
    while time.monotonic() < deadline:
        _reap_descendants(pty_child)
        try:
            children = psutil.Process().children(recursive=True)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return
        if children:
            ever_had_descendants = True
            time.sleep(_DRAIN_POLL_INTERVAL)
            continue
        if ever_had_descendants:
            time.sleep(_DRAIN_STABILIZATION)
        return


def _supervise_select_tick(
    selector: selectors.DefaultSelector,
    listen_socket: socket.socket | None,
    pty_main_fd: int,
    clients: dict[int, _AttachClient],
    *,
    event_store: EventStore,
    session_id: str,
    generation: int,
    store: SessionStore,
    timeout: float,
) -> None:
    """One selector pass. Drain PTY reads, accept new attach clients, route client frames."""
    for key, _mask in selector.select(timeout=timeout):
        if key.data == _TAG_PTY_MAIN:
            if not _drain_pty_chunk_to_clients(key.fd, selector, event_store, session_id, generation, clients, store):
                selector.unregister(key.fileobj)
        elif key.data == _TAG_LISTEN and listen_socket is not None:
            _accept_attach_client(listen_socket, selector, clients)
        elif key.data == _TAG_CLIENT:
            client = clients.get(key.fd)
            if client is not None and not _handle_attach_client_read(
                client, pty_main_fd, store, session_id, event_store
            ):
                _disconnect_attach_client(client, selector, clients, store, event_store, session_id, generation)
    event_store.conn.commit()


def _supervise_loop(
    pty_child: ManagedChild,
    *,
    listen_socket: socket.socket | None,
    session_id: str,
    event_store: EventStore,
    generation: int,
    store: SessionStore,
    control_state: control.ControlState,
    stop_timeout_seconds: float,
) -> None:
    """Tick-based supervise loop: drain the PTY main, accept attach clients, dispatch signal flags, watch child exit.

    Runs on the supervisor's main thread so signal handlers installed by
    ``control.install_handlers`` can flip request flags and have them
    serviced on the same thread each tick. After ``pty_child.poll()``
    reports exit, remaining main-fd output is drained and every attach
    client is closed.
    """
    clients: dict[int, _AttachClient] = {}
    with selectors.DefaultSelector() as selector:
        selector.register(pty_child.main_fd, selectors.EVENT_READ, _TAG_PTY_MAIN)
        if listen_socket is not None:
            selector.register(listen_socket.fileno(), selectors.EVENT_READ, _TAG_LISTEN)
        try:
            while pty_child.poll() is None:
                _supervise_select_tick(
                    selector,
                    listen_socket,
                    pty_child.main_fd,
                    clients,
                    event_store=event_store,
                    session_id=session_id,
                    generation=generation,
                    store=store,
                    timeout=_SUPERVISE_TICK,
                )
                status_change = _poll_child_status(pty_child)
                if status_change == "stopped":
                    control.handle_pause_observed(event_store, session_id, generation, store)
                elif status_change == "continued":
                    control.handle_resume_observed(event_store, session_id, generation, store)
                if control_state.stop_requested:
                    control.handle_stop(
                        control_state, pty_child, event_store, session_id, generation, stop_timeout_seconds, store
                    )
                if control_state.kill_requested:
                    control.handle_kill(control_state, pty_child, event_store, session_id, generation, store)
                if control_state.restart_requested:
                    control.handle_restart(
                        control_state, pty_child, event_store, session_id, generation, stop_timeout_seconds, store
                    )
                control.check_escalation(control_state, pty_child)
                _reap_descendants(pty_child)
            while selector.get_map() and _drain_tick(
                selector, event_store, session_id, generation, _SUPERVISE_TICK, clients, store
            ):
                pass
        finally:
            for client in list(clients.values()):
                _disconnect_attach_client(client, selector, clients, store, event_store, session_id, generation)


def _finalize_exit(
    pty_child: ManagedChild,
    session_id: str,
    store: SessionStore,
    start_monotonic: float,
    *,
    in_restart: bool = False,
) -> Session | None:
    """Record the result row and (unless ``in_restart``) transition the session to ``exited`` or ``failed``.

    If the session is already in ``stopping`` (because ``handle_stop`` or
    ``handle_kill`` has run), the intermediate ``running → stopping``
    transition is skipped to avoid a ``stopping → stopping`` error. When
    ``in_restart=True``, records the result row for the current
    generation and leaves the row at its pre-teardown state so the outer
    generation loop can transition it to ``restarting``.
    """
    duration = time.monotonic() - start_monotonic
    outcome = "exited" if pty_child.returncode is not None and pty_child.returncode == 0 else "failed"
    final_state = SessionState.exited if pty_child.returncode == 0 else SessionState.failed
    current = store.get(session_id)
    final: Session | None = None
    try:
        store.record_result(
            session_id=session_id,
            outcome=outcome,
            exit_code=pty_child.returncode,
            end_time=datetime.now(UTC),
            duration_seconds=duration,
        )
    finally:
        if not in_restart:
            if current is not None and current.state != SessionState.stopping:
                store.update(session_id, state=SessionState.stopping)
            final = store.update(session_id, state=final_state)
    return final


def _spawn_generation(
    argv: list[str],
    cwd: Path,
    state_dir: Path,
    session_id: str,
    generation: int,
) -> tuple[ManagedChild, ResourceSampler | None, threading.Thread | None]:
    """Spawn the managed child on a fresh PTY and prepare the ResourceSampler thread.

    ``PtyProcess.spawn`` wraps stdlib ``pty.fork()``, which calls
    ``setsid()`` in the child. The child becomes session + process-group
    leader (PGID = child PID), so ``os.killpg(child.pid, sig)`` still
    delivers to the group.
    """
    pty_proc = PtyProcess.spawn(argv, cwd=str(cwd))
    pty_child = ManagedChild(pid=pty_proc.pid, main_fd=pty_proc.fd, pty_process=pty_proc)
    sampler: ResourceSampler | None = None
    sampler_thread: threading.Thread | None = None
    try:
        ps_process = psutil.Process(pty_child.pid)
        sampler = ResourceSampler(ps_process, state_dir, session_id, generation)
        sampler_thread = threading.Thread(target=sampler.run, args=(_SAMPLE_INTERVAL,), daemon=True)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        pass  # child already exited or is a zombie awaiting reap. Skip sampling, supervise loop still finalizes.
    return pty_child, sampler, sampler_thread


def _respawn_with_backoff(
    argv: list[str],
    cwd: Path,
    state_dir: Path,
    session_id: str,
    new_generation: int,
) -> tuple[ManagedChild, ResourceSampler | None, threading.Thread | None]:
    """Retry ``_spawn_generation`` with ``_RESPAWN_BACKOFFS`` between attempts.

    Re-raises the last ``OSError`` or ``PtyProcessError`` on exhaustion.
    ``PtyProcess.spawn`` raises ``OSError`` from the underlying fork/exec
    and ``PtyProcessError`` for ptyprocess-specific failures. Both signal
    a failed spawn.
    """
    for attempt in range(len(_RESPAWN_BACKOFFS) + 1):
        try:
            return _spawn_generation(argv, cwd, state_dir, session_id, generation=new_generation)
        except (OSError, PtyProcessError):
            if attempt == len(_RESPAWN_BACKOFFS):
                raise
            time.sleep(_RESPAWN_BACKOFFS[attempt])
    msg = "unreachable: loop iterates once more than _RESPAWN_BACKOFFS length"
    raise RuntimeError(msg)


def _create_listen_socket(session_id: str) -> tuple[socket.socket | None, Path | None]:
    """Create the per-session control socket at ``/tmp/psoul-<uid>-<session_id>.sock``.

    Returns ``(socket, path)`` or ``(None, None)`` if ``bind()``,
    ``chmod()``, ``listen()``, or the non-blocking switch fails. The
    fixed ``/tmp`` prefix keeps the absolute path under
    ``sockaddr_un.sun_path`` (~104 bytes on macOS, ~108 on Linux)
    regardless of how deep the configured state directory sits. The
    socket file is set to mode ``0o600`` so only the same UID can
    connect. Caller closes and unlinks via ``_cleanup_listen_socket``.
    Any stale socket file from a prior crashed supervisor is unlinked
    first.
    """
    socket_path = Path(f"/tmp/psoul-{os.getuid()}-{session_id}.sock")  # noqa: S108 — /tmp required for AF_UNIX path-length on macOS, mode 0o600 mitigates
    socket_path.unlink(missing_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(socket_path))
        socket_path.chmod(0o600)
        sock.listen(4)
        sock.setblocking(False)
    except OSError:
        sock.close()
        socket_path.unlink(missing_ok=True)
        return None, None
    return sock, socket_path


def _cleanup_listen_socket(sock: socket.socket | None, path: Path | None) -> None:
    """Close *sock* and unlink *path* if they were ever created. Both are no-ops on ``None``."""
    if sock is not None:
        sock.close()
    if path is not None:
        path.unlink(missing_ok=True)


def _supervise(
    argv: list[str],
    cwd: Path,
    state_dir: Path,
    session_id: str,
    *,
    stop_timeout_seconds: float,
) -> None:
    """Background supervisor: spawn target, serve control signals, respawn across restarts, and finalize on exit.

    The outer ``while True:`` loop is the generation loop. Each iteration
    runs one supervise loop for one generation's child. On normal exit
    the outer loop finalizes the session and breaks. On restart,
    ``_finalize_exit`` is called with ``in_restart=True`` (record result
    without terminal transition), the row moves to ``restarting``, and
    ``_respawn_with_backoff`` spawns generation N+1.
    ``advance_restart_on_spawn_success`` and ``..._failure`` handle the
    boundary events and state transitions. The respawn catch widens to
    ``(OSError, PtyProcessError)`` so either spawn-error class closes
    the restart with ``command.failed(restart, ...)`` instead of
    escaping the supervisor.
    """
    _set_subreaper()
    with closing(open_db(state_dir, create=False)) as conn:
        listen_socket: socket.socket | None = None
        socket_path: Path | None = None
        try:
            listen_socket, socket_path = _create_listen_socket(session_id)
            sup_store = SessionStore(conn)
            event_store = EventStore(conn)
            control_state = control.ControlState()
            control.install_handlers(control_state)
            generation = 0
            pty_child, sampler, sampler_thread = _spawn_generation(
                argv, cwd, state_dir, session_id, generation=generation
            )
            sup_store.update(
                session_id,
                state=SessionState.running,
                supervisor_pid=os.getpid(),
                socket_path=socket_path,
            )
            while True:
                if sampler_thread is not None:
                    sampler_thread.start()
                start_mono = time.monotonic()
                try:
                    _supervise_loop(
                        pty_child,
                        listen_socket=listen_socket,
                        session_id=session_id,
                        event_store=event_store,
                        generation=generation,
                        store=sup_store,
                        control_state=control_state,
                        stop_timeout_seconds=stop_timeout_seconds,
                    )
                    control.emit_terminal_events(control_state, event_store, session_id, generation)
                finally:
                    if sampler is not None:
                        sampler.stop()
                    if sampler_thread is not None:
                        sampler_thread.join()
                if control_state.restart_pending is None:
                    _finalize_exit(pty_child, session_id, sup_store, start_mono)
                    _drain_descendants(pty_child)
                    break
                _finalize_exit(pty_child, session_id, sup_store, start_mono, in_restart=True)
                sup_store.update(session_id, state=SessionState.restarting)
                new_generation = generation + 1
                try:
                    pty_child, sampler, sampler_thread = _respawn_with_backoff(
                        argv, cwd, state_dir, session_id, new_generation
                    )
                except (OSError, PtyProcessError) as exc:
                    control.advance_restart_on_spawn_failure(
                        control_state,
                        sup_store,
                        event_store,
                        session_id,
                        generation,
                        "runtime_error",
                        str(exc),
                    )
                    break
                current = sup_store.get(session_id)
                if current is None:
                    msg = f"session row disappeared mid-restart: {session_id}"
                    raise RuntimeError(msg)
                control.advance_restart_on_spawn_success(
                    control_state,
                    sup_store,
                    event_store,
                    session_id,
                    new_generation,
                    current.control_epoch + 1,
                )
                generation = new_generation
            sup_store.update(session_id, supervisor_pid=None, socket_path=None)
        finally:
            _cleanup_listen_socket(listen_socket, socket_path)
