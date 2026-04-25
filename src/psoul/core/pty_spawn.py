"""PTY-backed spawn and supervise for headless sessions. Unix only."""

import errno
import os
import selectors
import socket
import struct
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psutil
from ptyprocess import PtyProcess, PtyProcessError

from psoul.core import control
from psoul.core.db import open_db
from psoul.core.events import EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.output import READ_CHUNK_SIZE
from psoul.core.resources import ResourceSampler
from psoul.core.session import Session, SessionState
from psoul.core.store import SessionStore

_SAMPLE_INTERVAL = 2.0  # seconds between resource samples
_SUPERVISE_TICK = 0.1  # seconds per supervise-loop iteration
_RESPAWN_BACKOFFS: tuple[float, ...] = (1.0, 2.0, 4.0)  # sleep before each retry when a restart respawns the child

_FRAME_DATA = 0x01
_FRAME_WINSIZE = 0x02
_FRAME_HELLO = 0x03
_FRAME_HEADER = struct.Struct("!BH")  # network-order: 1-byte type + 2-byte length


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


def _drain_tick(
    selector: selectors.DefaultSelector,
    event_store: EventStore,
    session_id: str,
    generation: int,
    timeout: float,
) -> bool:
    """Read one selector pass of PTY main output into the event log.

    Emits each chunk as a ``runtime.stdout`` event. A PTY merges the
    child's stdout and stderr at the kernel level, so there is no
    separate ``runtime.stderr`` emission for child writes.

    Returns ``True`` when at least one chunk was read. Unregisters the
    main fd on EOF so the caller can detect closure via
    ``selector.get_map()``. Linux raises ``OSError(EIO)`` on a main-fd
    read after the child closes its tty. macOS and BSD return ``b""``.
    Both are treated as EOF.
    """
    any_read = False
    for key, _mask in selector.select(timeout=timeout):
        try:
            chunk = os.read(key.fd, READ_CHUNK_SIZE)
        except OSError as exc:
            if exc.errno != errno.EIO:
                raise
            chunk = b""
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        event_store.append(
            session_id=session_id,
            event_type=EVENT_RUNTIME_STDOUT,
            payload={"text": chunk.decode("utf-8", errors="replace")},
            generation=generation,
            commit=False,
        )
        any_read = True
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


def _supervise_loop(
    pty_child: ManagedChild,
    *,
    session_id: str,
    event_store: EventStore,
    generation: int,
    store: SessionStore,
    control_state: control.ControlState,
    stop_timeout_seconds: float,
) -> None:
    """Tick-based supervise loop: drain the PTY main, dispatch signal flags, watch child exit.

    Runs on the supervisor's main thread so signal handlers installed by
    ``control.install_handlers`` can flip request flags and have them
    serviced on the same thread each tick. After ``pty_child.poll()``
    reports exit, any still-readable main-fd output is drained before
    returning.
    """
    with selectors.DefaultSelector() as selector:
        selector.register(pty_child.main_fd, selectors.EVENT_READ, EVENT_RUNTIME_STDOUT)
        while pty_child.poll() is None:
            _drain_tick(selector, event_store, session_id, generation, timeout=_SUPERVISE_TICK)
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
        while selector.get_map() and _drain_tick(
            selector, event_store, session_id, generation, timeout=_SUPERVISE_TICK
        ):
            pass


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
    except psutil.NoSuchProcess:
        pass  # child already exited; skip sampling, supervise loop still finalizes
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
    conn = open_db(state_dir)
    listen_socket: socket.socket | None = None
    socket_path: Path | None = None
    try:
        listen_socket, socket_path = _create_listen_socket(session_id)
        sup_store = SessionStore(conn)
        event_store = EventStore(conn)
        control_state = control.ControlState()
        control.install_handlers(control_state)
        generation = 0
        pty_child, sampler, sampler_thread = _spawn_generation(argv, cwd, state_dir, session_id, generation=generation)
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
        conn.close()
