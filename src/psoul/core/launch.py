"""Session launch: target parsing, request building, and process lifecycle."""

import contextlib
import os
import select
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import psoul.helper
from psoul.core.db import open_db
from psoul.core.events import EVENT_HELPER_TIMEOUT, EventStore
from psoul.core.helper import HelperLifecycle, HelperTransport
from psoul.core.names import generate_session_id
from psoul.core.provenance import gather
from psoul.core.session import LaunchMode, Session, SessionState, TargetType, validate_session_id
from psoul.core.store import SessionStore
from psoul.helper._psoul_helper import UnixHelperPipeAdapter
from psoul.version import VERSION

if sys.platform != "win32":
    from psoul.core.pty_spawn import _supervise


PSOUL_HELPER_PIPE_ENV = "PSOUL_HELPER_PIPE"
_HELPER_MODULE_NAME = "_psoul_helper"
_HELPER_SUPPORTED = sys.platform != "win32"

_MS_PER_SECOND = 1000
_HELPER_WATCHER_POLL_INTERVAL_SECONDS = 0.25  # cadence at which the watcher rechecks proc.poll() and the socket


def _helper_package_dir() -> Path:
    """Return the directory holding ``_psoul_helper.py`` for PYTHONPATH injection."""
    helper_init = psoul.helper.__file__
    if helper_init is None:
        msg = "psoul.helper has no __file__"
        raise RuntimeError(msg)
    return Path(helper_init).resolve().parent


def _check_cwd_collision(cwd: Path) -> None:
    """Raise ``ValueError`` if a ``_psoul_helper`` shadow lives in ``cwd``.

    With ``python -m _psoul_helper``, ``sys.path[0]`` is ``""`` (CWD of the spawned child), so any
    ``_psoul_helper.py`` file or ``_psoul_helper/`` directory in the user-target cwd would shadow our
    wrapper before the prepended PYTHONPATH entry is consulted. Both file and directory shapes
    (regular package, namespace package) are rejected. Caller passes ``request.cwd`` so the check
    targets the spawned child's CWD, not the supervisor's.
    """
    file_shadow = cwd / f"{_HELPER_MODULE_NAME}.py"
    dir_shadow = cwd / _HELPER_MODULE_NAME
    if file_shadow.exists():
        msg = f"{file_shadow} would shadow the psoul helper wrapper. Rename it before running with --record."
        raise ValueError(msg)
    if dir_shadow.is_dir():
        msg = f"{dir_shadow} would shadow the psoul helper wrapper. Rename it before running with --record."
        raise ValueError(msg)


def _build_wrapper_argv(target_argv: Sequence[str]) -> list[str]:
    """Prepend the wrapper invocation in front of the user's argv.

    ``target_argv`` is the full python command line (``[python, target.py, *args]`` for script mode
    or ``[python, "-m", module, *args]`` for module mode). The result inserts ``-m _psoul_helper``
    between the interpreter and the user-intended argv so the wrapper imports as ``__main__`` before
    user code runs.
    """
    if not target_argv:
        msg = "target_argv must include the python interpreter"
        raise ValueError(msg)
    python, *rest = target_argv
    return [python, "-m", _HELPER_MODULE_NAME, *rest]


def _build_helper_env(parent_env: Mapping[str, str], helper_fd: int) -> dict[str, str]:
    """Return a copy of ``parent_env`` with PSOUL_HELPER_PIPE set and PYTHONPATH prepended.

    The helper-package directory is prepended (not appended) so a user-set PYTHONPATH cannot shadow
    our wrapper. CWD shadowing is handled separately by :func:`_check_cwd_collision`.
    """
    env = dict(parent_env)
    env[PSOUL_HELPER_PIPE_ENV] = str(helper_fd)
    helper_dir = str(_helper_package_dir())
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{helper_dir}{os.pathsep}{existing_path}" if existing_path else helper_dir
    return env


@dataclass(frozen=True, slots=True)
class LaunchTarget:
    """Validated target to run: script path or module name plus arguments."""

    target_type: TargetType
    target: str
    target_args: tuple[str, ...]
    python_path: Path

    def as_cmd(self) -> list[str]:
        """Build the subprocess command list."""
        python = str(self.python_path)
        prefix = [python, "-m", self.target] if self.target_type == TargetType.module else [python, self.target]
        return [*prefix, *self.target_args]


def parse_launch_target(
    *, target: str | None, module: str | None, extra_args: Sequence[str], python_path: Path
) -> LaunchTarget:
    """Build a LaunchTarget from mutually exclusive CLI inputs.

    Exactly one of *target* or *module* must be provided.

    Args:
        target (str | None): Script path.
        module (str | None): Module name for ``-m`` invocation.
        extra_args (Sequence[str]): Additional arguments passed to the target.
        python_path (Path): Interpreter the target should run under.

    Returns:
        LaunchTarget: Validated target with its argument tuple.

    Raises:
        ValueError: Both or neither of *target* and *module* are set.

    """
    if target is not None and module is not None:
        raise ValueError("choose either a script target or -m module")
    args = tuple(extra_args)
    if module is not None:
        return LaunchTarget(target_type=TargetType.module, target=module, target_args=args, python_path=python_path)
    if target is None:
        raise ValueError("launch target is required")
    return LaunchTarget(target_type=TargetType.script, target=target, target_args=args, python_path=python_path)


def resolve_session_id(name: str | None) -> str:
    """Return a validated session ID from ``--name``, or generate one.

    Args:
        name (str | None): Explicit name from the CLI flag, or ``None``
            to auto-generate a four-word ID.

    Returns:
        str: A valid session ID.

    Raises:
        ValueError: *name* is provided but fails validation.

    """
    if name is not None:
        return validate_session_id(name)
    return generate_session_id()


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    """Frozen snapshot of everything needed to create a session."""

    session_id: str
    launch_mode: LaunchMode
    target: LaunchTarget
    cwd: Path
    record_requested: bool = False
    tags: Mapping[str, str] | None = None


def build_launch_request(
    *,
    target: str | None,
    module: str | None,
    extra_args: Sequence[str],
    name: str | None,
    headless: bool,
    tags: dict[str, str] | None,
    python_path: Path,
    default_mode: LaunchMode,
    record: bool = False,
) -> LaunchRequest:
    """Assemble a frozen LaunchRequest from CLI inputs.

    The launch mode is headless when ``--headless`` is set. Otherwise,
    *default_mode* applies. Recording is requested when *record* is True
    or the resolved launch mode is headless (since headless mode requires
    persistence to support async query).

    Args:
        target (str | None): Script path.
        module (str | None): Module name for ``-m`` invocation.
        extra_args (Sequence[str]): Additional arguments for the target.
        name (str | None): Explicit session ID, or ``None`` to auto-generate.
        headless (bool): Force headless launch mode.
        record (bool): User-driven persistence flag (CLI ``--record`` or
            ``[session] record = true`` in config). Headless launches
            imply recording regardless of this value.
        tags (dict[str, str] | None): Session tags from ``--tag`` flags.
        python_path (Path): Python interpreter that runs the target.
        default_mode (LaunchMode): Mode used when ``--headless`` is unset.

    Returns:
        LaunchRequest: Frozen snapshot of everything needed to create a session.

    """
    launch_target = parse_launch_target(target=target, module=module, extra_args=extra_args, python_path=python_path)
    launch_mode = LaunchMode.headless if headless else default_mode
    record_requested = record or launch_mode == LaunchMode.headless
    return LaunchRequest(
        session_id=resolve_session_id(name),
        launch_mode=launch_mode,
        target=launch_target,
        cwd=Path.cwd(),
        record_requested=record_requested,
        tags=MappingProxyType(dict(tags)) if tags is not None else None,
    )


def _create_session(request: LaunchRequest, store: SessionStore) -> Session:
    """Persist a new session in starting state and return it."""
    provenance = gather(request.target.target_type, request.target.target, request.cwd, request.target.python_path)
    session = Session(
        session_id=request.session_id,
        state=SessionState.starting,
        launch_mode=request.launch_mode,
        launch_time=datetime.now(UTC),
        psoul_version=VERSION,
        target_type=request.target.target_type,
        target=request.target.target,
        target_args=list(request.target.target_args),
        target_cwd=request.cwd,
        tags=dict(request.tags) if request.tags is not None else None,
        **provenance,
    )
    return store.create(session)


def launch_headless(
    request: LaunchRequest,
    store: SessionStore,
    state_dir: Path,
    *,
    stop_timeout_seconds: float = 10.0,
) -> tuple[Session, int]:
    """Fork a background supervisor and return the session in starting state.

    The forked child becomes the supervisor — it outlives the CLI
    process and monitors the user's script until it exits.

    Args:
        request (LaunchRequest): Resolved target, session ID, and launch options.
        store (SessionStore): Store for persisting the session.
        state_dir (Path): State directory for the supervisor's database connection.
        stop_timeout_seconds (float): Seconds between SIGTERM delivery and the
            SIGKILL escalation when a stop command is in flight. Default
            10.0 matches the ``process.stop_timeout = "10s"`` config default.

    Returns:
        tuple[Session, int]: The new session and the supervisor's PID.

    Raises:
        NotImplementedError: Platform does not support ``os.fork`` (Windows).

    """
    if not hasattr(os, "fork"):
        msg = "headless mode requires Unix (macOS/Linux)"
        raise NotImplementedError(msg)
    session = _create_session(request, store)
    child_pid = os.fork()
    if child_pid == 0:
        try:
            os.setsid()
            store.conn.close()
            _supervise(
                argv=request.target.as_cmd(),
                cwd=request.cwd,
                state_dir=state_dir,
                session_id=request.session_id,
                stop_timeout_seconds=stop_timeout_seconds,
            )
        except BaseException:  # noqa: BLE001 — fork-safety: never let the child re-enter parent context
            with contextlib.suppress(BaseException):
                traceback.print_exc()
            os._exit(1)
        os._exit(0)
    return session, child_pid


def launch_attached(
    request: LaunchRequest,
    store: SessionStore,
    state_dir: Path,
    *,
    helper_connect_timeout_seconds: float = 5.0,
) -> Session:
    """Spawn a process with inherited stdio and wait for it to exit.

    The CLI process itself acts as the supervisor in attached mode. On Unix when the launch is
    recorded (``request.record_requested=True``) the supervisor wraps the user's argv with the
    ``_psoul_helper`` injection wrapper, opens a socketpair for the helper transport, runs the
    capabilities exchange, and starts a daemon watcher thread that emits crash events on EOF.
    On Windows the helper transport is not yet implemented, so recorded launches still persist
    the session but skip helper plumbing.

    Args:
        request (LaunchRequest): Resolved target, session ID, and launch options.
        store (SessionStore): Store for persisting the session.
        state_dir (Path): State directory for the watcher's per-thread DB connection.
        helper_connect_timeout_seconds (float): Total seconds budget for the
            capabilities readiness exchange. Default 5.0 matches the
            ``[helper] connect_timeout = "5s"`` config default.

    Returns:
        Session: The completed session after the process has exited.

    """
    use_helper = request.record_requested and _HELPER_SUPPORTED
    if use_helper:
        _check_cwd_collision(request.cwd)
    _create_session(request, store)
    if use_helper:
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        argv = _build_wrapper_argv(request.target.as_cmd())
        env = _build_helper_env(os.environ, child_sock.fileno())
        proc = subprocess.Popen(  # noqa: S603
            argv,
            cwd=request.cwd,
            env=env,
            pass_fds=[child_sock.fileno()],
        )
        child_sock.close()
        store.update(request.session_id, state=SessionState.running, supervisor_pid=os.getpid())
        lifecycle = HelperLifecycle(HelperTransport(UnixHelperPipeAdapter(parent_sock)))
        capabilities = lifecycle.request_capabilities(
            timeout_ms=int(helper_connect_timeout_seconds * _MS_PER_SECOND),
        )
        if capabilities is None:
            EventStore(store.conn).append(
                session_id=request.session_id,
                event_type=EVENT_HELPER_TIMEOUT,
                payload=None,
                generation=0,
            )
            lifecycle.close()
        else:
            store.update(request.session_id, helper_pid=proc.pid, helper_capabilities=capabilities)
            threading.Thread(
                target=_watch_helper_eof,
                args=(state_dir, request.session_id, parent_sock, lifecycle, proc),
                daemon=True,
            ).start()
    else:
        proc = subprocess.Popen(request.target.as_cmd(), cwd=request.cwd)  # noqa: S603
        store.update(request.session_id, state=SessionState.running, supervisor_pid=os.getpid())
    wait_for_exit(request.session_id, proc, store)
    return store.update(request.session_id, supervisor_pid=None)


def _watch_helper_eof(
    state_dir: Path,
    session_id: str,
    helper_sock: socket.socket,
    lifecycle: HelperLifecycle,
    proc: subprocess.Popen[bytes],
) -> None:
    """Daemon thread: watch the helper socket for EOF and emit lifecycle events.

    Opens its own SQLite connection because the main thread's ``SessionStore`` is bound to its own
    connection (sqlite ``check_same_thread=True``). Closes the lifecycle (and the underlying socket)
    on exit via :func:`contextlib.closing`.
    """
    with closing(lifecycle):
        while proc.poll() is None:
            ready, _, _ = select.select([helper_sock], [], [], _HELPER_WATCHER_POLL_INTERVAL_SECONDS)
            if helper_sock in ready:
                try:
                    data = helper_sock.recv(1, socket.MSG_PEEK)
                except OSError:
                    break
                if not data:
                    break
                break
        child_alive = proc.poll() is None
        with closing(open_db(state_dir, create=False)) as conn:
            SessionStore(conn).update(session_id, helper_pid=None, helper_capabilities=None)
            ev_store = EventStore(conn)
            lifecycle.emit_for_eof(
                child_alive=child_alive,
                event_writer=lambda event_type, payload: ev_store.append(
                    session_id=session_id,
                    event_type=event_type,
                    payload=payload,
                    generation=0,
                ),
            )


def wait_for_exit(
    session_id: str,
    proc: subprocess.Popen[bytes],
    store: SessionStore,
) -> Session:
    """Block until the process exits, then record the result and finalize the session.

    Used by ``launch_attached``, where the CLI process itself supervises
    a child spawned with inherited stdio. The duration timer covers the
    full ``proc.wait()`` window.

    Args:
        session_id (str): Session that owns this process.
        proc (subprocess.Popen[bytes]): The running process to wait on.
        store (SessionStore): Store for recording the result and updating state.

    Returns:
        Session: The completed session after the process has exited.

    """
    start = time.monotonic()
    final_session: Session | None = None
    try:
        proc.wait()
    finally:
        duration = time.monotonic() - start
        outcome = "exited" if proc.returncode is not None and proc.returncode == 0 else "failed"
        final_state = SessionState.exited if proc.returncode == 0 else SessionState.failed
        try:
            store.record_result(
                session_id=session_id,
                outcome=outcome,
                exit_code=proc.returncode,
                end_time=datetime.now(UTC),
                duration_seconds=duration,
            )
        finally:
            store.update(session_id, state=SessionState.stopping)
            final_session = store.update(session_id, state=final_state)
    if final_session is None:  # pragma: no cover
        msg = f"session finalization failed: {session_id}"
        raise RuntimeError(msg)
    return final_session
