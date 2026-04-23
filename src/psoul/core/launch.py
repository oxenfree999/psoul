"""Session launch: target parsing, request building, and process lifecycle."""

import os
import selectors
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import psutil

from psoul.core import control
from psoul.core.db import open_db
from psoul.core.events import EVENT_RUNTIME_STDERR, EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.names import generate_session_id
from psoul.core.output import READ_CHUNK_SIZE, drain_output
from psoul.core.provenance import gather
from psoul.core.resources import ResourceSampler
from psoul.core.session import LaunchMode, Session, SessionState, TargetType, validate_session_id
from psoul.core.store import SessionStore
from psoul.version import VERSION


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
) -> LaunchRequest:
    """Assemble a frozen LaunchRequest from CLI inputs.

    The launch mode is headless when ``--headless`` is set. Otherwise,
    *default_mode* applies.

    Args:
        target (str | None): Script path.
        module (str | None): Module name for ``-m`` invocation.
        extra_args (Sequence[str]): Additional arguments for the target.
        name (str | None): Explicit session ID, or ``None`` to auto-generate.
        headless (bool): Force headless launch mode.
        tags (dict[str, str] | None): Session tags from ``--tag`` flags.
        python_path (Path): Python interpreter that runs the target.
        default_mode (LaunchMode): Mode used when ``--headless`` is unset.

    Returns:
        LaunchRequest: Frozen snapshot of everything needed to create a session.

    """
    launch_target = parse_launch_target(target=target, module=module, extra_args=extra_args, python_path=python_path)
    return LaunchRequest(
        session_id=resolve_session_id(name),
        launch_mode=LaunchMode.headless if headless else default_mode,
        target=launch_target,
        cwd=Path.cwd(),
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
        os.setsid()
        store.conn.close()
        _supervise(request, state_dir, stop_timeout_seconds=stop_timeout_seconds)
        os._exit(0)
    return session, child_pid


_SAMPLE_INTERVAL = 2.0  # seconds between resource samples
_SUPERVISE_TICK = 0.1  # seconds per supervise-loop iteration
_RESPAWN_BACKOFFS: tuple[float, ...] = (1.0, 2.0, 4.0)  # sleep before each retry when a restart respawns the child


def _drain_tick(
    selector: selectors.DefaultSelector,
    event_store: EventStore,
    session_id: str,
    generation: int,
    timeout: float,
) -> bool:
    """Read one selector pass of stdio chunks into the event log.

    Returns ``True`` when at least one chunk was read. Unregisters a
    stream on EOF so the caller can detect pipe closure via
    ``selector.get_map()``.
    """
    any_read = False
    for key, _mask in selector.select(timeout=timeout):
        chunk = os.read(key.fd, READ_CHUNK_SIZE)
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        event_store.append(
            session_id=session_id,
            event_type=str(key.data),
            payload={"text": chunk.decode("utf-8", errors="replace")},
            generation=generation,
            commit=False,
        )
        any_read = True
    event_store.conn.commit()
    return any_read


def _poll_child_status(proc: subprocess.Popen[bytes]) -> str | None:
    """Return the child's status change as ``"stopped"``, ``"continued"``, ``"exited"``, or ``None``.

    On ``"exited"``, sets ``proc.returncode`` to the child's exit code.
    """
    try:
        wait_pid, sts = os.waitpid(proc.pid, os.WNOHANG | os.WUNTRACED | os.WCONTINUED)
    except ChildProcessError:
        return None
    if wait_pid == 0:
        return None
    if os.WIFSTOPPED(sts):
        return "stopped"
    if os.WIFCONTINUED(sts):
        return "continued"
    if os.WIFEXITED(sts):
        proc.returncode = os.WEXITSTATUS(sts)
    elif os.WIFSIGNALED(sts):
        proc.returncode = -os.WTERMSIG(sts)
    else:
        return None
    return "exited"


def _supervise_loop(
    proc: subprocess.Popen[bytes],
    *,
    session_id: str,
    event_store: EventStore,
    generation: int,
    store: SessionStore,
    control_state: control.ControlState,
    stop_timeout_seconds: float,
) -> None:
    """Tick-based supervise loop: drain stdio, dispatch signal flags, watch proc exit.

    Runs on the supervisor's main thread so signal handlers installed by
    ``control.install_handlers`` can flip request flags and have them
    serviced on the same thread each tick. After ``proc.poll()`` reports
    exit, any still-readable stdio is drained before returning.
    """
    with selectors.DefaultSelector() as selector:
        if proc.stdout is not None:
            selector.register(proc.stdout, selectors.EVENT_READ, EVENT_RUNTIME_STDOUT)
        if proc.stderr is not None:
            selector.register(proc.stderr, selectors.EVENT_READ, EVENT_RUNTIME_STDERR)
        while proc.poll() is None:
            _drain_tick(selector, event_store, session_id, generation, timeout=_SUPERVISE_TICK)
            status_change = _poll_child_status(proc)
            if status_change == "stopped":
                control.handle_pause_observed(event_store, session_id, generation, store)
            elif status_change == "continued":
                control.handle_resume_observed(event_store, session_id, generation, store)
            if control_state.stop_requested:
                control.handle_stop(
                    control_state, proc, event_store, session_id, generation, stop_timeout_seconds, store
                )
            if control_state.kill_requested:
                control.handle_kill(control_state, proc, event_store, session_id, generation, store)
            if control_state.restart_requested:
                control.handle_restart(
                    control_state, proc, event_store, session_id, generation, stop_timeout_seconds, store
                )
            control.check_escalation(control_state, proc)
        while selector.get_map() and _drain_tick(
            selector, event_store, session_id, generation, timeout=_SUPERVISE_TICK
        ):
            pass


def _finalize_exit(
    proc: subprocess.Popen[bytes],
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
    outcome = "exited" if proc.returncode is not None and proc.returncode == 0 else "failed"
    final_state = SessionState.exited if proc.returncode == 0 else SessionState.failed
    current = store.get(session_id)
    final: Session | None = None
    try:
        store.record_result(
            session_id=session_id,
            outcome=outcome,
            exit_code=proc.returncode,
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
    request: LaunchRequest,
    state_dir: Path,
    session_id: str,
    generation: int,
) -> tuple[subprocess.Popen[bytes], ResourceSampler | None, threading.Thread | None]:
    """Spawn the child in its own process group and prepare the ResourceSampler thread."""
    proc = subprocess.Popen(  # noqa: S603
        request.target.as_cmd(),
        cwd=request.cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    sampler: ResourceSampler | None = None
    sampler_thread: threading.Thread | None = None
    try:
        ps_process = psutil.Process(proc.pid)
        sampler = ResourceSampler(ps_process, state_dir, session_id, generation)
        sampler_thread = threading.Thread(target=sampler.run, args=(_SAMPLE_INTERVAL,), daemon=True)
    except psutil.NoSuchProcess:
        pass  # child already exited; skip sampling, supervise loop still finalizes
    return proc, sampler, sampler_thread


def _respawn_with_backoff(
    request: LaunchRequest,
    state_dir: Path,
    new_generation: int,
) -> tuple[subprocess.Popen[bytes], ResourceSampler | None, threading.Thread | None]:
    """Retry ``_spawn_generation`` with ``_RESPAWN_BACKOFFS`` between attempts.

    Re-raises the last ``OSError`` on exhaustion.
    """
    for attempt in range(len(_RESPAWN_BACKOFFS) + 1):
        try:
            return _spawn_generation(request, state_dir, request.session_id, generation=new_generation)
        except OSError:
            if attempt == len(_RESPAWN_BACKOFFS):
                raise
            time.sleep(_RESPAWN_BACKOFFS[attempt])
    msg = "unreachable: loop iterates once more than _RESPAWN_BACKOFFS length"
    raise RuntimeError(msg)


def _supervise(request: LaunchRequest, state_dir: Path, *, stop_timeout_seconds: float) -> None:
    """Background supervisor: spawn target, serve control signals, respawn across restarts, and finalize on exit.

    The outer ``while True:`` loop is the generation loop. Each iteration runs
    one supervise loop for one generation's child. On normal exit the outer
    loop finalizes the session and breaks. On restart, ``_finalize_exit`` is
    called with ``in_restart=True`` (record result without terminal
    transition), the row moves to ``restarting``, and ``_respawn_with_backoff``
    spawns generation N+1. ``advance_restart_on_spawn_success`` / ``..._failure``
    handle the boundary events and state transitions.
    """
    conn = open_db(state_dir)
    try:
        sup_store = SessionStore(conn)
        event_store = EventStore(conn)
        control_state = control.ControlState()
        control.install_handlers(control_state)
        generation = 0
        proc, sampler, sampler_thread = _spawn_generation(request, state_dir, request.session_id, generation=generation)
        sup_store.update(request.session_id, state=SessionState.running, supervisor_pid=os.getpid())
        while True:
            if sampler_thread is not None:
                sampler_thread.start()
            start_mono = time.monotonic()
            try:
                _supervise_loop(
                    proc,
                    session_id=request.session_id,
                    event_store=event_store,
                    generation=generation,
                    store=sup_store,
                    control_state=control_state,
                    stop_timeout_seconds=stop_timeout_seconds,
                )
                control.emit_terminal_events(control_state, event_store, request.session_id, generation)
            finally:
                if sampler is not None:
                    sampler.stop()
                if sampler_thread is not None:
                    sampler_thread.join()
            if control_state.restart_pending is None:
                _finalize_exit(proc, request.session_id, sup_store, start_mono)
                break
            _finalize_exit(proc, request.session_id, sup_store, start_mono, in_restart=True)
            sup_store.update(request.session_id, state=SessionState.restarting)
            new_generation = generation + 1
            try:
                proc, sampler, sampler_thread = _respawn_with_backoff(request, state_dir, new_generation)
            except OSError as exc:
                control.advance_restart_on_spawn_failure(
                    control_state,
                    sup_store,
                    event_store,
                    request.session_id,
                    generation,
                    "runtime_error",
                    str(exc),
                )
                break
            current = sup_store.get(request.session_id)
            if current is None:
                msg = f"session row disappeared mid-restart: {request.session_id}"
                raise RuntimeError(msg)
            control.advance_restart_on_spawn_success(
                control_state,
                sup_store,
                event_store,
                request.session_id,
                new_generation,
                current.control_epoch + 1,
            )
            generation = new_generation
        sup_store.update(request.session_id, supervisor_pid=None)
    finally:
        conn.close()


def launch_attached(request: LaunchRequest, store: SessionStore) -> Session:
    """Spawn a process with inherited stdio and wait for it to exit.

    The CLI process itself acts as the supervisor in attached mode.

    Args:
        request (LaunchRequest): Resolved target, session ID, and launch options.
        store (SessionStore): Store for persisting the session.

    Returns:
        Session: The completed session after the process has exited.

    """
    _create_session(request, store)
    proc = subprocess.Popen(request.target.as_cmd(), cwd=request.cwd)  # noqa: S603
    store.update(request.session_id, state=SessionState.running, supervisor_pid=os.getpid())
    wait_for_exit(request.session_id, proc, store)
    return store.update(request.session_id, supervisor_pid=None)


def wait_for_exit(
    session_id: str,
    proc: subprocess.Popen[bytes],
    store: SessionStore,
    *,
    event_store: EventStore | None = None,
    generation: int = 0,
) -> Session:
    """Block until the process exits, then record the result and finalize the session.

    When *event_store* is provided, stdout and stderr are drained into
    the event log before waiting on the process. The duration timer
    covers the full drain-and-wait window so headless sessions report
    realistic session durations, not just the trailing ``proc.wait()``.

    Args:
        session_id (str): Session that owns this process.
        proc (subprocess.Popen[bytes]): The running process to wait on.
        store (SessionStore): Store for recording the result and updating state.
        event_store (EventStore | None): When provided, drain *proc*'s
            stdout/stderr into this store before waiting. Only meaningful
            when *proc* was opened with ``stdout=PIPE`` / ``stderr=PIPE``.
        generation (int): Session generation for the drained events. Ignored
            when *event_store* is ``None``.

    Returns:
        Session: The completed session after the process has exited.

    """
    start = time.monotonic()
    final_session: Session | None = None
    try:
        if event_store is not None:
            drain_output(proc, session_id=session_id, event_store=event_store, generation=generation)
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
