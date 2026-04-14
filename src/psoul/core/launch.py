"""Session launch: target parsing, request building, and process lifecycle."""

import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import psutil

from psoul.core.db import open_db
from psoul.core.events import EventStore
from psoul.core.names import generate_session_id
from psoul.core.output import drain_output
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

    The launch mode is headless when ``--headless`` is set or stdin is
    not a TTY; otherwise *default_mode* applies.

    Args:
        target (str | None): Script path.
        module (str | None): Module name for ``-m`` invocation.
        extra_args (Sequence[str]): Additional arguments for the target.
        name (str | None): Explicit session ID, or ``None`` to auto-generate.
        headless (bool): Force headless launch mode.
        tags (dict[str, str] | None): Session tags from ``--tag`` flags.
        python_path (Path): Python interpreter that runs the target.
        default_mode (LaunchMode): Mode used when ``--headless`` is unset and stdin is a TTY.

    Returns:
        LaunchRequest: Frozen snapshot of everything needed to create a session.

    """
    forced_headless = headless or not sys.stdin.isatty()
    launch_target = parse_launch_target(target=target, module=module, extra_args=extra_args, python_path=python_path)
    return LaunchRequest(
        session_id=resolve_session_id(name),
        launch_mode=LaunchMode.headless if forced_headless else default_mode,
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


def launch_headless(request: LaunchRequest, store: SessionStore, state_dir: Path) -> tuple[Session, int]:
    """Fork a background supervisor and return the session in starting state.

    The forked child becomes the supervisor — it outlives the CLI
    process and monitors the user's script until it exits.

    Args:
        request (LaunchRequest): Resolved target, session ID, and launch options.
        store (SessionStore): Store for persisting the session.
        state_dir (Path): State directory for the supervisor's database connection.

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
        _supervise(request, state_dir)
        os._exit(0)
    return session, child_pid


_SAMPLE_INTERVAL = 2.0  # seconds between resource samples


def _supervise(request: LaunchRequest, state_dir: Path) -> None:
    """Background supervisor: spawn target, capture output, sample resources, wait, update session state."""
    conn = open_db(state_dir)
    try:
        sup_store = SessionStore(conn)
        event_store = EventStore(conn)
        proc = subprocess.Popen(  # noqa: S603
            request.target.as_cmd(),
            cwd=request.cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        session = sup_store.update(request.session_id, state=SessionState.running, supervisor_pid=os.getpid())
        sampler: ResourceSampler | None = None
        sampler_thread: threading.Thread | None = None
        try:
            ps_process = psutil.Process(proc.pid)
            sampler = ResourceSampler(ps_process, state_dir, request.session_id, session.generation)
            sampler_thread = threading.Thread(target=sampler.run, args=(_SAMPLE_INTERVAL,), daemon=True)
            sampler_thread.start()
        except psutil.NoSuchProcess:
            pass  # child already exited; skip sampling, wait_for_exit still finalizes
        try:
            wait_for_exit(
                request.session_id,
                proc,
                sup_store,
                event_store=event_store,
                generation=session.generation,
            )
        finally:
            if sampler is not None:
                sampler.stop()
            if sampler_thread is not None:
                sampler_thread.join()
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
