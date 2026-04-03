import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from psoul.db import open_db
from psoul.names import generate_session_id
from psoul.provenance import gather
from psoul.session import LaunchMode, Session, SessionState, TargetType, validate_session_id
from psoul.store import SessionStore
from psoul.version import VERSION


@dataclass(frozen=True, slots=True)
class LaunchTarget:
    """Validated target to run: script path or module name plus arguments."""

    target_type: TargetType
    target: str
    target_args: tuple[str, ...]

    def as_cmd(self) -> list[str]:
        """Build the subprocess command list."""
        prefix = (
            [sys.executable, "-m", self.target]
            if self.target_type == TargetType.module
            else [sys.executable, self.target]
        )
        return [*prefix, *self.target_args]


def parse_launch_target(*, target: str | None, module: str | None, extra_args: Sequence[str]) -> LaunchTarget:
    """Build a LaunchTarget from mutually exclusive CLI inputs."""
    if target is not None and module is not None:
        raise ValueError("choose either a script target or -m module")
    if module is not None:
        return LaunchTarget(target_type=TargetType.module, target=module, target_args=tuple(extra_args))
    if target is None:
        raise ValueError("launch target is required")
    return LaunchTarget(target_type=TargetType.script, target=target, target_args=tuple(extra_args))


def resolve_session_id(name: str | None) -> str:
    """Return a validated session ID from --name, or generate one."""
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
) -> LaunchRequest:
    """Assemble a frozen LaunchRequest from CLI inputs."""
    return LaunchRequest(
        session_id=resolve_session_id(name),
        launch_mode=LaunchMode.headless if headless or not sys.stdin.isatty() else LaunchMode.attached,
        target=parse_launch_target(target=target, module=module, extra_args=extra_args),
        cwd=Path.cwd(),
        tags=MappingProxyType(dict(tags)) if tags is not None else None,
    )


def _create_session(request: LaunchRequest, store: SessionStore) -> Session:
    """Persist a new session in starting state and return it."""
    provenance = gather(request.target.target_type, request.target.target, request.cwd)
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
    """Fork a background supervisor and return the session in starting state."""
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


def _supervise(request: LaunchRequest, state_dir: Path) -> None:
    """Background supervisor: spawn target, wait, update session state."""
    conn = open_db(state_dir)
    try:
        sup_store = SessionStore(conn)
        proc = subprocess.Popen(  # noqa: S603
            request.target.as_cmd(),
            cwd=request.cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sup_store.update(request.session_id, state=SessionState.running, supervisor_pid=os.getpid())
        wait_for_exit(request.session_id, proc, sup_store)
    finally:
        conn.close()


def launch_attached(request: LaunchRequest, store: SessionStore) -> Session:
    """Spawn a process with inherited stdio and wait for it to exit."""
    _create_session(request, store)
    proc = subprocess.Popen(request.target.as_cmd(), cwd=request.cwd)  # noqa: S603
    store.update(request.session_id, state=SessionState.running, supervisor_pid=os.getpid())
    return wait_for_exit(request.session_id, proc, store)


def wait_for_exit(session_id: str, proc: subprocess.Popen[bytes], store: SessionStore) -> Session:
    """Block until the process exits, then update the session to its final state."""
    proc.wait()
    store.update(session_id, state=SessionState.stopping)
    return store.update(session_id, state=SessionState.exited if proc.returncode == 0 else SessionState.failed)
