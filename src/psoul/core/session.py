"""Session domain model: identity, state, provenance, and validation."""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

SESSION_ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SESSION_ID_MAX_LENGTH = 64


class SessionError(Exception):
    """Domain error for invalid session operations."""


class SessionState(StrEnum):
    """Session lifecycle states from the core.md state machine."""

    starting = "starting"
    running = "running"
    suspended = "suspended"
    debugging = "debugging"
    restarting = "restarting"
    stopping = "stopping"
    exited = "exited"
    failed = "failed"
    orphaned = "orphaned"


class LaunchMode(StrEnum):
    """How the session was started."""

    attached = "attached"
    headless = "headless"


class TargetType(StrEnum):
    """What the session is running."""

    script = "script"
    module = "module"


TERMINAL_STATES: frozenset[SessionState] = frozenset({SessionState.exited, SessionState.failed})

VALID_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.starting: frozenset({SessionState.running, SessionState.failed}),
    SessionState.running: frozenset(
        {
            SessionState.suspended,
            SessionState.debugging,
            SessionState.restarting,
            SessionState.stopping,
            SessionState.failed,
            SessionState.orphaned,
        }
    ),
    SessionState.suspended: frozenset(
        {SessionState.running, SessionState.restarting, SessionState.stopping, SessionState.failed}
    ),
    SessionState.debugging: frozenset({SessionState.running, SessionState.stopping, SessionState.failed}),
    SessionState.restarting: frozenset({SessionState.starting, SessionState.failed}),
    SessionState.stopping: frozenset({SessionState.exited, SessionState.failed}),
    SessionState.exited: frozenset(),
    SessionState.failed: frozenset(),
    SessionState.orphaned: frozenset(
        {
            SessionState.running,
            SessionState.stopping,
            SessionState.exited,
            SessionState.failed,
        }
    ),
}


def check_transition(from_state: SessionState, to_state: SessionState) -> None:
    """Enforce the session lifecycle state machine.

    Args:
        from_state (SessionState): Current session state.
        to_state (SessionState): Requested next state.

    Raises:
        SessionError: If *to_state* is not in ``VALID_TRANSITIONS[from_state]``.

    """
    allowed = VALID_TRANSITIONS[from_state]
    if to_state not in allowed:
        msg = f"invalid state transition: {from_state} -> {to_state}"
        raise SessionError(msg)


def validate_session_id(session_id: str) -> str:
    """Validate a session ID and return it unchanged.

    Session IDs are the single canonical identifier used everywhere: CLI
    arguments, file paths, URLs, tab completion, and shell scripts.  We
    intentionally restrict to lowercase-hyphen (``[a-z0-9]+(-[a-z0-9]+)*``)
    rather than the broader ``[a-zA-Z0-9]+([_.-][a-zA-Z0-9]+)*`` pattern in
    core.md.  This keeps IDs unambiguous in case-insensitive filesystems,
    avoids shell-quoting surprises, and makes tab completion predictable.

    Args:
        session_id (str): The ID to validate.

    Returns:
        str: The unchanged *session_id* if valid.

    Raises:
        ValueError: If empty, exceeds 64 characters, or contains invalid characters.

    """
    if not session_id:
        msg = "session ID must not be empty"
        raise ValueError(msg)
    if len(session_id) > SESSION_ID_MAX_LENGTH:
        msg = f"session ID exceeds {SESSION_ID_MAX_LENGTH} characters: {session_id!r}"
        raise ValueError(msg)
    if not SESSION_ID_PATTERN.match(session_id):
        msg = f"session ID must be lowercase alphanumeric segments separated by hyphens: {session_id!r}"
        raise ValueError(msg)
    return session_id


@dataclass(frozen=True, slots=True)
class Session:
    """Domain model for a psoul session.

    An immutable snapshot of session state as stored in the database.
    Created by the launch path and updated through ``SessionStore``.
    Fields are grouped into identity, state, target, provenance,
    runtime, and tags.  The ``session_id`` is validated on construction.
    """

    # Identity
    session_id: str

    # State
    state: SessionState
    launch_mode: LaunchMode
    launch_time: datetime
    psoul_version: str
    generation: int = 0
    control_epoch: int = 0

    # Target
    target_type: TargetType = TargetType.script
    target: str | None = None
    target_args: list[str] | None = None
    target_cwd: Path | None = None

    # Provenance
    protocol_version: int = 1
    python_version: str | None = None
    python_path: Path | None = None
    uv_version: str | None = None
    resolved_by: str | None = None
    host: str | None = None
    os: str | None = None
    arch: str | None = None
    config_sources: list[str] | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    lockfile_hash: str | None = None
    script_hash: str | None = None

    # Runtime
    supervisor_pid: int | None = None
    socket_path: Path | None = None
    helper_pid: int | None = None
    helper_capabilities: dict[str, object] | None = None
    controller_pid: int | None = None
    control_acquired_at: datetime | None = None
    sandbox_backend: str | None = None
    sandbox_policy: dict[str, object] | None = None

    # Tags
    tags: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate session_id on construction."""
        validate_session_id(self.session_id)
