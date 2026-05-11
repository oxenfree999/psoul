"""Session store: CRUD operations against the SQLite sessions table."""

import json
import sqlite3
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from psoul.core.session import (
    LaunchMode,
    Session,
    SessionState,
    TargetType,
    check_transition,
)

_FIELD_TYPES: dict[str, type] = {
    "state": SessionState,
    "launch_mode": LaunchMode,
    "target_type": TargetType,
    "target": str,
    "tags": dict,
    "helper_capabilities": dict,
    "sandbox_policy": dict,
    "target_args": list,
    "config_sources": list,
    "target_cwd": Path,
    "python_path": Path,
    "socket_path": Path,
    "launch_time": datetime,
    "control_acquired_at": datetime,
    "python_version": str,
    "uv_version": str,
    "resolved_by": str,
    "psoul_version": str,
    "protocol_version": int,
    "host": str,
    "os": str,
    "arch": str,
    "git_sha": str,
    "git_dirty": bool,
    "lockfile_hash": str,
    "script_hash": str,
    "supervisor_pid": int,
    "helper_pid": int,
    "generation": int,
    "control_epoch": int,
    "controller_pid": int,
    "sandbox_backend": str,
}

_JSON_FIELDS = frozenset({"tags", "helper_capabilities", "sandbox_policy", "target_args", "config_sources"})

_COLUMNS = (
    "tags",
    "state",
    "launch_mode",
    "launch_time",
    "target_type",
    "target",
    "target_args",
    "target_cwd",
    "python_version",
    "python_path",
    "uv_version",
    "resolved_by",
    "psoul_version",
    "protocol_version",
    "host",
    "os",
    "arch",
    "config_sources",
    "git_sha",
    "git_dirty",
    "lockfile_hash",
    "script_hash",
    "supervisor_pid",
    "socket_path",
    "helper_pid",
    "helper_capabilities",
    "generation",
    "control_epoch",
    "controller_pid",
    "control_acquired_at",
    "sandbox_backend",
    "sandbox_policy",
)

_INSERT_SQL = (
    f"INSERT INTO sessions (session_id, {', '.join(_COLUMNS)}) "  # noqa: S608
    f"VALUES (:session_id, {', '.join(f':{c}' for c in _COLUMNS)})"
)

_ALL_FIELDS = frozenset(_COLUMNS) | {"session_id"}
_IMMUTABLE_FIELDS = frozenset(
    {
        "session_id",
        "launch_time",
        "launch_mode",
        "psoul_version",
        "target_type",
        "target",
        "target_args",
        "target_cwd",
        "python_version",
        "python_path",
        "uv_version",
        "resolved_by",
        "protocol_version",
        "host",
        "os",
        "arch",
        "config_sources",
        "git_sha",
        "git_dirty",
        "lockfile_hash",
        "script_hash",
    }
)


def _serialize(session: Session) -> dict[str, object]:
    """Convert a Session to a dict of SQLite-compatible values."""
    raw = {
        "session_id": session.session_id,
        "tags": session.tags,
        "state": session.state,
        "launch_mode": session.launch_mode,
        "launch_time": session.launch_time,
        "target_type": session.target_type,
        "target": session.target,
        "target_args": session.target_args,
        "target_cwd": session.target_cwd,
        "python_version": session.python_version,
        "python_path": session.python_path,
        "uv_version": session.uv_version,
        "resolved_by": session.resolved_by,
        "psoul_version": session.psoul_version,
        "protocol_version": session.protocol_version,
        "host": session.host,
        "os": session.os,
        "arch": session.arch,
        "config_sources": session.config_sources,
        "git_sha": session.git_sha,
        "git_dirty": session.git_dirty,
        "lockfile_hash": session.lockfile_hash,
        "script_hash": session.script_hash,
        "supervisor_pid": session.supervisor_pid,
        "socket_path": session.socket_path,
        "helper_pid": session.helper_pid,
        "helper_capabilities": session.helper_capabilities,
        "generation": session.generation,
        "control_epoch": session.control_epoch,
        "controller_pid": session.controller_pid,
        "control_acquired_at": session.control_acquired_at,
        "sandbox_backend": session.sandbox_backend,
        "sandbox_policy": session.sandbox_policy,
    }
    return {key: _serialize_value(key, value) for key, value in raw.items()}


def _validate_json_shape(key: str, value: object) -> None:
    """Validate JSON-backed field contents against the Session model."""
    if key in {"target_args", "config_sources"}:
        if not isinstance(value, list):
            msg = f"{key} must be list, got {type(value).__name__}"
            raise TypeError(msg)
        if not all(isinstance(item, str) for item in value):
            msg = f"{key} must be list[str]"
            raise TypeError(msg)
        return
    if key == "tags":
        if not isinstance(value, dict):
            msg = f"{key} must be dict, got {type(value).__name__}"
            raise TypeError(msg)
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            msg = "tags must be dict[str, str]"
            raise TypeError(msg)
        return
    if key in {"helper_capabilities", "sandbox_policy"}:
        if not isinstance(value, dict):
            msg = f"{key} must be dict, got {type(value).__name__}"
            raise TypeError(msg)
        if all(isinstance(k, str) for k in value):
            return
        msg = f"{key} must be dict[str, object]"
        raise TypeError(msg)


def _validate_field_type(key: str, value: object) -> None:
    """Validate a field against its declared runtime type."""
    expected = _FIELD_TYPES.get(key)
    if expected is None:
        return
    if expected is int and type(value) is not int:
        msg = f"{key} must be int, got {type(value).__name__}"
        raise TypeError(msg)
    if expected is bool and type(value) is not bool:
        msg = f"{key} must be bool, got {type(value).__name__}"
        raise TypeError(msg)
    if expected not in {int, bool} and not isinstance(value, expected):
        msg = f"{key} must be {expected.__name__}, got {type(value).__name__}"
        raise TypeError(msg)


def _encode_value(key: str, value: object) -> object:
    """Encode a validated Python value for SQLite storage."""
    if isinstance(value, StrEnum):
        return value.value
    if key in _JSON_FIELDS:
        _validate_json_shape(key, value)
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _serialize_value(key: str, value: object) -> object:
    """Serialize a single field value for SQLite storage."""
    if value is None:
        return None
    _validate_field_type(key, value)
    return _encode_value(key, value)


def _from_json(value: str | None) -> object:
    """Deserialize a JSON string, or return None."""
    if value is None:
        return None
    return json.loads(value)


def _opt_path(value: object) -> Path | None:
    """Convert a non-None value to a Path, or return None."""
    return Path(str(value)) if value is not None else None


def _opt_dt(value: object) -> datetime | None:
    """Parse an ISO 8601 string to datetime, or return None."""
    return datetime.fromisoformat(str(value)) if value is not None else None


def _opt_int(value: object) -> int | None:
    """Convert a non-None value to int, or return None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(str(value))


def _deserialize(row: sqlite3.Row) -> Session:
    """Convert a SQLite row to a Session."""
    return Session(
        session_id=str(row["session_id"]),
        state=SessionState(str(row["state"])),
        launch_mode=LaunchMode(str(row["launch_mode"])),
        launch_time=datetime.fromisoformat(str(row["launch_time"])),
        psoul_version=str(row["psoul_version"]),
        generation=int(row["generation"]),
        control_epoch=int(row["control_epoch"]),
        target_type=TargetType(str(row["target_type"])),
        target=str(row["target"]) if row["target"] is not None else None,
        target_args=_from_json(row["target_args"]),  # ty: ignore[invalid-argument-type]
        target_cwd=_opt_path(row["target_cwd"]),
        protocol_version=int(row["protocol_version"]),
        python_version=str(row["python_version"]) if row["python_version"] is not None else None,
        python_path=_opt_path(row["python_path"]),
        uv_version=str(row["uv_version"]) if row["uv_version"] is not None else None,
        resolved_by=str(row["resolved_by"]) if row["resolved_by"] is not None else None,
        host=str(row["host"]) if row["host"] is not None else None,
        os=str(row["os"]) if row["os"] is not None else None,
        arch=str(row["arch"]) if row["arch"] is not None else None,
        config_sources=_from_json(row["config_sources"]),  # ty: ignore[invalid-argument-type]
        git_sha=str(row["git_sha"]) if row["git_sha"] is not None else None,
        git_dirty=bool(row["git_dirty"]) if row["git_dirty"] is not None else None,
        lockfile_hash=str(row["lockfile_hash"]) if row["lockfile_hash"] is not None else None,
        script_hash=str(row["script_hash"]) if row["script_hash"] is not None else None,
        supervisor_pid=_opt_int(row["supervisor_pid"]),
        socket_path=_opt_path(row["socket_path"]),
        helper_pid=_opt_int(row["helper_pid"]),
        helper_capabilities=_from_json(row["helper_capabilities"]),  # ty: ignore[invalid-argument-type]
        controller_pid=_opt_int(row["controller_pid"]),
        control_acquired_at=_opt_dt(row["control_acquired_at"]),
        sandbox_backend=str(row["sandbox_backend"]) if row["sandbox_backend"] is not None else None,
        sandbox_policy=_from_json(row["sandbox_policy"]),  # ty: ignore[invalid-argument-type]
        tags=_from_json(row["tags"]),  # ty: ignore[invalid-argument-type]
    )


class SessionStore:
    """CRUD operations for sessions against the SQLite sessions table.

    Wraps an open ``sqlite3.Connection`` and handles serialization,
    deserialization, state-transition validation, and result recording.
    All writes commit immediately.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Wrap an existing database connection."""
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    def create(self, session: Session) -> Session:
        """Insert a new session row and commit.

        Args:
            session (Session): Fully populated session to persist.

        Returns:
            Session: The same *session* object, confirming it was stored.

        Raises:
            sqlite3.IntegrityError: A session with this ID already exists.

        Example:
            >>> store.create(session)
            Session(session_id='calm-otter-builds-kites', ...)

        """
        row = _serialize(session)
        self.conn.execute(_INSERT_SQL, row)
        self.conn.commit()
        return session

    def get(self, session_id: str) -> Session | None:
        """Look up a session by its exact ID.

        Args:
            session_id (str): The session ID to look up.

        Returns:
            Session | None: The matching session, or ``None`` if not found.

        Example:
            >>> store.get("calm-otter-builds-kites")
            Session(session_id='calm-otter-builds-kites', ...)
            >>> store.get("nonexistent")  # returns None

        """
        row = self.conn.execute("SELECT * FROM sessions WHERE session_id = ?", [session_id]).fetchone()
        if row is None:
            return None
        return _deserialize(row)

    def list(self, *, state: SessionState | None = None, tags: dict[str, str] | None = None) -> list[Session]:
        """Return sessions ordered by launch_time descending.

        When *tags* is provided, only sessions whose tags contain **all**
        of the given key=value pairs are returned.

        Args:
            state (SessionState | None): Only return sessions in this state,
                or ``None`` for all states.
            tags (dict[str, str] | None): Only return sessions matching all
                of these tags, or ``None`` to skip tag filtering.

        Returns:
            list[Session]: Matching sessions, newest first.

        Examples:
            >>> store.list()  # all sessions
            >>> store.list(state=SessionState.running)
            >>> store.list(tags={"env": "dev"})

        """
        if state is not None:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE state = ? ORDER BY launch_time DESC",
                [state.value],
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM sessions ORDER BY launch_time DESC").fetchall()
        sessions = [_deserialize(row) for row in rows]
        if tags is not None:
            filter_items = tags.items()
            sessions = [s for s in sessions if s.tags is not None and filter_items <= s.tags.items()]
        return sessions

    def update(self, session_id: str, **fields: object) -> Session:
        """Update mutable fields on an existing session.

        State changes are validated against the lifecycle transition map.
        Domain types like ``Path`` and ``datetime`` are serialized automatically.

        Args:
            session_id (str): Session to update.
            **fields: Field names and new values.

        Returns:
            Session: The updated session as freshly read from the database.

        Raises:
            KeyError: Session does not exist.
            ValueError: No fields given, unknown field name, or immutable
                field included.

        Example:
            >>> store.update("calm-otter-builds-kites", state=SessionState.running)
            Session(session_id='calm-otter-builds-kites', state=<SessionState.running: 'running'>, ...)

        """
        if not fields:
            msg = "update requires at least one field"
            raise ValueError(msg)
        unknown = set(fields) - _ALL_FIELDS
        if unknown:
            msg = f"unknown fields: {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        immutable = set(fields) & _IMMUTABLE_FIELDS
        if immutable:
            msg = f"immutable fields: {', '.join(sorted(immutable))}"
            raise ValueError(msg)
        current = self.get(session_id)
        if current is None:
            msg = f"session not found: {session_id}"
            raise KeyError(msg)
        if "state" in fields:
            check_transition(current.state, SessionState(str(fields["state"])))
        serialized = {k: _serialize_value(k, v) for k, v in fields.items()}
        sets = ", ".join(f"{k} = ?" for k in serialized)
        self.conn.execute(
            f"UPDATE sessions SET {sets} WHERE session_id = ?",  # noqa: S608
            [*serialized.values(), session_id],
        )
        self.conn.commit()
        updated = self.get(session_id)
        if updated is None:  # pragma: no cover
            msg = f"session vanished after update: {session_id}"
            raise RuntimeError(msg)
        return updated

    def delete(self, session_id: str) -> None:
        """Remove a session row and let the schema cascade delete dependent rows.

        ``events``, ``results``, ``commands``, ``artifacts``,
        ``resource_samples``, and ``profiling_state`` are removed via
        ``ON DELETE CASCADE``.

        Args:
            session_id (str): Session to delete.

        Raises:
            KeyError: Session does not exist.

        Example:
            >>> store.delete("calm-otter-builds-kites")

        """
        cursor = self.conn.execute("DELETE FROM sessions WHERE session_id = ?", [session_id])
        if cursor.rowcount == 0:
            msg = f"session not found: {session_id}"
            raise KeyError(msg)
        self.conn.commit()

    def record_result(
        self,
        session_id: str,
        *,
        outcome: str,
        exit_code: int | None,
        end_time: datetime,
        duration_seconds: float,
    ) -> None:
        """Insert a result row for the session's current generation.

        Args:
            session_id (str): Session this result belongs to.
            outcome (str): How the session ended, e.g. ``"exited"`` or ``"failed"``.
            exit_code (int | None): Process exit code, or ``None`` if not captured.
            end_time (datetime): When the session finished.
            duration_seconds (float): Wall-clock duration of the session.

        Raises:
            KeyError: Session does not exist.

        Example:
            >>> store.record_result(
            ...     "calm-otter-builds-kites",
            ...     outcome="exited",
            ...     exit_code=0,
            ...     end_time=datetime.now(UTC),
            ...     duration_seconds=12.5,
            ... )

        """
        session = self.get(session_id)
        if session is None:
            msg = f"session not found: {session_id}"
            raise KeyError(msg)
        self.conn.execute(
            "INSERT INTO results (session_id, generation, outcome, exit_code, end_time, duration_seconds)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, session.generation, outcome, exit_code, end_time.isoformat(), duration_seconds),
        )
        self.conn.commit()
