"""Session store: CRUD operations against the SQLite sessions table."""

import json
import sqlite3
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from psoul.session import (
    LaunchMode,
    Session,
    SessionState,
    TargetType,
    check_transition,
)


def _to_json(value: object) -> str | None:
    """Serialize a list or dict to a JSON string, or return None."""
    if value is None:
        return None
    return json.dumps(value)


def _opt_str(value: Path | None) -> str | None:
    """Convert an optional Path to a string, or return None."""
    return str(value) if value is not None else None


def _serialize(session: Session) -> dict[str, object]:
    """Convert a Session to a dict of SQLite-compatible values."""
    return {
        "session_id": session.session_id,
        "tags": _to_json(session.tags),
        "state": session.state.value,
        "launch_mode": session.launch_mode.value,
        "launch_time": session.launch_time.isoformat(),
        "target_type": session.target_type.value,
        "target": session.target,
        "target_args": _to_json(session.target_args),
        "target_cwd": _opt_str(session.target_cwd),
        "python_version": session.python_version,
        "python_path": _opt_str(session.python_path),
        "uv_version": session.uv_version,
        "resolved_by": session.resolved_by,
        "psoul_version": session.psoul_version,
        "protocol_version": session.protocol_version,
        "host": session.host,
        "os": session.os,
        "arch": session.arch,
        "config_sources": _to_json(session.config_sources),
        "git_sha": session.git_sha,
        "git_dirty": int(session.git_dirty) if session.git_dirty is not None else None,
        "lockfile_hash": session.lockfile_hash,
        "script_hash": session.script_hash,
        "supervisor_pid": session.supervisor_pid,
        "socket_path": _opt_str(session.socket_path),
        "helper_pid": session.helper_pid,
        "helper_capabilities": _to_json(session.helper_capabilities),
        "generation": session.generation,
        "control_epoch": session.control_epoch,
        "controller_pid": session.controller_pid,
        "control_acquired_at": session.control_acquired_at.isoformat() if session.control_acquired_at else None,
        "sandbox_backend": session.sandbox_backend,
        "sandbox_policy": _to_json(session.sandbox_policy),
    }


_FIELD_TYPES: dict[str, type] = {
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
    "supervisor_pid": int,
    "helper_pid": int,
    "generation": int,
    "control_epoch": int,
    "controller_pid": int,
    "sandbox_backend": str,
}

_JSON_FIELDS = frozenset({"tags", "helper_capabilities", "sandbox_policy", "target_args", "config_sources"})


def _serialize_value(key: str, value: object) -> object:
    """Serialize a single field value for SQLite storage.

    Raises TypeError if the value doesn't match the expected domain type.
    """
    if value is None:
        return None
    if isinstance(value, StrEnum):
        return value.value
    expected = _FIELD_TYPES.get(key)
    if expected is not None:
        if expected is int and type(value) is not int:
            msg = f"{key} must be int, got {type(value).__name__}"
            raise TypeError(msg)
        if expected is not int and not isinstance(value, expected):
            msg = f"{key} must be {expected.__name__}, got {type(value).__name__}"
            raise TypeError(msg)
    if key in _JSON_FIELDS:
        return json.dumps(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


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


class SessionStore:
    """CRUD operations for sessions against the SQLite sessions table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Wrap an existing database connection."""
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    def create(self, session: Session) -> Session:
        """Insert a session. Raises IntegrityError on duplicate session_id."""
        row = _serialize(session)
        self.conn.execute(_INSERT_SQL, row)
        self.conn.commit()
        return session

    def get(self, session_id: str) -> Session | None:
        """Look up a session by its ID. Returns None if not found."""
        row = self.conn.execute("SELECT * FROM sessions WHERE session_id = ?", [session_id]).fetchone()
        if row is None:
            return None
        return _deserialize(row)

    def list(self, *, state: SessionState | None = None) -> list[Session]:
        """Return sessions ordered by launch_time descending, optionally filtered by state."""
        if state is not None:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE state = ? ORDER BY launch_time DESC",
                [state.value],
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM sessions ORDER BY launch_time DESC").fetchall()
        return [_deserialize(row) for row in rows]

    def update(self, session_id: str, **fields: object) -> Session:
        """Update mutable fields on an existing session.

        State changes are validated against the transition map.
        Domain types (Path, datetime, etc.) are serialized automatically.
        Raises KeyError if the session does not exist.
        Raises ValueError for empty updates, unknown fields, or immutable fields.
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
