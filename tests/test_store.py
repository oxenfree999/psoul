"""Tests for SessionStore CRUD operations."""

import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from psoul.core.db import open_db
from psoul.core.session import LaunchMode, Session, SessionError, SessionState, TargetType
from psoul.core.store import SessionStore


def _session(
    *,
    session_id: str = "calm-tiger-builds-kites",
    state: SessionState = SessionState.running,
    launch_mode: LaunchMode = LaunchMode.attached,
    psoul_version: str = "0.0.1",
    launch_time: datetime | None = None,
) -> Session:
    """Build a Session with sensible defaults for testing."""
    return Session(
        session_id=session_id,
        state=state,
        launch_mode=launch_mode,
        launch_time=launch_time if launch_time is not None else datetime(2026, 1, 1, tzinfo=UTC),
        psoul_version=psoul_version,
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SessionStore]:
    conn = open_db(tmp_path)
    yield SessionStore(conn)
    conn.close()


def test_create_and_get_round_trip(store: SessionStore) -> None:
    session = _session()
    store.create(session)
    loaded = store.get(session.session_id)
    assert loaded == session


def test_round_trip_all_fields(store: SessionStore) -> None:
    session = Session(
        session_id="bold-fox-reads-maps",
        state=SessionState.running,
        launch_mode=LaunchMode.headless,
        launch_time=datetime.now(UTC),
        psoul_version="0.0.1",
        target_type=TargetType.script,
        target="train.py",
        target_args=["--epochs", "10"],
        target_cwd=Path("/tmp/work"),
        python_version="3.14.2",
        python_path=Path("/usr/bin/python3"),
        uv_version="0.7.0",
        resolved_by="uv",
        protocol_version=1,
        host="devbox",
        os="linux",
        arch="aarch64",
        config_sources=["pyproject.toml", ".psoul.toml"],
        git_sha="abc123",
        git_dirty=True,
        lockfile_hash="sha256:deadbeef",
        script_hash="sha256:cafebabe",
        supervisor_pid=1234,
        socket_path=Path("/tmp/psoul.sock"),
        helper_pid=5678,
        helper_capabilities={"eval": True, "complete": False},
        generation=1,
        control_epoch=2,
        controller_pid=9999,
        control_acquired_at=datetime.now(UTC),
        sandbox_backend="bubblewrap",
        sandbox_policy={"net": False, "fs": "read-only"},
        tags={"env": "dev", "team": "ml"},
    )
    store.create(session)
    assert store.get(session.session_id) == session


def test_get_missing_returns_none(store: SessionStore) -> None:
    assert store.get("no-such-session") is None


def test_create_duplicate_raises(store: SessionStore) -> None:
    store.create(_session())
    with pytest.raises(sqlite3.IntegrityError):
        store.create(_session())


def test_create_rejects_bad_target_args_shape(store: SessionStore) -> None:
    session = replace(_session(), target_args=cast("list[str]", {"bad": "shape"}))
    with pytest.raises(TypeError, match=r"target_args must be list\b"):
        store.create(session)


def test_create_rejects_non_string_config_source(store: SessionStore) -> None:
    session = replace(_session(), config_sources=cast("list[str]", [1]))
    with pytest.raises(TypeError, match=r"config_sources must be list\[str\]"):
        store.create(session)


def test_list_returns_newest_first(store: SessionStore) -> None:
    store.create(_session(session_id="first-session-aaa-bbb", launch_time=datetime(2026, 1, 1, 0, 0, tzinfo=UTC)))
    store.create(_session(session_id="second-session-ccc-ddd", launch_time=datetime(2026, 1, 1, 0, 1, tzinfo=UTC)))
    assert [s.session_id for s in store.list()] == ["second-session-ccc-ddd", "first-session-aaa-bbb"]


def test_list_filters_by_state(store: SessionStore) -> None:
    store.create(_session(session_id="running-one", state=SessionState.running))
    store.create(_session(session_id="suspended-one", state=SessionState.suspended))
    assert [s.session_id for s in store.list(state=SessionState.suspended)] == ["suspended-one"]


def test_update_mutable_field(store: SessionStore) -> None:
    store.create(_session())
    updated = store.update("calm-tiger-builds-kites", supervisor_pid=42)
    assert updated.supervisor_pid == 42


def test_update_valid_state_transition(store: SessionStore) -> None:
    store.create(_session())
    updated = store.update("calm-tiger-builds-kites", state=SessionState.stopping)
    assert updated.state == SessionState.stopping


def test_update_invalid_state_transition(store: SessionStore) -> None:
    store.create(_session())
    with pytest.raises(SessionError, match="invalid state transition"):
        store.update("calm-tiger-builds-kites", state=SessionState.starting)


def test_update_rejects_empty(store: SessionStore) -> None:
    store.create(_session())
    with pytest.raises(ValueError, match="at least one field"):
        store.update("calm-tiger-builds-kites")


def test_update_rejects_unknown_field(store: SessionStore) -> None:
    store.create(_session())
    with pytest.raises(ValueError, match="unknown fields"):
        store.update("calm-tiger-builds-kites", bogus=1)


def test_update_rejects_immutable_field(store: SessionStore) -> None:
    store.create(_session())
    with pytest.raises(ValueError, match="immutable fields"):
        store.update("calm-tiger-builds-kites", launch_mode=LaunchMode.headless)


def test_update_missing_session_raises(store: SessionStore) -> None:
    with pytest.raises(KeyError, match="session not found"):
        store.update("no-such-session", supervisor_pid=1)


def test_update_rejects_non_string_tag_values(store: SessionStore) -> None:
    store.create(_session())
    with pytest.raises(TypeError, match=r"tags must be dict\[str, str\]"):
        store.update("calm-tiger-builds-kites", tags=cast("dict[str, str]", {"env": 1}))


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_type"),
    [
        ("tags", ["not", "a", "dict"], "dict"),
        ("helper_capabilities", "string", "dict"),
        ("socket_path", 123, "Path"),
        ("control_acquired_at", "not-a-datetime", "datetime"),
        ("supervisor_pid", 3.14, "int"),
        ("supervisor_pid", True, "int"),
    ],
)
def test_update_rejects_wrong_type(store: SessionStore, field: str, bad_value: object, expected_type: str) -> None:
    store.create(_session())
    with pytest.raises(TypeError, match=f"must be {expected_type}"):
        store.update("calm-tiger-builds-kites", **{field: bad_value})


def test_record_result_inserts_row(store: SessionStore) -> None:
    store.create(_session())
    end = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
    store.record_result(
        "calm-tiger-builds-kites",
        outcome="exited",
        exit_code=0,
        end_time=end,
        duration_seconds=300.0,
    )
    row = store.conn.execute("SELECT * FROM results WHERE session_id = ?", ["calm-tiger-builds-kites"]).fetchone()
    assert row is not None
    assert row["outcome"] == "exited"
    assert row["exit_code"] == 0
    assert row["end_time"] == end.isoformat()
    assert row["duration_seconds"] == pytest.approx(300.0)
    assert row["generation"] == 0


def test_record_result_uses_current_generation(store: SessionStore) -> None:
    """record_result() should write the session's current generation."""
    store.create(replace(_session(session_id="future-session"), generation=3))
    store.record_result(
        "future-session",
        outcome="failed",
        exit_code=7,
        end_time=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
        duration_seconds=12.5,
    )
    row = store.conn.execute("SELECT generation FROM results WHERE session_id = ?", ["future-session"]).fetchone()
    assert row is not None
    assert row["generation"] == 3


def test_record_result_missing_session_raises(store: SessionStore) -> None:
    with pytest.raises(KeyError, match="session not found"):
        store.record_result(
            "no-such-session",
            outcome="failed",
            exit_code=1,
            end_time=datetime(2026, 1, 1, tzinfo=UTC),
            duration_seconds=0.0,
        )


def test_delete_removes_session_and_cascades(store: SessionStore) -> None:
    """delete() should drop the session row and let cascade clear dependent events."""
    store.create(_session(session_id="bold-fox-builds-kites"))
    store.conn.execute(
        "INSERT INTO events (session_id, sequence, timestamp, event_type, payload) VALUES (?, ?, ?, ?, ?)",
        ["bold-fox-builds-kites", 0, "2026-01-01T00:00:00+00:00", "session.started", "{}"],
    )
    store.conn.commit()
    store.delete("bold-fox-builds-kites")
    assert store.get("bold-fox-builds-kites") is None
    event_count = store.conn.execute(
        "SELECT COUNT(*) FROM events WHERE session_id = ?", ["bold-fox-builds-kites"]
    ).fetchone()[0]
    assert event_count == 0


def test_delete_missing_session_raises(store: SessionStore) -> None:
    with pytest.raises(KeyError, match="session not found"):
        store.delete("no-such-session")
