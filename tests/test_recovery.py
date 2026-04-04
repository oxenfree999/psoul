"""Tests for orphan detection and stale session recovery."""

import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from psoul.db import open_db
from psoul.recovery import (
    STARTING_GRACE_SECONDS,
    ProcessStatus,
    _recover_session,
    check_pid,
    recover_sessions,
)
from psoul.session import LaunchMode, Session, SessionState, TargetType
from psoul.store import SessionStore
from psoul.version import VERSION


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = open_db(tmp_path)
    yield db
    db.close()


@pytest.fixture
def store(conn: sqlite3.Connection) -> SessionStore:
    return SessionStore(conn)


def test_check_pid_invalid_returns_unknown() -> None:
    assert check_pid(0) is ProcessStatus.unknown
    assert check_pid(-1) is ProcessStatus.unknown


def test_check_pid_windows_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.recovery.sys.platform", "win32")
    assert check_pid(12345) is ProcessStatus.unknown


@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [
        (ProcessLookupError, ProcessStatus.dead),
        (PermissionError, ProcessStatus.alive),
        (None, ProcessStatus.alive),
        (OSError(22, "Invalid argument"), ProcessStatus.unknown),
    ],
    ids=["esrch-dead", "eperm-alive", "success-alive", "oserror-unknown"],
)
def test_check_pid_os_kill_outcomes(
    side_effect: BaseException | None, expected: ProcessStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_kill(pid: int, sig: int) -> None:
        if side_effect is not None:
            raise side_effect

    monkeypatch.setattr("psoul.recovery.os.kill", fake_kill)
    assert check_pid(42) is expected


def _make_session(
    store: SessionStore,
    name: str,
    *,
    state: SessionState = SessionState.running,
    supervisor_pid: int | None = 99999,
    launch_time: datetime | None = None,
) -> Session:
    """Insert a session and advance it to the requested state."""
    session = store.create(
        Session(
            session_id=name,
            state=SessionState.starting,
            launch_mode=LaunchMode.headless,
            launch_time=launch_time or datetime.now(UTC),
            psoul_version=VERSION,
            target_type=TargetType.script,
            target="test.py",
        )
    )
    if state != SessionState.starting:
        session = store.update(name, state=state, supervisor_pid=supervisor_pid)
    return session


def _get_result(conn: sqlite3.Connection, session_id: str, generation: int = 0) -> dict[str, object] | None:
    """Read a result row as a plain dict, or None if absent."""
    row = conn.execute(
        "SELECT * FROM results WHERE session_id = ? AND generation = ?",
        [session_id, generation],
    ).fetchone()
    if row is None:
        return None
    return dict(zip(row.keys(), row, strict=True))


def test_dead_running_session_recovered(
    conn: sqlite3.Connection, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_session(store, "orphan-a", supervisor_pid=99999)
    monkeypatch.setattr("psoul.recovery.check_pid", lambda pid: ProcessStatus.dead)
    recovered = recover_sessions(conn)
    assert recovered == ["orphan-a"]
    session = store.get("orphan-a")
    assert session is not None
    assert session.state is SessionState.failed
    result = _get_result(conn, "orphan-a")
    assert result is not None
    assert result["outcome"] == "orphan_recovered"
    assert result["exit_code"] is None
    assert result["generation"] == 0
    assert result["orphan_detected_at"] is not None
    assert result["recovery_attempted_at"] is not None


def test_running_session_no_pid_left_alone(conn: sqlite3.Connection, store: SessionStore) -> None:
    _make_session(store, "no-pid", supervisor_pid=None)
    recovered = recover_sessions(conn)
    assert recovered == []
    session = store.get("no-pid")
    assert session is not None
    assert session.state is SessionState.running


def test_running_session_live_pid_left_alone(
    conn: sqlite3.Connection, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_session(store, "alive-sess")
    monkeypatch.setattr("psoul.recovery.check_pid", lambda pid: ProcessStatus.alive)
    recovered = recover_sessions(conn)
    assert recovered == []
    session = store.get("alive-sess")
    assert session is not None
    assert session.state is SessionState.running


def test_running_session_unknown_pid_status_left_alone(
    conn: sqlite3.Connection, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_session(store, "unknown-sess")
    monkeypatch.setattr("psoul.recovery.check_pid", lambda pid: ProcessStatus.unknown)
    recovered = recover_sessions(conn)
    assert recovered == []
    session = store.get("unknown-sess")
    assert session is not None
    assert session.state is SessionState.running


@pytest.mark.parametrize(
    ("age_offset", "should_recover"),
    [
        (-1, False),
        (+1, True),
    ],
    ids=["fresh-left-alone", "stale-recovered"],
)
def test_starting_session_grace_threshold(
    age_offset: int, should_recover: bool, conn: sqlite3.Connection, store: SessionStore
) -> None:
    launch = datetime.now(UTC) - timedelta(seconds=STARTING_GRACE_SECONDS + age_offset)
    _make_session(store, "start-thresh", state=SessionState.starting, launch_time=launch)
    recovered = recover_sessions(conn)
    session = store.get("start-thresh")
    assert session is not None
    if should_recover:
        assert recovered == ["start-thresh"]
        assert session.state is SessionState.failed
        result = _get_result(conn, "start-thresh")
        assert result is not None
        assert result["outcome"] == "orphan_recovered"
        assert result["generation"] == 0
    else:
        assert recovered == []
        assert session.state is SessionState.starting


@pytest.mark.parametrize(
    ("outcome", "expected_state"),
    [
        ("exited", SessionState.exited),
        ("failed", SessionState.failed),
    ],
    ids=["exited-result", "failed-result"],
)
def test_existing_result_row_finalizes_from_outcome(
    outcome: str,
    expected_state: SessionState,
    conn: sqlite3.Connection,
    store: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_session(store, "partial", supervisor_pid=99999)
    conn.execute(
        "INSERT INTO results (session_id, generation, outcome, exit_code, end_time) VALUES ('partial', 0, ?, 0, ?)",
        [outcome, datetime.now(UTC).isoformat()],
    )
    conn.commit()
    monkeypatch.setattr("psoul.recovery.check_pid", lambda pid: ProcessStatus.dead)
    recovered = recover_sessions(conn)
    assert recovered == ["partial"]
    session = store.get("partial")
    assert session is not None
    assert session.state is expected_state
    result = _get_result(conn, "partial")
    assert result is not None
    assert result["outcome"] == outcome
    assert result["orphan_detected_at"] is not None
    assert result["recovery_attempted_at"] is not None


def test_concurrent_recovery_does_not_double_write(tmp_path: Path) -> None:
    setup_conn = open_db(tmp_path)
    store = SessionStore(setup_conn)
    _make_session(store, "race-sess", supervisor_pid=99999)
    setup_conn.close()

    barrier = threading.Barrier(2)
    now = datetime.now(UTC)

    def worker() -> bool:
        conn = open_db(tmp_path)
        try:
            barrier.wait()
            return _recover_session(conn, "race-sess", SessionState.running, 0, now)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker) for _ in range(2)]
        results = [future.result() for future in futures]

    assert sorted(results) == [False, True]

    verify_conn = open_db(tmp_path)
    try:
        count = verify_conn.execute("SELECT COUNT(*) FROM results WHERE session_id = 'race-sess'").fetchone()[0]
        assert count == 1
        state = verify_conn.execute("SELECT state FROM sessions WHERE session_id = 'race-sess'").fetchone()[0]
        assert state == SessionState.failed.value
    finally:
        verify_conn.close()
