"""Tests for SQLite storage layer."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from psoul.core.db import DB_NAME, SCHEMA_VERSION, open_db, resolve_state_dir

_INSERT_SESSION = (
    "INSERT INTO sessions (session_id, state, launch_mode, launch_time, target_type, psoul_version) "
    "VALUES (?, 'running', 'attached', '2026-01-01T00:00:00', 'repl', '0.0.1')"
)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_db(tmp_path)
    yield conn
    conn.close()


def test_resolve_state_dir_uses_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom" / "nested"
    result = resolve_state_dir(custom)
    assert result == custom
    assert custom.is_dir()


def test_resolve_state_dir_uses_default(tmp_path: Path) -> None:
    default = tmp_path / "default-state"
    with patch("psoul.core.db.default_state_dir", return_value=default):
        result = resolve_state_dir()
    assert result == default
    assert default.is_dir()


def test_open_db_creates_file(tmp_path: Path) -> None:
    conn = open_db(tmp_path)
    conn.close()
    assert (tmp_path / DB_NAME).exists()


def test_resolve_state_dir_create_false_skips_mkdir(tmp_path: Path) -> None:
    """`resolve_state_dir(create=False)` returns the path without touching the filesystem."""
    target = tmp_path / "does-not-exist"
    result = resolve_state_dir(target, create=False)
    assert result == target
    assert not target.exists()


def test_open_db_create_false_raises_when_db_missing(tmp_path: Path) -> None:
    """`open_db(create=False)` raises FileNotFoundError when the DB file is absent."""
    with pytest.raises(FileNotFoundError):
        open_db(tmp_path, create=False)


def test_open_db_create_false_raises_when_schema_missing(tmp_path: Path) -> None:
    """`open_db(create=False)` raises FileNotFoundError when the file exists but schema is empty."""
    (tmp_path / DB_NAME).touch()
    with pytest.raises(FileNotFoundError):
        open_db(tmp_path, create=False)


def test_open_db_create_false_succeeds_when_initialized(tmp_path: Path) -> None:
    """`open_db(create=False)` opens a previously-initialized DB."""
    open_db(tmp_path).close()
    conn = open_db(tmp_path, create=False)
    try:
        cursor = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
        assert cursor.fetchone()[0] == str(SCHEMA_VERSION)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("pragma", "expected"),
    [
        ("journal_mode", "wal"),
        ("foreign_keys", 1),
        ("busy_timeout", 5000),
        ("synchronous", 1),  # 1 = NORMAL
    ],
)
def test_open_db_pragmas(tmp_path: Path, pragma: str, expected: object) -> None:
    conn = open_db(tmp_path)
    value = conn.execute(f"PRAGMA {pragma}").fetchone()[0]
    conn.close()
    assert value == expected


def test_schema_and_tables(db: sqlite3.Connection) -> None:
    version = db.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0]
    assert version == str(SCHEMA_VERSION)
    rows = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = {row[0] for row in rows}
    expected = {
        "schema_meta",
        "sessions",
        "results",
        "events",
        "commands",
        "artifacts",
        "resource_samples",
        "profiling_state",
    }
    assert expected <= tables


def test_open_db_idempotent(tmp_path: Path) -> None:
    conn1 = open_db(tmp_path)
    conn2 = open_db(tmp_path)  # second connection while first is still open
    conn1.close()
    conn2.close()


def test_results_fk_rejects_bad_session(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO results (session_id, generation, outcome, end_time) "
            "VALUES ('nonexistent', 0, 'exited', '2026-01-01T00:00:00')"
        )


def test_cascade_delete_removes_results(db: sqlite3.Connection) -> None:
    db.execute(_INSERT_SESSION, ["s1"])
    db.execute(
        "INSERT INTO results (session_id, generation, outcome, end_time) "
        "VALUES ('s1', 0, 'exited', '2026-01-01T01:00:00')"
    )
    db.execute("DELETE FROM sessions WHERE session_id = 's1'")
    assert db.execute("SELECT COUNT(*) FROM results WHERE session_id = 's1'").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("first_sql", "duplicate_sql"),
    [
        pytest.param(
            "INSERT INTO events (session_id, sequence, timestamp, event_type) "
            "VALUES ('s1', 1, '2026-01-01T00:00:00', 'session.started')",
            "INSERT INTO events (session_id, sequence, timestamp, event_type) "
            "VALUES ('s1', 1, '2026-01-01T00:00:01', 'runtime.stdout')",
            id="events-duplicate-sequence",
        ),
        pytest.param(
            "INSERT INTO commands (session_id, message_id, timestamp, command, status) "
            "VALUES ('s1', 'msg-1', '2026-01-01T00:00:00', 'eval', 'ok')",
            "INSERT INTO commands (session_id, message_id, timestamp, command, status) "
            "VALUES ('s1', 'msg-1', '2026-01-01T00:00:01', 'eval', 'ok')",
            id="commands-duplicate-message-id",
        ),
        pytest.param(
            "INSERT INTO artifacts (session_id, name, path, registered_at, source) "
            "VALUES ('s1', 'plot.png', 'artifacts/s1/plot.png', '2026-01-01T00:00:00', 'user')",
            "INSERT INTO artifacts (session_id, name, path, registered_at, source) "
            "VALUES ('s1', 'plot.png', 'artifacts/s1/plot2.png', '2026-01-01T00:00:01', 'user')",
            id="artifacts-duplicate-name",
        ),
        pytest.param(
            "INSERT INTO profiling_state (session_id, generation, backend, mode, started_at) "
            "VALUES ('s1', 0, 'austin', 'cpu', '2026-01-01T00:00:00')",
            "INSERT INTO profiling_state (session_id, generation, backend, mode, started_at) "
            "VALUES ('s1', 0, 'py-spy', 'cpu', '2026-01-01T00:00:01')",
            id="profiling-duplicate-active-mode",
        ),
    ],
)
def test_unique_constraint_rejects_duplicate(db: sqlite3.Connection, first_sql: str, duplicate_sql: str) -> None:
    db.execute(_INSERT_SESSION, ["s1"])
    db.execute(first_sql)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(duplicate_sql)


def test_commands_null_message_ids_allowed(db: sqlite3.Connection) -> None:
    db.execute(_INSERT_SESSION, ["s1"])
    for i in range(2):
        db.execute(
            "INSERT INTO commands (session_id, timestamp, command, status) VALUES (?, ?, 'eval', 'ok')",
            ("s1", f"2026-01-01T00:00:0{i}"),
        )


def test_future_version_rejected(tmp_path: Path) -> None:
    conn = open_db(tmp_path)
    conn.execute("UPDATE schema_meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="newer than this psoul"):
        open_db(tmp_path)


def test_migration_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = open_db(tmp_path)
    conn.execute("UPDATE schema_meta SET value = '0' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    applied = []
    monkeypatch.setattr("psoul.core.db._MIGRATIONS", {0: lambda c: applied.append(0)})
    conn = open_db(tmp_path)
    version = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0]
    conn.close()
    assert applied == [0]
    assert version == str(SCHEMA_VERSION)


def test_migration_rollback_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = open_db(tmp_path)
    conn.execute("UPDATE schema_meta SET value = '0' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    def bad_migration(c: sqlite3.Connection) -> None:
        msg = "intentional failure"
        raise RuntimeError(msg)

    monkeypatch.setattr("psoul.core.db._MIGRATIONS", {0: bad_migration})
    with pytest.raises(RuntimeError, match="intentional failure"):
        open_db(tmp_path)

    # Check with raw connection — open_db would retry the broken migration
    conn = sqlite3.connect(tmp_path / DB_NAME)
    version = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0]
    conn.close()
    assert version == "0"


@pytest.mark.parametrize(
    ("pre_init", "expect_wal_pragma"),
    [
        pytest.param(False, True, id="fresh-db-issues-wal-pragma"),
        pytest.param(True, False, id="already-wal-skips-pragma"),
    ],
)
def test_open_db_wal_pragma_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pre_init: bool,
    expect_wal_pragma: bool,
) -> None:
    if pre_init:
        open_db(tmp_path).close()  # establish WAL before the traced open

    statements: list[str] = []
    real_connect = sqlite3.connect

    def traced_connect(database: Path, *, timeout: float) -> sqlite3.Connection:
        conn = real_connect(database, timeout=timeout)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr("psoul.core.db.sqlite3.connect", traced_connect)
    open_db(tmp_path).close()

    issued_wal = any("PRAGMA journal_mode = WAL" in s for s in statements)
    assert issued_wal is expect_wal_pragma, statements
    assert any(s.strip() == "PRAGMA journal_mode" for s in statements), statements
