"""Tests for the ``psoul artifacts`` CLI command."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.core.db import open_db
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()


def _make_session(session_id: str) -> Session:
    """Build a minimal Session so artifacts can reference it by ID."""
    return Session(
        session_id=session_id,
        state=SessionState.exited,
        launch_mode=LaunchMode.headless,
        launch_time=datetime.now(UTC),
        psoul_version=VERSION,
    )


def _insert_artifact(
    conn: sqlite3.Connection,
    session_id: str,
    name: str,
    *,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    registered_at: str = "2026-01-01T00:00:00",
    source: str = "user",
    retention_class: str = "session",
) -> None:
    """Insert one artifact row directly, bypassing any producer."""
    path = f"artifacts/{session_id}/{name}"
    conn.execute(
        "INSERT INTO artifacts"
        " (session_id, name, path, mime_type, size_bytes, registered_at, source, retention_class)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, name, path, mime_type, size_bytes, registered_at, source, retention_class),
    )
    conn.commit()


@pytest.fixture
def seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Seed a session with three artifacts covering populated, null, and non-default enum values."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(_make_session("sesh-cli"))
        _insert_artifact(
            conn,
            "sesh-cli",
            "plot.png",
            mime_type="image/png",
            size_bytes=2048,
            registered_at="2026-01-01T00:00:01",
        )
        _insert_artifact(conn, "sesh-cli", "data.csv", registered_at="2026-01-01T00:00:02")
        _insert_artifact(
            conn,
            "sesh-cli",
            "trace.json",
            mime_type="application/json",
            size_bytes=512,
            registered_at="2026-01-01T00:00:03",
            source="psoul",
            retention_class="pinned",
        )
    return "sesh-cli"


def test_artifacts_prints_tab_separated_text(seeded: str) -> None:
    result = runner.invoke(cli, ["artifacts", seeded])
    assert result.exit_code == 0
    assert result.output == (
        "2026-01-01T00:00:01\tplot.png\tartifacts/sesh-cli/plot.png\timage/png\t2048\tuser\tsession\n"
        "2026-01-01T00:00:02\tdata.csv\tartifacts/sesh-cli/data.csv\t\t\tuser\tsession\n"
        "2026-01-01T00:00:03\ttrace.json\tartifacts/sesh-cli/trace.json\tapplication/json\t512\tpsoul\tpinned\n"
    )


def test_artifacts_empty_session_prints_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(_make_session("empty"))
    result = runner.invoke(cli, ["artifacts", "empty"])
    assert result.exit_code == 0
    assert result.output == ""


@pytest.mark.parametrize(
    ("selector", "expected_exit", "expected_contains"),
    [
        ("sesh", 0, "plot.png"),
        ("nope", 1, "session not found"),
    ],
    ids=["prefix-match", "unknown-session"],
)
def test_artifacts_resolves_session_selector(
    seeded: str,
    selector: str,
    expected_exit: int,
    expected_contains: str,
) -> None:
    result = runner.invoke(cli, ["artifacts", selector])
    assert result.exit_code == expected_exit
    assert expected_contains in result.output
