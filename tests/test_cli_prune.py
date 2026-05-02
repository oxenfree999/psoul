"""Tests for the ``psoul prune`` CLI command."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.cli.state import ExitCode
from psoul.core.db import open_db
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()


def _make_session(
    session_id: str,
    *,
    state: SessionState = SessionState.exited,
    launch_time: datetime | None = None,
    tags: dict[str, str] | None = None,
) -> Session:
    """Build a Session with sensible defaults for prune tests."""
    return Session(
        session_id=session_id,
        state=state,
        launch_mode=LaunchMode.headless,
        launch_time=launch_time if launch_time is not None else datetime(2026, 1, 1, tzinfo=UTC),
        psoul_version=VERSION,
        tags=tags,
    )


def _insert_result(conn: sqlite3.Connection, session_id: str, end_time: datetime, *, generation: int = 0) -> None:
    """Insert a results row so the session has an end_time for the --older-than filter."""
    conn.execute(
        "INSERT INTO results (session_id, generation, outcome, end_time, duration_seconds)"
        " VALUES (?, ?, 'exited', ?, 0.0)",
        (session_id, generation, end_time.isoformat()),
    )
    conn.commit()


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect psoul to use *tmp_path* as its state directory.

    Pre-creates the DB so existing validation tests hit validation rather
    than the no-DB short-circuit. Tests that exercise the no-DB path use
    their own fixture or skip this one.
    """
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    open_db(tmp_path).close()
    return tmp_path


@pytest.mark.parametrize(
    ("args", "message_substr"),
    [
        ([], "requires at least one of --all"),
        (["--all", "--state", "exited"], "--all cannot be combined"),
        (["--older-than", "abc"], "invalid duration"),
        (["--tag", "novalue"], "invalid tag (expected key=value)"),
    ],
    ids=["no-filters", "all-with-filter", "bad-older-than", "bad-tag"],
)
def test_invalid_input_returns_usage_error(state_dir: Path, args: list[str], message_substr: str) -> None:
    result = runner.invoke(cli, ["prune", *args])
    assert result.exit_code == ExitCode.USAGE
    assert message_substr in result.output


@pytest.mark.parametrize(
    ("seed", "args", "kept"),
    [
        ([_make_session("a"), _make_session("b", state=SessionState.failed)], ["--all"], []),
        ([_make_session("a"), _make_session("b", state=SessionState.failed)], ["--state", "exited"], ["b"]),
        ([_make_session("a"), _make_session("b", state=SessionState.failed)], ["--state", "failed"], ["a"]),
        ([_make_session("a", tags={"env": "dev"}), _make_session("b")], ["--tag", "env=dev"], ["b"]),
    ],
    ids=["all", "state-exited", "state-failed", "tag-env-dev"],
)
def test_filter_keeps_unmatched_sessions(
    state_dir: Path, seed: list[Session], args: list[str], kept: list[str]
) -> None:
    with closing(open_db(state_dir)) as conn:
        for s in seed:
            SessionStore(conn).create(s)
    result = runner.invoke(cli, ["prune", *args])
    assert result.exit_code == 0
    with closing(open_db(state_dir)) as conn:
        assert [s.session_id for s in SessionStore(conn).list()] == kept


def test_state_and_tag_compose_with_and(state_dir: Path) -> None:
    """Only sessions matching both --state and --tag are pruned."""
    with closing(open_db(state_dir)) as conn:
        store = SessionStore(conn)
        store.create(_make_session("a", tags={"env": "dev"}))
        store.create(_make_session("b"))
        store.create(_make_session("c", state=SessionState.failed, tags={"env": "dev"}))
    result = runner.invoke(cli, ["prune", "--state", "exited", "--tag", "env=dev"])
    assert result.exit_code == 0
    with closing(open_db(state_dir)) as conn:
        assert sorted(s.session_id for s in SessionStore(conn).list()) == ["b", "c"]


def test_older_than_keys_off_end_time_not_launch_time(state_dir: Path) -> None:
    """A long-running session that finished recently must NOT match --older-than."""
    now = datetime.now(UTC)
    with closing(open_db(state_dir)) as conn:
        store = SessionStore(conn)
        store.create(_make_session("long-run-recent-end", launch_time=now - timedelta(hours=10)))
        store.create(_make_session("short-run-old-end", launch_time=now - timedelta(hours=3)))
        _insert_result(conn, "long-run-recent-end", now - timedelta(minutes=5))
        _insert_result(conn, "short-run-old-end", now - timedelta(hours=2))
    result = runner.invoke(cli, ["prune", "--older-than", "1h"])
    assert result.exit_code == 0
    with closing(open_db(state_dir)) as conn:
        assert [s.session_id for s in SessionStore(conn).list()] == ["long-run-recent-end"]


@pytest.mark.parametrize(
    ("args", "exit_code", "kept", "message_substr"),
    [
        (["--all"], ExitCode.USAGE, ["c"], "cannot prune active session(s) without --force"),
        (["--all", "--force"], 0, [], ""),
    ],
    ids=["refused-without-force", "force-overrides"],
)
def test_active_state_refusal_and_force_override(
    state_dir: Path, args: list[str], exit_code: int, kept: list[str], message_substr: str
) -> None:
    with closing(open_db(state_dir)) as conn:
        SessionStore(conn).create(_make_session("c", state=SessionState.running))
    result = runner.invoke(cli, ["prune", *args])
    assert result.exit_code == exit_code
    if message_substr:
        assert message_substr in result.output
    with closing(open_db(state_dir)) as conn:
        assert [s.session_id for s in SessionStore(conn).list()] == kept


def test_artifact_directory_removed(state_dir: Path) -> None:
    artifact_dir = state_dir / "artifacts" / "a"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "plot.png").write_bytes(b"png")
    with closing(open_db(state_dir)) as conn:
        SessionStore(conn).create(_make_session("a"))
    result = runner.invoke(cli, ["prune", "--all"])
    assert result.exit_code == 0
    assert not artifact_dir.exists()


def test_json_output_shape(state_dir: Path) -> None:
    with closing(open_db(state_dir)) as conn:
        SessionStore(conn).create(_make_session("a"))
    result = runner.invoke(cli, ["prune", "--all", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"pruned": ["a"], "count": 1}
