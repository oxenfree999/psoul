"""Tests for the ``psoul logs`` CLI command."""

from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.core.db import open_db
from psoul.core.events import EVENT_RUNTIME_STDERR, EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()


def _make_session(session_id: str) -> Session:
    """Build a minimal Session so events can reference it by ID."""
    return Session(
        session_id=session_id,
        state=SessionState.exited,
        launch_mode=LaunchMode.headless,
        launch_time=datetime.now(UTC),
        psoul_version=VERSION,
    )


@pytest.fixture
def seeded_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Seed a session ``abc123`` with interleaved stdout/stderr events."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(_make_session("abc123"))
        store = EventStore(conn)
        store.append(session_id="abc123", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "hello\n"}, generation=0)
        store.append(session_id="abc123", event_type=EVENT_RUNTIME_STDERR, payload={"text": "err\n"}, generation=0)
        store.append(session_id="abc123", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "world\n"}, generation=0)
    return "abc123"


@pytest.mark.parametrize(
    ("flags", "expected_output"),
    [
        ([], "hello\nerr\nworld\n"),
        (["--stdout"], "hello\nworld\n"),
        (["--stderr"], "err\n"),
    ],
    ids=["default-interleaves", "stdout-only", "stderr-only"],
)
def test_logs_filters_streams(seeded_session: str, flags: list[str], expected_output: str) -> None:
    result = runner.invoke(cli, ["logs", seeded_session, *flags])
    assert result.exit_code == 0
    assert result.output == expected_output


def test_logs_rejects_both_stream_flags(seeded_session: str) -> None:
    result = runner.invoke(cli, ["logs", seeded_session, "--stdout", "--stderr"])
    assert result.exit_code == 2
    assert "cannot be used together" in result.output


@pytest.mark.parametrize(
    ("flags", "expected_output"),
    [
        ([], "old\nnew\n"),
        (["--generation", "0"], "old\n"),
        (["--generation", "1"], "new\n"),
    ],
    ids=["no-filter", "gen-zero", "gen-one"],
)
def test_logs_filters_by_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], expected_output: str
) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(_make_session("gen-test"))
        store = EventStore(conn)
        store.append(session_id="gen-test", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "old\n"}, generation=0)
        store.append(session_id="gen-test", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "new\n"}, generation=1)
    result = runner.invoke(cli, ["logs", "gen-test", *flags])
    assert result.exit_code == 0
    assert result.output == expected_output


@pytest.mark.parametrize(
    ("match", "expected_exit", "expected_contains"),
    [
        (True, 0, "hello"),
        (False, 1, "session not found"),
    ],
    ids=["prefix-match", "unknown-session"],
)
def test_logs_resolves_session_selector(
    seeded_session: str, match: bool, expected_exit: int, expected_contains: str
) -> None:
    selector = seeded_session[:3] if match else "nope"
    result = runner.invoke(cli, ["logs", selector])
    assert result.exit_code == expected_exit
    assert expected_contains in result.output
