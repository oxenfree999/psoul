"""Tests for the ``psoul events`` CLI command."""

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.core.db import open_db
from psoul.core.events import EventStore
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
def seeded_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Seed a session with runtime and non-runtime events plus a null payload."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(_make_session("sesh01"))
        store = EventStore(conn)
        store.append(session_id="sesh01", event_type="runtime.stdout", payload={"text": "hi"}, generation=0)
        store.append(session_id="sesh01", event_type="runtime.stderr", payload={"text": "err"}, generation=0)
        store.append(session_id="sesh01", event_type="custom.event", payload=None, generation=0)
    return "sesh01"


def test_events_text_output(seeded_events: str) -> None:
    result = runner.invoke(cli, ["events", seeded_events])
    assert result.exit_code == 0
    rows = [line.split("\t") for line in result.output.strip().split("\n")]
    assert len(rows) == 3
    assert [(r[0], r[1], r[3], r[4]) for r in rows] == [
        ("0", "0", "runtime.stdout", '{"text": "hi"}'),
        ("1", "0", "runtime.stderr", '{"text": "err"}'),
        ("2", "0", "custom.event", "null"),
    ]


def test_events_json_output(seeded_events: str) -> None:
    result = runner.invoke(cli, ["events", seeded_events, "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert [e["event_type"] for e in parsed] == ["runtime.stdout", "runtime.stderr", "custom.event"]
    assert [e["payload"] for e in parsed] == [{"text": "hi"}, {"text": "err"}, None]
    assert [e["sequence"] for e in parsed] == [0, 1, 2]
    assert all(e["generation"] == 0 for e in parsed)
    assert all(isinstance(e["timestamp"], str) and e["timestamp"] for e in parsed)


@pytest.mark.parametrize(
    ("match", "expected_exit", "expected_contains"),
    [
        (True, 0, "runtime.stdout"),
        (False, 1, "session not found"),
    ],
    ids=["prefix-match", "unknown-session"],
)
def test_events_resolves_session_selector(
    seeded_events: str, match: bool, expected_exit: int, expected_contains: str
) -> None:
    selector = seeded_events[:3] if match else "nope"
    result = runner.invoke(cli, ["events", selector])
    assert result.exit_code == expected_exit
    assert expected_contains in result.output
