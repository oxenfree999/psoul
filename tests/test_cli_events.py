"""Tests for the ``psoul events`` CLI command."""

import json
from collections.abc import Callable
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


def _parse_text(out: str) -> list[tuple[int, int, str, object]]:
    rows = [row.split("\t") for row in out.strip().split("\n")]
    return [(int(r[0]), int(r[1]), r[3], json.loads(r[4])) for r in rows]


def _parse_json_array(out: str) -> list[tuple[int, int, str, object]]:
    return [(o["sequence"], o["generation"], o["event_type"], o["payload"]) for o in json.loads(out)]


def _parse_ndjson(out: str) -> list[tuple[int, int, str, object]]:
    return [
        (o["sequence"], o["generation"], o["event_type"], o["payload"])
        for o in (json.loads(line) for line in out.strip().split("\n"))
    ]


@pytest.mark.parametrize(
    ("extra_args", "parse"),
    [
        ([], _parse_text),
        (["--json"], _parse_json_array),
        (["--follow"], _parse_text),
        (["--follow", "--json"], _parse_ndjson),
    ],
    ids=["text", "json-array", "follow-text", "follow-ndjson"],
)
def test_events_replay(
    seeded_events: str,
    extra_args: list[str],
    parse: Callable[[str], list[tuple[int, int, str, object]]],
) -> None:
    result = runner.invoke(cli, ["events", seeded_events, *extra_args])
    assert result.exit_code == 0
    assert parse(result.output) == [
        (0, 0, "runtime.stdout", {"text": "hi"}),
        (1, 0, "runtime.stderr", {"text": "err"}),
        (2, 0, "custom.event", None),
    ]


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
