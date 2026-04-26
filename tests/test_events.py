"""Tests for EventStore: append, sequence allocation, and list filters."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from psoul.core.db import open_db
from psoul.core.events import EVENT_RUNTIME_STDERR, EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION


def _make_session(session_id: str = "test-session") -> Session:
    """Build a minimal Session so events can reference it by ID.

    Events carry a ``session_id`` that must match a real row in the
    ``sessions`` table — the database enforces this, and inserting an
    event for a session that doesn't exist raises an error. Tests use
    this helper to create that row first.
    """
    return Session(
        session_id=session_id,
        state=SessionState.starting,
        launch_mode=LaunchMode.headless,
        launch_time=datetime.now(UTC),
        psoul_version=VERSION,
    )


@pytest.fixture
def event_store(tmp_path: Path) -> Iterator[EventStore]:
    conn = open_db(tmp_path)
    SessionStore(conn).create(_make_session())
    try:
        yield EventStore(conn)
    finally:
        conn.close()


@pytest.mark.parametrize("payload", [{"text": "hi"}, None], ids=["dict-payload", "null-payload"])
def test_append_round_trips_payload(event_store: EventStore, payload: dict[str, object] | None) -> None:
    event_store.append(
        session_id="test-session",
        event_type=EVENT_RUNTIME_STDOUT,
        payload=payload,
        generation=9,
    )
    event = event_store.list("test-session")[0]
    assert event["sequence"] == 0
    assert event["generation"] == 9
    assert event["event_type"] == EVENT_RUNTIME_STDOUT
    assert event["payload"] == payload
    assert isinstance(event["timestamp"], str)


def test_append_assigns_sequential_sequences(event_store: EventStore) -> None:
    for i in range(3):
        event_store.append(
            session_id="test-session",
            event_type=EVENT_RUNTIME_STDOUT,
            payload={"text": f"line {i}"},
            generation=0,
        )
    events = event_store.list("test-session")
    assert [e["sequence"] for e in events] == [0, 1, 2]
    assert [e["payload"] for e in events] == [{"text": "line 0"}, {"text": "line 1"}, {"text": "line 2"}]


def test_append_with_commit_false_still_readable_on_same_connection(event_store: EventStore) -> None:
    event_store.append(
        session_id="test-session",
        event_type=EVENT_RUNTIME_STDOUT,
        payload={"text": "batched"},
        generation=0,
        commit=False,
    )
    events = event_store.list("test-session")
    assert len(events) == 1
    assert events[0]["payload"] == {"text": "batched"}


def test_sequences_independent_per_session(event_store: EventStore) -> None:
    SessionStore(event_store.conn).create(_make_session("other-session"))
    event_store.append(session_id="test-session", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "a"}, generation=0)
    event_store.append(session_id="other-session", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "b"}, generation=0)
    event_store.append(session_id="test-session", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "c"}, generation=0)
    test_events = [(e["sequence"], e["payload"]) for e in event_store.list("test-session")]
    other_events = [(e["sequence"], e["payload"]) for e in event_store.list("other-session")]
    assert test_events == [(0, {"text": "a"}), (1, {"text": "c"})]
    assert other_events == [(0, {"text": "b"})]


@pytest.mark.parametrize(
    ("event_type", "after_sequence", "generation", "expected_payloads"),
    [
        (None, None, None, [{"text": "out1"}, {"text": "err"}, {"text": "out2"}, {"text": "gen1"}]),
        (EVENT_RUNTIME_STDOUT, None, None, [{"text": "out1"}, {"text": "out2"}, {"text": "gen1"}]),
        (None, 0, None, [{"text": "err"}, {"text": "out2"}, {"text": "gen1"}]),
        (EVENT_RUNTIME_STDOUT, 0, None, [{"text": "out2"}, {"text": "gen1"}]),
        (None, None, 0, [{"text": "out1"}, {"text": "err"}, {"text": "out2"}]),
        (None, None, 1, [{"text": "gen1"}]),
    ],
    ids=["no-filter", "type-only", "cursor-only", "type-and-cursor", "gen-zero", "gen-one"],
)
def test_list_applies_filters(
    event_store: EventStore,
    event_type: str | None,
    after_sequence: int | None,
    generation: int | None,
    expected_payloads: list[dict[str, object]],
) -> None:
    event_store.append(
        session_id="test-session", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "out1"}, generation=0
    )
    event_store.append(
        session_id="test-session", event_type=EVENT_RUNTIME_STDERR, payload={"text": "err"}, generation=0
    )
    event_store.append(
        session_id="test-session", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "out2"}, generation=0
    )
    event_store.append(
        session_id="test-session", event_type=EVENT_RUNTIME_STDOUT, payload={"text": "gen1"}, generation=1
    )
    events = event_store.list(
        "test-session", event_type=event_type, after_sequence=after_sequence, generation=generation
    )
    assert [e["payload"] for e in events] == expected_payloads


def _append(store: EventStore, etype: str, text: str) -> None:
    store.append(session_id="test-session", event_type=etype, payload={"text": text}, generation=0)


def test_list_recent_returns_empty_for_unknown_session(event_store: EventStore) -> None:
    assert event_store.list_recent("does-not-exist") == []


@pytest.mark.parametrize(
    ("seed_count", "limit", "expected_seqs"),
    [(5, 1000, list(range(5))), (20, 10, list(range(10, 20)))],
    ids=["under-limit", "over-limit"],
)
def test_list_recent_returns_correct_window(
    event_store: EventStore, seed_count: int, limit: int, expected_seqs: list[int]
) -> None:
    for i in range(seed_count):
        _append(event_store, EVENT_RUNTIME_STDOUT, f"line {i}")
    events = event_store.list_recent("test-session", limit=limit)
    assert [e["sequence"] for e in events] == expected_seqs
    assert [e["payload"] for e in events] == [{"text": f"line {i}"} for i in expected_seqs]


def test_list_recent_filters_by_event_type(event_store: EventStore) -> None:
    _append(event_store, EVENT_RUNTIME_STDOUT, "out1")
    _append(event_store, EVENT_RUNTIME_STDERR, "err1")
    _append(event_store, EVENT_RUNTIME_STDOUT, "out2")
    events = event_store.list_recent("test-session", event_type=EVENT_RUNTIME_STDOUT, limit=1000)
    assert [e["payload"] for e in events] == [{"text": "out1"}, {"text": "out2"}]


def test_list_recent_uses_single_bounded_query(event_store: EventStore) -> None:
    for i in range(5):
        _append(event_store, EVENT_RUNTIME_STDOUT, f"line {i}")
    captured: list[str] = []
    event_store.conn.set_trace_callback(captured.append)
    try:
        event_store.list_recent("test-session", event_type=EVENT_RUNTIME_STDOUT, limit=3)
    finally:
        event_store.conn.set_trace_callback(None)
    upper_sql = " ".join(captured).upper()
    assert "ORDER BY SEQUENCE DESC" in upper_sql
    assert "LIMIT" in upper_sql
