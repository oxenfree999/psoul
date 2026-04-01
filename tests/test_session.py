"""Tests for session domain model: enums, transitions, validation, and dataclass."""

from datetime import UTC, datetime

import pytest

from psoul.session import (
    SESSION_ID_MAX_LENGTH,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    LaunchMode,
    Session,
    SessionError,
    SessionState,
    TargetType,
    check_transition,
    validate_session_id,
)


def _session(
    *,
    session_id: str = "calm-tiger-builds-kites",
    state: SessionState = SessionState.running,
    launch_mode: LaunchMode = LaunchMode.attached,
    psoul_version: str = "0.0.1",
) -> Session:
    """Build a Session with sensible defaults for required fields."""
    return Session(
        session_id=session_id,
        state=state,
        launch_mode=launch_mode,
        launch_time=datetime.now(UTC),
        psoul_version=psoul_version,
    )


def test_session_state_values() -> None:
    expected = {
        "starting",
        "running",
        "suspended",
        "debugging",
        "restarting",
        "stopping",
        "exited",
        "failed",
        "orphaned",
    }
    assert {s.value for s in SessionState} == expected


def test_launch_mode_values() -> None:
    assert {m.value for m in LaunchMode} == {"attached", "headless"}


def test_target_type_values() -> None:
    assert {t.value for t in TargetType} == {"script", "module", "repl"}


def test_every_state_has_transitions_entry() -> None:
    assert set(VALID_TRANSITIONS) == set(SessionState)


def test_terminal_states_have_no_exits() -> None:
    for state in TERMINAL_STATES:
        assert VALID_TRANSITIONS[state] == frozenset()


_ALL_VALID_PAIRS = [(src, dst) for src, targets in VALID_TRANSITIONS.items() for dst in targets]
_ALL_INVALID_PAIRS = [(src, dst) for src in SessionState for dst in SessionState if dst not in VALID_TRANSITIONS[src]]


@pytest.mark.parametrize(("from_state", "to_state"), _ALL_VALID_PAIRS)
def test_valid_transition_accepted(from_state: SessionState, to_state: SessionState) -> None:
    check_transition(from_state, to_state)


@pytest.mark.parametrize(("from_state", "to_state"), _ALL_INVALID_PAIRS)
def test_invalid_transition_raises(from_state: SessionState, to_state: SessionState) -> None:
    with pytest.raises(SessionError, match="invalid state transition"):
        check_transition(from_state, to_state)


@pytest.mark.parametrize(
    "session_id",
    [
        "calm-tiger-builds-kites",
        "a",
        "hello",
        "my-session",
        "run-42",
        "a-b-c-d-e-f",
        "abc123",
        "a" * SESSION_ID_MAX_LENGTH,
    ],
)
def test_validate_session_id_accepts_valid(session_id: str) -> None:
    assert validate_session_id(session_id) == session_id


def test_validate_session_id_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_session_id("")


def test_validate_session_id_rejects_too_long() -> None:
    with pytest.raises(ValueError, match="exceeds 64 characters"):
        validate_session_id("a" * (SESSION_ID_MAX_LENGTH + 1))


@pytest.mark.parametrize(
    "session_id",
    [
        "Calm-Tiger",
        "calm_tiger",
        "calm.tiger",
        "calm tiger",
        "-leading",
        "trailing-",
        "double--hyphen",
    ],
)
def test_validate_session_id_rejects_bad_pattern(session_id: str) -> None:
    with pytest.raises(ValueError, match="lowercase alphanumeric segments"):
        validate_session_id(session_id)


def test_session_rejects_invalid_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _session(session_id="")
