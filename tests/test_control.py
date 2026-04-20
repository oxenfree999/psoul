"""Tests for the control module: signal handlers, command dispatch, terminal emission."""

import os
import signal
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from psoul.core.control import (
    COMMAND_ACCEPTED,
    COMMAND_COMPLETED,
    KILL_COMMAND,
    OUTCOME_ESCALATED,
    OUTCOME_NOOP,
    OUTCOME_OK,
    STOP_COMMAND,
    ControlState,
    PendingCommand,
    SupervisedProc,
    _new_message_id,
    check_escalation,
    emit_terminal_events,
    handle_kill,
    handle_stop,
    install_handlers,
)
from psoul.core.db import open_db
from psoul.core.events import EventStore
from psoul.core.launch import LaunchRequest, LaunchTarget, launch_headless
from psoul.core.session import LaunchMode, Session, SessionState, TargetType
from psoul.core.store import SessionStore
from psoul.version import VERSION

if sys.platform == "win32":
    pytest.skip("test_control.py requires Unix signals (SIGKILL / SIGUSR1 / SIGUSR2)", allow_module_level=True)

requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")


def _wait_until_running(store: SessionStore, session_id: str, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = store.get(session_id)
        if session is not None and session.state == SessionState.running:
            return
        time.sleep(0.02)
    msg = f"session {session_id} did not reach running within {timeout}s"
    raise RuntimeError(msg)


def _wait_for_path(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    msg = f"path {path} did not appear within {timeout}s"
    raise RuntimeError(msg)


def _assert_pid_dies_within(pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    msg = f"PID {pid} still alive after {timeout}s"
    raise AssertionError(msg)


@pytest.fixture
def stores(tmp_path: Path) -> Iterator[tuple[SessionStore, EventStore]]:
    conn = open_db(tmp_path)
    yield SessionStore(conn), EventStore(conn)
    conn.close()


def _state_requesting(command: str) -> ControlState:
    """Return a fresh ``ControlState`` with the request flag for *command* flipped on."""
    if command == STOP_COMMAND:
        return ControlState(stop_requested=True)
    return ControlState(kill_requested=True)


def _seed_running_session(store: SessionStore, session_id: str) -> None:
    store.create(
        Session(
            session_id=session_id,
            state=SessionState.starting,
            launch_mode=LaunchMode.headless,
            launch_time=datetime.now(UTC),
            psoul_version=VERSION,
            target_type=TargetType.script,
            target="noop.py",
        )
    )
    store.update(session_id, state=SessionState.running)


class _ExitedProc:
    """Fake ``proc`` with ``poll()`` returning an exit code. PID is unused in noop paths."""

    pid = 999

    def poll(self) -> int:
        return 0


@pytest.mark.parametrize("command", [STOP_COMMAND, KILL_COMMAND], ids=["stop", "kill"])
def test_handler_on_already_exited_child_is_noop(command: str, stores: tuple[SessionStore, EventStore]) -> None:
    store, event_store = stores
    _seed_running_session(store, "sesh-a")
    state = _state_requesting(command)
    if command == STOP_COMMAND:
        handle_stop(state, _ExitedProc(), event_store, "sesh-a", 0, stop_timeout_seconds=10.0, store=store)
    else:
        handle_kill(state, _ExitedProc(), event_store, "sesh-a", 0, store=store)
    events = event_store.list("sesh-a")
    assert [e["event_type"] for e in events] == [COMMAND_ACCEPTED, COMMAND_COMPLETED]
    accepted_payload = cast("dict[str, object]", events[0]["payload"])
    completed_payload = cast("dict[str, object]", events[1]["payload"])
    assert accepted_payload["command"] == command
    assert completed_payload["outcome"] == OUTCOME_NOOP
    assert completed_payload["duration_ms"] == 0
    session = store.get("sesh-a")
    assert session is not None
    assert session.state == SessionState.running
    assert state.stop_requested is False
    assert state.kill_requested is False
    assert state.stopping is False
    assert state.pending is None


class _RunningProc:
    """Fake ``proc`` with ``poll()`` returning None (child alive). Paired with monkeypatched ``os.killpg``."""

    pid = 999

    def poll(self) -> None:
        return None


def test_handle_stop_noop_when_already_stopping(
    stores: tuple[SessionStore, EventStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, event_store = stores
    _seed_running_session(store, "sesh-b")
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("psoul.core.control.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    existing = PendingCommand(command=STOP_COMMAND, message_id="prior-mid", start_monotonic=0.0)
    state = ControlState(stopping=True, pending=existing)
    handle_stop(state, _RunningProc(), event_store, "sesh-b", 0, stop_timeout_seconds=10.0, store=store)
    events = event_store.list("sesh-b")
    assert [e["event_type"] for e in events] == [COMMAND_ACCEPTED, COMMAND_COMPLETED]
    completed_payload = cast("dict[str, object]", events[1]["payload"])
    assert completed_payload["outcome"] == OUTCOME_NOOP
    assert killpg_calls == []
    assert state.pending is existing  # pending untouched
    assert state.stop_requested is False


@pytest.mark.parametrize(
    ("command", "expected_signal", "expects_escalation"),
    [
        (STOP_COMMAND, signal.SIGTERM, True),
        (KILL_COMMAND, signal.SIGKILL, False),
    ],
    ids=["stop-arms-escalation", "kill-no-escalation"],
)
def test_handler_happy_path_signals_pgroup_and_records_pending(
    command: str,
    expected_signal: int,
    expects_escalation: bool,
    stores: tuple[SessionStore, EventStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, event_store = stores
    _seed_running_session(store, "sesh-x")
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("psoul.core.control.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    state = _state_requesting(command)
    if command == STOP_COMMAND:
        handle_stop(state, _RunningProc(), event_store, "sesh-x", 0, stop_timeout_seconds=10.0, store=store)
    else:
        handle_kill(state, _RunningProc(), event_store, "sesh-x", 0, store=store)
    assert killpg_calls == [(999, expected_signal)]
    events = event_store.list("sesh-x")
    assert [e["event_type"] for e in events] == [COMMAND_ACCEPTED]
    assert state.stopping is True
    assert state.pending is not None
    assert state.pending.command == command
    assert (state.escalation_deadline is not None) is expects_escalation
    assert state.stop_requested is False
    assert state.kill_requested is False
    session = store.get("sesh-x")
    assert session is not None
    assert session.state == SessionState.stopping


def test_handle_kill_supersedes_pre_escalation_stop(
    stores: tuple[SessionStore, EventStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kill while a stop is pending (pre-escalation) closes the stop with noop and opens a fresh kill."""
    store, event_store = stores
    _seed_running_session(store, "sesh-e")
    store.update("sesh-e", state=SessionState.stopping)  # stop already happened
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("psoul.core.control.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    pending_stop = PendingCommand(command=STOP_COMMAND, message_id="stop-mid", start_monotonic=0.0)
    state = ControlState(stopping=True, pending=pending_stop, escalation_deadline=999.0, kill_requested=True)
    handle_kill(state, _RunningProc(), event_store, "sesh-e", 0, store=store)
    assert killpg_calls == [(999, signal.SIGKILL)]
    events = event_store.list("sesh-e")
    assert [e["event_type"] for e in events] == [COMMAND_COMPLETED, COMMAND_ACCEPTED]
    stop_closer = cast("dict[str, object]", events[0]["payload"])
    kill_opener = cast("dict[str, object]", events[1]["payload"])
    assert stop_closer["command"] == STOP_COMMAND
    assert stop_closer["message_id"] == "stop-mid"
    assert stop_closer["outcome"] == OUTCOME_NOOP
    assert kill_opener["command"] == KILL_COMMAND
    assert kill_opener["message_id"] != "stop-mid"
    assert state.pending is not None
    assert state.pending.command == KILL_COMMAND
    assert state.escalation_deadline is None
    assert state.escalation_fired is False


@pytest.mark.parametrize(
    ("deadline_offset", "make_proc", "fires_kill", "final_deadline_is_none"),
    [
        (None, _RunningProc, False, True),
        (+60.0, _RunningProc, False, False),
        (-60.0, _ExitedProc, False, True),
        (-60.0, _RunningProc, True, True),
    ],
    ids=["no-deadline", "future-deadline", "passed-but-child-exited", "passed-and-alive"],
)
def test_check_escalation(
    deadline_offset: float | None,
    make_proc: Callable[[], SupervisedProc],
    fires_kill: bool,
    final_deadline_is_none: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("psoul.core.control.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    deadline = None if deadline_offset is None else time.monotonic() + deadline_offset
    state = ControlState(escalation_deadline=deadline)
    check_escalation(state, make_proc())
    assert (killpg_calls == [(999, signal.SIGKILL)]) is fires_kill
    assert state.escalation_fired is fires_kill
    assert (state.escalation_deadline is None) is final_deadline_is_none


@pytest.mark.parametrize(
    ("pending_command", "escalation_fired", "expected_outcome"),
    [
        (STOP_COMMAND, False, OUTCOME_OK),
        (STOP_COMMAND, True, OUTCOME_ESCALATED),
        (KILL_COMMAND, False, OUTCOME_OK),
    ],
    ids=["stop-ok", "stop-escalated", "kill-ok"],
)
def test_emit_terminal_events_closes_pending_command(
    pending_command: str,
    escalation_fired: bool,
    expected_outcome: str,
    stores: tuple[SessionStore, EventStore],
) -> None:
    store, event_store = stores
    _seed_running_session(store, "sesh-t")
    start = time.monotonic() - 0.5
    pending = PendingCommand(command=pending_command, message_id="mid-xyz", start_monotonic=start)
    state = ControlState(stopping=True, pending=pending, escalation_fired=escalation_fired)
    emit_terminal_events(state, event_store, "sesh-t", 0)
    events = event_store.list("sesh-t")
    assert [e["event_type"] for e in events] == [COMMAND_COMPLETED]
    payload = cast("dict[str, object]", events[0]["payload"])
    assert payload["command"] == pending_command
    assert payload["message_id"] == "mid-xyz"
    assert payload["outcome"] == expected_outcome
    assert isinstance(payload["duration_ms"], int)
    assert payload["duration_ms"] >= 500
    assert state.completion_emitted is True


@pytest.mark.parametrize("command", [STOP_COMMAND, KILL_COMMAND], ids=["late-stop", "late-kill"])
def test_emit_terminal_events_emits_late_noop_pair(command: str, stores: tuple[SessionStore, EventStore]) -> None:
    store, event_store = stores
    _seed_running_session(store, "sesh-late")
    state = _state_requesting(command)
    emit_terminal_events(state, event_store, "sesh-late", 0)
    events = event_store.list("sesh-late")
    assert [e["event_type"] for e in events] == [COMMAND_ACCEPTED, COMMAND_COMPLETED]
    accepted_payload = cast("dict[str, object]", events[0]["payload"])
    completed_payload = cast("dict[str, object]", events[1]["payload"])
    assert accepted_payload["command"] == command
    assert completed_payload["command"] == command
    assert completed_payload["outcome"] == OUTCOME_NOOP
    assert completed_payload["duration_ms"] == 0
    assert state.stop_requested is False
    assert state.kill_requested is False


def test_emit_terminal_events_emits_both_late_noops_without_overwriting(
    stores: tuple[SessionStore, EventStore],
) -> None:
    """Both flags set late: two independent paired noops, one per command."""
    store, event_store = stores
    _seed_running_session(store, "sesh-both")
    state = ControlState(stop_requested=True, kill_requested=True)
    emit_terminal_events(state, event_store, "sesh-both", 0)
    events = event_store.list("sesh-both")
    types = [e["event_type"] for e in events]
    assert types.count(COMMAND_ACCEPTED) == 2
    assert types.count(COMMAND_COMPLETED) == 2
    commands = [cast("dict[str, object]", e["payload"])["command"] for e in events]
    assert STOP_COMMAND in commands
    assert KILL_COMMAND in commands
    assert state.stop_requested is False
    assert state.kill_requested is False


def test_handle_kill_after_escalation_is_noop_preserving_pending_stop(
    stores: tuple[SessionStore, EventStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kill arriving after stop has already escalated emits paired noop without overwriting the pending stop."""
    store, event_store = stores
    _seed_running_session(store, "sesh-esc")
    store.update("sesh-esc", state=SessionState.stopping)
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("psoul.core.control.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    pending_stop = PendingCommand(command=STOP_COMMAND, message_id="stop-mid", start_monotonic=0.0)
    state = ControlState(stopping=True, pending=pending_stop, escalation_fired=True, kill_requested=True)
    handle_kill(state, _RunningProc(), event_store, "sesh-esc", 0, store=store)
    assert killpg_calls == []
    events = event_store.list("sesh-esc")
    assert [e["event_type"] for e in events] == [COMMAND_ACCEPTED, COMMAND_COMPLETED]
    kill_accepted = cast("dict[str, object]", events[0]["payload"])
    kill_completed = cast("dict[str, object]", events[1]["payload"])
    assert kill_accepted["command"] == KILL_COMMAND
    assert kill_completed["outcome"] == OUTCOME_NOOP
    assert state.pending is pending_stop
    assert state.escalation_fired is True
    assert state.kill_requested is False


@pytest.mark.parametrize("command", [STOP_COMMAND, KILL_COMMAND], ids=["stop", "kill"])
def test_handler_toctou_guard_emits_noop_when_state_moved_out_of_running(
    command: str,
    stores: tuple[SessionStore, EventStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row moved out of running between CLI validation and handler dispatch: emit paired noop, no killpg."""
    store, event_store = stores
    _seed_running_session(store, "sesh-toc")
    store.update("sesh-toc", state=SessionState.failed)
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("psoul.core.control.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    state = _state_requesting(command)
    if command == STOP_COMMAND:
        handle_stop(state, _RunningProc(), event_store, "sesh-toc", 0, stop_timeout_seconds=10.0, store=store)
    else:
        handle_kill(state, _RunningProc(), event_store, "sesh-toc", 0, store=store)
    assert killpg_calls == []
    events = event_store.list("sesh-toc")
    assert [e["event_type"] for e in events] == [COMMAND_ACCEPTED, COMMAND_COMPLETED]
    completed_payload = cast("dict[str, object]", events[1]["payload"])
    assert completed_payload["outcome"] == OUTCOME_NOOP
    assert state.stopping is False
    assert state.pending is None
    assert state.stop_requested is False
    assert state.kill_requested is False


def test_handle_kill_duplicate_while_stopping_is_noop(
    stores: tuple[SessionStore, EventStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second kill while a kill is already pending emits paired noop with a fresh message_id."""
    store, event_store = stores
    _seed_running_session(store, "sesh-dup")
    store.update("sesh-dup", state=SessionState.stopping)
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("psoul.core.control.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    pending_kill = PendingCommand(command=KILL_COMMAND, message_id="kill-mid", start_monotonic=0.0)
    state = ControlState(stopping=True, pending=pending_kill, kill_requested=True)
    handle_kill(state, _RunningProc(), event_store, "sesh-dup", 0, store=store)
    assert killpg_calls == []
    events = event_store.list("sesh-dup")
    assert [e["event_type"] for e in events] == [COMMAND_ACCEPTED, COMMAND_COMPLETED]
    accepted_payload = cast("dict[str, object]", events[0]["payload"])
    completed_payload = cast("dict[str, object]", events[1]["payload"])
    assert accepted_payload["message_id"] != "kill-mid"
    assert completed_payload["outcome"] == OUTCOME_NOOP
    assert state.pending is pending_kill
    assert state.kill_requested is False


@pytest.mark.parametrize(
    ("command", "expected_signal"),
    [(STOP_COMMAND, signal.SIGTERM), (KILL_COMMAND, signal.SIGKILL)],
    ids=["stop", "kill"],
)
def test_handler_closes_with_noop_when_killpg_races_child_exit(
    command: str,
    expected_signal: int,
    stores: tuple[SessionStore, EventStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child dies between poll() gate and killpg: accepted + completed(noop) pair lands, state self-closes."""
    store, event_store = stores
    _seed_running_session(store, "sesh-race")
    killpg_calls: list[tuple[int, int]] = []

    def _record_then_raise(pid: int, sig: int) -> None:
        killpg_calls.append((pid, sig))
        raise ProcessLookupError

    monkeypatch.setattr("psoul.core.control.os.killpg", _record_then_raise)
    state = _state_requesting(command)
    if command == STOP_COMMAND:
        handle_stop(state, _RunningProc(), event_store, "sesh-race", 0, stop_timeout_seconds=10.0, store=store)
    else:
        handle_kill(state, _RunningProc(), event_store, "sesh-race", 0, store=store)
    assert killpg_calls == [(999, expected_signal)]
    events = event_store.list("sesh-race")
    assert [e["event_type"] for e in events] == [COMMAND_ACCEPTED, COMMAND_COMPLETED]
    completed = cast("dict[str, object]", events[1]["payload"])
    assert completed["command"] == command
    assert completed["outcome"] == OUTCOME_NOOP
    assert completed["duration_ms"] == 0
    assert state.completion_emitted is True
    assert state.escalation_deadline is None
    assert state.stop_requested is False
    assert state.kill_requested is False


def test_handle_kill_supersede_closes_with_noop_when_killpg_races_child_exit(
    stores: tuple[SessionStore, EventStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supersede path: stop closes noop, kill accepted, kill's killpg races child exit -> kill closes noop too."""
    store, event_store = stores
    _seed_running_session(store, "sesh-sup-race")
    store.update("sesh-sup-race", state=SessionState.stopping)

    def _raise(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("psoul.core.control.os.killpg", _raise)
    pending_stop = PendingCommand(command=STOP_COMMAND, message_id="stop-mid", start_monotonic=0.0)
    state = ControlState(stopping=True, pending=pending_stop, escalation_deadline=999.0, kill_requested=True)
    handle_kill(state, _RunningProc(), event_store, "sesh-sup-race", 0, store=store)
    events = event_store.list("sesh-sup-race")
    assert [e["event_type"] for e in events] == [COMMAND_COMPLETED, COMMAND_ACCEPTED, COMMAND_COMPLETED]
    stop_closer = cast("dict[str, object]", events[0]["payload"])
    kill_accepted = cast("dict[str, object]", events[1]["payload"])
    kill_closer = cast("dict[str, object]", events[2]["payload"])
    assert stop_closer["command"] == STOP_COMMAND
    assert stop_closer["outcome"] == OUTCOME_NOOP
    assert kill_accepted["command"] == KILL_COMMAND
    assert kill_closer["command"] == KILL_COMMAND
    assert kill_closer["outcome"] == OUTCOME_NOOP
    assert state.completion_emitted is True
    assert state.kill_requested is False


def test_check_escalation_swallows_race_when_killpg_finds_child_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """killpg races child exit after deadline: deadline clears, escalation_fired stays False."""

    def _raise(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("psoul.core.control.os.killpg", _raise)
    state = ControlState(escalation_deadline=time.monotonic() - 60.0)
    check_escalation(state, _RunningProc())
    assert state.escalation_deadline is None
    assert state.escalation_fired is False


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.parametrize(
    ("sig", "expected_command", "expected_final_state"),
    [
        (signal.SIGUSR1, STOP_COMMAND, SessionState.exited),
        (signal.SIGUSR2, KILL_COMMAND, SessionState.failed),
    ],
    ids=["sigusr1-stop-exits", "sigusr2-kill-fails"],
)
def test_signal_end_to_end_emits_ok_and_transitions_terminal(
    sig: int, expected_command: str, expected_final_state: SessionState, tmp_path: Path
) -> None:
    """Launch headless, signal the supervisor, assert completed(ok) + the terminal state."""
    # Stop: child's SIGTERM handler exits cleanly -> final state exited.
    # Kill: SIGKILL cannot be caught -> child dies with non-zero exit -> final state failed.
    child_script = (
        "import signal, sys, time\n"
        "def h(_s, _f): sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, h)\n"
        "while True: time.sleep(0.05)\n"
    )
    with closing(open_db(tmp_path)) as conn:
        store = SessionStore(conn)
        req = LaunchRequest(
            session_id="e2e",
            launch_mode=LaunchMode.headless,
            target=LaunchTarget(TargetType.script, "-c", (child_script,), Path(sys.executable)),
            cwd=tmp_path,
        )
        _, supervisor_pid = launch_headless(req, store, tmp_path)
        _wait_until_running(store, "e2e")
        os.kill(supervisor_pid, sig)
        os.waitpid(supervisor_pid, 0)
        event_store = EventStore(conn)
        events = event_store.list("e2e", event_type=COMMAND_COMPLETED)
        assert len(events) == 1
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["command"] == expected_command
        assert payload["outcome"] == OUTCOME_OK
        final = store.get("e2e")
        assert final is not None
        assert final.state == expected_final_state


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_stop_escalates_to_sigkill_when_child_ignores_sigterm(tmp_path: Path) -> None:
    """Child ignores SIGTERM; supervisor escalates to SIGKILL after stop_timeout and emits completed(escalated)."""
    child_script = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(0.05)\n"
    with closing(open_db(tmp_path)) as conn:
        store = SessionStore(conn)
        req = LaunchRequest(
            session_id="esc",
            launch_mode=LaunchMode.headless,
            target=LaunchTarget(TargetType.script, "-c", (child_script,), Path(sys.executable)),
            cwd=tmp_path,
        )
        _, supervisor_pid = launch_headless(req, store, tmp_path, stop_timeout_seconds=1.0)
        _wait_until_running(store, "esc")
        os.kill(supervisor_pid, signal.SIGUSR1)
        os.waitpid(supervisor_pid, 0)
        event_store = EventStore(conn)
        events = event_store.list("esc", event_type=COMMAND_COMPLETED)
        assert len(events) == 1
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["command"] == STOP_COMMAND
        assert payload["outcome"] == OUTCOME_ESCALATED
        duration_ms = payload["duration_ms"]
        assert isinstance(duration_ms, int)
        assert 1000 <= duration_ms < 5000
        final = store.get("esc")
        assert final is not None
        assert final.state == SessionState.failed


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_kill_process_group_scope(tmp_path: Path) -> None:
    """SIGUSR2 reaches a grandchild in the same pgroup: `os.killpg(child_pid, SIGKILL)` kills both."""
    pid_file = tmp_path / "grandchild.pid"
    child_script = (
        "import os, sys, time, pathlib\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "    while True: time.sleep(0.05)\n"
        "while True: time.sleep(0.05)\n"
    )
    with closing(open_db(tmp_path)) as conn:
        store = SessionStore(conn)
        req = LaunchRequest(
            session_id="grp",
            launch_mode=LaunchMode.headless,
            target=LaunchTarget(TargetType.script, "-c", (child_script, str(pid_file)), Path(sys.executable)),
            cwd=tmp_path,
        )
        _, supervisor_pid = launch_headless(req, store, tmp_path)
        _wait_until_running(store, "grp")
        _wait_for_path(pid_file, timeout=3.0)
        grandchild_pid = int(pid_file.read_text())
        os.kill(grandchild_pid, 0)  # sanity: grandchild is alive before the pgroup kill
        os.kill(supervisor_pid, signal.SIGUSR2)
        os.waitpid(supervisor_pid, 0)
        _assert_pid_dies_within(grandchild_pid, 5.0)


def test_new_message_id_is_valid_uuid4() -> None:
    mid = _new_message_id()
    parsed = uuid.UUID(mid)
    assert parsed.version == 4


@pytest.mark.parametrize(
    ("sig", "flipped_attr", "unaffected_attr"),
    [
        (signal.SIGUSR1, "stop_requested", "kill_requested"),
        (signal.SIGUSR2, "kill_requested", "stop_requested"),
    ],
    ids=["sigusr1-flips-stop", "sigusr2-flips-kill"],
)
def test_signal_flips_only_its_own_flag(sig: int, flipped_attr: str, unaffected_attr: str) -> None:
    state = ControlState()
    assert getattr(state, flipped_attr) is False
    assert getattr(state, unaffected_attr) is False
    original_usr1 = signal.getsignal(signal.SIGUSR1)
    original_usr2 = signal.getsignal(signal.SIGUSR2)
    try:
        install_handlers(state)
        os.kill(os.getpid(), sig)
        assert getattr(state, flipped_attr) is True
        assert getattr(state, unaffected_attr) is False
    finally:
        signal.signal(signal.SIGUSR1, original_usr1)
        signal.signal(signal.SIGUSR2, original_usr2)
