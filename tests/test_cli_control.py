"""Tests for the psoul stop, kill, pause, and resume CLI commands."""

import json
import os
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import psutil
import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.cli.state import ExitCode
from psoul.core.control import (
    COMMAND_ACCEPTED,
    COMMAND_COMPLETED,
    KILL_COMMAND,
    OUTCOME_ESCALATED,
    OUTCOME_OK,
    PAUSE_COMMAND,
    RESUME_COMMAND,
    STOP_COMMAND,
)
from psoul.core.db import open_db
from psoul.core.events import EventStore
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")


def _wait_for_state(state_dir: Path, session_id: str, target: SessionState, timeout: float = 3.0) -> None:
    """Poll the DB until *session_id* is in *target* state or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(open_db(state_dir)) as conn:
            session = SessionStore(conn).get(session_id)
        if session is not None and session.state == target:
            return
        time.sleep(0.02)
    msg = f"session {session_id} did not reach {target} within {timeout}s"
    raise RuntimeError(msg)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_stop_cli_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """psoul run --headless then psoul stop: assert supervisor emits completed(stop, ok) and session exits."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "graceful.py"
    script.write_text(
        "import signal, sys, time\n"
        "def h(_s, _f): sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, h)\n"
        "while True: time.sleep(0.05)\n"
    )
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    stop = runner.invoke(cli, ["stop", session_id])
    assert stop.exit_code == 0
    assert f"Sent stop to {session_id}." in stop.output
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        completed = EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
    assert len(completed) == 1
    payload = cast("dict[str, object]", completed[0]["payload"])
    assert payload["command"] == STOP_COMMAND
    assert payload["outcome"] == OUTCOME_OK


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_kill_cli_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """psoul run --headless then psoul kill: assert supervisor emits completed(kill, ok) and child dies via SIGKILL."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "sleeper.py"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    kill = runner.invoke(cli, ["kill", session_id])
    assert kill.exit_code == 0
    assert f"Sent kill to {session_id}." in kill.output
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        completed = EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
    assert len(completed) == 1
    payload = cast("dict[str, object]", completed[0]["payload"])
    assert payload["command"] == KILL_COMMAND
    assert payload["outcome"] == OUTCOME_OK


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_pause_cli_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "sleeper.py"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    pause = runner.invoke(cli, ["pause", session_id])
    assert pause.exit_code == 0
    assert f"Sent pause to {session_id}." in pause.output
    _wait_for_state(tmp_path, session_id, SessionState.suspended)
    runner.invoke(cli, ["kill", session_id])
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        completed = EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
    commands = [cast("dict[str, object]", e["payload"])["command"] for e in completed]
    assert PAUSE_COMMAND in commands


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_resume_cli_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "sleeper.py"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    assert runner.invoke(cli, ["pause", session_id]).exit_code == 0
    _wait_for_state(tmp_path, session_id, SessionState.suspended)
    resume = runner.invoke(cli, ["resume", session_id])
    assert resume.exit_code == 0
    assert f"Sent resume to {session_id}." in resume.output
    _wait_for_state(tmp_path, session_id, SessionState.running)
    runner.invoke(cli, ["kill", session_id])
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        completed = EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
    commands = [cast("dict[str, object]", e["payload"])["command"] for e in completed]
    assert PAUSE_COMMAND in commands
    assert RESUME_COMMAND in commands


def test_stop_unknown_session_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A selector that matches no session surfaces a session-not-found error."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    open_db(tmp_path).close()
    result = runner.invoke(cli, ["stop", "does-not-exist"])
    assert result.exit_code == ExitCode.USAGE
    assert "session not found" in result.output


@pytest.mark.parametrize(
    ("launch_mode", "state", "expected_msg"),
    [
        (LaunchMode.attached, SessionState.running, "requires a headless session"),
        (LaunchMode.headless, SessionState.orphaned, "session is orphaned"),
        (LaunchMode.headless, SessionState.stopping, "session is already stopping"),
        (LaunchMode.headless, SessionState.exited, "session is not running or suspended"),
        (LaunchMode.headless, SessionState.failed, "session is not running or suspended"),
        (LaunchMode.headless, SessionState.starting, "session is not running or suspended"),
        (LaunchMode.headless, SessionState.debugging, "session is not running or suspended"),
        (LaunchMode.headless, SessionState.restarting, "session is not running or suspended"),
    ],
    ids=["attached", "orphaned", "stopping", "exited", "failed", "starting", "debugging", "restarting"],
)
def test_stop_rejects_non_accept_state(
    launch_mode: LaunchMode,
    state: SessionState,
    expected_msg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each non-accept (launch_mode, state) combination gets its specific error message and exits non-zero."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=state,
                launch_mode=launch_mode,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )
    result = runner.invoke(cli, ["stop", "seed"])
    assert result.exit_code == ExitCode.USAGE
    assert expected_msg in result.output


def test_stop_missing_supervisor_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A running headless session with supervisor_pid=None surfaces the no-supervisor error."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=None,
            )
        )
    result = runner.invoke(cli, ["stop", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "session has no supervisor" in result.output


def test_stop_supervisor_process_lookup_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """os.kill raising ProcessLookupError surfaces the supervisor-not-running error."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)

    def _raise_lookup(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("psoul.cli.main.os.kill", _raise_lookup)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )
    result = runner.invoke(cli, ["stop", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "supervisor process is not running" in result.output


def test_stop_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """os.kill raising PermissionError surfaces the permission-denied error with the supervisor PID."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)

    def _raise_perm(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr("psoul.cli.main.os.kill", _raise_perm)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )
    result = runner.invoke(cli, ["stop", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "permission denied signalling supervisor" in result.output
    assert f"(PID={os.getpid()})" in result.output


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_stop_timeout_config_wiring_honors_1s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """[process] stop_timeout = '1s' threads from TOML through run into handle_stop's escalation deadline."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    config = tmp_path / "psoul.toml"
    config.write_text('[process]\nstop_timeout = "1s"\n')
    script = tmp_path / "ignore_term.py"
    script.write_text(
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(0.05)\n"
    )
    launch = runner.invoke(cli, ["--config", str(config), "run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    stop = runner.invoke(cli, ["--config", str(config), "stop", session_id])
    assert stop.exit_code == 0
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        completed = EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
    assert len(completed) == 1
    payload = cast("dict[str, object]", completed[0]["payload"])
    assert payload["command"] == STOP_COMMAND
    assert payload["outcome"] == OUTCOME_ESCALATED
    duration_ms = payload["duration_ms"]
    assert isinstance(duration_ms, int)
    assert 1000 <= duration_ms < 2500


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_stop_on_suspended_session_waits_for_sigkill_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    config = tmp_path / "psoul.toml"
    config.write_text('[process]\nstop_timeout = "1s"\n')
    script = tmp_path / "ignore_term.py"
    script.write_text(
        "import pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(__file__).parent.joinpath('child_ready').touch()\n"
        "while True: time.sleep(0.05)\n"
    )
    launch = runner.invoke(cli, ["--config", str(config), "run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    ready_file = tmp_path / "child_ready"
    ready_deadline = time.monotonic() + 5.0
    while time.monotonic() < ready_deadline and not ready_file.exists():
        time.sleep(0.02)
    assert ready_file.exists(), "child did not signal ready within 5s"
    assert runner.invoke(cli, ["--config", str(config), "pause", session_id]).exit_code == 0
    _wait_for_state(tmp_path, session_id, SessionState.suspended)
    stop = runner.invoke(cli, ["--config", str(config), "stop", session_id])
    assert stop.exit_code == 0
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        stop_events = [
            e
            for e in EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
            if cast("dict[str, object]", e["payload"])["command"] == STOP_COMMAND
        ]
        final = SessionStore(conn).get(session_id)
    assert len(stop_events) == 1
    assert cast("dict[str, object]", stop_events[0]["payload"])["outcome"] == OUTCOME_ESCALATED
    assert final is not None
    assert final.state == SessionState.failed


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_kill_on_suspended_session_terminates_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "sleeper.py"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    assert runner.invoke(cli, ["pause", session_id]).exit_code == 0
    _wait_for_state(tmp_path, session_id, SessionState.suspended)
    kill = runner.invoke(cli, ["kill", session_id])
    assert kill.exit_code == 0
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        kill_events = [
            e
            for e in EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
            if cast("dict[str, object]", e["payload"])["command"] == KILL_COMMAND
        ]
        final = SessionStore(conn).get(session_id)
    assert len(kill_events) == 1
    assert cast("dict[str, object]", kill_events[0]["payload"])["outcome"] == OUTCOME_OK
    assert final is not None
    assert final.state == SessionState.failed


@pytest.mark.parametrize("command", ["pause", "resume"], ids=["pause", "resume"])
def test_pause_resume_unknown_selector(command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    open_db(tmp_path).close()
    result = runner.invoke(cli, [command, "does-not-exist"])
    assert result.exit_code == ExitCode.USAGE
    assert "session not found" in result.output


@pytest.mark.parametrize(
    ("command", "accept_state"),
    [("pause", SessionState.running), ("resume", SessionState.suspended)],
    ids=["pause", "resume"],
)
def test_pause_resume_missing_supervisor_pid(
    command: str, accept_state: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=accept_state,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=None,
            )
        )
    result = runner.invoke(cli, [command, "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "session has no supervisor" in result.output


@pytest.mark.parametrize(
    ("launch_mode", "state", "expected_msg"),
    [
        (LaunchMode.attached, SessionState.running, "requires a headless session"),
        (LaunchMode.headless, SessionState.orphaned, "session is orphaned"),
        (LaunchMode.headless, SessionState.stopping, "session is already stopping"),
        (LaunchMode.headless, SessionState.starting, "session is not running"),
        (LaunchMode.headless, SessionState.suspended, "session is not running"),
        (LaunchMode.headless, SessionState.exited, "session is not running"),
        (LaunchMode.headless, SessionState.failed, "session is not running"),
        (LaunchMode.headless, SessionState.debugging, "session is not running"),
        (LaunchMode.headless, SessionState.restarting, "session is not running"),
    ],
    ids=["attached", "orphaned", "stopping", "starting", "suspended", "exited", "failed", "debugging", "restarting"],
)
def test_pause_rejects_non_accept_state(
    launch_mode: LaunchMode,
    state: SessionState,
    expected_msg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=state,
                launch_mode=launch_mode,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )
    result = runner.invoke(cli, ["pause", "seed"])
    assert result.exit_code == ExitCode.USAGE
    assert expected_msg in result.output


@pytest.mark.parametrize(
    ("launch_mode", "state", "expected_msg"),
    [
        (LaunchMode.attached, SessionState.suspended, "requires a headless session"),
        (LaunchMode.headless, SessionState.orphaned, "session is orphaned"),
        (LaunchMode.headless, SessionState.stopping, "session is already stopping"),
        (LaunchMode.headless, SessionState.starting, "session is not suspended"),
        (LaunchMode.headless, SessionState.running, "session is not suspended"),
        (LaunchMode.headless, SessionState.exited, "session is not suspended"),
        (LaunchMode.headless, SessionState.failed, "session is not suspended"),
        (LaunchMode.headless, SessionState.debugging, "session is not suspended"),
        (LaunchMode.headless, SessionState.restarting, "session is not suspended"),
    ],
    ids=["attached", "orphaned", "stopping", "starting", "running", "exited", "failed", "debugging", "restarting"],
)
def test_resume_rejects_non_accept_state(
    launch_mode: LaunchMode,
    state: SessionState,
    expected_msg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=state,
                launch_mode=launch_mode,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )
    result = runner.invoke(cli, ["resume", "seed"])
    assert result.exit_code == ExitCode.USAGE
    assert expected_msg in result.output


class _FakeChild:
    """Minimal stand-in for a ``psutil.Process`` child entry."""

    def __init__(self, pid: int) -> None:
        self.pid = pid


class _FakeProcess:
    """Minimal stand-in for ``psutil.Process`` that exposes ``.children()``."""

    def __init__(self, children: list[_FakeChild]) -> None:
        self._children = children

    def children(self) -> list[_FakeChild]:
        return self._children


def _seed_runnable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )


def test_pause_supervisor_vanished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_runnable(tmp_path, monkeypatch)

    def _raise(_pid: int) -> None:
        raise psutil.NoSuchProcess(_pid)

    monkeypatch.setattr("psoul.cli.main.psutil.Process", _raise)
    result = runner.invoke(cli, ["pause", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "supervisor process is not running" in result.output


def test_pause_supervisor_has_no_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_runnable(tmp_path, monkeypatch)
    monkeypatch.setattr("psoul.cli.main.psutil.Process", lambda _pid: _FakeProcess([]))
    result = runner.invoke(cli, ["pause", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "supervisor has no managed child" in result.output


def test_pause_killpg_process_lookup_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_runnable(tmp_path, monkeypatch)
    monkeypatch.setattr("psoul.cli.main.psutil.Process", lambda _pid: _FakeProcess([_FakeChild(99999)]))

    def _raise(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("psoul.cli.main.os.killpg", _raise)
    result = runner.invoke(cli, ["pause", "seed"])
    assert result.exit_code == 0
    assert "child has already exited" in result.output


def test_pause_killpg_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_runnable(tmp_path, monkeypatch)
    monkeypatch.setattr("psoul.cli.main.psutil.Process", lambda _pid: _FakeProcess([_FakeChild(99999)]))

    def _raise(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr("psoul.cli.main.os.killpg", _raise)
    result = runner.invoke(cli, ["pause", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "permission denied signalling child" in result.output


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_signal_sigusr1_ignored_by_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Child installs SIG_IGN for SIGUSR1; signal delivers but the session stays running."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "ignore_usr1.py"
    script.write_text(
        "import pathlib, signal, time\n"
        "signal.signal(signal.SIGUSR1, signal.SIG_IGN)\n"
        "pathlib.Path(__file__).parent.joinpath('child_ready').touch()\n"
        "while True: time.sleep(0.05)\n"
    )
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    ready_file = tmp_path / "child_ready"
    ready_deadline = time.monotonic() + 5.0
    while time.monotonic() < ready_deadline and not ready_file.exists():
        time.sleep(0.02)
    assert ready_file.exists(), "child did not signal ready within 5s"
    result = runner.invoke(cli, ["signal", session_id, "USR1"])
    assert result.exit_code == 0
    assert f"Sent SIGUSR1 to {session_id}." in result.output
    with closing(open_db(tmp_path)) as conn:
        after_signal = SessionStore(conn).get(session_id)
    assert after_signal is not None
    assert after_signal.state == SessionState.running
    runner.invoke(cli, ["kill", session_id])
    os.waitpid(supervisor_pid, 0)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.parametrize("signal_name", ["TERM", "KILL"], ids=["term", "kill"])
def test_signal_terminating_signals_fail_session(
    signal_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM / SIGKILL via signal end the session. No command.* events are emitted for the signal path."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "sleeper.py"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    result = runner.invoke(cli, ["signal", session_id, signal_name])
    assert result.exit_code == 0
    assert f"Sent SIG{signal_name} to {session_id}." in result.output
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        accepted = EventStore(conn).list(session_id, event_type=COMMAND_ACCEPTED)
        completed = EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
        final = SessionStore(conn).get(session_id)
    assert accepted == []
    assert completed == []
    assert final is not None
    assert final.state == SessionState.failed


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_signal_sigstop_and_sigcont_are_observed_as_pause_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGSTOP via signal triggers pause events; SIGCONT via signal triggers resume events."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "sleeper.py"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_state(tmp_path, session_id, SessionState.running)
    stop_result = runner.invoke(cli, ["signal", session_id, "STOP"])
    assert stop_result.exit_code == 0
    assert f"Sent SIGSTOP to {session_id}." in stop_result.output
    _wait_for_state(tmp_path, session_id, SessionState.suspended)
    cont_result = runner.invoke(cli, ["signal", session_id, "CONT"])
    assert cont_result.exit_code == 0
    assert f"Sent SIGCONT to {session_id}." in cont_result.output
    _wait_for_state(tmp_path, session_id, SessionState.running)
    runner.invoke(cli, ["kill", session_id])
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        completed = EventStore(conn).list(session_id, event_type=COMMAND_COMPLETED)
    commands = [cast("dict[str, object]", e["payload"])["command"] for e in completed]
    assert PAUSE_COMMAND in commands
    assert RESUME_COMMAND in commands


@pytest.mark.parametrize(
    "raw_signal",
    ["NOTREAL", "15", "RTMIN+1", "SIG_IGN"],
    ids=["typo", "numeric", "realtime-offset", "handler-name"],
)
def test_signal_unknown_signal_name(raw_signal: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown / numeric / realtime-offset / handler-enum names all surface as USAGE unknown signal."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    result = runner.invoke(cli, ["signal", "any", raw_signal])
    assert result.exit_code == ExitCode.USAGE
    assert f"unknown signal: {raw_signal}" in result.output


@pytest.mark.parametrize(
    "state",
    [SessionState.running, SessionState.suspended],
    ids=["running", "suspended"],
)
def test_signal_accepts_running_and_suspended(
    state: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validation cascade reaches killpg for live-supervisor accept states."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=state,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )
    monkeypatch.setattr("psoul.cli.main.psutil.Process", lambda _pid: _FakeProcess([_FakeChild(99999)]))
    monkeypatch.setattr("psoul.cli.main.os.killpg", lambda _pid, _sig: None)
    result = runner.invoke(cli, ["signal", "seed", "USR1"])
    assert result.exit_code == 0
    assert "Sent SIGUSR1 to seed." in result.output


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("USR1", "SIGUSR1"), ("SIGTERM", "SIGTERM"), ("hup", "SIGHUP"), ("SigInt", "SIGINT")],
    ids=["short", "full", "lower", "mixed"],
)
def test_signal_accepts_signal_name_forms(
    raw: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Short, full, lowercase, and mixed-case names all resolve to the canonical enum name."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )
    monkeypatch.setattr("psoul.cli.main.psutil.Process", lambda _pid: _FakeProcess([_FakeChild(99999)]))
    monkeypatch.setattr("psoul.cli.main.os.killpg", lambda _pid, _sig: None)
    result = runner.invoke(cli, ["signal", "seed", raw])
    assert result.exit_code == 0
    assert f"Sent {expected} to seed." in result.output


@pytest.mark.parametrize(
    ("launch_mode", "state", "expected_msg"),
    [
        (LaunchMode.attached, SessionState.running, "requires a headless session"),
        (LaunchMode.headless, SessionState.stopping, "session is already stopping"),
        (LaunchMode.headless, SessionState.starting, "not running, suspended, or orphaned"),
        (LaunchMode.headless, SessionState.debugging, "not running, suspended, or orphaned"),
        (LaunchMode.headless, SessionState.restarting, "not running, suspended, or orphaned"),
        (LaunchMode.headless, SessionState.exited, "not running, suspended, or orphaned"),
        (LaunchMode.headless, SessionState.failed, "not running, suspended, or orphaned"),
    ],
    ids=["attached", "stopping", "starting", "debugging", "restarting", "exited", "failed"],
)
def test_signal_rejects_non_accept_state(
    launch_mode: LaunchMode,
    state: SessionState,
    expected_msg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=state,
                launch_mode=launch_mode,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )
    result = runner.invoke(cli, ["signal", "seed", "USR1"])
    assert result.exit_code == ExitCode.USAGE
    assert expected_msg in result.output


def test_signal_unknown_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    open_db(tmp_path).close()
    result = runner.invoke(cli, ["signal", "does-not-exist", "TERM"])
    assert result.exit_code == ExitCode.USAGE
    assert "session not found" in result.output


def test_signal_missing_supervisor_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=None,
            )
        )
    result = runner.invoke(cli, ["signal", "seed", "USR1"])
    assert result.exit_code == ExitCode.ERROR
    assert "session has no supervisor" in result.output


def test_signal_orphaned_validation_passes_runtime_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Orphaned sessions clear the state-check branch but fail at child-pid resolution."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.cli.main.recover_sessions", lambda _conn: None)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="seed",
                state=SessionState.orphaned,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                supervisor_pid=os.getpid(),
            )
        )

    def _raise(_pid: int) -> None:
        raise psutil.NoSuchProcess(_pid)

    monkeypatch.setattr("psoul.cli.main.psutil.Process", _raise)
    result = runner.invoke(cli, ["signal", "seed", "USR1"])
    assert result.exit_code == ExitCode.ERROR
    assert "supervisor process is not running" in result.output


def test_signal_supervisor_has_no_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_runnable(tmp_path, monkeypatch)
    monkeypatch.setattr("psoul.cli.main.psutil.Process", lambda _pid: _FakeProcess([]))
    result = runner.invoke(cli, ["signal", "seed", "USR1"])
    assert result.exit_code == ExitCode.ERROR
    assert "supervisor has no managed child" in result.output


def test_signal_killpg_process_lookup_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_runnable(tmp_path, monkeypatch)
    monkeypatch.setattr("psoul.cli.main.psutil.Process", lambda _pid: _FakeProcess([_FakeChild(99999)]))

    def _raise(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("psoul.cli.main.os.killpg", _raise)
    result = runner.invoke(cli, ["signal", "seed", "USR1"])
    assert result.exit_code == 0
    assert "child has already exited" in result.output


def test_signal_killpg_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_runnable(tmp_path, monkeypatch)
    monkeypatch.setattr("psoul.cli.main.psutil.Process", lambda _pid: _FakeProcess([_FakeChild(99999)]))

    def _raise(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr("psoul.cli.main.os.killpg", _raise)
    result = runner.invoke(cli, ["signal", "seed", "USR1"])
    assert result.exit_code == ExitCode.ERROR
    assert "permission denied signalling child" in result.output
