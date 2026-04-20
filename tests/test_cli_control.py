"""Tests for the psoul stop and psoul kill CLI commands."""

import json
import os
import sys
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.cli.state import ExitCode
from psoul.core.control import COMMAND_COMPLETED, KILL_COMMAND, OUTCOME_ESCALATED, OUTCOME_OK, STOP_COMMAND
from psoul.core.db import open_db
from psoul.core.events import EventStore
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

if sys.platform == "win32":
    pytest.skip(
        "test_cli_control.py exercises Unix-only psoul stop / kill semantics. "
        "The platform-rejection branch is covered on Unix via monkeypatched sys.platform.",
        allow_module_level=True,
    )

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


def test_stop_unknown_session_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A selector that matches no session surfaces a session-not-found error."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    result = runner.invoke(cli, ["stop", "does-not-exist"])
    assert result.exit_code == ExitCode.USAGE
    assert "session not found" in result.output


@pytest.mark.parametrize(
    ("launch_mode", "state", "expected_msg"),
    [
        (LaunchMode.attached, SessionState.running, "requires a headless session"),
        (LaunchMode.headless, SessionState.orphaned, "session is orphaned"),
        (LaunchMode.headless, SessionState.stopping, "session is already stopping"),
        (LaunchMode.headless, SessionState.exited, "session is not running"),
        (LaunchMode.headless, SessionState.failed, "session is not running"),
        (LaunchMode.headless, SessionState.starting, "session is not running"),
        (LaunchMode.headless, SessionState.suspended, "session is not running"),
        (LaunchMode.headless, SessionState.debugging, "session is not running"),
        (LaunchMode.headless, SessionState.restarting, "session is not running"),
    ],
    ids=["attached", "orphaned", "stopping", "exited", "failed", "starting", "suspended", "debugging", "restarting"],
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


def test_stop_on_windows_surfaces_platform_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Platform check short-circuits before DB access on Windows with a Unix-only error message."""
    monkeypatch.setattr("psoul.cli.main.sys.platform", "win32")
    result = runner.invoke(cli, ["stop", "any"])
    assert result.exit_code == ExitCode.USAGE
    assert "stop is Unix-only (macOS / Linux)" in result.output
    assert "Windows support deferred" in result.output


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
