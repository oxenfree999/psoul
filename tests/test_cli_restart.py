"""Tests for the psoul restart CLI command."""

import json
import os
import signal
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
from psoul.core.control import (
    COMMAND_ACCEPTED,
    COMMAND_COMPLETED,
    RESTART_COMMAND,
    SESSION_RESTARTED,
)
from psoul.core.db import open_db
from psoul.core.events import EventStore
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

if sys.platform == "win32":
    pytest.skip(
        "test_cli_restart.py exercises Unix-only psoul restart semantics. "
        "The platform-rejection branch is covered on Unix via monkeypatched sys.platform.",
        allow_module_level=True,
    )

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")


def _wait_for_generation_running(state_dir: Path, session_id: str, target_gen: int, timeout: float = 5.0) -> None:
    """Poll the DB until *session_id* reaches *target_gen* in the running state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(open_db(state_dir)) as conn:
            session = SessionStore(conn).get(session_id)
        if session is not None and session.state == SessionState.running and session.generation == target_gen:
            return
        time.sleep(0.02)
    msg = f"session {session_id} did not reach generation={target_gen} running within {timeout}s"
    raise RuntimeError(msg)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_restart_cli_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """psoul run --headless then psoul restart: new generation spawns and the cross-generation pair is emitted."""
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
    _wait_for_generation_running(tmp_path, session_id, target_gen=0)
    restart = runner.invoke(cli, ["restart", session_id])
    assert restart.exit_code == 0
    assert f"Sent restart to {session_id}." in restart.output
    _wait_for_generation_running(tmp_path, session_id, target_gen=1)
    runner.invoke(cli, ["kill", session_id])
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        events = EventStore(conn).list(session_id)
    accepted = [
        e
        for e in events
        if e["event_type"] == COMMAND_ACCEPTED and cast("dict[str, object]", e["payload"])["command"] == RESTART_COMMAND
    ]
    session_restarted = [e for e in events if e["event_type"] == SESSION_RESTARTED]
    completed = [
        e
        for e in events
        if e["event_type"] == COMMAND_COMPLETED
        and cast("dict[str, object]", e["payload"])["command"] == RESTART_COMMAND
    ]
    assert len(accepted) == 1
    assert len(session_restarted) == 1
    assert len(completed) == 1
    # Cross-generation pair: accepted on gen 0 (eager at dispatch), completed on gen 1 (post-running boundary).
    assert accepted[0]["generation"] == 0
    assert session_restarted[0]["generation"] == 1
    assert completed[0]["generation"] == 1
    # Accepted and completed share a message_id even across generations.
    accepted_mid = cast("dict[str, object]", accepted[0]["payload"])["message_id"]
    completed_mid = cast("dict[str, object]", completed[0]["payload"])["message_id"]
    assert accepted_mid == completed_mid
    # Temporal ordering: accepted fires at dispatch, session.restarted at the restarting→starting
    # boundary, completed post-running. EventStore.list returns events in sequence order.
    restart_boundary_sequence = [
        e["event_type"]
        for e in events
        if e["event_type"] == SESSION_RESTARTED
        or (
            e["event_type"] in {COMMAND_ACCEPTED, COMMAND_COMPLETED}
            and cast("dict[str, object]", e["payload"])["command"] == RESTART_COMMAND
        )
    ]
    assert restart_boundary_sequence == [COMMAND_ACCEPTED, SESSION_RESTARTED, COMMAND_COMPLETED]


def test_restart_unknown_session_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A selector that matches no session surfaces a session-not-found error."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    open_db(tmp_path).close()
    result = runner.invoke(cli, ["restart", "does-not-exist"])
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
def test_restart_rejects_non_accept_state(
    launch_mode: LaunchMode,
    state: SessionState,
    expected_msg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-accept (launch_mode, state) combination exits with USAGE and its specific error."""
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
    result = runner.invoke(cli, ["restart", "seed"])
    assert result.exit_code == ExitCode.USAGE
    assert expected_msg in result.output


def test_restart_missing_supervisor_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    result = runner.invoke(cli, ["restart", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "session has no supervisor" in result.output


def test_restart_supervisor_process_lookup_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    result = runner.invoke(cli, ["restart", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "supervisor process is not running" in result.output


def test_restart_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    result = runner.invoke(cli, ["restart", "seed"])
    assert result.exit_code == ExitCode.ERROR
    assert "permission denied signalling supervisor" in result.output
    assert f"(PID={os.getpid()})" in result.output


def test_restart_on_windows_surfaces_platform_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Platform check short-circuits before DB access on Windows with a Unix-only error message."""
    monkeypatch.setattr("psoul.cli.main.sys.platform", "win32")
    result = runner.invoke(cli, ["restart", "any"])
    assert result.exit_code == ExitCode.USAGE
    assert "restart is Unix-only (macOS / Linux)" in result.output
    assert "Windows support deferred" in result.output


def _wait_for_state(state_dir: Path, session_id: str, target: SessionState, timeout: float = 3.0) -> None:
    """Poll the DB until *session_id* reaches *target* state or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(open_db(state_dir)) as conn:
            session = SessionStore(conn).get(session_id)
        if session is not None and session.state == target:
            return
        time.sleep(0.02)
    msg = f"session {session_id} did not reach {target} within {timeout}s"
    raise RuntimeError(msg)


def _assert_gen0_killed_via_escalation(state_dir: Path, session_id: str) -> None:
    """Verify gen-0's result row records SIGKILL termination (escalation path fired)."""
    with closing(open_db(state_dir)) as conn:
        row = conn.execute(
            "SELECT outcome, exit_code FROM results WHERE session_id = ? AND generation = 0",
            (session_id,),
        ).fetchone()
    assert row is not None, "no gen-0 result row"
    # subprocess.Popen sets proc.returncode to -signal_number on signal termination;
    # SIGKILL delivered via escalation → proc.returncode == -signal.SIGKILL == -9.
    assert row[0] == "failed"
    assert row[1] == -signal.SIGKILL


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_restart_from_suspended_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """psoul pause then psoul restart: SIGKILL escalation wakes the stopped child, new generation spawns.

    The child installs ``SIG_IGN`` for SIGTERM so the only way it can die is via SIGKILL escalation
    at the ``process.stop_timeout`` deadline. Uses ``stop_timeout = "1s"`` to keep the test fast.
    """
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
    _wait_for_generation_running(tmp_path, session_id, target_gen=0)
    ready_file = tmp_path / "child_ready"
    ready_deadline = time.monotonic() + 5.0
    while time.monotonic() < ready_deadline and not ready_file.exists():
        time.sleep(0.02)
    assert ready_file.exists(), "child did not signal ready within 5s"
    assert runner.invoke(cli, ["--config", str(config), "pause", session_id]).exit_code == 0
    _wait_for_state(tmp_path, session_id, SessionState.suspended)
    restart = runner.invoke(cli, ["--config", str(config), "restart", session_id])
    assert restart.exit_code == 0
    _wait_for_generation_running(tmp_path, session_id, target_gen=1)
    runner.invoke(cli, ["--config", str(config), "kill", session_id])
    os.waitpid(supervisor_pid, 0)
    _assert_gen0_killed_via_escalation(tmp_path, session_id)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_restart_child_ignoring_sigterm_escalates_to_sigkill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A child that ignores SIGTERM gets SIGKILLed at the stop_timeout deadline, then the new generation spawns."""
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
    _wait_for_generation_running(tmp_path, session_id, target_gen=0)
    ready_file = tmp_path / "child_ready"
    ready_deadline = time.monotonic() + 5.0
    while time.monotonic() < ready_deadline and not ready_file.exists():
        time.sleep(0.02)
    assert ready_file.exists(), "child did not signal ready within 5s"
    restart = runner.invoke(cli, ["--config", str(config), "restart", session_id])
    assert restart.exit_code == 0
    _wait_for_generation_running(tmp_path, session_id, target_gen=1)
    runner.invoke(cli, ["--config", str(config), "kill", session_id])
    os.waitpid(supervisor_pid, 0)
    _assert_gen0_killed_via_escalation(tmp_path, session_id)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_restart_generation_monotonicity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three successive restarts produce generation 0 → 1 → 2 → 3, each with its own result row."""
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
    _wait_for_generation_running(tmp_path, session_id, target_gen=0)
    for expected_gen in (1, 2, 3):
        assert runner.invoke(cli, ["restart", session_id]).exit_code == 0
        _wait_for_generation_running(tmp_path, session_id, target_gen=expected_gen)
    runner.invoke(cli, ["kill", session_id])
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        results = conn.execute(
            "SELECT generation FROM results WHERE session_id = ? ORDER BY generation",
            (session_id,),
        ).fetchall()
    assert [row[0] for row in results] == [0, 1, 2, 3]  # one result row per spawned generation


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_restart_keeps_existing_artifact_rows_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An artifact row inserted before restart is still readable after gen 1 is running.

    The ``artifacts`` table has ``UNIQUE (session_id, name)`` and no generation column, so it
    cannot distinguish same-named artifacts across generations. This test only exercises the
    preservation property.
    """
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
    _wait_for_generation_running(tmp_path, session_id, target_gen=0)
    with closing(open_db(tmp_path)) as conn:
        conn.execute(
            "INSERT INTO artifacts"
            " (session_id, name, path, mime_type, size_bytes, registered_at, source, retention_class)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                "gen0-output.txt",
                f"artifacts/{session_id}/gen0-output.txt",
                "text/plain",
                42,
                datetime.now(UTC).isoformat(),
                "user",
                "session",
            ),
        )
        conn.commit()
    assert runner.invoke(cli, ["restart", session_id]).exit_code == 0
    _wait_for_generation_running(tmp_path, session_id, target_gen=1)
    runner.invoke(cli, ["kill", session_id])
    os.waitpid(supervisor_pid, 0)
    with closing(open_db(tmp_path)) as conn:
        rows = conn.execute(
            "SELECT name, size_bytes FROM artifacts WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [("gen0-output.txt", 42)]
