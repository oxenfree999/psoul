"""Tests for the psoul attach CLI command."""

import json
import os
import sys
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import ptyprocess
import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.cli.state import ExitCode
from psoul.core.db import open_db
from psoul.core.events import (
    EVENT_SESSION_CONTROLLER_ACQUIRED,
    EVENT_SESSION_CONTROLLER_RELEASED,
    EventStore,
)
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")

_DETACH_ESCAPE_BYTE = bytes([0x1D])
_PROC_STARTUP_GRACE = 0.5  # seconds to let the spawned attach client connect and send HELLO
_RUNNING_TIMEOUT = 5.0


def _wait_for_running(state_dir: Path, session_id: str, timeout: float = _RUNNING_TIMEOUT) -> None:
    """Poll the DB until *session_id* enters the running state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(open_db(state_dir)) as conn:
            session = SessionStore(conn).get(session_id)
        if session is not None and session.state == SessionState.running:
            return
        time.sleep(0.02)
    msg = f"session {session_id} did not reach running within {timeout}s"
    raise RuntimeError(msg)


def _write_test_config(tmp_path: Path) -> Path:
    """Write a minimal psoul.toml pointing at *tmp_path* and return the config path."""
    config_path = tmp_path / "psoul.toml"
    config_path.write_text(f'[paths]\nstate_dir = "{tmp_path}"\n')
    return config_path


def test_attach_rejects_non_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CliRunner's default non-TTY stdin/stdout triggers a USAGE-coded error."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    result = runner.invoke(cli, ["attach", "any-session"])
    assert result.exit_code == ExitCode.USAGE
    assert "TTY" in result.output


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_attach_rejects_missing_session(tmp_path: Path) -> None:
    """Attach to a non-existent session exits with USAGE."""
    config_path = _write_test_config(tmp_path)
    proc = ptyprocess.PtyProcess.spawn(
        [sys.executable, "-m", "psoul", "--config", str(config_path), "attach", "missing-session"],
    )
    proc.wait()
    assert proc.exitstatus == ExitCode.USAGE


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_attach_runtime_failure_returns_error_exit_code(tmp_path: Path) -> None:
    """A stale or missing socket_path causes a runtime OSError that the CLI normalizes to ExitCode.ERROR."""
    config_path = _write_test_config(tmp_path)
    session_id = "stale-sock"
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id=session_id,
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                socket_path=tmp_path / "nonexistent.sock",
            )
        )
    proc = ptyprocess.PtyProcess.spawn(
        [sys.executable, "-m", "psoul", "--config", str(config_path), "attach", session_id],
    )
    proc.wait()
    assert proc.exitstatus == ExitCode.ERROR


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_attach_happy_path_detach_then_session_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Attach via PTY subprocess, detach with Ctrl-], session keeps running, controller events recorded."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "long.py"
    script.write_text("import time\ntime.sleep(60)\n")
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_running(tmp_path, session_id)

    config_path = _write_test_config(tmp_path)
    proc = ptyprocess.PtyProcess.spawn(
        [sys.executable, "-m", "psoul", "--config", str(config_path), "attach", session_id],
    )
    try:
        time.sleep(_PROC_STARTUP_GRACE)
        proc.write(_DETACH_ESCAPE_BYTE)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and proc.isalive():
            time.sleep(0.05)
        assert not proc.isalive(), "attach client did not exit after Ctrl-]"
        assert proc.exitstatus == 0
        with closing(open_db(tmp_path)) as conn:
            session = SessionStore(conn).get(session_id)
            events = EventStore(conn).list(session_id)
        assert session is not None
        assert session.state == SessionState.running
        assert session.control_epoch == 1
        assert any(e["event_type"] == EVENT_SESSION_CONTROLLER_ACQUIRED for e in events)
        assert any(e["event_type"] == EVENT_SESSION_CONTROLLER_RELEASED for e in events)
    finally:
        if proc.isalive():
            proc.terminate(force=True)
        proc.wait()
        runner.invoke(cli, ["kill", session_id])
        os.waitpid(supervisor_pid, 0)
