"""Tests for headless session output capture end-to-end."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.core.db import open_db
from psoul.core.events import EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.recovery import ProcessStatus, check_pid
from psoul.core.resources import EVENT_RESOURCE_TELEMETRY
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")


def _make_session(session_id: str = "test-session") -> Session:
    """Build a minimal Session so events can reference it by ID."""
    return Session(
        session_id=session_id,
        state=SessionState.starting,
        launch_mode=LaunchMode.headless,
        launch_time=datetime.now(UTC),
        psoul_version=VERSION,
    )


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    """Create a psoul.toml pointing at a tmp_path-backed state dir."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config = tmp_path / "psoul.toml"
    config.write_text(f"[paths]\nstate_dir = '{state_dir}'\n")
    return config, state_dir


def _psoul_bin() -> str:
    """Path to the psoul console script in the current venv."""
    return str(Path(sys.executable).parent / "psoul")


def _wait_for_event(state_dir: Path, session_id: str, event_type: str, timeout: float = 5.0) -> None:
    """Block until the session has at least one event of the given type, or raise AssertionError on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(open_db(state_dir)) as conn:
            if EventStore(conn).list(session_id, event_type=event_type):
                return
        time.sleep(0.1)
    msg = f"timed out waiting for {event_type} event in session {session_id}"
    raise AssertionError(msg)


@pytest.fixture
def event_store(tmp_path: Path) -> Iterator[EventStore]:
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(_make_session())
        yield EventStore(conn)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_headless_supervisor_persists_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "print.py"
    script.write_text("print('captured!')")
    result = runner.invoke(cli, ["run", "--headless", "--name", "e2e-capture", str(script)])
    assert result.exit_code == 0
    record = json.loads(result.output)
    os.waitpid(record["supervisor_pid"], 0)
    with closing(open_db(tmp_path)) as conn:
        events = EventStore(conn).list("e2e-capture", event_type=EVENT_RUNTIME_STDOUT)
    assert any("captured!" in str(e["payload"]) for e in events)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_headless_supervisor_samples_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(1)")
    result = runner.invoke(cli, ["run", "--headless", "--name", "e2e-resources", str(script)])
    assert result.exit_code == 0
    record = json.loads(result.output)
    os.waitpid(record["supervisor_pid"], 0)
    with closing(open_db(tmp_path)) as conn:
        row = conn.execute(
            "SELECT cpu_percent, memory_rss_mb FROM resource_samples WHERE session_id = ?", ("e2e-resources",)
        ).fetchone()
        events = EventStore(conn).list("e2e-resources", event_type=EVENT_RESOURCE_TELEMETRY)
    assert row is not None
    assert len(events) >= 1


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_headless_fast_exit_finalizes_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A target that exits immediately still finalizes the session."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "instant.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["run", "--headless", "--name", "e2e-fast", str(script)])
    assert result.exit_code == 0
    record = json.loads(result.output)
    os.waitpid(record["supervisor_pid"], 0)
    with closing(open_db(tmp_path)) as conn:
        session = SessionStore(conn).get("e2e-fast")
    assert session is not None
    assert session.state == SessionState.exited
    assert session.supervisor_pid is None


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_events_follow_captures_live_stream(tmp_path: Path) -> None:
    config, _state_dir = _write_config(tmp_path)
    script = tmp_path / "slow.py"
    script.write_text("import sys\nimport time\nprint('hello')\nprint('oops', file=sys.stderr)\ntime.sleep(3)\n")
    launch = runner.invoke(cli, ["--config", str(config), "run", "--headless", "--name", "e2e-follow", str(script)])
    assert launch.exit_code == 0
    record = json.loads(launch.output)
    assert check_pid(record["supervisor_pid"]) is ProcessStatus.alive
    follow = subprocess.run(  # noqa: S603
        [_psoul_bin(), "--config", str(config), "events", "--follow", "e2e-follow", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    os.waitpid(record["supervisor_pid"], 0)
    assert follow.returncode == 0
    lines = [json.loads(line) for line in follow.stdout.strip().split("\n")]
    event_types = {event["event_type"] for event in lines}
    assert {EVENT_RUNTIME_STDOUT, EVENT_RESOURCE_TELEMETRY} <= event_types
    stdout_text = "".join(str(e["payload"]) for e in lines if e["event_type"] == EVENT_RUNTIME_STDOUT)
    assert "hello" in stdout_text
    assert "oops" in stdout_text


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_events_follow_exits_cleanly_on_sigint(tmp_path: Path) -> None:
    config, _state_dir = _write_config(tmp_path)
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(1)\n")
    launch = runner.invoke(cli, ["--config", str(config), "run", "--headless", "--name", "sigint-test", str(script)])
    assert launch.exit_code == 0
    record = json.loads(launch.output)
    assert check_pid(record["supervisor_pid"]) is ProcessStatus.alive
    follow = subprocess.Popen(  # noqa: S603
        [_psoul_bin(), "--config", str(config), "events", "--follow", "sigint-test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    follow.send_signal(signal.SIGINT)
    _stdout, stderr = follow.communicate(timeout=5)
    os.waitpid(record["supervisor_pid"], 0)
    assert follow.returncode == 0, f"stderr: {stderr.decode(errors='replace')}"


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_logs_follow_captures_live_stream(tmp_path: Path) -> None:
    config, state_dir = _write_config(tmp_path)
    script = tmp_path / "slow.py"
    script.write_text(
        "import sys, time\n"
        "print('hello', flush=True)\n"
        "time.sleep(1)\n"
        "print('oops', file=sys.stderr, flush=True)\n"
        "time.sleep(1)\n"
    )
    launch = runner.invoke(
        cli, ["--config", str(config), "run", "--headless", "--name", "e2e-logs-follow", str(script)]
    )
    assert launch.exit_code == 0
    record = json.loads(launch.output)
    _wait_for_event(state_dir, "e2e-logs-follow", EVENT_RUNTIME_STDOUT)
    assert check_pid(record["supervisor_pid"]) is ProcessStatus.alive
    with closing(open_db(state_dir)) as conn:
        types_before = [e["event_type"] for e in EventStore(conn).list("e2e-logs-follow")]
    assert types_before.count(EVENT_RUNTIME_STDOUT) == 1
    follow = subprocess.run(  # noqa: S603
        [_psoul_bin(), "--config", str(config), "logs", "--follow", "e2e-logs-follow"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    os.waitpid(record["supervisor_pid"], 0)
    assert follow.returncode == 0
    assert follow.stdout == "hello\noops\n"


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_logs_follow_exits_cleanly_on_sigint(tmp_path: Path) -> None:
    config, _state_dir = _write_config(tmp_path)
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(1)\n")
    launch = runner.invoke(
        cli, ["--config", str(config), "run", "--headless", "--name", "logs-sigint-test", str(script)]
    )
    assert launch.exit_code == 0
    record = json.loads(launch.output)
    assert check_pid(record["supervisor_pid"]) is ProcessStatus.alive
    follow = subprocess.Popen(  # noqa: S603
        [_psoul_bin(), "--config", str(config), "logs", "--follow", "logs-sigint-test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    follow.send_signal(signal.SIGINT)
    _stdout, stderr = follow.communicate(timeout=5)
    os.waitpid(record["supervisor_pid"], 0)
    assert follow.returncode == 0, f"stderr: {stderr.decode(errors='replace')}"
