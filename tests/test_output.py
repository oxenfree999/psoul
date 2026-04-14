"""Tests for drain_output and headless stdout/stderr capture end-to-end."""

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
from psoul.core.events import EVENT_RUNTIME_STDERR, EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.output import drain_output
from psoul.core.resources import EVENT_RESOURCE_TELEMETRY
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")
requires_unix_pipes = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows SelectSelector only handles sockets, not pipes",
)


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


@pytest.fixture
def event_store(tmp_path: Path) -> Iterator[EventStore]:
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(_make_session())
        yield EventStore(conn)


@requires_unix_pipes
@pytest.mark.parametrize(
    ("code", "expected_type", "expected_text"),
    [
        ("print('hello')", EVENT_RUNTIME_STDOUT, "hello"),
        ("import sys; print('oops', file=sys.stderr)", EVENT_RUNTIME_STDERR, "oops"),
    ],
    ids=["stdout", "stderr"],
)
def test_drain_captures_single_stream(
    event_store: EventStore,
    code: str,
    expected_type: str,
    expected_text: str,
) -> None:
    with subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        drain_output(proc, session_id="test-session", event_store=event_store, generation=0)
    events = event_store.list("test-session", event_type=expected_type)
    assert events
    assert any(expected_text in str(e["payload"]) for e in events)


@requires_unix_pipes
def test_drain_captures_interleaved_streams(event_store: EventStore) -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        drain_output(proc, session_id="test-session", event_store=event_store, generation=0)
    stdout_events = event_store.list("test-session", event_type=EVENT_RUNTIME_STDOUT)
    stderr_events = event_store.list("test-session", event_type=EVENT_RUNTIME_STDERR)
    assert any("out" in str(e["payload"]) for e in stdout_events)
    assert any("err" in str(e["payload"]) for e in stderr_events)


def test_drain_no_pipes_returns_immediately(event_store: EventStore) -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "print('ignored')"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as proc:
        drain_output(proc, session_id="test-session", event_store=event_store, generation=0)
    assert event_store.list("test-session") == []


@requires_unix_pipes
def test_drain_handles_non_utf8_bytes(event_store: EventStore) -> None:
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            r"import sys; sys.stdout.buffer.write(b'\xff\xfe\xfdok'); sys.stdout.buffer.flush()",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        drain_output(proc, session_id="test-session", event_store=event_store, generation=0)
    events = event_store.list("test-session", event_type=EVENT_RUNTIME_STDOUT)
    assert events
    combined = "".join(str(e["payload"]) for e in events)
    assert "ok" in combined
    assert "\ufffd" in combined


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
    follow = subprocess.run(  # noqa: S603
        [_psoul_bin(), "--config", str(config), "events", "--follow", "e2e-follow", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    os.waitpid(record["supervisor_pid"], 0)
    assert follow.returncode == 0
    event_types = {json.loads(line)["event_type"] for line in follow.stdout.strip().split("\n")}
    assert {EVENT_RUNTIME_STDOUT, EVENT_RUNTIME_STDERR, EVENT_RESOURCE_TELEMETRY} <= event_types


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_events_follow_exits_cleanly_on_sigint(tmp_path: Path) -> None:
    config, _state_dir = _write_config(tmp_path)
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(1)\n")
    launch = runner.invoke(cli, ["--config", str(config), "run", "--headless", "--name", "sigint-test", str(script)])
    assert launch.exit_code == 0
    record = json.loads(launch.output)
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
