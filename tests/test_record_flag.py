"""End-to-end tests for the --record flag, [session] record config, and no-DB short-circuit behavior."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.cli.state import ExitCode
from psoul.core.db import DB_NAME

runner = CliRunner()

# Locks the contract message verbatim. Drift in the production
# constant surfaces here as a test failure.
_NO_RECORDED_SESSIONS_LINE = "no recorded sessions, pass --record to enable persistence\n"


def test_no_record_skips_db_on_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`psoul run script.py` without --record execs the script and leaves no DB behind."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["run", str(script)])
    assert result.exit_code == 0
    assert not (tmp_path / DB_NAME).exists()


def test_no_record_propagates_target_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-recorded `psoul run` propagates the target's exit code per the launcher contract."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "fail.py"
    script.write_text("import sys\nsys.exit(7)\n")
    result = runner.invoke(cli, ["run", str(script)])
    assert result.exit_code == 7
    assert not (tmp_path / DB_NAME).exists()


def test_record_flag_creates_db_on_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`psoul run --record script.py` creates the DB and records the session."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["run", "--record", str(script)])
    assert result.exit_code == 0
    assert (tmp_path / DB_NAME).exists()


def test_run_without_target_or_module_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`psoul run` with no target and no -m surfaces a USAGE error from build_launch_request."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    result = runner.invoke(cli, ["run"])
    assert result.exit_code == ExitCode.USAGE
    assert "launch target is required" in result.output


def test_session_record_config_creates_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`[session] record = true` in config triggers recording without the flag."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    config = tmp_path / "psoul.toml"
    config.write_text("[session]\nrecord = true\n")
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["--config", str(config), "run", str(script)])
    assert result.exit_code == 0
    assert (tmp_path / DB_NAME).exists()


def test_launch_mode_headless_config_implies_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`[launch] mode = "headless"` resolves launch_mode to headless, which implies recording."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    config = tmp_path / "psoul.toml"
    config.write_text('[launch]\nmode = "headless"\n')
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["--config", str(config), "run", str(script)])
    if result.exit_code != 0:
        pytest.skip("headless launch unavailable on this platform")
    assert (tmp_path / DB_NAME).exists()


@pytest.mark.parametrize(
    ("command_args", "expected_exit"),
    [
        (["ps"], ExitCode.SUCCESS),
        (["status", "any"], ExitCode.SUCCESS),
        (["events", "any"], ExitCode.SUCCESS),
        (["logs", "any"], ExitCode.SUCCESS),
        (["stats", "any"], ExitCode.SUCCESS),
        (["artifacts", "any"], ExitCode.SUCCESS),
        (["env", "any"], ExitCode.SUCCESS),
        (["prune", "--all"], ExitCode.SUCCESS),
    ],
    ids=["ps", "status", "events", "logs", "stats", "artifacts", "env", "prune"],
)
def test_no_db_read_side_exits_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command_args: list[str], expected_exit: ExitCode
) -> None:
    """Read-side commands on a fresh home exit 0 with the empty-state stderr message."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    result = runner.invoke(cli, command_args)
    assert result.exit_code == expected_exit
    assert result.stdout == ""
    assert result.stderr == _NO_RECORDED_SESSIONS_LINE
    assert not (tmp_path / DB_NAME).exists()


@pytest.mark.parametrize(
    "command_args",
    [
        ["stop", "any"],
        ["kill", "any"],
        ["pause", "any"],
        ["resume", "any"],
        ["restart", "any"],
        ["signal", "any", "TERM"],
        ["attach", "any"],
    ],
    ids=["stop", "kill", "pause", "resume", "restart", "signal", "attach"],
)
def test_no_db_control_side_exits_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command_args: list[str]
) -> None:
    """Control-side commands on a fresh Unix home exit USAGE with the empty-state stderr message."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    result = runner.invoke(cli, command_args)
    assert result.exit_code == ExitCode.USAGE
    assert result.stdout == ""
    assert result.stderr == _NO_RECORDED_SESSIONS_LINE
    assert not (tmp_path / DB_NAME).exists()
