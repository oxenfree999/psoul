"""Tests for global CLI flags."""

import sqlite3
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.version import VERSION

runner = CliRunner()


@pytest.mark.parametrize("args", [["--version"], ["-V"], ["version"]])
def test_version_output(args: list[str]) -> None:
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert f"psoul {VERSION}" in result.output


def test_verbose_quiet_conflict() -> None:
    result = runner.invoke(cli, ["-v", "-q", "version"])
    assert result.exit_code != 0
    assert "cannot be used together" in result.output


def test_color_invalid() -> None:
    result = runner.invoke(cli, ["--color", "bogus", "version"])
    assert result.exit_code != 0


def test_json_shorthand() -> None:
    result = runner.invoke(cli, ["doctor", "--json"])
    assert result.exit_code == 0


def test_config_error_missing_file() -> None:
    result = runner.invoke(cli, ["--config", "nonexistent.toml", "config"])
    assert result.exit_code == 1
    assert "Config error" in result.output


@pytest.mark.parametrize(
    ("toml_content", "match"),
    [
        ("invalid = = toml", "Config error"),
        ("[bogus]\nkey = 1", "unknown section"),
        ("[process]\nstop_timeout = 42", "expected str, got int"),
        ('[process]\nstop_timeout = "nope"', "invalid duration"),
    ],
)
def test_config_error_invalid_content(tmp_path: Path, toml_content: str, match: str) -> None:
    toml_file = tmp_path / "psoul.toml"
    toml_file.write_text(toml_content)
    result = runner.invoke(cli, ["--config", str(toml_file), "config"])
    assert result.exit_code == 1
    assert match in result.output


def test_help_shows_flags() -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    output = typer.unstyle(result.output)
    for flag in ["--verbose", "--quiet", "--color", "--config", "--version"]:
        assert flag in output


@pytest.mark.parametrize(
    ("invocation", "needs_script", "needs_record"),
    [
        pytest.param([], False, True, id="bare-psoul"),
        pytest.param(["run", "--record"], True, False, id="run"),
        pytest.param(["ps"], False, False, id="ps"),
        pytest.param(["status", "fake-id"], False, False, id="status"),
    ],
)
def test_open_db_lock_translates_to_clean_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invocation: list[str],
    needs_script: bool,
    needs_record: bool,
) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    args = list(invocation)
    if needs_record:
        config = tmp_path / "psoul.toml"
        config.write_text("[session]\nrecord = true\n")
        args = ["--config", str(config), *args]
    if needs_script:
        script = tmp_path / "noop.py"
        script.write_text("pass")
        args.append(str(script))

    def raise_locked(_state_dir: Path, *, create: bool = True) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("psoul.cli.main.open_db", raise_locked)
    result = runner.invoke(cli, args)
    assert result.exit_code == 1
    assert "database is busy or locked" in result.output
    assert "Traceback" not in result.output
