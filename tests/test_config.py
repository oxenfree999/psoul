"""Tests for configuration schema, discovery, loading, and CLI commands."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.config import (
    LaunchConfig,
    OutputConfig,
    PathsConfig,
    ProcessConfig,
    PsoulConfig,
    PythonConfig,
    RetentionConfig,
    SessionConfig,
    default_config_dir,
    default_state_dir,
    find_config_file,
    generate_config,
    load_config,
)


def test_default_config_dir_returns_path() -> None:
    result = default_config_dir()
    assert isinstance(result, Path)
    assert "psoul" in result.parts


def test_default_state_dir_returns_path() -> None:
    result = default_state_dir()
    assert isinstance(result, Path)
    assert "psoul" in result.parts


def test_paths_config_defaults() -> None:
    config = PathsConfig()
    assert config.state_dir is None


def test_python_config_defaults() -> None:
    config = PythonConfig()
    assert config.python_path is None


def test_launch_config_defaults() -> None:
    config = LaunchConfig()
    assert config.mode == "attached"


def test_process_config_defaults() -> None:
    config = ProcessConfig()
    assert config.stop_timeout == "10s"
    assert config.stop_signal == "SIGTERM"


def test_session_config_defaults() -> None:
    config = SessionConfig()
    assert config.tags is None


def test_output_config_defaults() -> None:
    config = OutputConfig()
    assert config.format == "text"
    assert config.color == "auto"
    assert config.timestamps is True


def test_retention_config_defaults() -> None:
    config = RetentionConfig()
    assert config.max_age == "7d"
    assert config.max_sessions == 100
    assert config.max_artifact_mb == 500


def test_psoul_config_defaults() -> None:
    config = PsoulConfig()
    assert config.paths.state_dir is None
    assert config.python.python_path is None
    assert config.launch.mode == "attached"
    assert config.process.stop_timeout == "10s"
    assert config.session.tags is None
    assert config.output.format == "text"
    assert config.retention.max_age == "7d"


def test_psoul_config_override() -> None:
    config = PsoulConfig(
        paths=PathsConfig(state_dir=Path("/tmp/psoul")),
        launch=LaunchConfig(mode="headless"),
    )
    assert config.paths.state_dir == Path("/tmp/psoul")
    assert config.launch.mode == "headless"
    assert config.python.python_path is None  # untouched


def test_config_is_frozen() -> None:
    config = PsoulConfig()
    with pytest.raises(AttributeError):
        config.paths = PathsConfig(state_dir=Path("/tmp"))  # ty: ignore[invalid-assignment]


def test_load_config_none_returns_defaults() -> None:
    config = load_config(None)
    assert config == PsoulConfig()


def test_load_config_from_toml(tmp_path: Path) -> None:
    toml_file = tmp_path / "psoul.toml"
    toml_file.write_text('[launch]\nmode = "headless"\n\n[process]\nstop_timeout = "30s"\n')
    config = load_config(toml_file)
    assert config.launch.mode == "headless"
    assert config.process.stop_timeout == "30s"
    assert config.paths.state_dir is None  # untouched sections keep defaults


def test_load_config_from_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.psoul.launch]\nmode = "headless"\n')
    config = load_config(pyproject)
    assert config.launch.mode == "headless"
    assert config.process.stop_signal == "SIGTERM"  # defaults preserved


@pytest.mark.parametrize(
    ("toml_content", "section", "key"),
    [
        ('[paths]\nstate_dir = "/tmp/psoul-state"', "paths", "state_dir"),
        ('[python]\npython_path = "/usr/bin/python3"', "python", "python_path"),
    ],
)
def test_load_config_coerces_string_to_path(
    tmp_path: Path,
    toml_content: str,
    section: str,
    key: str,
) -> None:
    toml_file = tmp_path / "psoul.toml"
    toml_file.write_text(toml_content)
    config = load_config(toml_file)
    assert isinstance(getattr(getattr(config, section), key), Path)


@pytest.mark.parametrize(
    ("toml_content", "match"),
    [
        ("[paths]\nstate_dir = 42", r"\[paths\] state_dir: expected path string, got int"),
        ('[retention]\nmax_sessions = "nope"', r"\[retention\] max_sessions: expected int, got str"),
        ("[retention]\nmax_sessions = true", r"\[retention\] max_sessions: expected int, got bool"),
        ("[launch]\nmode = 42", r"\[launch\] mode: expected str, got int"),
        ("[output]\ntimestamps = 1", r"\[output\] timestamps: expected bool, got int"),
        ('[output]\ntimestamps = "yes"', r"\[output\] timestamps: expected bool, got str"),
    ],
)
def test_load_config_rejects_invalid_type(tmp_path: Path, toml_content: str, match: str) -> None:
    toml_file = tmp_path / "psoul.toml"
    toml_file.write_text(toml_content)
    with pytest.raises(TypeError, match=match):
        load_config(toml_file)


@pytest.mark.parametrize(
    ("toml_content", "match"),
    [
        ("[launch]\nbogus = 1", r"\[launch\] unknown key: bogus"),
        ("[launch]\nfoo = 1\nbar = 2", r"\[launch\] unknown key: bar, foo"),
        ("[bogus]\nfoo = 1", r"unknown section: \[bogus\]"),
    ],
)
def test_load_config_rejects_unknown(tmp_path: Path, toml_content: str, match: str) -> None:
    toml_file = tmp_path / "psoul.toml"
    toml_file.write_text(toml_content)
    with pytest.raises(ValueError, match=match):
        load_config(toml_file)


@pytest.mark.parametrize(
    ("toml_content", "match"),
    [
        ('[process]\nstop_timeout = "bogus"', r"\[process\] stop_timeout: invalid duration"),
        ('[retention]\nmax_age = "nope"', r"\[retention\] max_age: invalid duration"),
    ],
)
def test_load_config_rejects_invalid_duration(tmp_path: Path, toml_content: str, match: str) -> None:
    toml_file = tmp_path / "psoul.toml"
    toml_file.write_text(toml_content)
    with pytest.raises(ValueError, match=match):
        load_config(toml_file)


def test_find_config_file_override_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_config_file(tmp_path / "nonexistent.toml")


def test_find_config_file_override_exists(tmp_path: Path) -> None:
    config_file = tmp_path / "custom.toml"
    config_file.write_text("")
    assert find_config_file(config_file) == config_file


def test_find_config_file_discovery_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    # No config files — returns None
    assert find_config_file() is None

    # pyproject.toml without [tool.psoul] — ignored
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
    assert find_config_file() is None

    # Add [tool.psoul] — now pyproject.toml is discovered
    (tmp_path / "pyproject.toml").write_text("[tool.psoul.launch]\nmode = 'headless'\n")
    result = find_config_file()
    assert result is not None
    assert result.name == "pyproject.toml"

    # Add psoul.toml — wins over pyproject.toml
    (tmp_path / "psoul.toml").write_text('[launch]\nmode = "attached"\n')
    result = find_config_file()
    assert result is not None
    assert result.name == "psoul.toml"


runner = CliRunner()


def test_config_command_text() -> None:
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 0
    assert "launch.mode = 'attached'" in result.output
    assert "process.stop_timeout = '10s'" in result.output


def test_config_command_json() -> None:
    result = runner.invoke(cli, ["config", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["launch"]["mode"] == "attached"
    assert data["process"]["stop_timeout"] == "10s"


def test_config_default_text() -> None:
    result = runner.invoke(cli, ["config", "--default"])
    assert result.exit_code == 0
    assert "launch.mode = 'attached'" in result.output
    assert "process.stop_timeout = '10s'" in result.output


def test_config_default_json() -> None:
    result = runner.invoke(cli, ["config", "--default", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["launch"]["mode"] == "attached"
    assert data["retention"]["max_sessions"] == 100


def test_config_default_ignores_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "psoul.toml").write_text('[launch]\nmode = "headless"\n')
    # Without --default: picks up the local psoul.toml
    result = runner.invoke(cli, ["config"])
    assert "launch.mode = 'headless'" in result.output
    # With --default: always shows defaults regardless of config file
    result = runner.invoke(cli, ["config", "--default"])
    assert "launch.mode = 'attached'" in result.output


def test_config_init_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli, ["config", "init"])
    assert result.exit_code == 0
    assert "Wrote psoul.toml" in result.output
    content = (tmp_path / "psoul.toml").read_text()
    assert "[launch]" in content
    assert "# mode" in content


def test_generate_config_emits_type_correct_examples() -> None:
    config = generate_config()
    assert '# state_dir = "~/.local/state/psoul"' in config
    assert '# python_path = "/usr/bin/python3"' in config
    assert '# tags = { env = "dev", team = "platform" }' in config


def test_config_init_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "psoul.toml").write_text("")
    result = runner.invoke(cli, ["config", "init"])
    assert result.exit_code == 1


def test_doctor_ignores_broken_local_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "psoul.toml").write_text("[launch\nmode = 'headless'\n")
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0


def test_config_default_ignores_broken_local_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "psoul.toml").write_text("[launch\nmode = 'headless'\n")
    result = runner.invoke(cli, ["config", "--default"])
    assert result.exit_code == 0
    assert "launch.mode = 'attached'" in result.output


def test_config_reports_broken_local_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "psoul.toml").write_text("[launch\nmode = 'headless'\n")
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 1
    assert "Config" in result.output
    assert "Traceback" not in result.output


def test_doctor_ignores_missing_config_override() -> None:
    result = runner.invoke(cli, ["--config", "missing.toml", "doctor"])
    assert result.exit_code == 0


def test_config_reports_missing_config_override() -> None:
    result = runner.invoke(cli, ["--config", "missing.toml", "config"])
    assert result.exit_code == 1
    assert "Config error" in result.output
    assert "Traceback" not in result.output


def test_config_init_ignores_missing_config_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli, ["--config", "missing.toml", "config", "init"])
    assert result.exit_code == 0
    assert "Wrote psoul.toml" in result.output


@pytest.mark.parametrize("cli_args", [["doctor"], ["version"], ["config", "--default"]])
def test_safe_commands_survive_invalid_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_args: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "psoul.toml").write_text('[retention]\nmax_sessions = "nope"\n')
    result = runner.invoke(cli, cli_args)
    assert result.exit_code == 0


def test_config_reports_invalid_type_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "psoul.toml").write_text('[retention]\nmax_sessions = "nope"\n')
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 1
    assert "Config error" in result.output
    assert "expected int" in result.output
    assert "Traceback" not in result.output
