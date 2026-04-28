"""Tests for the ``psoul env`` CLI command."""

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from psoul.cli import env as env_module
from psoul.cli.main import cli
from psoul.cli.state import ExitCode
from psoul.core.db import open_db
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()


@pytest.fixture
def mock_live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every live-env source so ``get_current_env`` output is deterministic."""

    def fake_tool(name: str) -> dict[str, Any]:
        return {"available": True, "version": f"{name}-1.2.3", "path": f"/fake/bin/{name}", "error": None}

    monkeypatch.setattr(env_module, "_get_tool_info", fake_tool)
    monkeypatch.setattr(env_module, "_get_venv", lambda: "/fake/.venv")
    monkeypatch.setenv("SHELL", "/fake/zsh")
    monkeypatch.delenv("COMSPEC", raising=False)
    monkeypatch.setattr(env_module.shutil, "which", lambda name: f"/fake/bin/{name}" if name == "uv" else None)
    monkeypatch.setattr(
        env_module.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(returncode=0, stdout="psoul v0.0.3\n├── dep1\n└── dep2\n"),
    )
    monkeypatch.setattr(env_module.importlib.metadata, "distributions", lambda: iter([1, 2, 3, 4, 5]))


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect psoul to use *tmp_path* as its state directory."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    return tmp_path


def test_get_current_env_happy_path(mock_live_env: None) -> None:
    info = env_module.get_current_env()
    assert info["source"] == "current"
    assert info["shell"] == "/fake/zsh"
    assert info["venv"] == "/fake/.venv"
    assert info["python"]["version"]
    assert set(info["tools"]) == set(env_module.ENV_TOOLS)
    assert info["tools"]["uv"]["version"] == "uv-1.2.3"
    assert info["dependency_graph"].startswith("psoul v0.0.3")
    assert info["package_count"] == 5


@pytest.mark.parametrize(
    ("env_vars", "expected"),
    [
        ({"SHELL": "/bin/zsh"}, "/bin/zsh"),
        ({"COMSPEC": "C:\\Windows\\System32\\cmd.exe"}, "C:\\Windows\\System32\\cmd.exe"),
        ({}, None),
    ],
    ids=["shell-only", "comspec-only", "neither"],
)
def test_get_current_env_shell_detection(
    mock_live_env: None,
    monkeypatch: pytest.MonkeyPatch,
    env_vars: dict[str, str],
    expected: str | None,
) -> None:
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.delenv("COMSPEC", raising=False)
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    info = env_module.get_current_env()
    assert info["shell"] == expected


@pytest.mark.parametrize(
    ("which_returns", "run_mock"),
    [
        (None, Mock()),
        ("/fake/bin/uv", Mock(side_effect=OSError("nope"))),
        ("/fake/bin/uv", Mock(side_effect=env_module.subprocess.TimeoutExpired(cmd="uv", timeout=3))),
        ("/fake/bin/uv", Mock(return_value=SimpleNamespace(returncode=1, stdout=""))),
    ],
    ids=["uv-missing", "oserror", "timeout", "nonzero-exit"],
)
def test_get_current_env_dependency_graph_none_on_failure(
    mock_live_env: None,
    monkeypatch: pytest.MonkeyPatch,
    which_returns: str | None,
    run_mock: Mock,
) -> None:
    monkeypatch.setattr(env_module.shutil, "which", lambda _: which_returns)
    monkeypatch.setattr(env_module.subprocess, "run", run_mock)
    info = env_module.get_current_env()
    assert info["dependency_graph"] is None


def test_get_current_env_package_count_none_on_failure(mock_live_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        env_module.importlib.metadata, "distributions", Mock(side_effect=RuntimeError("cannot iterate"))
    )
    info = env_module.get_current_env()
    assert info["package_count"] is None


def _make_session(**overrides: object) -> Session:
    """Build a Session with the required fields filled in. Override anything else as kwargs."""
    defaults: dict[str, Any] = {
        "session_id": "abc123",
        "state": SessionState.exited,
        "launch_mode": LaunchMode.headless,
        "launch_time": datetime(2026, 1, 1, tzinfo=UTC),
        "psoul_version": VERSION,
        "os": "darwin",
        "arch": "arm64",
        "python_version": "3.13.1",
    }
    defaults.update(overrides)
    return Session(**defaults)


def test_get_session_env_basic_shape() -> None:
    session = _make_session()
    info = env_module.get_session_env(session)
    assert info["source"] == f"session:{session.session_id}"
    assert info["platform"] == {"system": "darwin", "machine": "arm64"}
    assert info["python"] == {"version": "3.13.1"}


@pytest.mark.parametrize(
    ("field", "value", "expected_key"),
    [
        ("uv_version", "0.1.4", "tools"),
        ("target_cwd", Path("/proj"), "directory"),
        ("python_path", Path("/opt/python3.13/bin/python"), "interpreter"),
    ],
    ids=["uv_version", "target_cwd", "python_path"],
)
def test_get_session_env_conditionally_includes_optional_field(field: str, value: object, expected_key: str) -> None:
    info_set = env_module.get_session_env(_make_session(**{field: value}))
    info_unset = env_module.get_session_env(_make_session())
    assert expected_key in info_set
    assert expected_key not in info_unset


def test_get_session_env_includes_all_provenance_when_populated() -> None:
    populated = _make_session(
        host="myhost",
        git_sha="abc123",
        git_dirty=True,
        lockfile_hash="sha256:lock",
        script_hash="sha256:script",
        resolved_by="explicit",
        config_sources=["/cfg"],
    )
    info = env_module.get_session_env(populated)
    for key in ("host", "git_sha", "git_dirty", "lockfile_hash", "script_hash", "resolved_by", "config_sources"):
        assert key in info


def test_get_session_env_omits_provenance_when_unpopulated() -> None:
    info = env_module.get_session_env(_make_session())
    for key in ("host", "git_sha", "git_dirty", "lockfile_hash", "script_hash", "resolved_by", "config_sources"):
        assert key not in info


@pytest.fixture
def live_info() -> dict[str, Any]:
    """Minimal live-form info dict for ``format_text`` tests."""
    return {
        "source": "current",
        "platform": {"system": "Darwin", "release": "25.3.0", "machine": "arm64"},
        "shell": "/bin/zsh",
        "directory": "/home",
        "venv": "/home/.venv",
        "interpreter": "/usr/bin/python3",
        "python": {"version": "3.13.1"},
        "tools": {
            name: {"available": True, "version": f"{name}-1.0", "path": f"/p/{name}", "error": None}
            for name in env_module.ENV_TOOLS
        },
        "sys_path": ["/path/a", "/path/b"],
        "dependency_graph": "psoul v0.0.3\n├── dep1",
        "package_count": 42,
    }


def test_format_text_live_form_has_no_not_recorded(live_info: dict[str, Any]) -> None:
    assert "not recorded" not in env_module.format_text(live_info)


def test_format_text_session_form_marks_live_only_fields_as_not_recorded() -> None:
    text = env_module.format_text(env_module.get_session_env(_make_session()))
    for label in ("Shell", "Venv", "Sys path", "Dependencies", "Packages"):
        assert f"{label:<{env_module.LABEL_WIDTH}} not recorded" in text


@pytest.mark.parametrize(
    ("key", "expected_marker"),
    [
        ("shell", "not available"),
        ("venv", "not active"),
        ("dependency_graph", "not available"),
        ("package_count", "not available"),
    ],
    ids=["shell", "venv", "dependency_graph", "package_count"],
)
def test_format_text_renders_none_with_context_marker(
    live_info: dict[str, Any], key: str, expected_marker: str
) -> None:
    live_info[key] = None
    assert expected_marker in env_module.format_text(live_info)


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ({"available": True, "version": "1.2.3", "path": "/p", "error": None}, "1.2.3"),
        ({"available": True, "version": None, "path": "/p", "error": "timeout"}, "timeout"),
        ({"available": False, "version": None, "path": None, "error": None}, "not found"),
    ],
    ids=["version", "error", "not-found"],
)
def test_format_text_tool_rendering(live_info: dict[str, Any], tool: dict[str, Any], expected: str) -> None:
    live_info["tools"]["uv"] = tool
    assert f"{'uv':<{env_module.LABEL_WIDTH}} {expected}" in env_module.format_text(live_info)


def test_format_text_empty_sys_path_renders_as_empty_marker(live_info: dict[str, Any]) -> None:
    live_info["sys_path"] = []
    assert f"{'Sys path':<{env_module.LABEL_WIDTH}} (empty)" in env_module.format_text(live_info)


def test_format_text_empty_config_sources_renders_as_empty_marker() -> None:
    info = env_module.get_session_env(_make_session())
    info["config_sources"] = []
    assert f"{'Config':<{env_module.LABEL_WIDTH}} (empty)" in env_module.format_text(info)


@pytest.fixture
def seeded_session(state_dir: Path) -> Path:
    """Seed a session with provenance fields into the redirected state_dir."""
    with closing(open_db(state_dir)) as conn:
        SessionStore(conn).create(_make_session(host="myhost", git_sha="abc"))
    return state_dir


@pytest.mark.parametrize(
    ("args", "expected_substring"),
    [
        (["env"], "Source         current"),
        (["env", "--json"], '"source": "current"'),
    ],
    ids=["text", "json"],
)
def test_cli_no_arg(mock_live_env: None, args: list[str], expected_substring: str) -> None:
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert expected_substring in result.output


def test_cli_no_arg_does_not_open_db(mock_live_env: None, state_dir: Path) -> None:
    result = runner.invoke(cli, ["env"])
    assert result.exit_code == 0
    assert list(state_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("args", "source_substring"),
    [
        (["env", "abc123"], "Source         session:abc123"),
        (["env", "abc123", "--json"], '"source": "session:abc123"'),
    ],
    ids=["text", "json"],
)
def test_cli_session_arg_emits_source_and_provenance(
    seeded_session: Path, args: list[str], source_substring: str
) -> None:
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert source_substring in result.output
    assert "myhost" in result.output


def test_cli_session_arg_json_omits_live_only_keys(seeded_session: Path) -> None:
    result = runner.invoke(cli, ["env", "abc123", "--json"])
    assert result.exit_code == 0
    info = json.loads(result.output)
    assert "sys_path" not in info
    assert "dependency_graph" not in info
    assert "package_count" not in info


@pytest.mark.parametrize(
    ("seed_session_id", "selector", "expected_exit_code", "expected_substring"),
    [
        ("long-session-name", "long", 0, "session:long-session-name"),
        (None, "nonexistent", ExitCode.ERROR, "session not found"),
    ],
    ids=["unique-prefix", "not-found"],
)
def test_cli_session_selector(
    state_dir: Path,
    seed_session_id: str | None,
    selector: str,
    expected_exit_code: int,
    expected_substring: str,
) -> None:
    if seed_session_id is not None:
        with closing(open_db(state_dir)) as conn:
            SessionStore(conn).create(_make_session(session_id=seed_session_id))
    result = runner.invoke(cli, ["env", selector])
    assert result.exit_code == expected_exit_code
    assert expected_substring in result.output
