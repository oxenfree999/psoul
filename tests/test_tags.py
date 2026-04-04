"""Tests for tag parsing, CLI wiring, and filtering."""

import json
import os
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from psoul.cli.main import cli, parse_tags
from psoul.db import open_db
from psoul.store import SessionStore

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    """Create a config file pointing at a tmp_path-backed state dir."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config = tmp_path / "psoul.toml"
    config.write_text(f"[paths]\nstate_dir = '{state_dir}'\n")
    return config, state_dir


class TestParseTags:
    """Unit tests for the parse_tags() CLI helper."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ([], None),
            (["env=dev"], {"env": "dev"}),
            (["env=dev", "team=backend"], {"env": "dev", "team": "backend"}),
            (["query=x=1&y=2"], {"query": "x=1&y=2"}),
            (["marker="], {"marker": ""}),
            (["env=dev", "env=prod"], {"env": "prod"}),
            ([" env = dev "], {"env": "dev"}),
            (["note=hello world"], {"note": "hello world"}),
            (["flag=   "], {"flag": ""}),
        ],
        ids=[
            "none",
            "empty-list",
            "single-tag",
            "multiple-tags",
            "value-containing-equals",
            "empty-value-allowed",
            "duplicate-key-last-wins",
            "strips-whitespace",
            "value-with-interior-spaces",
            "whitespace-only-value-becomes-empty",
        ],
    )
    def test_valid_input(self, raw: list[str] | None, expected: dict[str, str] | None) -> None:
        assert parse_tags(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "match"),
        [
            (["noequals"], "expected key=value"),
            (["=value"], "empty key"),
            (["   =value"], "empty key"),
        ],
        ids=["missing-equals", "empty-key", "whitespace-only-key"],
    )
    def test_invalid_input(self, raw: list[str], match: str) -> None:
        with pytest.raises(typer.BadParameter, match=match):
            parse_tags(raw)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_run_tag_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(
        cli, ["run", "--headless", "--name", "tag-run", "--tag", "env=dev", "--tag", "team=backend", str(script)]
    )
    assert result.exit_code == 0
    record = json.loads(result.output)
    os.waitpid(record["supervisor_pid"], 0)
    conn = open_db(tmp_path)
    try:
        session = SessionStore(conn).get("tag-run")
    finally:
        conn.close()
    assert session is not None
    assert session.tags == {"env": "dev", "team": "backend"}


def test_repl_tag_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def prompt(self, _message: str) -> str:
            raise EOFError

    monkeypatch.setattr("psoul.cli.repl.PromptSession", FakePromptSession)
    config, state_dir = _write_config(tmp_path)
    result = runner.invoke(
        cli,
        ["--config", str(config), "repl", "--name", "repl-tag", "--tag", "env=staging", "--tag", "team=backend"],
    )
    assert result.exit_code == 0
    conn = open_db(state_dir)
    try:
        session = SessionStore(conn).get("repl-tag")
    finally:
        conn.close()
    assert session is not None
    assert session.tags == {"env": "staging", "team": "backend"}


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--name", "bad-tag-run", "--tag", "broken", "-m", "http.server"],
        ["repl", "--name", "bad-tag-repl", "--tag", "broken"],
    ],
    ids=["run", "repl"],
)
def test_tag_cli_rejects_malformed_input(tmp_path: Path, args: list[str]) -> None:
    config, _state_dir = _write_config(tmp_path)
    result = runner.invoke(cli, ["--config", str(config), *args])
    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert "expected key=value" in result.output
