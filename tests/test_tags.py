"""Tests for tag parsing, CLI wiring, and filtering."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from psoul.cli.main import cli, parse_tags
from psoul.core.db import open_db
from psoul.core.session import LaunchMode, Session, SessionState, TargetType
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    """Create a config file pointing at a tmp_path-backed state dir.

    The config opts in to recording by default and pre-creates the DB so
    tests that exercise read-side commands skip the no-DB short-circuit.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config = tmp_path / "psoul.toml"
    config.write_text(f"[paths]\nstate_dir = '{state_dir}'\n[session]\nrecord = true\n")
    open_db(state_dir).close()
    return config, state_dir


def _store_tagged_session(state_dir: Path, name: str, tags: dict[str, str] | None = None) -> None:
    """Insert a running session with optional tags directly into the store."""
    conn = open_db(state_dir)
    store = SessionStore(conn)
    store.create(
        Session(
            session_id=name,
            state=SessionState.starting,
            launch_mode=LaunchMode.headless,
            launch_time=datetime.now(UTC),
            psoul_version=VERSION,
            target_type=TargetType.script,
            target="test.py",
            tags=tags,
        )
    )
    store.update(name, state=SessionState.running)
    conn.close()


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

    @pytest.mark.parametrize(
        ("raw", "defaults", "expected"),
        [
            (None, None, None),
            (None, {"env": "dev"}, {"env": "dev"}),
            ([], {"env": "dev"}, {"env": "dev"}),
            (["team=ml"], {"env": "dev"}, {"env": "dev", "team": "ml"}),
            (["env=prod"], {"env": "dev"}, {"env": "prod"}),
            (["env=prod", "team=ml"], {"env": "dev", "region": "us"}, {"env": "prod", "region": "us", "team": "ml"}),
        ],
        ids=["both-empty", "defaults-only-none-raw", "defaults-only-empty-raw", "merge", "cli-wins", "complex-merge"],
    )
    def test_merges_defaults(
        self, raw: list[str] | None, defaults: dict[str, str] | None, expected: dict[str, str] | None
    ) -> None:
        assert parse_tags(raw, defaults=defaults) == expected


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_run_tag_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
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


@pytest.mark.parametrize(
    ("mock_target", "cli_args"),
    [
        ("psoul.cli.main.build_launch_request", ["run", "--tag", "env=prod", "noop.py"]),
        ("psoul.cli.main.run_repl", ["repl", "--name", "tag-merge", "--tag", "env=prod"]),
    ],
    ids=["run", "repl"],
)
def test_command_merges_config_tags_with_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_target: str, cli_args: list[str]
) -> None:
    config, _ = _write_config(tmp_path)
    config.write_text(f'{config.read_text()}\n[session.tags]\nenv = "dev"\nteam = "ops"\n')
    captured: dict[str, object] = {}

    def fake(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise typer.Exit(0)

    monkeypatch.setattr(mock_target, fake)
    result = runner.invoke(cli, ["--config", str(config), *cli_args])
    assert result.exit_code == 0
    assert captured["tags"] == {"env": "prod", "team": "ops"}


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
        ["ps", "--tag", "broken"],
    ],
    ids=["run", "repl", "ps"],
)
def test_tag_cli_rejects_malformed_input(tmp_path: Path, args: list[str]) -> None:
    config, _state_dir = _write_config(tmp_path)
    result = runner.invoke(cli, ["--config", str(config), *args])
    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert "expected key=value" in result.output


class TestPsTagFilter:
    """Tests for ps --tag filtering: AND logic, superset match, and negative cases."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config, state_dir = _write_config(tmp_path)
        self.config = str(config)
        _store_tagged_session(state_dir, "all-tags", {"env": "dev", "team": "backend", "region": "us"})
        _store_tagged_session(state_dir, "partial-tags", {"env": "dev"})
        _store_tagged_session(state_dir, "no-tags")
        _store_tagged_session(state_dir, "wrong-value", {"env": "prod"})

    @pytest.mark.parametrize(
        ("tags", "expected_ids"),
        [
            (["env=dev"], {"all-tags", "partial-tags"}),
            (["env=dev", "team=backend"], {"all-tags"}),
            (["env=dev", "region=us"], {"all-tags"}),
            (["env=prod"], {"wrong-value"}),
            (["env=staging"], set()),
            (["team=backend"], {"all-tags"}),
        ],
        ids=[
            "single-tag-matches-superset-and-exact",
            "multiple-tags-and-logic",
            "superset-match",
            "wrong-value-only",
            "no-match-returns-empty",
            "excludes-no-tags-and-partial",
        ],
    )
    def test_filter(self, tags: list[str], expected_ids: set[str]) -> None:
        args = ["--config", self.config, "ps", "--json"]
        for t in tags:
            args.extend(["--tag", t])
        result = runner.invoke(cli, args)
        assert result.exit_code == 0
        actual_ids = {r["session_id"] for r in json.loads(result.output)}
        assert actual_ids == expected_ids


def test_status_displays_tags(tmp_path: Path) -> None:
    config, state_dir = _write_config(tmp_path)
    _store_tagged_session(state_dir, "status-tag", {"env": "dev", "team": "backend"})
    conn = open_db(state_dir)
    try:
        session = SessionStore(conn).get("status-tag")
    finally:
        conn.close()
    assert session is not None
    assert session.tags == {"env": "dev", "team": "backend"}
    text_result = runner.invoke(cli, ["--config", str(config), "status", "status-tag"])
    assert text_result.exit_code == 0
    assert "tags: {'env': 'dev', 'team': 'backend'}" in text_result.output
    json_result = runner.invoke(cli, ["--config", str(config), "status", "status-tag", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.output)
    assert data["tags"] == {"env": "dev", "team": "backend"}
