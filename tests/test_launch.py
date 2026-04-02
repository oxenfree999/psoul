"""Tests for the launch module: input validation, request assembly, and process lifecycle."""

import json
import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.db import open_db
from psoul.launch import (
    LaunchRequest,
    LaunchTarget,
    build_launch_request,
    launch_attached,
    launch_headless,
    parse_launch_target,
    resolve_session_id,
)
from psoul.session import LaunchMode, Session, SessionState, TargetType
from psoul.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SessionStore]:
    conn = open_db(tmp_path)
    yield SessionStore(conn)
    conn.close()


@pytest.mark.parametrize(
    ("target", "module", "extra", "expected_type", "expected_target"),
    [
        ("app.py", None, ["--port", "8000"], TargetType.script, "app.py"),
        (None, "http.server", ["8000"], TargetType.module, "http.server"),
    ],
)
def test_parse_launch_target(
    target: str | None,
    module: str | None,
    extra: list[str],
    expected_type: TargetType,
    expected_target: str,
) -> None:
    t = parse_launch_target(target=target, module=module, extra_args=extra)
    assert t.target_type == expected_type
    assert t.target == expected_target
    assert t.target_args == tuple(extra)


@pytest.mark.parametrize(
    ("target", "module", "match"),
    [
        ("app.py", "http.server", "choose either"),
        (None, None, "launch target is required"),
    ],
)
def test_parse_launch_target_rejects_bad_input(target: str | None, module: str | None, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_launch_target(target=target, module=module, extra_args=[])


def test_resolve_session_id_explicit() -> None:
    assert resolve_session_id("my-session") == "my-session"


def test_resolve_session_id_invalid() -> None:
    with pytest.raises(ValueError, match="lowercase alphanumeric"):
        resolve_session_id("BAD NAME!")


def test_resolve_session_id_generates_four_words() -> None:
    assert resolve_session_id(None).count("-") == 3


def test_as_cmd_script() -> None:
    t = LaunchTarget(TargetType.script, "app.py", ("--flag",))
    assert t.as_cmd() == [sys.executable, "app.py", "--flag"]


def test_as_cmd_module() -> None:
    t = LaunchTarget(TargetType.module, "http.server", ("8000",))
    assert t.as_cmd() == [sys.executable, "-m", "http.server", "8000"]


def test_build_launch_request_freezes_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.launch.sys.stdin.isatty", lambda: False)
    tags = {"env": "dev"}
    req = build_launch_request(target="x.py", module=None, extra_args=[], name="a-b-c-d", headless=False, tags=tags)
    tags["env"] = "prod"
    assert req.tags is not None
    assert req.tags["env"] == "dev"


def test_build_headless_when_no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.launch.sys.stdin.isatty", lambda: False)
    req = build_launch_request(target="x.py", module=None, extra_args=[], name="a-b-c-d", headless=False, tags=None)
    assert req.launch_mode == LaunchMode.headless


def test_build_headless_flag_overrides_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.launch.sys.stdin.isatty", lambda: True)
    req = build_launch_request(target="x.py", module=None, extra_args=[], name="a-b-c-d", headless=True, tags=None)
    assert req.launch_mode == LaunchMode.headless


def _script_request(
    code: str, *, name: str = "test-session", launch_mode: LaunchMode = LaunchMode.headless
) -> LaunchRequest:
    """Build a request that runs ``python -c <code>``."""
    return LaunchRequest(
        session_id=name,
        launch_mode=launch_mode,
        target=LaunchTarget(TargetType.script, "-c", (code,)),
        cwd=Path.cwd(),
    )


def _store_session(state_dir: Path, name: str, *, target: str = "test.py") -> None:
    """Insert a session directly into the store (no fork needed)."""
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
            target=target,
        )
    )
    store.update(name, state=SessionState.running)
    conn.close()


@requires_fork
def test_headless_supervisor_reaps_success(store: SessionStore, tmp_path: Path) -> None:
    req = _script_request("pass")
    session, supervisor_pid = launch_headless(req, store, tmp_path)
    assert session.state == SessionState.starting
    os.waitpid(supervisor_pid, 0)
    final = store.get(req.session_id)
    assert final is not None
    assert final.state == SessionState.exited


@requires_fork
def test_headless_supervisor_reaps_failure(store: SessionStore, tmp_path: Path) -> None:
    req = _script_request("import sys; sys.exit(1)", name="fail-session")
    session, supervisor_pid = launch_headless(req, store, tmp_path)
    assert session.state == SessionState.starting
    os.waitpid(supervisor_pid, 0)
    final = store.get(req.session_id)
    assert final is not None
    assert final.state == SessionState.failed


def test_attached_launch_records_current_process_as_supervisor(store: SessionStore) -> None:
    final = launch_attached(_script_request("pass", launch_mode=LaunchMode.attached), store)
    assert final.supervisor_pid == os.getpid()


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_headless_cli_prints_record_and_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "slow.py"
    script.write_text("import time; time.sleep(5)")
    start = time.monotonic()
    result = runner.invoke(cli, ["run", "--headless", "--name", "quick-exit", str(script)])
    elapsed = time.monotonic() - start
    assert result.exit_code == 0
    assert elapsed < 2
    record = json.loads(result.output)
    assert record["session_id"] == "quick-exit"
    assert record["state"] == "starting"
    assert record["supervisor_pid"] > 0


def test_duplicate_session_id_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    monkeypatch.setattr("psoul.launch.sys.stdin.isatty", lambda: True)
    _store_session(tmp_path, "dup-test")
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["run", "--name", "dup-test", str(script)])
    assert result.exit_code == 1
    assert "session ID already exists: dup-test" in result.output


def test_headless_without_fork_prints_cli_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    if hasattr(os, "fork"):
        monkeypatch.delattr("psoul.launch.os.fork")
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["run", "--headless", str(script)])
    assert result.exit_code == 1
    assert "headless mode requires Unix" in result.output


def test_ps_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    _store_session(tmp_path, "ps-test-a")
    _store_session(tmp_path, "ps-test-b")
    text_result = runner.invoke(cli, ["ps"])
    assert text_result.exit_code == 0
    assert "ps-test-a" in text_result.output
    json_result = runner.invoke(cli, ["ps", "--json"])
    assert json_result.exit_code == 0
    records = json.loads(json_result.output)
    assert {record["session_id"] for record in records} == {"ps-test-a", "ps-test-b"}


def test_status_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    _store_session(tmp_path, "status-test")
    text_result = runner.invoke(cli, ["status", "status-t"])
    assert text_result.exit_code == 0
    assert "status-test" in text_result.output
    json_result = runner.invoke(cli, ["status", "status-test", "--json"])
    assert json_result.exit_code == 0
    assert json.loads(json_result.output)["session_id"] == "status-test"


def test_status_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    result = runner.invoke(cli, ["status", "nonexistent"])
    assert result.exit_code == 1
    assert "session not found: nonexistent" in result.output


def test_status_ambiguous_prefix_lists_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    _store_session(tmp_path, "status-aa")
    _store_session(tmp_path, "status-ab")
    result = runner.invoke(cli, ["status", "status-a"])
    assert result.exit_code == 1
    assert "ambiguous session selector: status-a" in result.output
    assert "status-aa" in result.output
    assert "status-ab" in result.output


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_launch_to_query_exited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "hello.py"
    script.write_text("print('hello')")
    record = json.loads(runner.invoke(cli, ["run", "--headless", "--name", "e2e-ok", str(script)]).output)
    os.waitpid(record["supervisor_pid"], 0)
    ps_result = runner.invoke(cli, ["ps", "--state", "exited"])
    assert ps_result.exit_code == 0
    assert "e2e-ok" in ps_result.output
    detail = json.loads(runner.invoke(cli, ["status", "e2e-ok", "--json"]).output)
    assert detail["state"] == "exited"
    assert detail["target"] == str(script)


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_launch_to_query_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(42)")
    record = json.loads(runner.invoke(cli, ["run", "--headless", "--name", "e2e-fail", str(script)]).output)
    os.waitpid(record["supervisor_pid"], 0)
    assert "e2e-fail" in runner.invoke(cli, ["ps", "--state", "failed"]).output
    detail = json.loads(runner.invoke(cli, ["status", "e2e-fail", "--json"]).output)
    assert detail["state"] == "failed"
