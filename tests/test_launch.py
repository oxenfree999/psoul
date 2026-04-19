"""Tests for the launch module: input validation, request assembly, and process lifecycle."""

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import click
import pytest
import typer
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.core.db import open_db
from psoul.core.launch import (
    LaunchRequest,
    LaunchTarget,
    build_launch_request,
    launch_attached,
    launch_headless,
    parse_launch_target,
    resolve_session_id,
    wait_for_exit,
)
from psoul.core.provenance import SessionProvenance
from psoul.core.session import LaunchMode, Session, SessionState, TargetType
from psoul.core.store import SessionStore
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
    t = parse_launch_target(target=target, module=module, extra_args=extra, python_path=Path(sys.executable))
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
        parse_launch_target(target=target, module=module, extra_args=[], python_path=Path(sys.executable))


def test_resolve_session_id_explicit() -> None:
    assert resolve_session_id("my-session") == "my-session"


def test_resolve_session_id_invalid() -> None:
    with pytest.raises(ValueError, match="lowercase alphanumeric"):
        resolve_session_id("BAD NAME!")


def test_resolve_session_id_generates_four_words() -> None:
    assert resolve_session_id(None).count("-") == 3


def test_as_cmd_script() -> None:
    t = LaunchTarget(TargetType.script, "app.py", ("--flag",), Path(sys.executable))
    assert t.as_cmd() == [sys.executable, "app.py", "--flag"]


def test_as_cmd_module() -> None:
    t = LaunchTarget(TargetType.module, "http.server", ("8000",), Path(sys.executable))
    assert t.as_cmd() == [sys.executable, "-m", "http.server", "8000"]


@pytest.mark.parametrize(
    ("target_type", "target", "expected_prefix"),
    [
        (TargetType.script, "app.py", ["fake-python", "app.py"]),
        (TargetType.module, "http.server", ["fake-python", "-m", "http.server"]),
    ],
    ids=["script", "module"],
)
def test_as_cmd_uses_configured_python_path(target_type: TargetType, target: str, expected_prefix: list[str]) -> None:
    t = LaunchTarget(target_type, target, ("--flag",), Path("fake-python"))
    assert t.as_cmd() == [*expected_prefix, "--flag"]


def test_build_launch_request_freezes_tags() -> None:
    tags = {"env": "dev"}
    req = build_launch_request(
        target="x.py",
        module=None,
        extra_args=[],
        name="a-b-c-d",
        headless=False,
        tags=tags,
        python_path=Path(sys.executable),
        default_mode=LaunchMode.attached,
    )
    tags["env"] = "prod"
    assert req.tags is not None
    assert req.tags["env"] == "dev"


@pytest.mark.parametrize(
    ("headless", "default_mode", "expected"),
    [
        (True, LaunchMode.attached, LaunchMode.headless),
        (False, LaunchMode.headless, LaunchMode.headless),
        (False, LaunchMode.attached, LaunchMode.attached),
    ],
    ids=["flag-forces-headless", "default-headless", "default-attached"],
)
def test_build_launch_request_resolves_mode(
    headless: bool,
    default_mode: LaunchMode,
    expected: LaunchMode,
) -> None:
    req = build_launch_request(
        target="x.py",
        module=None,
        extra_args=[],
        name="a-b-c-d",
        headless=headless,
        tags=None,
        python_path=Path(sys.executable),
        default_mode=default_mode,
    )
    assert req.launch_mode == expected


def test_build_launch_request_threads_python_path() -> None:
    req = build_launch_request(
        target="x.py",
        module=None,
        extra_args=[],
        name="a-b-c-d",
        headless=False,
        tags=None,
        python_path=Path("/fake/python"),
        default_mode=LaunchMode.attached,
    )
    assert req.target.python_path == Path("/fake/python")


def test_build_launch_request_stays_attached_when_stdin_not_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: non-TTY stdin does not auto-promote attached launches to headless."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    req = build_launch_request(
        target="x.py",
        module=None,
        extra_args=[],
        name="a-b-c-d",
        headless=False,
        tags=None,
        python_path=Path(sys.executable),
        default_mode=LaunchMode.attached,
    )
    assert req.launch_mode == LaunchMode.attached


def _script_request(
    code: str, *, name: str = "test-session", launch_mode: LaunchMode = LaunchMode.headless
) -> LaunchRequest:
    """Build a request that runs ``python -c <code>``."""
    return LaunchRequest(
        session_id=name,
        launch_mode=launch_mode,
        target=LaunchTarget(TargetType.script, "-c", (code,), Path(sys.executable)),
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


def test_attached_launch_clears_supervisor_pid_on_exit(store: SessionStore) -> None:
    final = launch_attached(_script_request("pass", launch_mode=LaunchMode.attached), store)
    assert final.supervisor_pid is None


def test_attached_launch_populates_provenance(store: SessionStore, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_provenance: SessionProvenance = {
        "git_sha": "b" * 40,
        "git_dirty": True,
        "script_hash": None,
        "lockfile_hash": "sha256:abc123",
        "python_version": "3.14.0",
        "python_path": Path("/usr/bin/python3"),
        "host": "testhost",
        "os": "linux",
        "arch": "aarch64",
    }
    calls: list[tuple[object, ...]] = []

    def fake_gather(*args: object) -> SessionProvenance:
        calls.append(args)
        return fake_provenance

    monkeypatch.setattr("psoul.core.launch.gather", fake_gather)
    req = _script_request("pass", launch_mode=LaunchMode.attached)
    final = launch_attached(req, store)

    assert calls == [(TargetType.script, "-c", req.cwd, Path(sys.executable))]
    assert final.git_sha == "b" * 40
    assert final.git_dirty is True
    assert final.script_hash is None
    assert final.lockfile_hash == "sha256:abc123"
    assert final.python_version == "3.14.0"
    assert final.python_path == Path("/usr/bin/python3")
    assert final.host == "testhost"
    assert final.os == "linux"
    assert final.arch == "aarch64"


def _get_result(store: SessionStore, session_id: str) -> dict[str, object]:
    """Read the results row for a session, returning a plain dict."""
    store.conn.row_factory = None
    cur = store.conn.execute("SELECT * FROM results WHERE session_id = ?", [session_id])
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    store.conn.row_factory = __import__("sqlite3").Row
    assert row is not None, f"no result row for {session_id}"
    return dict(zip(cols, row, strict=True))


def test_attached_exit_records_result_success(store: SessionStore) -> None:
    final = launch_attached(_script_request("pass", launch_mode=LaunchMode.attached, name="res-ok"), store)
    assert final.state == SessionState.exited
    result = _get_result(store, "res-ok")
    assert result["outcome"] == "exited"
    assert result["exit_code"] == 0
    assert result["duration_seconds"] is not None
    assert result["end_time"] is not None


def test_attached_exit_records_result_failure(store: SessionStore) -> None:
    final = launch_attached(
        _script_request("import sys; sys.exit(42)", launch_mode=LaunchMode.attached, name="res-fail"), store
    )
    assert final.state == SessionState.failed
    result = _get_result(store, "res-fail")
    assert result["outcome"] == "failed"
    assert result["exit_code"] == 42
    assert result["duration_seconds"] is not None
    assert result["end_time"] is not None


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_headless_supervisor_records_realistic_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: headless duration must measure the full session, not just proc.wait()."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(0.3)")
    result = runner.invoke(cli, ["run", "--headless", "--name", "sleepy", str(script)])
    assert result.exit_code == 0
    record = json.loads(result.output)
    os.waitpid(record["supervisor_pid"], 0)
    with closing(open_db(tmp_path)) as conn:
        row = conn.execute("SELECT duration_seconds FROM results WHERE session_id = ?", ["sleepy"]).fetchone()
    assert row is not None
    assert row[0] is not None
    assert row[0] >= 0.2


def test_wait_for_exit_records_result_when_wait_raises(store: SessionStore) -> None:
    """Regression: if proc.wait() raises, the session must still transition out of running."""
    store.create(
        Session(
            session_id="crash-test",
            state=SessionState.starting,
            launch_mode=LaunchMode.attached,
            launch_time=datetime.now(UTC),
            psoul_version=VERSION,
        )
    )
    store.update("crash-test", state=SessionState.running)

    class FakeProc:
        returncode = 1

        def wait(self) -> None:
            raise OSError("fake crash")

    with pytest.raises(OSError, match="fake crash"):
        wait_for_exit("crash-test", FakeProc(), store)  # ty: ignore[invalid-argument-type]

    final = store.get("crash-test")
    assert final is not None
    assert final.state == SessionState.failed
    result = _get_result(store, "crash-test")
    assert result["outcome"] == "failed"
    assert result["exit_code"] == 1


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_headless_cli_prints_record_and_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
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


def test_run_config_attached_beats_piped_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: `config.launch.mode = "attached"` wins over non-TTY stdin through the CLI.

    Relies on CliRunner's default non-TTY stdin. `sys.stdin.isatty()` is `False` inside
    `runner.invoke(...)` without extra setup.
    """
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    config = tmp_path / "psoul.toml"
    config.write_text('[launch]\nmode = "attached"\n')
    script = tmp_path / "noop.py"
    script.write_text("pass")
    captured: dict[str, LaunchMode] = {}

    def capture(request: LaunchRequest, *_args: object) -> None:
        captured["mode"] = request.launch_mode
        raise typer.Exit(0)

    monkeypatch.setattr("psoul.cli.main.launch_attached", capture)
    monkeypatch.setattr("psoul.cli.main.launch_headless", capture)
    result = runner.invoke(cli, ["--config", str(config), "run", str(script)])
    assert result.exit_code == 0
    assert captured["mode"] == LaunchMode.attached


def test_duplicate_session_id_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    _store_session(tmp_path, "dup-test")
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["run", "--name", "dup-test", str(script)])
    assert result.exit_code == 1
    assert "session ID already exists: dup-test" in result.output


def test_headless_without_fork_prints_cli_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    if hasattr(os, "fork"):
        monkeypatch.delattr("psoul.core.launch.os.fork")
    script = tmp_path / "noop.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["run", "--headless", str(script)])
    assert result.exit_code == 1
    assert "headless mode requires Unix" in result.output


def test_ps_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
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
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    _store_session(tmp_path, "status-test")
    text_result = runner.invoke(cli, ["status", "status-t"])
    assert text_result.exit_code == 0
    assert "status-test" in text_result.output
    json_result = runner.invoke(cli, ["status", "status-test", "--json"])
    assert json_result.exit_code == 0
    assert json.loads(json_result.output)["session_id"] == "status-test"


def test_status_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    result = runner.invoke(cli, ["status", "nonexistent"])
    assert result.exit_code == 1
    assert "session not found: nonexistent" in result.output


def test_status_ambiguous_prefix_lists_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
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
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
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
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(42)")
    record = json.loads(runner.invoke(cli, ["run", "--headless", "--name", "e2e-fail", str(script)]).output)
    os.waitpid(record["supervisor_pid"], 0)
    assert "e2e-fail" in runner.invoke(cli, ["ps", "--state", "failed"]).output
    detail = json.loads(runner.invoke(cli, ["status", "e2e-fail", "--json"]).output)
    assert detail["state"] == "failed"


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_bare_file_routes_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``psoul script.py`` is equivalent to ``psoul run script.py``."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "hello.py"
    script.write_text("pass")
    result = runner.invoke(cli, [str(script)])
    assert result.exit_code == 0
    records = json.loads(runner.invoke(cli, ["ps", "--json"]).output)
    assert len(records) == 1
    assert records[0]["target"] == str(script)
    assert records[0]["state"] == "exited"


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_bare_file_with_global_verbose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``psoul -v script.py`` — global options parsed before disambiguation."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "hello.py"
    script.write_text("pass")
    result = runner.invoke(cli, ["-v", str(script)])
    assert result.exit_code == 0
    records = json.loads(runner.invoke(cli, ["ps", "--json"]).output)
    assert len(records) == 1


def test_bare_nonexistent_file_errors_unknown_command(tmp_path: Path) -> None:
    """``psoul nonexistent.py`` errors instead of silently routing to run."""
    result = runner.invoke(cli, [str(tmp_path / "nonexistent.py")])
    assert result.exit_code == 2
    assert "No such command" in click.unstyle(result.output)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_bare_file_passes_extra_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``psoul script.py extra --child-flag`` passes trailing tokens to run."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "echo.py"
    script.write_text("pass")
    result = runner.invoke(cli, [str(script), "extra", "--child-flag"])
    assert result.exit_code == 0
    records = json.loads(runner.invoke(cli, ["ps", "--json"]).output)
    assert len(records) == 1
    detail = json.loads(runner.invoke(cli, ["status", records[0]["session_id"], "--json"]).output)
    assert detail["target_args"] == ["extra", "--child-flag"]


def test_unknown_flag_not_swallowed_by_disambiguation() -> None:
    """``psoul --bad script.py`` still errors — disambiguation is narrow."""
    result = runner.invoke(cli, ["--bad", "script.py"])
    assert result.exit_code == 2
    assert "No such option: --bad" in click.unstyle(result.output)
