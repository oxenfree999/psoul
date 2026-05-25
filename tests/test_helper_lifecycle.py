"""Lifecycle tests for the launch.py helper integration."""

import os
import socket
import subprocess
import sys
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from psoul.core.db import open_db
from psoul.core.events import EVENT_HELPER_CRASHED, EVENT_RUNTIME_STATUS, EventStore
from psoul.core.helper import HelperLifecycle
from psoul.core.launch import (
    PSOUL_HELPER_PIPE_ENV,
    LaunchRequest,
    LaunchTarget,
    _build_helper_env,
    _build_wrapper_argv,
    _check_cwd_collision,
    _watch_helper_eof,
    launch_attached,
)
from psoul.core.session import LaunchMode, Session, SessionState, TargetType
from psoul.core.store import SessionStore


def _make_session(session_id: str) -> Session:
    return Session(
        session_id=session_id,
        state=SessionState.starting,
        launch_mode=LaunchMode.attached,
        launch_time=datetime.now(UTC),
        psoul_version="0.0.0",
        target_type=TargetType.script,
        target="x.py",
        target_args=[],
        target_cwd=Path("/tmp"),
        tags=None,
    )


def _make_request(session_id: str, cwd: Path, record: bool) -> LaunchRequest:
    return LaunchRequest(
        session_id=session_id,
        launch_mode=LaunchMode.attached,
        target=LaunchTarget(
            target_type=TargetType.script,
            target=str(cwd / "user.py"),
            target_args=(),
            python_path=Path(sys.executable),
        ),
        cwd=cwd,
        record_requested=record,
    )


def test_check_cwd_collision_no_shadow_passes(tmp_path: Path) -> None:
    _check_cwd_collision(tmp_path)


@pytest.mark.parametrize(
    "shadow",
    ["_psoul_helper.py", "_psoul_helper/__init__.py", "_psoul_helper/__main__.py"],
    ids=["file", "regular_pkg", "namespace_pkg_main"],
)
def test_check_cwd_collision_rejects_shadow(tmp_path: Path, shadow: str) -> None:
    target = tmp_path / shadow
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# stub\n")
    with pytest.raises(ValueError, match="psoul helper"):
        _check_cwd_collision(tmp_path)


def test_build_wrapper_argv_inserts_helper_module_invocation() -> None:
    assert _build_wrapper_argv(["python", "script.py", "arg1"]) == [
        "python",
        "-m",
        "_psoul_helper",
        "script.py",
        "arg1",
    ]


def test_build_helper_env_sets_pipe_var_and_prepends_pythonpath() -> None:
    parent = {"PYTHONPATH": "/user/path", "OTHER": "x"}
    env = _build_helper_env(parent, helper_fd=42)
    assert env[PSOUL_HELPER_PIPE_ENV] == "42"
    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath_entries[0].endswith(str(Path("psoul") / "helper"))
    assert pythonpath_entries[-1] == "/user/path"
    assert env["OTHER"] == "x"


def test_build_helper_env_handles_unset_pythonpath() -> None:
    env = _build_helper_env({}, helper_fd=7)
    assert env[PSOUL_HELPER_PIPE_ENV] == "7"
    assert os.pathsep not in env["PYTHONPATH"]


@pytest.fixture
def state_with_session(tmp_path: Path) -> tuple[Path, str]:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_id = "watcher-session"
    with closing(open_db(state_dir)) as conn:
        SessionStore(conn).create(_make_session(session_id))
    return state_dir, session_id


@pytest.mark.parametrize(
    ("wait_result", "expect_events", "expected_crash_payload"),
    [
        ("timeout", True, {}),
        (0, False, None),
        (1, True, {"exit_code": 1}),
    ],
    ids=["helper_crash", "clean_exit", "nonclean_exit"],
)
def test_watch_helper_eof_emits_events_only_on_crash(
    state_with_session: tuple[Path, str],
    wait_result: str | int,
    expect_events: bool,
    expected_crash_payload: dict[str, object] | None,
) -> None:
    state_dir, session_id = state_with_session
    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    proc = MagicMock()
    proc.poll.return_value = None
    if wait_result == "timeout":
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="child", timeout=2.0)
    else:
        proc.wait.return_value = wait_result
    lifecycle = HelperLifecycle(MagicMock())

    with parent_sock:
        thread = threading.Thread(
            target=_watch_helper_eof,
            args=(state_dir, session_id, parent_sock, lifecycle, proc),
        )
        thread.start()
        child_sock.close()
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    with closing(open_db(state_dir, create=False)) as conn:
        session = SessionStore(conn).get(session_id)
        events = EventStore(conn).list(session_id)
    assert session is not None
    assert session.helper_pid is None
    assert session.helper_capabilities is None
    event_types = {e["event_type"] for e in events}
    assert (EVENT_HELPER_CRASHED in event_types) is expect_events
    assert (EVENT_RUNTIME_STATUS in event_types) is expect_events
    if expect_events:
        crashed = next(e for e in events if e["event_type"] == EVENT_HELPER_CRASHED)
        assert crashed["payload"] == expected_crash_payload


def test_watch_helper_eof_ignores_in_flight_frame(
    state_with_session: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readable in-flight frame is not EOF, so the watcher keeps watching until real EOF."""
    state_dir, session_id = state_with_session
    helper_sock = MagicMock()
    helper_sock.recv.side_effect = [b"\x01", b""]
    select_calls = iter([([], [], []), ([helper_sock], [], []), ([helper_sock], [], [])])
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="child", timeout=2.0)
    monkeypatch.setattr("select.select", lambda *_: next(select_calls))
    monkeypatch.setattr("time.sleep", lambda _: None)

    _watch_helper_eof(state_dir, session_id, helper_sock, HelperLifecycle(MagicMock()), proc)

    with closing(open_db(state_dir, create=False)) as conn:
        events = EventStore(conn).list(session_id)
    assert helper_sock.recv.call_count == 2
    crashed = [e for e in events if e["event_type"] == EVENT_HELPER_CRASHED]
    assert len(crashed) == 1


def test_launch_attached_non_recorded_skips_helper_plumbing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-recorded launches skip helper plumbing entirely."""
    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    target = tmp_path / "user.py"
    target.write_text("# stub\n")
    request = _make_request("non-recorded", tmp_path, record=False)
    with closing(open_db(state_dir)) as conn:
        store = SessionStore(conn)
        launch_attached(
            request=request,
            store=store,
            state_dir=state_dir,
            helper_connect_timeout_seconds=5.0,
        )
        session = store.get(request.session_id)
        events = EventStore(conn).list(request.session_id)
    assert session is not None
    assert session.helper_pid is None
    assert session.helper_capabilities is None
    assert all(e["event_type"] != "helper.timeout" for e in events)


def test_launch_attached_collision_emits_usage_error_and_no_session_row(
    tmp_path: Path,
) -> None:
    """Collision check fires before `_create_session`. No persisted starting row on collision."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "_psoul_helper.py").write_text("# shadow\n")
    target = tmp_path / "user.py"
    target.write_text("# stub\n")
    request = _make_request("collide", tmp_path, record=True)
    with closing(open_db(state_dir)) as conn:
        store = SessionStore(conn)
        with pytest.raises(ValueError, match="psoul helper"):
            launch_attached(
                request=request,
                store=store,
                state_dir=state_dir,
                helper_connect_timeout_seconds=5.0,
            )
        assert store.get(request.session_id) is None
