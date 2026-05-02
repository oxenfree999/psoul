"""Tests for the psoul attach CLI command."""

import json
import os
import socket
import sys
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import ptyprocess
import pytest
from typer.testing import CliRunner

from psoul.cli.attach import (
    _DETACH_ESCAPE,
    _handle_socket_read,
    _handle_stdin_read,
    run_attach_loop,
)
from psoul.cli.main import cli
from psoul.cli.state import ExitCode
from psoul.core.db import open_db
from psoul.core.events import (
    EVENT_SESSION_CONTROLLER_ACQUIRED,
    EVENT_SESSION_CONTROLLER_RELEASED,
    EventStore,
)
from psoul.core.pty_spawn import (
    _FRAME_DATA,
    _FRAME_HELLO,
    _cleanup_listen_socket,
    _create_listen_socket,
    _decode_frames,
    _encode_frame,
)
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()
requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (Unix)")

_DETACH_ESCAPE_BYTE = bytes([0x1D])
_TEST_RECV_BUFSIZE = 1024  # bytes per recv() in attach-client tests, sized for small fixtures
_PROC_STARTUP_GRACE = 0.5  # seconds to let the spawned attach client connect and send HELLO
_RUNNING_TIMEOUT = 5.0


def _wait_for_running(state_dir: Path, session_id: str, timeout: float = _RUNNING_TIMEOUT) -> None:
    """Poll the DB until *session_id* enters the running state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(open_db(state_dir)) as conn:
            session = SessionStore(conn).get(session_id)
        if session is not None and session.state == SessionState.running:
            return
        time.sleep(0.02)
    msg = f"session {session_id} did not reach running within {timeout}s"
    raise RuntimeError(msg)


def _write_test_config(tmp_path: Path) -> Path:
    """Write a minimal psoul.toml pointing at *tmp_path* and return the config path."""
    config_path = tmp_path / "psoul.toml"
    config_path.write_text(f'[paths]\nstate_dir = "{tmp_path}"\n')
    return config_path


def test_attach_rejects_non_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CliRunner's default non-TTY stdin/stdout triggers a USAGE-coded error."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    open_db(tmp_path).close()
    result = runner.invoke(cli, ["attach", "any-session"])
    assert result.exit_code == ExitCode.USAGE
    assert "TTY" in result.output


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_attach_rejects_missing_session(tmp_path: Path) -> None:
    """Attach to a non-existent session exits with USAGE."""
    config_path = _write_test_config(tmp_path)
    proc = ptyprocess.PtyProcess.spawn(
        [sys.executable, "-m", "psoul", "--config", str(config_path), "attach", "missing-session"],
    )
    proc.wait()
    assert proc.exitstatus == ExitCode.USAGE


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_attach_runtime_failure_returns_error_exit_code(tmp_path: Path) -> None:
    """A stale or missing socket_path causes a runtime OSError that the CLI normalizes to ExitCode.ERROR."""
    config_path = _write_test_config(tmp_path)
    session_id = "stale-sock"
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id=session_id,
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
                socket_path=tmp_path / "nonexistent.sock",
            )
        )
    proc = ptyprocess.PtyProcess.spawn(
        [sys.executable, "-m", "psoul", "--config", str(config_path), "attach", session_id],
    )
    proc.wait()
    assert proc.exitstatus == ExitCode.ERROR


@requires_fork
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_attach_happy_path_detach_then_session_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Attach via PTY subprocess, detach with Ctrl-], session keeps running, controller events recorded."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    script = tmp_path / "long.py"
    script.write_text("import time\ntime.sleep(60)\n")
    launch = runner.invoke(cli, ["run", "--headless", str(script)])
    assert launch.exit_code == 0
    info = json.loads(launch.output)
    session_id = info["session_id"]
    supervisor_pid = info["supervisor_pid"]
    _wait_for_running(tmp_path, session_id)

    config_path = _write_test_config(tmp_path)
    proc = ptyprocess.PtyProcess.spawn(
        [sys.executable, "-m", "psoul", "--config", str(config_path), "attach", session_id],
    )
    try:
        time.sleep(_PROC_STARTUP_GRACE)
        proc.write(_DETACH_ESCAPE_BYTE)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and proc.isalive():
            time.sleep(0.05)
        assert not proc.isalive(), "attach client did not exit after Ctrl-]"
        assert proc.exitstatus == 0
        with closing(open_db(tmp_path)) as conn:
            session = SessionStore(conn).get(session_id)
            events = EventStore(conn).list(session_id)
        assert session is not None
        assert session.state == SessionState.running
        assert session.control_epoch == 1
        assert any(e["event_type"] == EVENT_SESSION_CONTROLLER_ACQUIRED for e in events)
        assert any(e["event_type"] == EVENT_SESSION_CONTROLLER_RELEASED for e in events)
    finally:
        if proc.isalive():
            proc.terminate(force=True)
        proc.wait()
        runner.invoke(cli, ["kill", session_id])
        os.waitpid(supervisor_pid, 0)


def test_handle_socket_read_writes_data_frames_to_stdout(capfd: pytest.CaptureFixture[str]) -> None:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.setblocking(False)
    with a, b:
        a.sendall(_encode_frame(_FRAME_DATA, b"hello"))
        buf = bytearray()
        assert _handle_socket_read(b, buf) is True
    captured = capfd.readouterr()
    assert "hello" in captured.out


def test_handle_socket_read_returns_false_on_eof() -> None:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.setblocking(False)
    a.close()
    with b:
        buf = bytearray()
        assert _handle_socket_read(b, buf) is False


@pytest.mark.parametrize(
    ("input_bytes", "expected_detach", "expected_forwarded"),
    [
        (_DETACH_ESCAPE, True, b""),
        (b"abc" + _DETACH_ESCAPE, True, b"abc"),
        (b"abc", False, b"abc"),
    ],
    ids=["bare-escape", "pre-escape-bytes", "no-escape"],
)
def test_handle_stdin_read_partitions_on_escape(
    monkeypatch: pytest.MonkeyPatch,
    input_bytes: bytes,
    expected_detach: bool,
    expected_forwarded: bytes,
) -> None:
    pipe_r, pipe_w = os.pipe()
    try:
        os.write(pipe_w, input_bytes)
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"fileno": staticmethod(lambda: pipe_r)})())
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        a.setblocking(False)
        with a, b:
            assert _handle_stdin_read(b) is expected_detach
            try:
                received = a.recv(_TEST_RECV_BUFSIZE)
            except BlockingIOError:
                received = b""
        if expected_forwarded:
            frames = _decode_frames(bytearray(received))
            assert any(k == _FRAME_DATA and p == expected_forwarded for k, p in frames)
        else:
            assert received == b""
    finally:
        os.close(pipe_r)
        os.close(pipe_w)


def test_handle_stdin_read_returns_true_on_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe_r, pipe_w = os.pipe()
    os.close(pipe_w)
    try:
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"fileno": staticmethod(lambda: pipe_r)})())
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with a, b:
            assert _handle_stdin_read(b) is True
    finally:
        os.close(pipe_r)


def test_handle_socket_read_returns_false_on_recv_oserror() -> None:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()
    with a:
        assert _handle_socket_read(b, bytearray()) is False


def test_run_attach_loop_connects_and_sends_hello(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_attach_loop`` connects to the listen socket and emits a HELLO frame, then yields to ``_io_loop``."""
    listen_sock, listen_path = _create_listen_socket("run-attach-cov")
    assert listen_sock is not None
    assert listen_path is not None
    try:
        monkeypatch.setattr("psoul.cli.attach._io_loop", lambda _sock: None)
        run_attach_loop(listen_path, 12345)
        listen_sock.setblocking(True)
        client_sock, _addr = listen_sock.accept()
        try:
            data = client_sock.recv(_TEST_RECV_BUFSIZE)
        finally:
            client_sock.close()
        frames = _decode_frames(bytearray(data))
        assert any(k == _FRAME_HELLO for k, _ in frames)
    finally:
        _cleanup_listen_socket(listen_sock, listen_path)
