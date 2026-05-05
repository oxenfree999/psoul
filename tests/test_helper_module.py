"""Tests for the in-process helper module."""

import contextlib
import os
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from psoul.helper._psoul_helper import (
    PSOUL_HELPER_PIPE_ENV,
    UnixHelperPipeAdapter,
    WindowsHelperPipeAdapter,
    _dispatch,
    _dispatch_loop,
    _main,
    _open_adapter,
    _open_windows_adapter,
    _read_frame,
    _run_user_target,
    _write_frame,
)

_AdapterPair = tuple[UnixHelperPipeAdapter, UnixHelperPipeAdapter]

_HELPER_MODULE = "psoul.helper._psoul_helper"


class _BrokenPipeOSError(OSError):
    """OSError carrying ``winerror = ERROR_BROKEN_PIPE`` (109).

    The Windows adapter calls ``getattr(e, "winerror", None)`` to map
    Win32 broken-pipe errors to ``EOFError``. Tests use this subclass
    to exercise that mapping on non-Windows platforms.
    """

    winerror = 109


@pytest.fixture
def adapter_pair() -> Iterator[_AdapterPair]:
    if sys.platform == "win32":
        pytest.skip("helper transport is Unix-only (socket.AF_UNIX unavailable)")
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    adapter_a = UnixHelperPipeAdapter(sock_a)
    adapter_b = UnixHelperPipeAdapter(sock_b)
    yield adapter_a, adapter_b
    adapter_a.close()
    adapter_b.close()


@pytest.fixture
def fake_winapi(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a mocked ``_winapi`` so Windows code paths run on any platform."""
    fake = MagicMock()
    fake.ERROR_IO_PENDING = 997
    fake.ERROR_BROKEN_PIPE = 109
    fake.WAIT_OBJECT_0 = 0
    fake.WAIT_TIMEOUT = 0x102
    fake.INFINITE = 0xFFFFFFFF
    fake.GENERIC_READ = 0x80000000
    fake.GENERIC_WRITE = 0x40000000
    fake.OPEN_EXISTING = 3
    fake.FILE_FLAG_OVERLAPPED = 0x40000000
    monkeypatch.setattr(f"{_HELPER_MODULE}._winapi", fake)
    return fake


@pytest.mark.parametrize(
    ("command", "expected_status"),
    [
        ("capabilities", "ok"),
        ("ping", "ok"),
        ("eval", "error"),
        ("debug.step", "error"),
        ("unknown_xyz", "error"),
    ],
)
def test_dispatch_status_and_id_echo(command: str, expected_status: str) -> None:
    response = _dispatch({"id": "echo", "command": command})
    assert response["id"] == "echo"
    assert response["type"] == "response"
    assert response["status"] == expected_status


def test_dispatch_capabilities_result_shape() -> None:
    response = _dispatch({"id": "c", "command": "capabilities"})
    assert response["result"]["commands"] == ["capabilities", "ping"]
    assert response["result"]["backends"] == []
    assert len(response["result"]["python_version"]) == 3


def test_dispatch_ping_result_is_pong() -> None:
    response = _dispatch({"id": "p", "command": "ping"})
    assert response["result"] == {"pong": True}


@pytest.mark.parametrize(
    "request_in",
    [
        {"id": "r1", "command": "eval"},
        {"id": "r2"},
    ],
    ids=["unknown_command", "missing_command"],
)
def test_dispatch_error_uses_not_implemented_code(request_in: dict) -> None:
    response = _dispatch(request_in)
    assert response["status"] == "error"
    assert response["error"]["code"] == "not_implemented"


def test_frame_round_trip(adapter_pair: _AdapterPair) -> None:
    helper_adapter, test_adapter = adapter_pair
    _write_frame(test_adapter, {"hello": "world"})
    received = _read_frame(helper_adapter)
    assert received == {"hello": "world"}


def test_dispatch_loop_responds_to_ping(adapter_pair: _AdapterPair) -> None:
    helper_adapter, test_adapter = adapter_pair
    thread = threading.Thread(target=_dispatch_loop, args=(helper_adapter,))
    thread.start()
    try:
        _write_frame(test_adapter, {"id": "req-1", "command": "ping"})
        response = _read_frame(test_adapter)
        assert response["id"] == "req-1"
        assert response["result"] == {"pong": True}
    finally:
        test_adapter.close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def test_dispatch_loop_exits_on_eof(adapter_pair: _AdapterPair) -> None:
    helper_adapter, test_adapter = adapter_pair
    thread = threading.Thread(target=_dispatch_loop, args=(helper_adapter,))
    thread.start()
    test_adapter.close()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_open_adapter_returns_none_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PSOUL_HELPER_PIPE_ENV, raising=False)
    assert _open_adapter() is None


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket path")
def test_open_adapter_returns_unix_adapter_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with sock_a, sock_b:
        adapter_fd = os.dup(sock_a.fileno())
        monkeypatch.setenv(PSOUL_HELPER_PIPE_ENV, str(adapter_fd))
        adapter = _open_adapter()
        assert adapter is not None
        with contextlib.closing(adapter):
            assert isinstance(adapter, UnixHelperPipeAdapter)


def test_open_adapter_dispatches_to_windows_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PSOUL_HELPER_PIPE_ENV, r"\\.\pipe\fake")
    monkeypatch.setattr(sys, "platform", "win32")
    sentinel = MagicMock(spec=WindowsHelperPipeAdapter)
    monkeypatch.setattr(f"{_HELPER_MODULE}._open_windows_adapter", lambda pipe: sentinel)
    assert _open_adapter() is sentinel


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket path")
def test_unix_adapter_close_is_idempotent() -> None:
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with sock_b:
        adapter = UnixHelperPipeAdapter(sock_a)
        adapter.close()
        adapter.close()


def test_unix_adapter_recv_raises_on_timeout(adapter_pair: _AdapterPair) -> None:
    helper_adapter, _test_adapter = adapter_pair
    with pytest.raises(TimeoutError):
        helper_adapter.recv(4, timeout=0.05)


def test_windows_adapter_init_raises_without_winapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_HELPER_MODULE}._winapi", None)
    with pytest.raises(RuntimeError, match="_winapi"):
        WindowsHelperPipeAdapter(handle=42)


def test_windows_adapter_send_writes_via_winapi(fake_winapi: MagicMock) -> None:
    payload = b"hello"
    overlapped = MagicMock()
    overlapped.GetOverlappedResult.return_value = (len(payload), 0)
    fake_winapi.WriteFile.return_value = (overlapped, 0)
    adapter = WindowsHelperPipeAdapter(handle=99)
    adapter.send(payload)
    fake_winapi.WriteFile.assert_called_once_with(99, payload, overlapped=True)


def test_windows_adapter_send_raises_eof_on_broken_pipe(
    fake_winapi: MagicMock,
) -> None:
    fake_winapi.WriteFile.side_effect = _BrokenPipeOSError()
    adapter = WindowsHelperPipeAdapter(handle=99)
    with pytest.raises(EOFError):
        adapter.send(b"data")


def test_windows_adapter_recv_returns_bytes(fake_winapi: MagicMock) -> None:
    payload = b"abcd"
    overlapped = MagicMock()
    overlapped.GetOverlappedResult.return_value = (len(payload), 0)
    overlapped.getbuffer.return_value = payload
    fake_winapi.ReadFile.return_value = (overlapped, 0)
    adapter = WindowsHelperPipeAdapter(handle=99)
    assert adapter.recv(4) == payload


def test_windows_adapter_recv_raises_on_timeout(fake_winapi: MagicMock) -> None:
    overlapped = MagicMock()
    fake_winapi.ReadFile.return_value = (overlapped, fake_winapi.ERROR_IO_PENDING)
    fake_winapi.WaitForMultipleObjects.return_value = fake_winapi.WAIT_TIMEOUT
    adapter = WindowsHelperPipeAdapter(handle=99)
    with pytest.raises(TimeoutError):
        adapter.recv(4, timeout=0.01)
    overlapped.cancel.assert_called_once()


def test_windows_adapter_recv_raises_eof_on_broken_pipe(
    fake_winapi: MagicMock,
) -> None:
    fake_winapi.ReadFile.side_effect = _BrokenPipeOSError()
    adapter = WindowsHelperPipeAdapter(handle=99)
    with pytest.raises(EOFError):
        adapter.recv(4)


def test_windows_adapter_close_calls_close_handle(fake_winapi: MagicMock) -> None:
    adapter = WindowsHelperPipeAdapter(handle=99)
    adapter.close()
    fake_winapi.CloseHandle.assert_called_once_with(99)
    adapter.close()
    assert fake_winapi.CloseHandle.call_count == 1


def test_open_windows_adapter_calls_create_file(fake_winapi: MagicMock) -> None:
    fake_winapi.CreateFile.return_value = 99
    adapter = _open_windows_adapter(r"\\.\pipe\foo")
    assert isinstance(adapter, WindowsHelperPipeAdapter)
    fake_winapi.CreateFile.assert_called_once()


@patch(f"{_HELPER_MODULE}.runpy.run_module")
@patch(f"{_HELPER_MODULE}.runpy.run_path")
@pytest.mark.parametrize("args", [[], ["-m"]], ids=["empty", "dash_m_only"])
def test_run_user_target_returns_for_invalid_invocation(
    run_path: MagicMock,
    run_module: MagicMock,
    args: list[str],
) -> None:
    _run_user_target(args)
    assert run_path.call_count == 0
    assert run_module.call_count == 0


@patch(f"{_HELPER_MODULE}.runpy.run_path")
def test_run_user_target_script_mode(
    run_path: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["original"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    script = tmp_path / "user_script.py"
    script.write_text("# stub\n")
    _run_user_target([str(script), "arg1", "arg2"])
    run_path.assert_called_once_with(str(script), run_name="__main__")
    assert sys.argv == [str(script), "arg1", "arg2"]
    assert sys.path[0] == str(script.parent.resolve())


@patch(f"{_HELPER_MODULE}.runpy.run_module")
def test_run_user_target_module_mode(
    run_module: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["original"])
    _run_user_target(["-m", "json.tool", "--help"])
    run_module.assert_called_once_with(
        "json.tool",
        run_name="__main__",
        alter_sys=True,
    )
    assert sys.argv == ["json.tool", "--help"]


def test_main_starts_dispatch_thread_when_adapter_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_adapter = MagicMock()
    monkeypatch.setattr(f"{_HELPER_MODULE}._open_adapter", lambda: fake_adapter)
    started = threading.Event()

    def fake_dispatch_loop(adapter: object) -> None:
        started.set()

    monkeypatch.setattr(f"{_HELPER_MODULE}._dispatch_loop", fake_dispatch_loop)
    monkeypatch.setattr(f"{_HELPER_MODULE}._run_user_target", lambda args: None)
    _main(["_psoul_helper"])
    assert started.wait(timeout=2.0)


def test_main_skips_dispatch_when_adapter_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_HELPER_MODULE}._open_adapter", lambda: None)
    dispatch_called = MagicMock()
    monkeypatch.setattr(f"{_HELPER_MODULE}._dispatch_loop", dispatch_called)
    run_target = MagicMock()
    monkeypatch.setattr(f"{_HELPER_MODULE}._run_user_target", run_target)
    _main(["_psoul_helper", "user_script.py", "arg"])
    assert dispatch_called.call_count == 0
    run_target.assert_called_once_with(["user_script.py", "arg"])
