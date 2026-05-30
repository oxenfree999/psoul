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
    _dispatch,
    _dispatch_loop,
    _main,
    _open_adapter,
    _read_frame,
    _run_user_target,
    _safe_repr,
    _session_globals,
    _write_frame,
)

_AdapterPair = tuple[UnixHelperPipeAdapter, UnixHelperPipeAdapter]

_HELPER_MODULE = "psoul.helper._psoul_helper"


@pytest.fixture
def adapter_pair() -> Iterator[_AdapterPair]:
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    adapter_a = UnixHelperPipeAdapter(sock_a)
    adapter_b = UnixHelperPipeAdapter(sock_b)
    yield adapter_a, adapter_b
    adapter_a.close()
    adapter_b.close()


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
    assert response["result"]["commands"] == ["capabilities", "eval", "ping"]
    assert response["result"]["backends"] == []
    assert len(response["result"]["python_version"]) == 3


def test_dispatch_ping_result_is_pong() -> None:
    response = _dispatch({"id": "p", "command": "ping"})
    assert response["result"] == {"pong": True}


@pytest.mark.parametrize(
    ("code", "expected_value", "expected_type"),
    [
        ("1 + 1", "2", "int"),
        ("x = 5", "None", "NoneType"),
        ("x = 5\nx + 3", "8", "int"),
    ],
    ids=["single_expression", "statement_only", "multi_statement"],
)
def test_dispatch_eval_returns_value_and_type(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    expected_value: str,
    expected_type: str,
) -> None:
    ns: dict = {}
    monkeypatch.setattr(f"{_HELPER_MODULE}._session_globals", lambda: ns)
    response = _dispatch(
        {
            "id": "e",
            "command": "eval",
            "params": {"code": code},
        }
    )
    assert response["status"] == "ok"
    assert response["result"] == {"value": expected_value, "type": expected_type}


def test_dispatch_eval_mutates_session_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ns: dict = {}
    monkeypatch.setattr(f"{_HELPER_MODULE}._session_globals", lambda: ns)
    _dispatch(
        {
            "id": "e",
            "command": "eval",
            "params": {"code": "x = 42"},
        }
    )
    assert ns["x"] == 42


@pytest.mark.parametrize(
    ("code", "expected_exc_name"),
    [
        ("undefined_name_xyz", "NameError"),
        ("def foo(:", "SyntaxError"),
        ("raise SystemExit('bye')", "SystemExit"),
    ],
    ids=["name_error", "syntax_error", "system_exit"],
)
def test_dispatch_eval_returns_eval_failed_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    expected_exc_name: str,
) -> None:
    ns: dict = {}
    monkeypatch.setattr(f"{_HELPER_MODULE}._session_globals", lambda: ns)
    response = _dispatch(
        {
            "id": "e",
            "command": "eval",
            "params": {"code": code},
        }
    )
    assert response["status"] == "error"
    assert response["error"]["code"] == "eval_failed"
    assert expected_exc_name in response["error"]["message"]
    assert "Traceback" in response["error"]["traceback"]


@pytest.mark.parametrize(
    "request_in",
    [
        {"id": "e", "command": "eval"},
        {"id": "e", "command": "eval", "params": {}},
        {"id": "e", "command": "eval", "params": None},
        {"id": "e", "command": "eval", "params": {"code": 123}},
        {"id": "e", "command": "eval", "params": 1},
        {"id": "e", "command": "eval", "params": True},
        {"id": "e", "command": "eval", "params": "abcd"},
        {"id": "e", "command": "eval", "params": [1, 2, 3]},
    ],
    ids=[
        "no_params",
        "empty_params",
        "null_params",
        "non_string_code",
        "int_params",
        "bool_params",
        "string_params",
        "list_params",
    ],
)
def test_dispatch_eval_returns_eval_failed_on_malformed_request(
    monkeypatch: pytest.MonkeyPatch,
    request_in: dict,
) -> None:
    ns: dict = {}
    monkeypatch.setattr(f"{_HELPER_MODULE}._session_globals", lambda: ns)
    response = _dispatch(request_in)
    assert response["status"] == "error"
    assert response["error"]["code"] == "eval_failed"


def test_safe_repr_returns_fallback_when_repr_raises() -> None:
    class Broken:
        def __repr__(self) -> str:
            raise RuntimeError("nope")

    result = _safe_repr(Broken())
    assert "unrepresentable" in result
    assert "Broken" in result
    assert "RuntimeError" in result


def test_session_globals_returns_main_module_dict() -> None:
    assert _session_globals() is sys.modules["__main__"].__dict__


@pytest.mark.parametrize(
    "request_in",
    [
        {"id": "r1", "command": "nonexistent_command"},
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


@pytest.mark.parametrize(
    ("exc_type", "exc_message"),
    [
        (RuntimeError, "boom"),
        (SystemExit, "bye"),
    ],
    ids=["exception_subclass", "base_exception_only"],
)
def test_dispatch_loop_survives_handler_exception(
    adapter_pair: _AdapterPair,
    monkeypatch: pytest.MonkeyPatch,
    exc_type: type[BaseException],
    exc_message: str,
) -> None:
    helper_adapter, test_adapter = adapter_pair
    original_dispatch = _dispatch
    raised = False

    def flaky_dispatch(request: dict) -> dict:
        nonlocal raised
        if not raised:
            raised = True
            raise exc_type(exc_message)
        return original_dispatch(request)

    monkeypatch.setattr(f"{_HELPER_MODULE}._dispatch", flaky_dispatch)
    thread = threading.Thread(target=_dispatch_loop, args=(helper_adapter,))
    thread.start()
    try:
        _write_frame(test_adapter, {"id": "req-err", "command": "ping"})
        error_response = _read_frame(test_adapter)
        assert error_response["id"] == "req-err"
        assert error_response["status"] == "error"
        assert error_response["error"]["code"] == "runtime_error"
        assert exc_type.__name__ in error_response["error"]["message"]
        assert exc_message in error_response["error"]["message"]
        assert "Traceback" in error_response["error"]["traceback"]

        _write_frame(test_adapter, {"id": "req-ok", "command": "ping"})
        ok_response = _read_frame(test_adapter)
        assert ok_response["id"] == "req-ok"
        assert ok_response["result"] == {"pong": True}
    finally:
        test_adapter.close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def test_dispatch_loop_survives_submitted_system_exit(
    adapter_pair: _AdapterPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_adapter, test_adapter = adapter_pair
    ns: dict = {}
    monkeypatch.setattr(f"{_HELPER_MODULE}._session_globals", lambda: ns)
    thread = threading.Thread(target=_dispatch_loop, args=(helper_adapter,))
    thread.start()
    try:
        _write_frame(
            test_adapter,
            {
                "id": "exit",
                "command": "eval",
                "params": {"code": "raise SystemExit('bye')"},
            },
        )
        eval_response = _read_frame(test_adapter)
        assert eval_response["id"] == "exit"
        assert eval_response["status"] == "error"
        assert eval_response["error"]["code"] == "eval_failed"
        assert "SystemExit" in eval_response["error"]["message"]

        _write_frame(test_adapter, {"id": "after-exit", "command": "ping"})
        ping_response = _read_frame(test_adapter)
        assert ping_response["id"] == "after-exit"
        assert ping_response["result"] == {"pong": True}
    finally:
        test_adapter.close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def test_open_adapter_returns_none_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PSOUL_HELPER_PIPE_ENV, raising=False)
    assert _open_adapter() is None


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
