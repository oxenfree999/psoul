"""Tests for the supervisor-side HelperTransport."""

import json
import socket
import struct
import threading
from collections.abc import Iterator

import pytest

from psoul.core.helper import HelperTransport
from psoul.helper._psoul_helper import UnixHelperPipeAdapter

_AdapterPair = tuple[UnixHelperPipeAdapter, UnixHelperPipeAdapter]
_FRAME_HEADER = struct.Struct(">I")


@pytest.fixture
def adapter_pair() -> Iterator[_AdapterPair]:
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    transport_side = UnixHelperPipeAdapter(sock_a)
    helper_side = UnixHelperPipeAdapter(sock_b)
    yield transport_side, helper_side
    transport_side.close()
    helper_side.close()


def _read_frame(adapter: UnixHelperPipeAdapter) -> dict:
    header = adapter.recv(_FRAME_HEADER.size)
    (length,) = _FRAME_HEADER.unpack(header)
    return json.loads(adapter.recv(length))


def _write_frame(adapter: UnixHelperPipeAdapter, message: dict) -> None:
    payload = json.dumps(message).encode("utf-8")
    adapter.send(_FRAME_HEADER.pack(len(payload)) + payload)


def _echo_response(adapter: UnixHelperPipeAdapter, result: dict | None = None) -> dict:
    request = _read_frame(adapter)
    _write_frame(adapter, {"id": request["id"], "type": "response", "status": "ok", "result": result or {}})
    return request


def _misbehaving_helper(adapter: UnixHelperPipeAdapter, behavior: str) -> None:
    _read_frame(adapter)
    if behavior == "wrong_id":
        _write_frame(adapter, {"id": "wrong", "type": "response", "status": "ok", "result": {}})
    elif behavior == "malformed":
        bad = b"not json"
        adapter.send(_FRAME_HEADER.pack(len(bad)) + bad)
    elif behavior == "close":
        adapter.close()


def test_request_round_trip_with_params_and_id_allocation(adapter_pair: _AdapterPair) -> None:
    transport_side, helper_side = adapter_pair
    transport = HelperTransport(transport_side)
    captured: list[dict] = []

    def helper() -> None:
        captured.extend(_echo_response(helper_side, {"ack": True}) for _ in range(3))

    thread = threading.Thread(target=helper)
    thread.start()
    r1 = transport.request("capabilities", timeout_ms=2000)
    r2 = transport.request("eval", params={"code": "x + 1"}, timeout_ms=2000)
    r3 = transport.request("ping", timeout_ms=2000)
    thread.join(timeout=2.0)

    assert [c["id"] for c in captured] == ["req-0", "req-1", "req-2"]
    assert [r["id"] for r in (r1, r2, r3)] == ["req-0", "req-1", "req-2"]
    assert all(r["status"] == "ok" and r["result"] == {"ack": True} for r in (r1, r2, r3))
    assert [c["command"] for c in captured] == ["capabilities", "eval", "ping"]
    assert captured[0]["type"] == "request"
    assert captured[0]["params"] == {}
    assert captured[1]["params"] == {"code": "x + 1"}
    assert captured[0]["timeout_ms"] == 2000


@pytest.mark.parametrize(
    ("behavior", "expected", "timeout_ms"),
    [
        ("none", TimeoutError, 50),
        ("close", EOFError, 2000),
        ("wrong_id", ValueError, 2000),
        ("malformed", json.JSONDecodeError, 2000),
    ],
)
def test_request_error_paths(
    adapter_pair: _AdapterPair,
    behavior: str,
    expected: type[Exception],
    timeout_ms: int,
) -> None:
    transport_side, helper_side = adapter_pair
    transport = HelperTransport(transport_side)
    thread: threading.Thread | None = None
    if behavior in ("wrong_id", "malformed", "close"):
        thread = threading.Thread(target=_misbehaving_helper, args=(helper_side, behavior))
        thread.start()
    with pytest.raises(expected):
        transport.request("ping", timeout_ms=timeout_ms)
    if thread is not None:
        thread.join(timeout=2.0)


def test_close_is_idempotent(adapter_pair: _AdapterPair) -> None:
    transport_side, _helper_side = adapter_pair
    transport = HelperTransport(transport_side)
    transport.close()
    transport.close()


def test_concurrent_requests_serialize(adapter_pair: _AdapterPair) -> None:
    transport_side, helper_side = adapter_pair
    transport = HelperTransport(transport_side)
    captured_ids: list[str] = []

    def helper() -> None:
        captured_ids.extend(_echo_response(helper_side)["id"] for _ in range(2))

    helper_thread = threading.Thread(target=helper)
    helper_thread.start()
    results: list[dict] = []

    def caller() -> None:
        results.append(transport.request("ping", timeout_ms=2000))

    callers = [threading.Thread(target=caller) for _ in range(2)]
    for t in callers:
        t.start()
    for t in callers:
        t.join(timeout=2.0)
    helper_thread.join(timeout=2.0)

    assert len(results) == 2
    assert sorted(r["id"] for r in results) == ["req-0", "req-1"]
