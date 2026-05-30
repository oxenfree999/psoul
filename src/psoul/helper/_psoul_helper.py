"""psoul Python helper loaded into recorded subprocesses.

Invoked as ``python -m _psoul_helper <user-argv>`` from the
supervisor's launch path. Opens the inherited Unix socket fd,
starts a daemon dispatch thread that handles requests over
length-prefixed JSON, then restores the user's argv and runpys
the user target.
"""

import ast
import contextlib
import json
import os
import runpy
import socket
import struct
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Protocol

PSOUL_HELPER_PIPE_ENV = "PSOUL_HELPER_PIPE"

_FRAME_HEADER = struct.Struct(">I")


class _PipeAdapter(Protocol):
    """Structural type for the wrapper's send/recv/close API."""

    def send(self, data: bytes) -> None: ...

    def recv(self, maxsize: int, timeout: float | None = None) -> bytes: ...

    def close(self) -> None: ...


class UnixHelperPipeAdapter:
    """Wrap a Unix socket as a length-prefixed byte stream.

    Implements the ``send`` / ``recv`` / ``close`` shape that
    :class:`HelperTransport` consumes.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def send(self, data: bytes) -> None:
        """Block until ``data`` is fully sent."""
        self._sock.sendall(data)

    def recv(self, maxsize: int, timeout: float | None = None) -> bytes:
        """Read exactly ``maxsize`` bytes, raising on EOF or timeout.

        Args:
            maxsize: Number of bytes to read. The call returns when
                exactly that many bytes are read.
            timeout: Total seconds budget across the read loop. ``None``
                blocks indefinitely.

        Returns:
            The ``maxsize`` bytes read from the socket.

        Raises:
            EOFError: The peer closed the socket before ``maxsize``
                bytes were read.
            TimeoutError: ``timeout`` elapsed before ``maxsize`` bytes
                were read.

        """
        deadline = None if timeout is None else time.monotonic() + timeout
        chunks: list[bytes] = []
        remaining = maxsize
        while remaining > 0:
            if deadline is not None:
                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    raise TimeoutError
                self._sock.settimeout(remaining_time)
            else:
                self._sock.settimeout(None)
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        """Close the underlying socket. Idempotent."""
        with contextlib.suppress(OSError):
            self._sock.close()


_CMD_CAPABILITIES = "capabilities"
_CMD_EVAL = "eval"
_CMD_PING = "ping"


def _read_frame(adapter: _PipeAdapter) -> dict:
    """Read one length-prefixed JSON frame from ``adapter``.

    Raises ``EOFError`` if the adapter sees end of stream before a
    full header or payload arrives.
    """
    header = adapter.recv(_FRAME_HEADER.size)
    (length,) = _FRAME_HEADER.unpack(header)
    payload = adapter.recv(length)
    return json.loads(payload)


def _write_frame(adapter: _PipeAdapter, message: dict) -> None:
    """Write one length-prefixed JSON frame to ``adapter``."""
    payload = json.dumps(message).encode("utf-8")
    adapter.send(_FRAME_HEADER.pack(len(payload)) + payload)


def _session_globals() -> dict:
    """Return the read-path namespace shared by eval and vars-global."""
    return sys.modules["__main__"].__dict__


def _safe_repr(value: object) -> str:
    """Return ``repr(value)`` or a fallback when ``__repr__`` raises."""
    try:
        return repr(value)
    except BaseException as exc:  # noqa: BLE001 (one bad repr must not blow up a multi-value result)
        return f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"


def _dispatch(request: dict) -> dict:
    """Dispatch one request and return the response envelope.

    Recognized commands are ``capabilities``, ``eval``, and ``ping``.
    Any other command returns a ``not_implemented`` error response.
    """
    request_id = request.get("id")
    command = request.get("command", "")
    if command == _CMD_CAPABILITIES:
        return {
            "id": request_id,
            "type": "response",
            "status": "ok",
            "result": {
                "commands": [_CMD_CAPABILITIES, _CMD_EVAL, _CMD_PING],
                "python_version": list(sys.version_info[:3]),
                "backends": [],
            },
        }
    if command == _CMD_EVAL:
        params = request.get("params")
        if not isinstance(params, dict) or "code" not in params:
            return {
                "id": request_id,
                "type": "response",
                "status": "error",
                "error": {
                    "code": "eval_failed",
                    "message": "missing required param: code",
                    "traceback": "",
                },
            }
        try:
            code = params["code"]
            ns = _session_globals()
            tree = ast.parse(code, mode="exec")
            result = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                leading = ast.Module(body=tree.body[:-1], type_ignores=[])
                trailing = ast.Expression(body=tree.body[-1].value)
                exec(compile(leading, "<eval>", "exec"), ns)  # noqa: S102
                result = eval(compile(trailing, "<eval>", "eval"), ns)  # noqa: S307
            else:
                exec(compile(tree, "<eval>", "exec"), ns)  # noqa: S102
            return {
                "id": request_id,
                "type": "response",
                "status": "ok",
                "result": {
                    "value": _safe_repr(result),
                    "type": type(result).__name__,
                },
            }
        except BaseException as exc:  # noqa: BLE001 (eval handler must answer with eval_failed for any user-code failure)
            return {
                "id": request_id,
                "type": "response",
                "status": "error",
                "error": {
                    "code": "eval_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": "".join(traceback.format_exception(exc)),
                },
            }
    if command == _CMD_PING:
        return {
            "id": request_id,
            "type": "response",
            "status": "ok",
            "result": {"pong": True},
        }
    return {
        "id": request_id,
        "type": "response",
        "status": "error",
        "error": {
            "code": "not_implemented",
            "message": f"{command!r} is not implemented",
        },
    }


def _dispatch_loop(adapter: _PipeAdapter) -> None:
    """Read requests, dispatch, write responses, until the peer closes.

    Closes ``adapter`` on exit. Helper-side EOF (supervisor closed
    the pipe) is the normal termination path. A handler that raises is
    caught here so the loop answers with an error response and keeps
    serving, rather than letting the dispatch thread die silently.
    """
    try:
        while True:
            try:
                request = _read_frame(adapter)
            except EOFError:
                return
            try:
                response = _dispatch(request)
            except BaseException as exc:  # noqa: BLE001 (a handler must never kill the dispatch loop)
                response = {
                    "id": request.get("id"),
                    "type": "response",
                    "status": "error",
                    "error": {
                        "code": "runtime_error",
                        "message": f"{type(exc).__name__}: {exc}",
                        "traceback": "".join(traceback.format_exception(exc)),
                    },
                }
            _write_frame(adapter, response)
    finally:
        adapter.close()


def _open_adapter() -> UnixHelperPipeAdapter | None:
    """Open the adapter from the ``PSOUL_HELPER_PIPE`` env var.

    Returns ``None`` when the env var is unset (the wrapper bails
    out and the user's script runs without a helper).
    """
    pipe = os.environ.get(PSOUL_HELPER_PIPE_ENV)
    if not pipe:
        return None
    sock = socket.socket(fileno=int(pipe))
    return UnixHelperPipeAdapter(sock)


def _run_user_target(args: list[str]) -> None:
    """Restore user argv / sys.path[0] and runpy the target.

    ``args`` is the wrapper's ``sys.argv[1:]``. Either a script path
    followed by user args, or ``"-m"`` followed by a module name and
    user args. With no args the wrapper exits cleanly.
    """
    if not args:
        return
    if args[0] == "-m":
        module_args = args[1:]
        if not module_args:
            return
        module = module_args[0]
        sys.argv = [module, *module_args[1:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return
    target = args[0]
    sys.argv = list(args)
    sys.path[0] = str(Path(target).resolve().parent)
    runpy.run_path(target, run_name="__main__")


def _main(argv: list[str]) -> None:
    """Run the wrapper: open the adapter, start dispatch, run the user target.

    ``argv`` is the full ``sys.argv`` for the wrapper invocation
    (``["_psoul_helper", *user_argv]``).
    """
    adapter = _open_adapter()
    if adapter is not None:
        threading.Thread(
            target=_dispatch_loop,
            args=(adapter,),
            daemon=True,
        ).start()
    _run_user_target(argv[1:])


if __name__ == "__main__":
    _main(sys.argv)
