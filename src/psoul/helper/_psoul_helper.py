"""psoul Python helper loaded into recorded subprocesses.

Invoked as ``python -m _psoul_helper <user-argv>`` from the
supervisor's launch path. Opens the inherited transport (Unix
socket fd or Windows named pipe), starts a daemon dispatch
thread that handles requests over length-prefixed JSON, then
restores the user's argv and runpys the user target.
"""

import contextlib
import json
import os
import runpy
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

PSOUL_HELPER_PIPE_ENV = "PSOUL_HELPER_PIPE"

_FRAME_HEADER = struct.Struct(">I")


class _PipeAdapter(Protocol):
    """Structural type for the wrapper's send/recv/close API."""

    def send(self, data: bytes) -> None: ...

    def recv(self, maxsize: int, timeout: float | None = None) -> bytes: ...

    def close(self) -> None: ...


class UnixHelperPipeAdapter:
    """Wrap a Unix socket as a length-prefixed byte stream.

    Provides the same ``send`` / ``recv`` / ``close`` shape as
    the Windows adapter so :class:`HelperTransport` can consume
    either without branching.
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


_winapi: Any = None
if sys.platform == "win32":
    import _winapi  # type: ignore[no-redef]


class WindowsHelperPipeAdapter:
    """Wrap a Windows named-pipe handle as a length-prefixed byte stream.

    Provides the same shape as :class:`UnixHelperPipeAdapter`. Uses
    the overlapped-I/O pattern from
    :class:`multiprocessing.connection.PipeConnection`. Module-level
    ``_winapi`` is the live ``_winapi`` stdlib module on Windows and
    ``None`` elsewhere. Tests replace it with a mock to exercise
    these methods on non-Windows platforms.
    """

    def __init__(self, handle: int) -> None:
        if _winapi is None:
            msg = "WindowsHelperPipeAdapter requires _winapi"
            raise RuntimeError(msg)
        self._handle = handle

    def send(self, data: bytes) -> None:
        """Block until ``data`` is fully sent.

        Raises:
            EOFError: The peer closed the pipe before the write completed.

        """
        try:
            ov, err = _winapi.WriteFile(self._handle, data, overlapped=True)
            if err == _winapi.ERROR_IO_PENDING:
                _winapi.WaitForMultipleObjects(
                    [ov.event],
                    False,
                    _winapi.INFINITE,
                )
            nwritten, _err = ov.GetOverlappedResult(True)
        except OSError as e:
            if getattr(e, "winerror", None) == _winapi.ERROR_BROKEN_PIPE:
                raise EOFError from e
            raise
        if nwritten != len(data):
            msg = f"partial write: {nwritten}/{len(data)}"
            raise OSError(msg)

    def recv(self, maxsize: int, timeout: float | None = None) -> bytes:
        """Read exactly ``maxsize`` bytes, raising on EOF or timeout.

        Args:
            maxsize: Number of bytes to read.
            timeout: Total seconds budget. ``None`` blocks
                indefinitely.

        Returns:
            The ``maxsize`` bytes read from the pipe.

        Raises:
            EOFError: The peer closed the pipe before ``maxsize``
                bytes were read.
            TimeoutError: ``timeout`` elapsed before ``maxsize``
                bytes were read.

        """
        deadline = None if timeout is None else time.monotonic() + timeout
        chunks: list[bytes] = []
        remaining = maxsize
        while remaining > 0:
            if deadline is None:
                timeout_ms = _winapi.INFINITE
            else:
                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    raise TimeoutError
                timeout_ms = int(remaining_time * 1000)
            try:
                ov, err = _winapi.ReadFile(
                    self._handle,
                    remaining,
                    overlapped=True,
                )
                if err == _winapi.ERROR_IO_PENDING:
                    waitres = _winapi.WaitForMultipleObjects(
                        [ov.event],
                        False,
                        timeout_ms,
                    )
                    if waitres == _winapi.WAIT_TIMEOUT:
                        ov.cancel()
                        raise TimeoutError
                ov.GetOverlappedResult(True)
                chunk = bytes(ov.getbuffer())
            except OSError as e:
                if getattr(e, "winerror", None) == _winapi.ERROR_BROKEN_PIPE:
                    raise EOFError from e
                raise
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        """Close the underlying handle. Idempotent."""
        handle = self._handle
        if handle == 0:
            return
        self._handle = 0
        with contextlib.suppress(OSError):
            _winapi.CloseHandle(handle)


_CMD_CAPABILITIES = "capabilities"
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


def _dispatch(request: dict) -> dict:
    """Dispatch one request and return the response envelope.

    Currently implemented handlers are ``capabilities`` and
    ``ping``. Any other command returns a ``not_implemented`` error
    response.
    """
    request_id = request.get("id")
    command = request.get("command", "")
    if command == _CMD_CAPABILITIES:
        return {
            "id": request_id,
            "type": "response",
            "status": "ok",
            "result": {
                "commands": [_CMD_CAPABILITIES, _CMD_PING],
                "python_version": list(sys.version_info[:3]),
                "backends": [],
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
    the pipe) is the normal termination path.
    """
    try:
        while True:
            try:
                request = _read_frame(adapter)
            except EOFError:
                return
            response = _dispatch(request)
            _write_frame(adapter, response)
    finally:
        adapter.close()


def _open_windows_adapter(pipe: str) -> WindowsHelperPipeAdapter:
    """Open a Windows named-pipe adapter.

    Extracted so tests can mock ``_winapi`` independently of the
    platform branch in :func:`_open_adapter`.
    """
    handle = _winapi.CreateFile(
        pipe,
        _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
        0,
        None,
        _winapi.OPEN_EXISTING,
        _winapi.FILE_FLAG_OVERLAPPED,
        None,
    )
    return WindowsHelperPipeAdapter(handle)


def _open_adapter() -> _PipeAdapter | None:
    """Open the adapter from the ``PSOUL_HELPER_PIPE`` env var.

    Returns ``None`` when the env var is unset (the wrapper bails
    out and the user's script runs without a helper).
    """
    pipe = os.environ.get(PSOUL_HELPER_PIPE_ENV)
    if not pipe:
        return None
    if sys.platform == "win32":
        return _open_windows_adapter(pipe)
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
