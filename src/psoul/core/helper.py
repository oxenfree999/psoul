"""Supervisor-side transport for the in-process Python helper.

Wraps a pipe adapter (Unix socket or Windows named-pipe handle) and exposes a ``request`` method that sends a
length-prefixed JSON envelope and returns the matching response. Single-flight model: a ``threading.Lock``
serializes concurrent callers so requests never interleave on the wire.
"""

import json
import struct
import threading
import time
from typing import Protocol

_FRAME_HEADER = struct.Struct(">I")


class _PipeAdapter(Protocol):
    """Structural type for the adapter's send/recv/close API."""

    def send(self, data: bytes) -> None: ...

    def recv(self, maxsize: int, timeout: float | None = None) -> bytes: ...

    def close(self) -> None: ...


class HelperTransport:
    """Supervisor-side request/response transport.

    Each ``request`` call allocates a fresh integer-suffixed id, sends a length-prefixed JSON envelope, then
    reads the next response frame and returns it. Single-flight model: only one request is in flight at a
    time, so any response that does not echo the request id is treated as malformed and raises ``ValueError``.
    """

    def __init__(self, adapter: _PipeAdapter) -> None:
        """Wrap a pipe adapter."""
        self._adapter = adapter
        self._lock = threading.Lock()
        self._next_id = 0

    def request(
        self,
        command: str,
        params: dict | None = None,
        timeout_ms: int = 5000,
    ) -> dict:
        """Send a request and return the matching response envelope.

        Args:
            command: Command name from the helper protocol.
            params: Optional command-specific parameters.
            timeout_ms: Maximum total milliseconds for the round-trip.

        Returns:
            The response envelope.

        Raises:
            EOFError: The adapter saw end of stream before the response arrived.
            TimeoutError: ``timeout_ms`` elapsed before the response arrived.
            ValueError: A response arrived but did not echo the request id.
            json.JSONDecodeError: A response frame arrived but its payload was not valid JSON.

        """
        with self._lock:
            request_id = f"req-{self._next_id}"
            self._next_id += 1
            envelope = {
                "id": request_id,
                "type": "request",
                "command": command,
                "params": params or {},
                "timeout_ms": timeout_ms,
            }
            deadline = time.monotonic() + (timeout_ms / 1000.0)
            self._write_frame(envelope)
            response = self._read_frame(deadline=deadline)
            if response.get("id") != request_id:
                msg = f"unexpected response id: {response.get('id')!r} (expected {request_id!r})"
                raise ValueError(msg)
            return response

    def close(self) -> None:
        """Close the underlying adapter. Idempotent."""
        self._adapter.close()

    def _write_frame(self, message: dict) -> None:
        payload = json.dumps(message).encode("utf-8")
        self._adapter.send(_FRAME_HEADER.pack(len(payload)) + payload)

    def _read_frame(self, deadline: float) -> dict:
        header_timeout = max(0.0, deadline - time.monotonic())
        header = self._adapter.recv(_FRAME_HEADER.size, timeout=header_timeout)
        (length,) = _FRAME_HEADER.unpack(header)
        payload_timeout = max(0.0, deadline - time.monotonic())
        payload = self._adapter.recv(length, timeout=payload_timeout)
        return json.loads(payload)
