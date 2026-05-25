"""Supervisor-side transport for the in-process Python helper.

Wraps a Unix socket pipe adapter and exposes a ``request`` method that sends a length-prefixed JSON envelope
and returns the matching response. Single-flight model: a ``threading.Lock`` serializes concurrent callers so
requests never interleave on the wire.
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


class _EventWriter(Protocol):
    """Structural type for the event-write callback used by :meth:`HelperLifecycle.emit_for_eof`."""

    def __call__(self, event_type: str, payload: dict) -> None: ...


class HelperLifecycle:
    """Owns the supervisor-side helper transport plus crash-vs-exit decision logic.

    Wraps a :class:`HelperTransport` and provides the readiness-exchange convenience method
    plus the EOF-disambiguation logic both spawn paths use to emit ``helper.crashed`` /
    ``runtime.status(helper_lost=true)`` events on the same triggers.
    """

    def __init__(self, transport: HelperTransport) -> None:
        """Wrap a HelperTransport."""
        self._transport = transport

    def request_capabilities(self, timeout_ms: int) -> dict | None:
        """Send a ``capabilities`` request and return the result dict, or ``None`` on failure.

        Used at supervisor startup as the readiness signal. Returns ``None`` for any helper-side failure
        (timeout, EOF, malformed response) so the integrator can fall back to basic mode without catching
        multiple exception types.
        """
        try:
            response = self._transport.request("capabilities", timeout_ms=timeout_ms)
        except (TimeoutError, EOFError, ValueError, json.JSONDecodeError, OSError):
            return None
        result = response.get("result")
        return result if isinstance(result, dict) else None

    def emit_for_eof(self, *, returncode: int | None, event_writer: _EventWriter) -> None:
        """Emit lifecycle events for the child status recovered after helper-pipe EOF.

        ``returncode`` is the child's exit status from a bounded wait after EOF, or ``None`` when
        the child outlived that wait. ``event_writer`` takes ``(event_type, payload_dict)``.

        - ``returncode == 0``: a clean exit closed the pipe. Returns silently on the crash channel.
        - ``returncode`` non-zero (non-zero exit, or negative ``-N`` signal death): the child died
          abnormally. Emits ``helper.crashed`` carrying ``exit_code`` plus ``runtime.status`` with
          ``helper_lost: true``.
        - ``returncode is None``: the child outlived the wait, so the helper closed its end mid-run
          while user code continues. Emits ``helper.crashed`` with an empty payload plus the same
          ``runtime.status``.
        """
        if returncode == 0:
            return
        payload: dict[str, object] = {} if returncode is None else {"exit_code": returncode}
        event_writer("helper.crashed", payload)
        event_writer("runtime.status", {"helper_lost": True})

    def close(self) -> None:
        """Close the underlying transport. Idempotent."""
        self._transport.close()
