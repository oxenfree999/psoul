"""Drain subprocess stdout/stderr pipes and persist chunks as events."""

import os
import selectors
import subprocess

from psoul.core.events import EVENT_RUNTIME_STDERR, EVENT_RUNTIME_STDOUT, EventStore

READ_CHUNK_SIZE = 8192  # bytes per os.read; matches typical Linux pipe buffer


def drain_output(
    proc: subprocess.Popen[bytes],
    *,
    session_id: str,
    event_store: EventStore,
    generation: int,
) -> None:
    """Capture *proc*'s stdout and stderr into the event log.

    Each chunk read from a pipe becomes one event of type
    ``runtime.stdout`` or ``runtime.stderr`` with payload
    ``{"text": <decoded>}``. Bytes are decoded as UTF-8 with
    ``errors="replace"`` — non-decodable bytes become U+FFFD rather
    than raising. Events are batched per selector wakeup via
    ``append(commit=False)`` and a single ``commit()``.

    Blocks until both pipes reach EOF, which typically happens when
    the child exits and the kernel closes its stdio. Callers should
    ``proc.wait()`` afterward to collect the exit code. Returns
    immediately if *proc* has neither ``stdout`` nor ``stderr`` as a
    pipe.

    Args:
        proc (subprocess.Popen[bytes]): Running subprocess opened with
            ``stdout=PIPE`` and/or ``stderr=PIPE``.
        session_id (str): Session owning *proc*.
        event_store (EventStore): Store that will receive the events.
        generation (int): Session generation at the time of capture.

    """
    with selectors.DefaultSelector() as selector:
        if proc.stdout is not None:
            selector.register(proc.stdout, selectors.EVENT_READ, EVENT_RUNTIME_STDOUT)
        if proc.stderr is not None:
            selector.register(proc.stderr, selectors.EVENT_READ, EVENT_RUNTIME_STDERR)
        while selector.get_map():
            for key, _mask in selector.select():
                chunk = os.read(key.fd, READ_CHUNK_SIZE)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                event_store.append(
                    session_id=session_id,
                    event_type=str(key.data),
                    payload={"text": chunk.decode("utf-8", errors="replace")},
                    generation=generation,
                    commit=False,
                )
            event_store.conn.commit()
