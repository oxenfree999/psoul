"""Event store: append-only log of session events backed by the events table."""

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime

EVENT_RUNTIME_STDOUT = "runtime.stdout"
EVENT_RUNTIME_STDERR = "runtime.stderr"
EVENT_RUNTIME_STATUS = "runtime.status"
EVENT_SESSION_CONTROLLER_ACQUIRED = "session.controller_acquired"
EVENT_SESSION_CONTROLLER_RELEASED = "session.controller_released"
EVENT_HELPER_TIMEOUT = "helper.timeout"
EVENT_HELPER_CRASHED = "helper.crashed"


class EventStore:
    """Append-only event log backed by the SQLite events table.

    Wraps an open ``sqlite3.Connection`` and provides ``append`` and
    ``list`` operations against the ``events`` table. Sequence numbers
    are strictly increasing per session, allocated by the store at
    append time via a subquery against ``MAX(sequence)``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Wrap an existing database connection."""
        self.conn = conn

    def append(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, object] | None,
        generation: int,
        commit: bool = True,
    ) -> None:
        """Insert an event and (by default) commit.

        The sequence number is allocated atomically by SQLite via a
        subquery against ``MAX(sequence)`` for *session_id*, so the
        first event gets sequence 0 and subsequent events increment
        from the current maximum.

        Args:
            session_id (str): Session this event belongs to.
            event_type (str): Event type, e.g. ``"runtime.stdout"``.
            payload (dict[str, object] | None): Event payload, or ``None``
                to store SQL ``NULL``.
            generation (int): Session generation at the time of the event.
            commit (bool): When ``True`` (default), commit after the
                insert. Pass ``False`` to batch multiple appends in one
                transaction — the caller is then responsible for calling
                ``conn.commit()`` at a boundary.

        """
        timestamp = datetime.now(UTC).isoformat()
        encoded = json.dumps(payload) if payload is not None else None
        self.conn.execute(
            "INSERT INTO events (session_id, sequence, generation, timestamp, event_type, payload)"
            " VALUES (?,"
            "   COALESCE((SELECT MAX(sequence) FROM events WHERE session_id = ?), -1) + 1,"
            "   ?, ?, ?, ?)",
            (session_id, session_id, generation, timestamp, event_type, encoded),
        )
        if commit:
            self.conn.commit()

    def list(
        self,
        session_id: str,
        *,
        event_type: str | None = None,
        after_sequence: int | None = None,
        generation: int | None = None,
    ) -> list[dict[str, object]]:
        """Return events for *session_id* in sequence order.

        Args:
            session_id (str): Session whose events to return.
            event_type (str | None): Filter to this event type, or ``None``
                to return all types.
            after_sequence (int | None): Only return events with sequence
                strictly greater than this value, or ``None`` to return
                from the beginning.
            generation (int | None): Filter to this session generation,
                or ``None`` to return events from all generations.

        Returns:
            list[dict[str, object]]: Events ordered by sequence ascending.
                Each dict has ``sequence``, ``generation``, ``timestamp``,
                ``event_type``, and ``payload`` (decoded JSON or ``None``).

        """
        sql = "SELECT sequence, generation, timestamp, event_type, payload FROM events WHERE session_id = ?"
        params: list[object] = [session_id]
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        if after_sequence is not None:
            sql += " AND sequence > ?"
            params.append(after_sequence)
        if generation is not None:
            sql += " AND generation = ?"
            params.append(generation)
        sql += " ORDER BY sequence ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "sequence": int(row[0]),
                "generation": int(row[1]),
                "timestamp": str(row[2]),
                "event_type": str(row[3]),
                "payload": json.loads(row[4]) if row[4] is not None else None,
            }
            for row in rows
        ]

    def list_recent(
        self,
        session_id: str,
        *,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> Sequence[dict[str, object]]:
        """Return up to *limit* most recent events for *session_id*, ordered ascending by sequence.

        Issues a single ``ORDER BY sequence DESC LIMIT ?`` query and reverses
        the result in Python, so the SQLite scan touches at most *limit*
        rows regardless of how long the session's event history is.

        Args:
            session_id (str): Session whose events to return.
            event_type (str | None): Filter to this event type, or ``None``
                to return all types.
            limit (int): Maximum number of events to return.

        Returns:
            list[dict[str, object]]: Up to *limit* most recent events,
                ordered by sequence ascending. Same dict shape as ``list``.

        """
        sql = "SELECT sequence, generation, timestamp, event_type, payload FROM events WHERE session_id = ?"
        params: list[object] = [session_id]
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY sequence DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "sequence": int(sequence),
                "generation": int(generation),
                "timestamp": str(timestamp),
                "event_type": str(row_event_type),
                "payload": json.loads(payload) if payload is not None else None,
            }
            for sequence, generation, timestamp, row_event_type, payload in reversed(rows)
        ]
