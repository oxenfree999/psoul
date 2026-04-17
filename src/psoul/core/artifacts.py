"""Artifact store: read-only access to session artifact metadata."""

import sqlite3


class ArtifactStore:
    """Read-only access to the ``artifacts`` table.

    Wraps an open ``sqlite3.Connection`` and exposes a ``list`` operation
    that returns the artifacts registered for a single session, ordered
    by registration time.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Wrap an existing database connection."""
        self.conn = conn

    def list(self, session_id: str) -> list[dict[str, object]]:
        """Return artifacts for *session_id* ordered by registration time.

        Rows are ordered by ``registered_at`` ascending, with
        ``artifact_id`` ascending as a stable tiebreaker so equal
        timestamps resolve in insertion order.

        Args:
            session_id (str): Session whose artifacts to return.

        Returns:
            list[dict[str, object]]: Artifacts oldest first. Each dict has
                ``name``, ``path``, ``mime_type``, ``size_bytes``,
                ``registered_at``, ``source``, and ``retention_class``.

        """
        rows = self.conn.execute(
            "SELECT name, path, mime_type, size_bytes, registered_at, source, retention_class"
            " FROM artifacts WHERE session_id = ?"
            " ORDER BY registered_at ASC, artifact_id ASC",
            [session_id],
        ).fetchall()
        return [
            {
                "name": str(row[0]),
                "path": str(row[1]),
                "mime_type": str(row[2]) if row[2] is not None else None,
                "size_bytes": int(row[3]) if row[3] is not None else None,
                "registered_at": str(row[4]),
                "source": str(row[5]),
                "retention_class": str(row[6]),
            }
            for row in rows
        ]
