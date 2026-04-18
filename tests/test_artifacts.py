"""Tests for ArtifactStore: list ordering, filtering, and schema constraints."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from psoul.core.artifacts import ArtifactStore
from psoul.core.db import open_db
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION


def _make_session(session_id: str = "test-session") -> Session:
    """Build a minimal Session so artifacts can reference it by ID."""
    return Session(
        session_id=session_id,
        state=SessionState.exited,
        launch_mode=LaunchMode.headless,
        launch_time=datetime.now(UTC),
        psoul_version=VERSION,
    )


@pytest.fixture
def artifact_store(tmp_path: Path) -> Iterator[ArtifactStore]:
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(_make_session())
        yield ArtifactStore(conn)


def _insert_artifact(
    conn: sqlite3.Connection,
    session_id: str,
    name: str,
    *,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    registered_at: str = "2026-01-01T00:00:00",
    source: str = "user",
    retention_class: str = "session",
) -> None:
    """Insert one artifact row directly, bypassing any producer."""
    path = f"artifacts/{session_id}/{name}"
    conn.execute(
        "INSERT INTO artifacts"
        " (session_id, name, path, mime_type, size_bytes, registered_at, source, retention_class)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, name, path, mime_type, size_bytes, registered_at, source, retention_class),
    )
    conn.commit()


@pytest.mark.parametrize(
    ("seed", "expected_names"),
    [
        ([], []),
        (
            [("mid", "2026-01-01T00:00:02"), ("first", "2026-01-01T00:00:01"), ("last", "2026-01-01T00:00:03")],
            ["first", "mid", "last"],
        ),
        (
            [("alpha", "2026-01-01T00:00:00"), ("bravo", "2026-01-01T00:00:00"), ("charlie", "2026-01-01T00:00:00")],
            ["alpha", "bravo", "charlie"],
        ),
    ],
    ids=["empty", "timestamp-ascending", "tiebreak-by-insertion"],
)
def test_list_ordering(
    artifact_store: ArtifactStore,
    seed: list[tuple[str, str]],
    expected_names: list[str],
) -> None:
    for name, registered_at in seed:
        _insert_artifact(artifact_store.conn, "test-session", name, registered_at=registered_at)
    assert [r["name"] for r in artifact_store.list("test-session")] == expected_names


def test_list_filters_strictly_by_session_id(artifact_store: ArtifactStore) -> None:
    SessionStore(artifact_store.conn).create(_make_session("other-session"))
    _insert_artifact(artifact_store.conn, "test-session", "mine")
    _insert_artifact(artifact_store.conn, "other-session", "theirs")
    assert [r["name"] for r in artifact_store.list("test-session")] == ["mine"]
    assert [r["name"] for r in artifact_store.list("other-session")] == ["theirs"]


def test_list_round_trips_rows_including_nulls(artifact_store: ArtifactStore) -> None:
    _insert_artifact(
        artifact_store.conn,
        "test-session",
        "sized",
        mime_type="image/png",
        size_bytes=2048,
        registered_at="2026-01-01T00:00:01",
    )
    _insert_artifact(artifact_store.conn, "test-session", "unsized", registered_at="2026-01-01T00:00:02")
    sized, unsized = artifact_store.list("test-session")
    assert sized == {
        "name": "sized",
        "path": "artifacts/test-session/sized",
        "mime_type": "image/png",
        "size_bytes": 2048,
        "registered_at": "2026-01-01T00:00:01",
        "source": "user",
        "retention_class": "session",
    }
    assert unsized == {
        "name": "unsized",
        "path": "artifacts/test-session/unsized",
        "mime_type": None,
        "size_bytes": None,
        "registered_at": "2026-01-01T00:00:02",
        "source": "user",
        "retention_class": "session",
    }


def test_unique_session_name_constraint_enforced(artifact_store: ArtifactStore) -> None:
    _insert_artifact(artifact_store.conn, "test-session", "plot.png", registered_at="2026-01-01T00:00:00")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_artifact(artifact_store.conn, "test-session", "plot.png", registered_at="2026-01-01T00:00:01")
