"""Orphan detection and stale session recovery."""

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from psoul.core.session import SessionState

STARTING_GRACE_SECONDS = 30


class ProcessStatus(StrEnum):
    """Tri-state result of a PID liveness check."""

    alive = "alive"
    dead = "dead"
    unknown = "unknown"


def check_pid(pid: int) -> ProcessStatus:
    """Check whether a PID is currently occupied using signal 0.

    Returns ``dead`` when the OS confirms no process with *pid* exists
    (``ProcessLookupError`` / ESRCH).  Returns ``alive`` when the
    signal succeeds or ``PermissionError`` (EPERM) indicates a process
    is present.  Returns ``unknown`` for invalid PIDs (<=0) or any
    other ``OSError``.  Callers must not recover a session unless the
    status is ``dead``.

    Limitation: this checks current PID occupancy, not process
    identity.  A ``dead`` result reliably means the original process
    is gone.  An ``alive`` result may be a false positive if the OS
    has recycled the PID to an unrelated process.  A stronger guard
    (PID + birth time) would eliminate false positives but requires
    platform-specific identity checks not implemented here.

    Args:
        pid (int): Process ID to check.

    Returns:
        ProcessStatus: ``alive``, ``dead``, or ``unknown``.

    """
    if pid <= 0:
        return ProcessStatus.unknown
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return ProcessStatus.dead
    except PermissionError:
        return ProcessStatus.alive
    except OSError:
        return ProcessStatus.unknown
    else:
        return ProcessStatus.alive


def _recover_session(
    conn: sqlite3.Connection,
    session_id: str,
    expected_state: SessionState,
    generation: int,
    now: datetime,
) -> bool:
    """Atomically recover one session to a terminal state.

    Uses a guarded UPDATE (WHERE state = expected_state) so only one
    concurrent caller wins the race.  The result INSERT is idempotent
    via INSERT OR IGNORE on the (session_id, generation) primary key.
    Result writes and state changes share a single implicit transaction.

    If a result row already exists (supervisor wrote it before dying),
    the existing outcome determines the terminal state instead of
    forcing failed.

    Returns True if this call performed the recovery, False if the
    session was already recovered or its state changed.
    """
    now_iso = now.isoformat()
    cursor = conn.execute(
        "UPDATE sessions SET state = ?, supervisor_pid = NULL WHERE session_id = ? AND state = ?"
        " AND NOT EXISTS ("
        "   SELECT 1 FROM results WHERE session_id = ? AND generation = ?"
        " )",
        [
            SessionState.failed.value,
            session_id,
            expected_state.value,
            session_id,
            generation,
        ],
    )
    if cursor.rowcount != 0:
        conn.execute(
            "INSERT OR IGNORE INTO results"
            " (session_id, generation, outcome, exit_code, end_time,"
            "  orphan_detected_at, recovery_attempted_at)"
            " VALUES (?, ?, 'orphan_recovered', NULL, ?, ?, ?)",
            [session_id, generation, now_iso, now_iso, now_iso],
        )
        conn.commit()
        return True

    existing = conn.execute(
        "SELECT outcome FROM results WHERE session_id = ? AND generation = ?",
        [session_id, generation],
    ).fetchone()
    if existing is None:
        return False

    outcome = existing[0]
    final_state = SessionState.exited if outcome == "exited" else SessionState.failed
    current_state = expected_state
    if expected_state == SessionState.running and final_state == SessionState.exited:
        cursor = conn.execute(
            "UPDATE sessions SET state = ? WHERE session_id = ? AND state = ?",
            [SessionState.stopping.value, session_id, current_state.value],
        )
        if cursor.rowcount == 0:
            return False
        current_state = SessionState.stopping
    cursor = conn.execute(
        "UPDATE sessions SET state = ?, supervisor_pid = NULL WHERE session_id = ? AND state = ?",
        [final_state.value, session_id, current_state.value],
    )
    if cursor.rowcount == 0:
        return False
    conn.execute(
        "UPDATE results SET orphan_detected_at = ?, recovery_attempted_at = ? WHERE session_id = ? AND generation = ?",
        [now_iso, now_iso, session_id, generation],
    )
    conn.commit()
    return True


def recover_sessions(conn: sqlite3.Connection) -> list[str]:
    """Detect and recover orphaned and stale sessions.

    Scans the sessions table for running sessions whose supervisor is
    no longer present and starting sessions past the grace period.
    Uses guarded SQL so concurrent callers cannot double-recover.

    Args:
        conn (sqlite3.Connection): Open database connection with the psoul schema.

    Returns:
        list[str]: Session IDs recovered by this call.

    """
    now = datetime.now(UTC)
    recovered: list[str] = []

    rows = conn.execute(
        "SELECT session_id, supervisor_pid, generation FROM sessions WHERE state = ?",
        [SessionState.running.value],
    ).fetchall()
    for session_id, supervisor_pid, generation in rows:
        if supervisor_pid is None:
            continue
        if check_pid(supervisor_pid) != ProcessStatus.dead:
            continue
        if _recover_session(conn, session_id, SessionState.running, generation, now):
            recovered.append(session_id)

    threshold = (now - timedelta(seconds=STARTING_GRACE_SECONDS)).isoformat()
    rows = conn.execute(
        "SELECT session_id, generation FROM sessions WHERE state = ? AND launch_time < ?",
        [SessionState.starting.value, threshold],
    ).fetchall()
    for session_id, generation in rows:
        if _recover_session(conn, session_id, SessionState.starting, generation, now):
            recovered.append(session_id)

    return recovered
