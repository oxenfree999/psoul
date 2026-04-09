"""SQLite storage layer for session data, events, history, and artifacts."""

import collections.abc
import sqlite3
from pathlib import Path

from psoul.config import default_state_dir

SCHEMA_VERSION = 1
DB_NAME = "psoul.db"

_MIGRATIONS: dict[int, collections.abc.Callable[[sqlite3.Connection], None]] = {}


def resolve_state_dir(config_state_dir: Path | None = None) -> Path:
    """Return the state directory, creating it if needed.

    Uses the config override when set, falling back to the platform default.

    Args:
        config_state_dir (Path | None): Explicit directory from config, or
            ``None`` to use the platform default.

    Returns:
        Path: Resolved state directory (guaranteed to exist).

    """
    state_dir = config_state_dir if config_state_dir is not None else default_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Set journal mode, foreign keys, and synchronous mode.

    WAL persists across reopens, so the journal-mode pragma is skipped
    when the database is already in WAL — re-running it can contend on
    a fresh database. Busy timeout is set in ``open_db`` via ``sqlite3.connect``.
    """
    current = conn.execute("PRAGMA journal_mode").fetchone()[0]
    if current.lower() != "wal":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't already exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR IGNORE INTO schema_meta (key, value)
            VALUES ('schema_version', '1');

        CREATE TABLE IF NOT EXISTS sessions (
            session_id          TEXT PRIMARY KEY,
            tags                TEXT,
            state               TEXT NOT NULL,
            launch_mode         TEXT NOT NULL,
            launch_time         TEXT NOT NULL,
            target_type         TEXT NOT NULL,
            target              TEXT,
            target_args         TEXT,
            target_cwd          TEXT,
            python_version      TEXT,
            python_path         TEXT,
            uv_version          TEXT,
            resolved_by         TEXT,
            psoul_version       TEXT NOT NULL,
            protocol_version    INTEGER NOT NULL DEFAULT 1,
            host                TEXT,
            os                  TEXT,
            arch                TEXT,
            config_sources      TEXT,
            git_sha             TEXT,
            git_dirty           INTEGER,
            lockfile_hash       TEXT,
            script_hash         TEXT,
            supervisor_pid      INTEGER,
            socket_path         TEXT,
            helper_pid          INTEGER,
            helper_capabilities TEXT,
            generation          INTEGER NOT NULL DEFAULT 0,
            control_epoch       INTEGER NOT NULL DEFAULT 0,
            controller_pid      INTEGER,
            control_acquired_at TEXT,
            sandbox_backend     TEXT,
            sandbox_policy      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_state ON sessions(state);
        CREATE INDEX IF NOT EXISTS idx_sessions_launch_time ON sessions(launch_time);

        CREATE TABLE IF NOT EXISTS results (
            session_id          TEXT NOT NULL,
            generation          INTEGER NOT NULL,
            outcome             TEXT NOT NULL,
            exit_code           INTEGER,
            signal              TEXT,
            end_time            TEXT NOT NULL,
            duration_seconds    REAL,
            peak_memory_mb      REAL,
            avg_cpu_percent     REAL,
            gpu_peak_memory_mb  REAL,
            artifacts_count     INTEGER,
            events_count        INTEGER,
            crash_reason        TEXT,
            orphan_detected_at  TEXT,
            recovery_attempted_at TEXT,
            PRIMARY KEY (session_id, generation),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            sequence    INTEGER NOT NULL,
            generation  INTEGER NOT NULL DEFAULT 0,
            timestamp   TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            payload     TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, event_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_events_sequence ON events(session_id, sequence);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);

        CREATE TABLE IF NOT EXISTS commands (
            command_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            message_id  TEXT,
            source      TEXT,
            generation  INTEGER NOT NULL DEFAULT 0,
            timestamp   TEXT NOT NULL,
            command     TEXT NOT NULL,
            params      TEXT,
            status      TEXT NOT NULL,
            result      TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_commands_session ON commands(session_id, command_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_commands_message_id ON commands(message_id)
            WHERE message_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS history (
            history_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            timestamp   TEXT NOT NULL,
            input       TEXT NOT NULL,
            cwd         TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_history_session ON history(session_id);
        CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp);

        CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
            input,
            content='history',
            content_rowid='history_id'
        );

        CREATE TRIGGER IF NOT EXISTS history_fts_insert AFTER INSERT ON history BEGIN
            INSERT INTO history_fts(rowid, input) VALUES (new.history_id, new.input);
        END;
        CREATE TRIGGER IF NOT EXISTS history_fts_delete AFTER DELETE ON history BEGIN
            INSERT INTO history_fts(history_fts, rowid, input)
                VALUES ('delete', old.history_id, old.input);
        END;
        CREATE TRIGGER IF NOT EXISTS history_fts_update AFTER UPDATE ON history BEGIN
            INSERT INTO history_fts(history_fts, rowid, input)
                VALUES ('delete', old.history_id, old.input);
            INSERT INTO history_fts(rowid, input) VALUES (new.history_id, new.input);
        END;

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            name            TEXT NOT NULL,
            path            TEXT NOT NULL,
            mime_type       TEXT,
            size_bytes      INTEGER,
            registered_at   TEXT NOT NULL,
            source          TEXT NOT NULL,
            retention_class TEXT NOT NULL DEFAULT 'session',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            UNIQUE (session_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);

        CREATE TABLE IF NOT EXISTS resource_samples (
            sample_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT NOT NULL,
            generation          INTEGER NOT NULL DEFAULT 0,
            timestamp           TEXT NOT NULL,
            cpu_percent         REAL,
            memory_rss_mb       REAL,
            memory_vms_mb       REAL,
            disk_read_mb        REAL,
            disk_write_mb       REAL,
            gpu_utilization_pct REAL,
            gpu_memory_used_mb  REAL,
            gpu_memory_total_mb REAL,
            gpu_temperature_c   REAL,
            gpu_power_watts     REAL,
            extended            TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_resources_session
            ON resource_samples(session_id, generation, timestamp);

        CREATE TABLE IF NOT EXISTS profiling_state (
            session_id      TEXT NOT NULL,
            generation      INTEGER NOT NULL DEFAULT 0,
            backend         TEXT NOT NULL,
            mode            TEXT NOT NULL,
            started_at      TEXT NOT NULL,
            sampling_rate   INTEGER,
            artifact_path   TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_profiling_active
            ON profiling_state(session_id, generation, mode);
    """)


def _run_migrations(conn: sqlite3.Connection, from_version: int) -> None:
    """Apply sequential migrations from from_version to SCHEMA_VERSION."""
    conn.execute("BEGIN")
    try:
        for version in range(from_version, SCHEMA_VERSION):
            _MIGRATIONS[version](conn)
        conn.execute("UPDATE schema_meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _check_version(conn: sqlite3.Connection) -> None:
    """Verify schema version and run migrations if needed."""
    row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    db_version = int(row[0])
    if db_version > SCHEMA_VERSION:
        msg = f"database schema v{db_version} is newer than this psoul (v{SCHEMA_VERSION})"
        raise RuntimeError(msg)
    if db_version < SCHEMA_VERSION:
        _run_migrations(conn, db_version)


def _schema_exists(conn: sqlite3.Connection) -> bool:
    """Check whether the schema has already been created."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'").fetchone()
    return row is not None


def open_db(state_dir: Path) -> sqlite3.Connection:
    """Open or create the psoul database.

    Configures the connection for safe concurrent access, creates the
    schema on first use, and runs pending migrations on an existing
    database.  The caller owns the returned connection and must close it.

    Args:
        state_dir (Path): Directory containing (or that will contain) ``psoul.db``.

    Returns:
        sqlite3.Connection: Ready-to-use connection with schema in place.

    Raises:
        RuntimeError: Database schema is newer than this psoul version.

    """
    conn = sqlite3.connect(state_dir / DB_NAME, timeout=5.0)
    try:
        _apply_pragmas(conn)
        if not _schema_exists(conn):
            _create_schema(conn)
        else:
            _check_version(conn)
    except Exception:
        conn.close()
        raise
    return conn
