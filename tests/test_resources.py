"""Tests for ResourceSampler: periodic CPU/memory/disk collection."""

import threading
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pytest

from psoul.core.db import open_db
from psoul.core.events import EventStore
from psoul.core.resources import EVENT_RESOURCE_TELEMETRY, ResourceSampler
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

SESSION_ID = "res-test"


def _seed_session(tmp_path: Path) -> None:
    """Create a minimal session so foreign keys are satisfied."""
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id=SESSION_ID,
                state=SessionState.running,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
            )
        )


def test_run_persists_metrics(tmp_path: Path) -> None:
    """run() writes at least one resource_samples row and one event."""
    _seed_session(tmp_path)
    sampler = ResourceSampler(psutil.Process(), tmp_path, SESSION_ID, generation=0)
    thread = threading.Thread(target=sampler.run, args=(0.01,))
    thread.start()
    time.sleep(0.1)  # let a few samples land
    sampler.stop()
    thread.join(timeout=2.0)
    with closing(open_db(tmp_path)) as conn:
        row = conn.execute("SELECT cpu_percent, memory_rss_mb FROM resource_samples").fetchone()
        assert row is not None
        assert isinstance(row[0], float)
        assert row[1] > 0  # running process always uses some RSS
        events = EventStore(conn).list(SESSION_ID, event_type=EVENT_RESOURCE_TELEMETRY)
        assert len(events) >= 1


def test_run_exits_on_process_death(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run() exits cleanly when the target process disappears mid-sample."""
    _seed_session(tmp_path)
    sampler = ResourceSampler(psutil.Process(), tmp_path, SESSION_ID, generation=0)
    call_count = 0
    original_collect = psutil.Process.cpu_percent

    def dying_cpu(self: psutil.Process, interval: float | None = None) -> float:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise psutil.NoSuchProcess(pid=self.pid)
        return original_collect(self, interval)

    monkeypatch.setattr(psutil.Process, "cpu_percent", dying_cpu)
    sampler.run(interval=0.01)  # blocks until NoSuchProcess
    assert call_count == 2


def test_stop_interrupts_run(tmp_path: Path) -> None:
    """stop() causes run() to exit without waiting for the full interval."""
    _seed_session(tmp_path)
    sampler = ResourceSampler(psutil.Process(), tmp_path, SESSION_ID, generation=0)
    thread = threading.Thread(target=sampler.run, args=(60.0,))  # long interval
    thread.start()
    sampler.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
