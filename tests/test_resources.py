"""Tests for ResourceSampler: periodic CPU/memory/disk collection."""

import threading
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Self, cast

import psutil
import pytest

from psoul.core.db import open_db
from psoul.core.events import EventStore
from psoul.core.resources import EVENT_RESOURCE_TELEMETRY, ResourceSampler
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

SESSION_ID = "res-test"

_GPU_PAYLOAD: dict[str, float | None] = {
    "gpu_utilization_pct": 42.0,
    "gpu_memory_used_mb": 2048.0,
    "gpu_memory_total_mb": 8192.0,
    "gpu_temperature_c": 65.0,
    "gpu_power_watts": 75.0,
}


class _StubGpuReader:
    """Drop-in for ``GpuReader`` that returns a fixed payload from ``read()``."""

    def __init__(self, payload: dict[str, float | None] | None = None) -> None:
        self._payload: dict[str, float | None] = payload or {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> dict[str, float | None]:
        return self._payload


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


def test_gpu_columns_null_when_reader_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``GpuReader.read()`` returns ``{}``, all 5 GPU columns are NULL in the row."""
    monkeypatch.setattr("psoul.core.resources.GpuReader", _StubGpuReader)
    _seed_session(tmp_path)
    sampler = ResourceSampler(psutil.Process(), tmp_path, SESSION_ID, generation=0)
    thread = threading.Thread(target=sampler.run, args=(0.01,))
    thread.start()
    time.sleep(0.1)
    sampler.stop()
    thread.join(timeout=2.0)
    with closing(open_db(tmp_path)) as conn:
        row = conn.execute(
            "SELECT gpu_utilization_pct, gpu_memory_used_mb, gpu_memory_total_mb,"
            " gpu_temperature_c, gpu_power_watts FROM resource_samples",
        ).fetchone()
        assert row == (None, None, None, None, None)


def test_gpu_metrics_flow_to_row_and_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``GpuReader.read()`` returns populated values, the row and event payload include them."""
    monkeypatch.setattr("psoul.core.resources.GpuReader", lambda: _StubGpuReader(_GPU_PAYLOAD))
    _seed_session(tmp_path)
    sampler = ResourceSampler(psutil.Process(), tmp_path, SESSION_ID, generation=0)
    thread = threading.Thread(target=sampler.run, args=(0.01,))
    thread.start()
    time.sleep(0.1)
    sampler.stop()
    thread.join(timeout=2.0)
    with closing(open_db(tmp_path)) as conn:
        row = conn.execute(
            "SELECT gpu_utilization_pct, gpu_memory_used_mb, gpu_memory_total_mb,"
            " gpu_temperature_c, gpu_power_watts FROM resource_samples",
        ).fetchone()
        assert row == (42.0, 2048.0, 8192.0, 65.0, 75.0)
        events = EventStore(conn).list(SESSION_ID, event_type=EVENT_RESOURCE_TELEMETRY)
        payload = cast("dict[str, float]", events[0]["payload"])
        assert payload["gpu_utilization_pct"] == 42.0
        assert payload["gpu_power_watts"] == 75.0


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
