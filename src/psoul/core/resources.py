"""Periodic resource sampler: CPU, memory, and disk metrics via psutil."""

import sqlite3  # noqa: TC003 — used in _persist type annotation
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import psutil

from psoul.core.db import open_db
from psoul.core.events import EventStore
from psoul.core.gpu import GpuReader

EVENT_RESOURCE_TELEMETRY = "resource.telemetry"

_BYTES_PER_MB = 1024 * 1024  # psutil returns bytes; DB stores megabytes


def _collect(process: psutil.Process) -> dict[str, object]:
    """Read CPU, memory, and disk metrics from *process*.

    Returns a dict matching the ``resource_samples`` column names.
    Fields that aren't available on the current platform (e.g.
    ``disk_read_mb`` on macOS) are ``None``. Raises
    ``psutil.NoSuchProcess`` if the process has exited and been
    reaped, or ``psutil.ZombieProcess`` if the process exited but
    is still in the kernel's process table awaiting reap.
    """
    cpu = process.cpu_percent()
    mem = process.memory_info()
    disk_read: float | None = None
    disk_write: float | None = None
    if hasattr(process, "io_counters"):  # not available on macOS
        io = process.io_counters()
        disk_read = io.read_bytes / _BYTES_PER_MB
        disk_write = io.write_bytes / _BYTES_PER_MB
    return {
        "cpu_percent": cpu,
        "memory_rss_mb": mem.rss / _BYTES_PER_MB,
        "memory_vms_mb": mem.vms / _BYTES_PER_MB,
        "disk_read_mb": disk_read,
        "disk_write_mb": disk_write,
    }


class ResourceSampler:
    """Periodic resource sampler for a managed process.

    Collects CPU, memory, and disk metrics via psutil and persists each
    snapshot as a ``resource_samples`` row and a ``resource.telemetry``
    event.
    """

    def __init__(
        self,
        process: psutil.Process,
        state_dir: Path,
        session_id: str,
        generation: int,
    ) -> None:
        """Bind the sampler to a process and state directory."""
        self._process = process
        self._state_dir = state_dir
        self._session_id = session_id
        self._generation = generation
        self._stop_event = threading.Event()

    def _persist(self, metrics: dict[str, object], conn: "sqlite3.Connection", event_store: EventStore) -> None:
        """Write one sample row and one event in a single transaction."""
        timestamp = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO resource_samples"
            " (session_id, generation, timestamp,"
            "  cpu_percent, memory_rss_mb, memory_vms_mb,"
            "  disk_read_mb, disk_write_mb,"
            "  gpu_utilization_pct, gpu_memory_used_mb, gpu_memory_total_mb,"
            "  gpu_temperature_c, gpu_power_watts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._session_id,
                self._generation,
                timestamp,
                metrics["cpu_percent"],
                metrics["memory_rss_mb"],
                metrics["memory_vms_mb"],
                metrics["disk_read_mb"],
                metrics["disk_write_mb"],
                metrics.get("gpu_utilization_pct"),
                metrics.get("gpu_memory_used_mb"),
                metrics.get("gpu_memory_total_mb"),
                metrics.get("gpu_temperature_c"),
                metrics.get("gpu_power_watts"),
            ),
        )
        event_store.append(
            session_id=self._session_id,
            event_type=EVENT_RESOURCE_TELEMETRY,
            payload=metrics,
            generation=self._generation,
            commit=False,
        )
        conn.commit()

    def run(self, interval: float) -> None:
        """Sample in a loop every *interval* seconds until stopped.

        Opens its own DB connection so it's safe to call from a
        background thread.  Exits cleanly on ``stop()`` or when the
        process disappears.
        """
        with closing(open_db(self._state_dir)) as conn, GpuReader() as gpu:
            event_store = EventStore(conn)
            while not self._stop_event.is_set():
                try:
                    metrics = _collect(self._process)
                    metrics.update(gpu.read())
                    self._persist(metrics, conn, event_store)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    break
                self._stop_event.wait(interval)  # interruptible sleep

    def stop(self) -> None:
        """Signal the sampling loop to exit."""
        self._stop_event.set()
