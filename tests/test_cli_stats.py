"""Tests for the ``psoul stats`` CLI command."""

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.core.db import open_db
from psoul.core.session import LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()

SESSION_ID = "res-cli"

_SAMPLE_COLUMNS = "generation, timestamp, cpu_percent, memory_rss_mb, memory_vms_mb, disk_read_mb, disk_write_mb"


@pytest.fixture
def seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Seed a session with two samples so generation ordering is observable."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id=SESSION_ID,
                state=SessionState.exited,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
            )
        )
        conn.execute(
            f"INSERT INTO resource_samples (session_id, {_SAMPLE_COLUMNS})"  # noqa: S608
            " VALUES (?, 0, '2026-01-01T00:02:00', 5.0, 50.0, 200.0, 0.5, 1.0)",
            (SESSION_ID,),
        )
        conn.execute(
            f"INSERT INTO resource_samples (session_id, {_SAMPLE_COLUMNS})"  # noqa: S608
            " VALUES (?, 1, '2026-01-01T00:01:00', 12.5, 100.0, 400.0, 1.5, 2.5)",
            (SESSION_ID,),
        )
        conn.commit()
    return SESSION_ID


def test_stats_text_returns_latest(seeded: str) -> None:
    result = runner.invoke(cli, ["stats", seeded])
    assert result.exit_code == 0
    assert "generation: 1" in result.output
    assert "cpu_percent: 12.5" in result.output
    assert "5.0" not in result.output  # old sample excluded
    assert "gpu_utilization_pct" not in result.output  # NULLs omitted


def test_stats_json_shape(seeded: str) -> None:
    result = runner.invoke(cli, ["stats", seeded, "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["cpu_percent"] == 12.5
    assert parsed["memory_rss_mb"] == 100.0
    assert parsed["generation"] == 1
    assert parsed["gpu_utilization_pct"] is None
    assert set(parsed) == {
        "generation",
        "timestamp",
        "cpu_percent",
        "memory_rss_mb",
        "memory_vms_mb",
        "disk_read_mb",
        "disk_write_mb",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_temperature_c",
        "gpu_power_watts",
    }


@pytest.mark.parametrize(
    ("match", "expected_exit", "expected_contains"),
    [
        (True, 0, "cpu_percent"),
        (False, 1, "session not found"),
    ],
    ids=["prefix-match", "unknown-session"],
)
def test_stats_resolves_session_selector(seeded: str, match: bool, expected_exit: int, expected_contains: str) -> None:
    selector = seeded[:3] if match else "nope"
    result = runner.invoke(cli, ["stats", selector])
    assert result.exit_code == expected_exit
    assert expected_contains in result.output


def test_stats_no_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Error when session exists but has no resource samples."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="empty",
                state=SessionState.exited,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
            )
        )
    result = runner.invoke(cli, ["stats", "empty"])
    assert result.exit_code == 1
    assert "Error: no resource samples found." in result.output


def test_stats_renders_gpu_columns_when_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Text and ``--json`` output both include GPU values when the row has GPU data."""
    monkeypatch.setattr("psoul.core.db.default_state_dir", lambda: tmp_path)
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="gpu-cli",
                state=SessionState.exited,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
            )
        )
        conn.execute(
            "INSERT INTO resource_samples"
            " (session_id, generation, timestamp,"
            "  cpu_percent, memory_rss_mb, memory_vms_mb,"
            "  disk_read_mb, disk_write_mb,"
            "  gpu_utilization_pct, gpu_memory_used_mb, gpu_memory_total_mb,"
            "  gpu_temperature_c, gpu_power_watts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("gpu-cli", 0, "2026-01-01T00:00:00", 5.0, 50.0, 200.0, 0.5, 1.0, 42.0, 2048.0, 8192.0, 65.0, 75.0),
        )
        conn.commit()
    text_result = runner.invoke(cli, ["stats", "gpu-cli"])
    assert text_result.exit_code == 0
    assert "gpu_utilization_pct: 42.0" in text_result.output
    assert "gpu_power_watts: 75.0" in text_result.output
    json_result = runner.invoke(cli, ["stats", "gpu-cli", "--json"])
    assert json_result.exit_code == 0
    parsed = json.loads(json_result.output)
    assert parsed["gpu_utilization_pct"] == 42.0
    assert parsed["gpu_power_watts"] == 75.0
