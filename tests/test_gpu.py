"""Tests for the NVIDIA GPU telemetry reader."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from psoul.core import gpu


class FakeNvmlError(Exception):
    """Stand-in for ``pynvml.NVMLError`` so tests can ``except`` it."""


class FakeNvmlNotSupportedError(FakeNvmlError):
    """Stand-in for ``pynvml.NVMLError_NotSupported`` (subclass of NVMLError)."""


@pytest.fixture
def fake_pynvml(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Inject a fake pynvml module with sensible happy-path defaults."""
    fake = SimpleNamespace(
        nvmlInit=MagicMock(),
        nvmlShutdown=MagicMock(),
        nvmlDeviceGetCount=MagicMock(return_value=1),
        nvmlDeviceGetHandleByIndex=MagicMock(return_value="device-0-handle"),
        nvmlDeviceGetUtilizationRates=MagicMock(return_value=SimpleNamespace(gpu=42)),
        nvmlDeviceGetMemoryInfo=MagicMock(
            return_value=SimpleNamespace(used=2 * 1024 * 1024 * 1024, total=8 * 1024 * 1024 * 1024),
        ),
        nvmlDeviceGetTemperature=MagicMock(return_value=65),
        nvmlDeviceGetPowerUsage=MagicMock(return_value=75_000),
        NVMLError=FakeNvmlError,
        NVMLError_NotSupported=FakeNvmlNotSupportedError,
        NVML_TEMPERATURE_GPU=0,
    )
    monkeypatch.setattr("psoul.core.gpu.pynvml", fake)
    return fake


def test_gpu_unavailable_when_pynvml_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``pynvml`` is None, ``read()`` returns ``{}``."""
    monkeypatch.setattr("psoul.core.gpu.pynvml", None)
    with gpu.GpuReader() as reader:
        assert reader.read() == {}


def test_gpu_import_blocked_with_reload() -> None:
    """Block ``pynvml`` in ``sys.modules`` and reload to exercise the real ``except ImportError`` branch."""
    sys.modules["pynvml"] = None  # ty: ignore[invalid-assignment]
    try:
        importlib.reload(gpu)
        assert gpu.pynvml is None
        with gpu.GpuReader() as reader:
            assert reader.read() == {}
    finally:
        del sys.modules["pynvml"]
        importlib.reload(gpu)


def test_gpu_unavailable_when_init_fails(fake_pynvml: SimpleNamespace) -> None:
    """When ``nvmlInit`` raises NVMLError, the reader stays inactive and ``nvmlShutdown`` is not called."""
    fake_pynvml.nvmlInit.side_effect = FakeNvmlError("no driver")
    with gpu.GpuReader() as reader:
        assert reader.read() == {}
    fake_pynvml.nvmlShutdown.assert_not_called()


def test_gpu_unavailable_when_no_devices(fake_pynvml: SimpleNamespace) -> None:
    """When no devices are present, ``read()`` returns ``{}`` but ``nvmlShutdown`` runs to balance ``nvmlInit``."""
    fake_pynvml.nvmlDeviceGetCount.return_value = 0
    with gpu.GpuReader() as reader:
        assert reader.read() == {}
    fake_pynvml.nvmlShutdown.assert_called_once()


def test_gpu_happy_path_all_fields(fake_pynvml: SimpleNamespace) -> None:
    """All five fields populate with correct unit conversions (bytes→MB, mW→W)."""
    with gpu.GpuReader() as reader:
        assert reader.read() == {
            "gpu_utilization_pct": 42.0,
            "gpu_memory_used_mb": 2048.0,
            "gpu_memory_total_mb": 8192.0,
            "gpu_temperature_c": 65.0,
            "gpu_power_watts": 75.0,
        }


def test_gpu_partial_when_power_unsupported(fake_pynvml: SimpleNamespace) -> None:
    """Power and temperature become ``None`` when they raise ``NVMLError_NotSupported``."""
    fake_pynvml.nvmlDeviceGetTemperature.side_effect = FakeNvmlNotSupportedError()
    fake_pynvml.nvmlDeviceGetPowerUsage.side_effect = FakeNvmlNotSupportedError()
    with gpu.GpuReader() as reader:
        assert reader.read() == {
            "gpu_utilization_pct": 42.0,
            "gpu_memory_used_mb": 2048.0,
            "gpu_memory_total_mb": 8192.0,
            "gpu_temperature_c": None,
            "gpu_power_watts": None,
        }


def test_gpu_returns_empty_on_generic_nvml_error(fake_pynvml: SimpleNamespace) -> None:
    """Generic NVMLError during a read drops the whole GPU section (``read()`` returns ``{}``)."""
    fake_pynvml.nvmlDeviceGetUtilizationRates.side_effect = FakeNvmlError("driver glitch")
    with gpu.GpuReader() as reader:
        assert reader.read() == {}


def test_gpu_shutdown_called_on_exit(fake_pynvml: SimpleNamespace) -> None:
    """``__exit__`` runs ``nvmlShutdown`` on both normal and exception exits."""
    with gpu.GpuReader():
        pass
    assert fake_pynvml.nvmlShutdown.call_count == 1
    with pytest.raises(RuntimeError), gpu.GpuReader():
        raise RuntimeError("boom")
    assert fake_pynvml.nvmlShutdown.call_count == 2
