"""NVIDIA GPU telemetry for device 0 via pynvml.

When pynvml is missing, the driver or device is unavailable, or a read
fails, ``read()`` returns ``{}`` and the GPU columns stay NULL in
``resource_samples``.
"""

import contextlib
from typing import Self

try:
    import pynvml
except ImportError:
    pynvml = None  # ty: ignore[invalid-assignment]

_BYTES_PER_MB = 1024 * 1024  # nvml returns bytes; DB stores megabytes
_MILLIWATTS_PER_WATT = 1000  # nvml returns milliwatts; DB stores watts


class GpuReader:
    """Context-managed reader for NVIDIA device 0 telemetry."""

    def __init__(self) -> None:
        """Initialize reader state; no nvml calls happen until ``__enter__``."""
        self._handle: object | None = None

    def __enter__(self) -> Self:
        """Initialize nvml and acquire device 0's handle, if both succeed."""
        if pynvml is None:
            return self
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return self
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            pass
        if self._handle is None:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the device handle and shut nvml down if it was initialized."""
        if self._handle is None:
            return
        self._handle = None
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()

    def read(self) -> dict[str, float | None]:
        """Return device 0's telemetry, or ``{}`` when the reader is inactive or NVML errors out."""
        if self._handle is None:
            return {}
        try:
            util = float(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            try:
                temp: float | None = float(pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError_NotSupported:  # ty: ignore[unresolved-attribute]
                temp = None
            try:
                power: float | None = pynvml.nvmlDeviceGetPowerUsage(self._handle) / _MILLIWATTS_PER_WATT
            except pynvml.NVMLError_NotSupported:  # ty: ignore[unresolved-attribute]
                power = None
            return {
                "gpu_utilization_pct": util,
                "gpu_memory_used_mb": mem.used / _BYTES_PER_MB,
                "gpu_memory_total_mb": mem.total / _BYTES_PER_MB,
                "gpu_temperature_c": temp,
                "gpu_power_watts": power,
            }
        except pynvml.NVMLError:
            return {}
