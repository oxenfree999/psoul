"""Environment health checks for psoul doctor."""

import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from psoul.version import VERSION

LABEL_WIDTH = 14

KNOWN_TOOLS = (
    # packaging
    "uv",
    "pip",
    "poetry",
    "pdm",
    # linting
    "ruff",
    "flake8",
    "pylint",
    # formatting
    "black",
    "isort",
    # type checking
    "ty",
    "mypy",
    "pyright",
    # profiling
    "austin",
    "py-spy",
    "memray",
    "scalene",
)

PREFERRED_TOOLS = ("uv", "ruff", "ty")


def _get_tool_info(name: str) -> dict[str, Any]:
    """Detect a CLI tool's availability and version."""
    path = shutil.which(name)
    if path is None:
        return {"available": False, "version": None, "path": None, "error": None}
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=3, check=False)  # noqa: S603 — path comes from shutil.which
    except Exception as e:  # noqa: BLE001 — version check should never crash doctor
        return {"available": True, "version": None, "path": path, "error": str(e)}
    stdout = proc.stdout.strip()
    version = m.group(0) if proc.returncode == 0 and (m := re.search(r"\d+\.\d+[\w.]*", stdout)) else None
    return {"available": True, "version": version, "path": path, "error": None}


def _get_venv() -> str | None:
    """Return the venv path if running inside a virtual environment."""
    return sys.prefix if sys.prefix != sys.base_prefix else None


def get_system_info() -> dict[str, Any]:
    """Collect environment information for ``psoul doctor``.

    Gathers platform details, Python interpreter path, psoul version,
    virtual environment status, and availability of known Python tools.

    Returns:
        dict[str, Any]: Nested dict keyed by category: ``platform``,
            ``python``, ``psoul``, ``tools``, ``venv``, ``interpreter``,
            ``directory``.

    """
    tools = {name: _get_tool_info(name) for name in KNOWN_TOOLS}
    return {
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "directory": str(Path.cwd()),
        "venv": _get_venv(),
        "interpreter": sys.executable,
        "python": {"version": platform.python_version()},
        "psoul": {"version": VERSION},
        "tools": tools,
    }


def format_text(info: dict[str, Any]) -> str:
    """Format system info as column-aligned human-readable text.

    Args:
        info (dict[str, Any]): Output from ``get_system_info()``.

    Returns:
        str: Multi-line string with labels and values aligned.

    """
    lines = []

    p = info["platform"]
    lines.append(f"{'Platform':<{LABEL_WIDTH}} {p['system']} {p['release']} ({p['machine']})")
    lines.append(f"{'Directory':<{LABEL_WIDTH}} {info['directory']}")

    venv = info["venv"]
    lines.append(f"{'Venv':<{LABEL_WIDTH}} {venv or 'not active'}")
    lines.append(f"{'Interpreter':<{LABEL_WIDTH}} {info['interpreter']}")

    lines.append(f"{'psoul':<{LABEL_WIDTH}} {info['psoul']['version']}")
    lines.append(f"{'Python':<{LABEL_WIDTH}} {info['python']['version']}")

    lines.append("")
    for name in KNOWN_TOOLS:
        tool = info["tools"][name]
        if tool["available"]:
            if tool["error"]:
                lines.append(f"{name:<{LABEL_WIDTH}} {tool['error']}")
            else:
                lines.append(f"{name:<{LABEL_WIDTH}} {tool['version']}")
        elif name in PREFERRED_TOOLS:
            lines.append(f"{name:<{LABEL_WIDTH}} not found")

    return "\n".join(lines)
