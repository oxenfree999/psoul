"""Helpers behind ``psoul env``.

The CLI either snapshots the running process via ``get_current_env``
or pulls what was recorded at launch via ``get_session_env``. Either
way the result is a plain dict that the shared ``format_text``
renders for terminal output and ``json.dumps`` emits for ``--json``.
"""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from psoul.cli.doctor import _get_tool_info, _get_venv
from psoul.core.session import Session

ENV_TOOLS: tuple[str, ...] = ("pip", "uv", "ruff", "ty")
LABEL_WIDTH = 14
_UV_TREE_TIMEOUT_SEC = 3

_SESSION_ONLY_ROWS: tuple[tuple[str, str], ...] = (
    ("Host", "host"),
    ("Git SHA", "git_sha"),
    ("Git dirty", "git_dirty"),
    ("Lockfile", "lockfile_hash"),
    ("Script hash", "script_hash"),
    ("Resolved by", "resolved_by"),
    ("Config", "config_sources"),
)


def get_current_env() -> dict[str, Any]:
    """Snapshot what the current Python process can see about its environment.

    Returns a dict consumed by the shared ``format_text`` (text) and
    ``json.dumps`` (JSON). The live form and session form produce
    related but distinct shapes — the formatter accepts both. Every
    documented live-form key is always present here. ``None`` just
    means we looked and could not find it (no venv, no ``$SHELL``, no
    ``uv tree``, etc.).

    Returns:
        dict[str, Any]: Live env summary, ready for ``format_text`` or
            ``json.dumps``.

    """
    uv_path = shutil.which("uv")
    dependency_graph: str | None = None
    if uv_path is not None:
        try:
            proc = subprocess.run(  # noqa: S603 — uv path comes from shutil.which
                [uv_path, "tree", "--depth", "1"],
                capture_output=True,
                text=True,
                timeout=_UV_TREE_TIMEOUT_SEC,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        else:
            if proc.returncode == 0:
                dependency_graph = proc.stdout.strip() or None

    try:
        package_count: int | None = sum(1 for _ in importlib.metadata.distributions())
    except Exception:  # noqa: BLE001 — package count must never crash env
        package_count = None

    return {
        "source": "current",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "shell": os.environ.get("SHELL") or os.environ.get("COMSPEC") or None,
        "directory": str(Path.cwd()),
        "venv": _get_venv(),
        "interpreter": sys.executable,
        "python": {"version": platform.python_version()},
        "tools": {name: _get_tool_info(name) for name in ENV_TOOLS},
        "sys_path": list(sys.path),
        "dependency_graph": dependency_graph,
        "package_count": package_count,
    }


def get_session_env(session: Session) -> dict[str, Any]:
    """Pull what a recorded session captured about its environment at launch.

    Sessions only store what could be observed at launch time — platform,
    interpreter, git/lockfile provenance — not live runtime details like
    ``sys.path`` or the current dependency graph. Whatever the row has
    goes into the dict. Anything missing is left out, so the formatter
    can render "not recorded" and the JSON form simply omits the key.

    Args:
        session (Session): Session row to read provenance from.

    Returns:
        dict[str, Any]: Session env summary, ready for ``format_text``
            or ``json.dumps``.

    """
    info: dict[str, Any] = {
        "source": f"session:{session.session_id}",
        "platform": {"system": session.os, "machine": session.arch},
        "python": {"version": session.python_version},
    }
    if session.target_cwd is not None:
        info["directory"] = str(session.target_cwd)
    if session.python_path is not None:
        info["interpreter"] = str(session.python_path)
    if session.uv_version is not None:
        info["tools"] = {
            "uv": {"available": True, "version": session.uv_version, "path": None, "error": None},
        }
    for key in ("host", "git_sha", "git_dirty", "lockfile_hash", "script_hash", "resolved_by", "config_sources"):
        value = getattr(session, key)
        if value is not None:
            info[key] = value
    return info


def _row(label: str, value: str) -> str:
    """Format ``label`` and ``value`` aligned to ``LABEL_WIDTH`` columns."""
    return f"{label:<{LABEL_WIDTH}} {value}"


def _str_value(info: dict[str, Any], key: str, *, null: str = "not recorded", missing: str = "not recorded") -> str:
    """Render a value with separate strings for missing-from-dict vs ``None``."""
    if key not in info:
        return missing
    value = info[key]
    if value is None:
        return null
    return str(value)


def format_text(info: dict[str, Any]) -> str:  # noqa: C901, PLR0912, PLR0915 — straight-line rendering with per-row branches
    """Render env info as column-aligned human-readable text.

    Walks a fixed list of label rows in a stable order. Missing keys render
    as ``not recorded``. ``None`` values render as a context-specific
    marker (``not active`` / ``not available`` / ``not found``). Session-
    only rows are appended only when ``source`` is a session.

    Args:
        info (dict[str, Any]): Output from ``get_current_env`` or
            ``get_session_env``.

    Returns:
        str: Multi-line text with labels and values column-aligned.

    """
    is_session = (info.get("source") or "").startswith("session:")
    lines: list[str] = [_row("Source", info.get("source") or "?")]

    p = info.get("platform")
    if not p:
        platform_str = "not recorded"
    else:
        parts = [p.get("system") or "?"]
        if p.get("release"):
            parts.append(p["release"])
        if p.get("machine"):
            parts.append(f"({p['machine']})")
        platform_str = " ".join(parts)
    lines.append(_row("Platform", platform_str))

    lines.append(_row("Shell", _str_value(info, "shell", null="not available")))
    lines.append(_row("Directory", _str_value(info, "directory")))
    lines.append(_row("Venv", _str_value(info, "venv", null="not active")))
    lines.append(_row("Interpreter", _str_value(info, "interpreter")))

    py = info.get("python")
    lines.append(_row("Python", py["version"] if py and py.get("version") else "not recorded"))

    lines.append("")
    tools = info.get("tools") or {}
    for name in ENV_TOOLS:
        tool = tools.get(name)
        if tool is None:
            tool_str = "not recorded"
        elif not tool.get("available"):
            tool_str = "not found"
        elif tool.get("error"):
            tool_str = str(tool["error"])
        else:
            tool_str = tool.get("version") or "unknown"
        lines.append(_row(name, tool_str))

    lines.append("")
    if "sys_path" not in info:
        lines.append(_row("Sys path", "not recorded"))
    else:
        sys_path_list = info["sys_path"] or []
        if not sys_path_list:
            lines.append(_row("Sys path", "(empty)"))
        else:
            lines.append(_row("Sys path", sys_path_list[0]))
            lines.extend(f"{'':<{LABEL_WIDTH}} {entry}" for entry in sys_path_list[1:])

    lines.append("")
    if "dependency_graph" not in info:
        lines.append(_row("Dependencies", "not recorded"))
    elif info["dependency_graph"] is None:
        lines.append(_row("Dependencies", "not available"))
    else:
        graph_lines = info["dependency_graph"].splitlines() or [""]
        lines.append(_row("Dependencies", graph_lines[0]))
        lines.extend(f"{'':<{LABEL_WIDTH}} {line}" for line in graph_lines[1:])

    lines.append("")
    lines.append(_row("Packages", _str_value(info, "package_count", null="not available")))

    if is_session:
        lines.append("")
        for label, key in _SESSION_ONLY_ROWS:
            if key not in info:
                value_str = "not recorded"
            elif key == "git_dirty":
                value_str = "yes" if info[key] else "no"
            elif key == "config_sources":
                value_str = ", ".join(str(item) for item in info[key]) or "(empty)"
            else:
                value_str = str(info[key])
            lines.append(_row(label, value_str))

    return "\n".join(lines)
