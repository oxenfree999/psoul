"""Configuration schema, platform directory resolution, and config generation."""

import contextlib
import dataclasses
import os
import re
import sys
import tempfile
import tomllib
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import tomlkit
from platformdirs import PlatformDirs
from platformdirs.unix import Unix
from tomlkit.items import Table
from tomlkit.toml_file import TOMLFile

from psoul.core.duration import parse_duration
from psoul.core.session import LaunchMode

APP_NAME = "psoul"

_DIRS = Unix(APP_NAME) if sys.platform == "darwin" else PlatformDirs(APP_NAME)


def _unwrap_optional(tp: type) -> tuple[type, bool]:
    """Unwrap ``X | None`` to ``(X, True)``.  Non-optional types return ``(tp, False)``."""
    origin = get_origin(tp)
    if origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _coerce_field(section: str, key: str, value: object, expected: type, *, duration: bool = False) -> object:
    """Validate a config value's type and coerce string paths to ``Path`` objects."""
    base, optional = _unwrap_optional(expected)

    if value is None:
        if optional:
            return None
        msg = f"[{section}] {key}: unexpected null value"
        raise ValueError(msg)

    if base is Path:  # TOML has no path type, accept strings
        if isinstance(value, str):
            return Path(value).expanduser()  # so "~/foo" resolves to $HOME/foo
        msg = f"[{section}] {key}: expected path string, got {type(value).__name__}"
        raise TypeError(msg)

    # For dict[K, V], check against dict
    check_type = dict if get_origin(base) is dict else base
    # bool is a subclass of int — reject crossover in both directions
    if check_type is int and isinstance(value, bool):
        msg = f"[{section}] {key}: expected int, got bool"
        raise TypeError(msg)
    if not isinstance(value, check_type):
        expect = "table" if check_type is dict else check_type.__name__
        msg = f"[{section}] {key}: expected {expect}, got {type(value).__name__}"
        raise TypeError(msg)

    if duration and isinstance(value, str):
        try:
            parse_duration(value)
        except ValueError:
            msg = f"[{section}] {key}: invalid duration: {value!r}"
            raise ValueError(msg) from None

    return value


def _normalize_section(section_name: str, section_cls: type, raw: dict) -> dict:
    """Validate keys and coerce values for a single config section."""
    fields = {f.name: f for f in dataclasses.fields(section_cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        msg = f"[{section_name}] unknown key: {', '.join(unknown)}"
        raise ValueError(msg)

    hints = get_type_hints(section_cls)
    return {
        key: _coerce_field(section_name, key, val, hints[key], duration=fields[key].metadata.get("duration", False))
        for key, val in raw.items()
    }


def default_config_dir() -> Path:
    """Return the user config directory for psoul (~/.config/psoul on Unix).

    Returns:
        Path: Platform-appropriate config directory.

    """
    return _DIRS.user_config_path


def default_state_dir() -> Path:
    """Return the user state directory for psoul (~/.local/state/psoul on Unix).

    Returns:
        Path: Platform-appropriate state directory.

    """
    return _DIRS.user_state_path


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """[paths] section.

    state_dir (Path | None): overrides the default platform state directory. Default: None.
    """

    state_dir: Path | None = field(
        default=None,
        metadata={"description": "override session/state directory", "example": "~/.local/state/psoul"},
    )


@dataclass(frozen=True, slots=True)
class PythonConfig:
    """[python] section.

    python_path (Path | None): override Python interpreter used by ``psoul run``. Default: None.
    """

    python_path: Path | None = field(
        default=None,
        metadata={"description": "override Python interpreter used by psoul run", "example": "/usr/bin/python3"},
    )


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    """[launch] section.

    mode (str): 'attached' or 'headless'. Default: 'attached'.
    """

    mode: str = field(default="attached", metadata={"description": "attached (default) or headless"})

    def __post_init__(self) -> None:
        """Reject mode values that aren't in the LaunchMode enum."""
        try:
            LaunchMode(self.mode)
        except ValueError as exc:
            msg = f"[launch] mode: expected one of [{', '.join(LaunchMode)}], got {self.mode!r}"
            raise ValueError(msg) from exc


@dataclass(frozen=True, slots=True)
class ProcessConfig:
    """[process] section.

    stop_timeout (str): duration before suggesting kill (e.g. '10s', '1m'). Default: '10s'.
    stop_signal (str): signal sent by stop — 'SIGTERM', 'SIGINT', etc. Default: 'SIGTERM'.
    """

    stop_timeout: str = field(
        default="10s", metadata={"description": "how long stop waits before suggesting kill", "duration": True}
    )
    stop_signal: str = field(default="SIGTERM", metadata={"description": "signal sent by stop (SIGTERM, SIGINT, etc.)"})


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """[session] section.

    tags (dict | None): default key-value tags applied to all sessions. Default: None.
    record (bool): opt in to persistent recording of sessions. Default: False.
    """

    tags: dict[str, str] | None = field(
        default=None,
        metadata={
            "description": "default key-value tags applied to all sessions",
            "example": {"env": "dev", "team": "platform"},
        },
    )
    record: bool = field(
        default=False,
        metadata={"description": "opt in to persistent recording of sessions"},
    )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """[output] section.

    format (str): 'text', 'json', or 'ndjson'. Default: 'text'.
    color (str): 'auto', 'always', or 'never'. Default: 'auto'.
    timestamps (bool): show timestamps in human-readable output. Default: True.
    """

    format: str = field(default="text", metadata={"description": "text (default), json, or ndjson"})
    color: str = field(default="auto", metadata={"description": "auto, always, or never"})
    timestamps: bool = field(default=True, metadata={"description": "show timestamps in human-readable output"})


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    """[retention] section.

    max_age (str): auto-prune sessions older than this. Default: '7d'.
    max_sessions (int): max completed sessions to keep. Default: 100.
    max_artifact_mb (int): per-session artifact cap in MB. Default: 500.
    """

    max_age: str = field(
        default="7d", metadata={"description": "auto-prune sessions older than this", "duration": True}
    )
    max_sessions: int = field(default=100, metadata={"description": "max completed sessions to keep"})
    max_artifact_mb: int = field(default=500, metadata={"description": "per-session artifact cap in MB"})


@dataclass(frozen=True, slots=True)
class PsoulConfig:
    """Top-level configuration, composed from TOML sections.

    paths (PathsConfig): [paths] section.
    python (PythonConfig): [python] section.
    launch (LaunchConfig): [launch] section.
    process (ProcessConfig): [process] section.
    session (SessionConfig): [session] section.
    output (OutputConfig): [output] section.
    retention (RetentionConfig): [retention] section.
    """

    paths: PathsConfig = PathsConfig()
    python: PythonConfig = PythonConfig()
    launch: LaunchConfig = LaunchConfig()
    process: ProcessConfig = ProcessConfig()
    session: SessionConfig = SessionConfig()
    output: OutputConfig = OutputConfig()
    retention: RetentionConfig = RetentionConfig()


_SECTION_CLASSES: dict[str, type] = {
    "paths": PathsConfig,
    "python": PythonConfig,
    "launch": LaunchConfig,
    "process": ProcessConfig,
    "session": SessionConfig,
    "output": OutputConfig,
    "retention": RetentionConfig,
}

_GENERATED_SECTIONS: frozenset[str] = frozenset({"paths", "python", "launch", "session"})


def find_config_file(override: Path | None = None) -> Path | None:
    """Discover the config file to use, following precedence order.

    Precedence: explicit *override* path, then ``psoul.toml`` in the
    current directory, then ``pyproject.toml`` (if it contains
    ``[tool.psoul]``), then ``~/.config/psoul/config.toml``.

    Args:
        override (Path | None): Explicit ``--config`` path.  When set, no
            further search is performed.

    Returns:
        Path | None: First existing config file, or ``None`` if none found.

    Raises:
        FileNotFoundError: *override* is set but the file does not exist.

    """
    if override is not None:
        if not override.is_file():
            msg = f"Config file not found: {override}"
            raise FileNotFoundError(msg)
        return override

    # Project-local: psoul.toml wins over pyproject.toml
    psoul_toml = Path("psoul.toml")
    if psoul_toml.is_file():
        return psoul_toml

    pyproject = Path("pyproject.toml")
    if pyproject.is_file():
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        if "psoul" in data.get("tool", {}):
            return pyproject

    # User-level config
    user_config = default_config_dir() / "config.toml"
    if user_config.is_file():
        return user_config

    return None


def _extract_psoul_table(path: Path, data: dict) -> dict:
    """Extract the psoul config table from raw TOML data.

    For pyproject.toml, reads from [tool.psoul]. For other files, returns data as-is.
    """
    if path.name == "pyproject.toml":
        return data.get("tool", {}).get("psoul", {})
    return data


def load_config(path: Path | None = None) -> PsoulConfig:
    """Load configuration from a TOML file into a PsoulConfig.

    When *path* is ``None`` (no config file found), all defaults apply.
    For ``pyproject.toml`` files, reads from ``[tool.psoul]``.  Unknown
    sections or keys raise ``ValueError``.

    Args:
        path (Path | None): Config file path from ``find_config_file()``,
            or ``None`` for all defaults.

    Returns:
        PsoulConfig: Fully resolved configuration.

    Raises:
        ValueError: Unknown section or key in the config file.
        TypeError: A value has the wrong TOML type for its field.

    Example:
        >>> cfg = load_config(find_config_file())
        >>> cfg.process.stop_timeout
        '10s'

    """
    if path is None:
        return PsoulConfig()

    with path.open("rb") as f:
        data = tomllib.load(f)

    raw = _extract_psoul_table(path, data)

    unknown = sorted(set(raw) - set(_SECTION_CLASSES))
    if unknown:
        msg = f"unknown section: {', '.join(f'[{s}]' for s in unknown)}"
        raise ValueError(msg)

    return PsoulConfig(
        **{name: cls(**_normalize_section(name, cls, raw.get(name, {}))) for name, cls in _SECTION_CLASSES.items()},
    )


def _format_toml_value(value: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, dict):
        pairs = [f'{k} = "{v}"' for k, v in value.items()]
        return "{ " + ", ".join(pairs) + " }"
    return str(value)


def _detect_line_ending(content: str) -> str:
    r"""Detect the line-ending style of *content*.

    Returns ``"\r\n"`` if every newline is a Windows-style CRLF,
    ``"\n"`` if every newline is Unix-style LF, or ``os.linesep``
    when the content has no newlines or mixed endings.
    """
    num_lf = content.count("\n")
    if num_lf == 0:
        return os.linesep
    num_crlf = content.count("\r\n")
    if num_crlf == num_lf:
        return "\r\n"
    if num_crlf == 0:
        return "\n"
    return os.linesep


def _apply_line_ending(content: str, linesep: str) -> str:
    r"""Convert *content* (assumed ``"\n"``-normalized) to use *linesep*."""
    if linesep == "\n":
        return content.replace("\r\n", "\n")
    if linesep == "\r\n":
        return re.sub(r"(?<!\r)\n", "\r\n", content)
    return content


def _iter_config_comments() -> Iterator[tuple[str, list[str]]]:
    """Yield ``(section_name, comment_lines)`` for sections in ``_GENERATED_SECTIONS``."""
    defaults = PsoulConfig()
    for section_field in dataclasses.fields(defaults):
        if section_field.name not in _GENERATED_SECTIONS:
            continue
        section = getattr(defaults, section_field.name)
        comments: list[str] = []
        for f in dataclasses.fields(section):
            desc = f.metadata.get("description", "")
            value = f.metadata["example"] if getattr(section, f.name) is None else getattr(section, f.name)
            line = f"{f.name} = {_format_toml_value(value)}"
            if desc:
                line += f"  # {desc}"
            comments.append(line)
        yield section_field.name, comments


def generate_config() -> str:
    """Generate a commented psoul.toml by introspecting PsoulConfig fields.

    Every value is commented out so the file uses all defaults.
    Descriptions come from field metadata, not a separate data structure.

    Returns:
        str: Complete TOML file content ready to write to disk.

    """
    lines = ["# psoul configuration", ""]
    for section_name, comments in _iter_config_comments():
        lines.append(f"[{section_name}]")
        lines.extend(f"# {c}" for c in comments)
        lines.append("")
    return "\n".join(lines)


def build_pyproject_psoul_table() -> Table:
    """Build a ``[tool.psoul]`` table with commented-out default config.

    Returns a tomlkit ``Table`` with one sub-table per config section,
    each containing commented-out default values.  Uses the same field
    iteration as ``generate_config()`` so examples stay in sync.
    """
    psoul = tomlkit.table()
    for section_name, comments in _iter_config_comments():
        sub = tomlkit.table()
        for comment in comments:
            sub.add(tomlkit.comment(comment))
        psoul.add(section_name, sub)
    return psoul


def inject_pyproject_config(path: Path) -> None:
    """Inject a ``[tool.psoul]`` section into an existing pyproject.toml.

    Reads with ``TOMLFile`` for round-trip preservation of comments,
    formatting, and ordering.  Writes atomically via a same-directory
    temp file and ``os.replace()`` to prevent partial writes.  This does
    **not** protect against concurrent edits between read and replace.

    The target file itself must be writable.  This helper checks that
    explicitly before the temp-file replace path, since ``os.replace()``
    operates on the directory entry and would otherwise bypass a
    read-only file mode in a writable directory.

    Args:
        path (Path): Path to an existing pyproject.toml.

    Raises:
        FileNotFoundError: *path* does not exist.
        tomlkit.exceptions.TOMLKitError: *path* contains invalid TOML.
        TypeError: ``[tool]`` is not a TOML table (e.g. scalar, array,
            or inline table — none can hold ``[tool.psoul]`` sub-tables).
        ValueError: ``[tool.psoul]`` already exists in the document.
        PermissionError: Target file or parent directory is not writable.
        OSError: Other I/O failure.

    Example:
        >>> inject_pyproject_config(Path("pyproject.toml"))  # adds [tool.psoul]

    """
    with path.open(encoding="utf-8", newline="") as fh:
        raw = fh.read()
    linesep = _detect_line_ending(raw)

    doc = TOMLFile(str(path)).read()
    with path.open("r+b"):
        pass

    if "tool" not in doc:
        doc["tool"] = {}
    tool = doc["tool"]
    if not isinstance(tool, Table):
        msg = f"[tool] is not a table in {path}; cannot inject [tool.psoul] sub-tables"
        raise TypeError(msg)
    if "psoul" in tool:
        msg = f"[tool.psoul] already exists in {path}"
        raise ValueError(msg)
    tool["psoul"] = build_pyproject_psoul_table()

    content = _apply_line_ending(doc.as_string(), linesep)

    fd, tmp_str = tempfile.mkstemp(dir=path.parent, suffix=".toml.tmp")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
