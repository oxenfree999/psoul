"""Configuration schema, platform directory resolution, and config generation."""

import dataclasses
import sys
import tomllib
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from platformdirs import PlatformDirs
from platformdirs.unix import Unix

from psoul.duration import parse_duration

APP_NAME = "psoul"


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

    # Path coercion: TOML has no path type, accept strings
    if base is Path:
        if isinstance(value, str):
            return Path(value)
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


_DIRS = Unix(APP_NAME) if sys.platform == "darwin" else PlatformDirs(APP_NAME)


def default_config_dir() -> Path:
    """Return the user config directory for psoul (~/.config/psoul on Unix)."""
    return _DIRS.user_config_path


def default_state_dir() -> Path:
    """Return the user state directory for psoul (~/.local/state/psoul on Unix)."""
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

    python_path (Path | None): overrides uv/PATH discovery with an explicit binary. Default: None.
    """

    python_path: Path | None = field(
        default=None,
        metadata={"description": "override Python binary, bypasses uv discovery", "example": "/usr/bin/python3"},
    )


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    """[launch] section.

    mode (str): 'attached' or 'headless'. Default: 'attached'.
    """

    mode: str = field(default="attached", metadata={"description": "attached (default) or headless"})


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
    """

    tags: dict[str, str] | None = field(
        default=None,
        metadata={
            "description": "default key-value tags applied to all sessions",
            "example": {"env": "dev", "team": "platform"},
        },
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


def find_config_file(override: Path | None = None) -> Path | None:
    """Discover the config file to use, following precedence order.

    override (Path | None): explicit --config flag. Raises FileNotFoundError if set but missing.

    Returns the first existing config file, or None if no config found.
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


_SECTION_CLASSES: dict[str, type] = {
    "paths": PathsConfig,
    "python": PythonConfig,
    "launch": LaunchConfig,
    "process": ProcessConfig,
    "session": SessionConfig,
    "output": OutputConfig,
    "retention": RetentionConfig,
}


def load_config(path: Path | None = None) -> PsoulConfig:
    """Load configuration from a TOML file into a PsoulConfig.

    path (Path | None): file path from find_config_file(). None means all defaults.
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


def generate_config() -> str:
    """Generate a commented psoul.toml by introspecting PsoulConfig fields.

    Every value is commented out so the file uses all defaults.
    Descriptions come from field metadata, not a separate data structure.
    """
    defaults = PsoulConfig()
    lines = ["# psoul configuration", ""]

    for section_field in dataclasses.fields(defaults):
        section = getattr(defaults, section_field.name)
        lines.append(f"[{section_field.name}]")
        for f in dataclasses.fields(section):
            desc = f.metadata.get("description", "")
            value = f.metadata["example"] if getattr(section, f.name) is None else getattr(section, f.name)
            comment = f"  # {desc}" if desc else ""
            lines.append(f"# {f.name} = {_format_toml_value(value)}{comment}")
        lines.append("")

    return "\n".join(lines)
