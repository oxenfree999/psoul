"""Global CLI state passed to all subcommands via ctx.obj."""

import os
import sys
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path


class ExitCode(IntEnum):
    """Process exit codes."""

    SUCCESS = 0
    ERROR = 1
    USAGE = 2


class ColorMode(StrEnum):
    """Terminal color output mode."""

    auto = "auto"
    always = "always"
    never = "never"


class OutputFormat(StrEnum):
    """Output serialization format."""

    text = "text"
    json = "json"


def resolve_color(mode: ColorMode) -> bool:
    """Decide whether to emit color based on ``--color`` flag and environment.

    For ``auto`` mode, respects ``NO_COLOR``, ``FORCE_COLOR``, TTY detection,
    and ``TERM=dumb``.

    Args:
        mode (ColorMode): The ``--color`` flag value.

    Returns:
        bool: ``True`` if color output should be enabled.

    """
    if mode is ColorMode.always:
        return True
    if mode is ColorMode.never:
        return False
    if os.environ.get("NO_COLOR", "") != "":
        return False
    if os.environ.get("FORCE_COLOR", "") != "":
        return True
    return sys.stdout.isatty() and os.environ.get("TERM", "").lower() != "dumb"


@dataclass(frozen=True, slots=True)
class GlobalState:
    """Immutable state resolved from global CLI flags."""

    verbose: int
    quiet: bool
    color: ColorMode
    color_enabled: bool
    log_level: int
    config_override: Path | None
