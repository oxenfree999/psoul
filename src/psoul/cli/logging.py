"""Structlog-based internal diagnostics for psoul."""

import logging
import os
import sys

import structlog

from psoul.cli.state import OutputFormat

_LEVEL_NAMES: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


_VERBOSE_INFO = 1
_VERBOSE_DEBUG = 2


def resolve_log_level(verbose: int, quiet: bool) -> int:
    """Return the effective log level.

    Precedence: ``-q`` → ``-v``/``-vv`` → ``PSOUL_LOG`` env var → WARNING.

    Args:
        verbose (int): Count of ``-v`` flags (0, 1, or 2+).
        quiet (bool): Whether ``-q`` was passed.

    Returns:
        int: A standard ``logging`` log level.

    """
    if quiet:
        return logging.ERROR
    if verbose >= _VERBOSE_DEBUG:
        return logging.DEBUG
    if verbose == _VERBOSE_INFO:
        return logging.INFO

    return _LEVEL_NAMES.get(os.environ.get("PSOUL_LOG", "").strip().lower(), logging.WARNING)


def configure_logging(level: int, output_format: OutputFormat) -> None:
    """Configure structlog for psoul's diagnostic output.

    All diagnostic logs go to stderr.  Uses ``ConsoleRenderer`` for text
    and ``JSONRenderer`` for JSON output.

    Args:
        level (int): Log level from ``resolve_log_level``.
        output_format (OutputFormat): Controls which renderer is used.

    """
    renderer = (
        structlog.processors.JSONRenderer() if output_format is OutputFormat.json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[structlog.processors.add_log_level, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
