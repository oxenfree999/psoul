"""Parse human-readable duration strings like '10s' or '1h30m' into timedelta."""

import re
from datetime import timedelta

_UNITS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}

_PATTERN = re.compile(r"(\d+\.?\d*)([smhdw])")


def parse_duration(s: str) -> timedelta:
    """Convert a duration string like ``'10s'`` or ``'1h30m'`` into a timedelta.

    Supports ``s`` (seconds), ``m`` (minutes), ``h`` (hours), ``d`` (days),
    and ``w`` (weeks).  Multiple units can be combined: ``'1h30m'`` becomes
    90 minutes.  Fractional amounts like ``'1.5h'`` are allowed.

    Args:
        s (str): Duration string to parse.

    Returns:
        timedelta: The parsed duration.

    Raises:
        ValueError: If *s* contains no recognised unit segments.

    Examples:
        >>> parse_duration("10s")
        datetime.timedelta(seconds=10)
        >>> parse_duration("1h30m")
        datetime.timedelta(seconds=5400)

    """
    matches = _PATTERN.findall(s)
    if not matches:
        msg = f"invalid duration: {s!r}"
        raise ValueError(msg)
    kwargs = {_UNITS[unit]: float(value) for value, unit in matches}
    return timedelta(**kwargs)
