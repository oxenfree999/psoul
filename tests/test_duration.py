"""Tests for duration string parsing."""

from datetime import timedelta

import pytest

from psoul.core.duration import parse_duration


@pytest.mark.parametrize(
    ("input_string", "expected"),
    [
        ("30s", timedelta(seconds=30)),
        ("5m", timedelta(minutes=5)),
        ("2h", timedelta(hours=2)),
        ("1d", timedelta(days=1)),
        ("3w", timedelta(weeks=3)),
        ("1h30m", timedelta(hours=1, minutes=30)),
        ("2d5h30m", timedelta(days=2, hours=5, minutes=30)),
        ("1.5h", timedelta(hours=1.5)),
        ("0s", timedelta(seconds=0)),
        ("10000s", timedelta(seconds=10000)),
    ],
)
def test_parse_duration_valid(input_string: str, expected: timedelta) -> None:
    assert parse_duration(input_string) == expected


@pytest.mark.parametrize("input_string", ["", "abc", "10", "10x", "h5"])
def test_parse_duration_invalid(input_string: str) -> None:
    with pytest.raises(ValueError, match=f"invalid duration: {input_string!r}"):
        parse_duration(input_string)


def test_parse_duration_equivalence() -> None:
    assert parse_duration("60s") == parse_duration("1m")
    assert parse_duration("60m") == parse_duration("1h")
    assert parse_duration("24h") == parse_duration("1d")
    assert parse_duration("7d") == parse_duration("1w")
    assert parse_duration("1h30m") == parse_duration("90m")
