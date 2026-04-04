"""Tests for tag parsing, CLI wiring, and filtering."""

import pytest
import typer

from psoul.cli.main import parse_tags


class TestParseTags:
    """Unit tests for the parse_tags() CLI helper."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ([], None),
            (["env=dev"], {"env": "dev"}),
            (["env=dev", "team=backend"], {"env": "dev", "team": "backend"}),
            (["query=x=1&y=2"], {"query": "x=1&y=2"}),
            (["marker="], {"marker": ""}),
            (["env=dev", "env=prod"], {"env": "prod"}),
            ([" env = dev "], {"env": "dev"}),
            (["note=hello world"], {"note": "hello world"}),
            (["flag=   "], {"flag": ""}),
        ],
        ids=[
            "none",
            "empty-list",
            "single-tag",
            "multiple-tags",
            "value-containing-equals",
            "empty-value-allowed",
            "duplicate-key-last-wins",
            "strips-whitespace",
            "value-with-interior-spaces",
            "whitespace-only-value-becomes-empty",
        ],
    )
    def test_valid_input(self, raw: list[str] | None, expected: dict[str, str] | None) -> None:
        assert parse_tags(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "match"),
        [
            (["noequals"], "expected key=value"),
            (["=value"], "empty key"),
            (["   =value"], "empty key"),
        ],
        ids=["missing-equals", "empty-key", "whitespace-only-key"],
    )
    def test_invalid_input(self, raw: list[str], match: str) -> None:
        with pytest.raises(typer.BadParameter, match=match):
            parse_tags(raw)
