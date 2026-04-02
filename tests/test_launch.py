"""Tests for the launch module: input validation, request assembly, and frozen containers."""

import sys

import pytest

from psoul.launch import (
    LaunchTarget,
    build_launch_request,
    parse_launch_target,
    resolve_session_id,
)
from psoul.session import LaunchMode, TargetType


@pytest.mark.parametrize(
    ("target", "module", "extra", "expected_type", "expected_target"),
    [
        ("app.py", None, ["--port", "8000"], TargetType.script, "app.py"),
        (None, "http.server", ["8000"], TargetType.module, "http.server"),
    ],
)
def test_parse_launch_target(
    target: str | None,
    module: str | None,
    extra: list[str],
    expected_type: TargetType,
    expected_target: str,
) -> None:
    t = parse_launch_target(target=target, module=module, extra_args=extra)
    assert t.target_type == expected_type
    assert t.target == expected_target
    assert t.target_args == tuple(extra)


@pytest.mark.parametrize(
    ("target", "module", "match"),
    [
        ("app.py", "http.server", "choose either"),
        (None, None, "launch target is required"),
    ],
)
def test_parse_launch_target_rejects_bad_input(target: str | None, module: str | None, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_launch_target(target=target, module=module, extra_args=[])


def test_resolve_session_id_explicit() -> None:
    assert resolve_session_id("my-session") == "my-session"


def test_resolve_session_id_invalid() -> None:
    with pytest.raises(ValueError, match="lowercase alphanumeric"):
        resolve_session_id("BAD NAME!")


def test_resolve_session_id_generates_four_words() -> None:
    assert resolve_session_id(None).count("-") == 3


def test_as_cmd_script() -> None:
    t = LaunchTarget(TargetType.script, "app.py", ("--flag",))
    assert t.as_cmd() == [sys.executable, "app.py", "--flag"]


def test_as_cmd_module() -> None:
    t = LaunchTarget(TargetType.module, "http.server", ("8000",))
    assert t.as_cmd() == [sys.executable, "-m", "http.server", "8000"]


def test_build_launch_request_freezes_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.launch.sys.stdin.isatty", lambda: False)
    tags = {"env": "dev"}
    req = build_launch_request(target="x.py", module=None, extra_args=[], name="a-b-c-d", headless=False, tags=tags)
    tags["env"] = "prod"
    assert req.tags is not None
    assert req.tags["env"] == "dev"


def test_build_headless_when_no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.launch.sys.stdin.isatty", lambda: False)
    req = build_launch_request(target="x.py", module=None, extra_args=[], name="a-b-c-d", headless=False, tags=None)
    assert req.launch_mode == LaunchMode.headless


def test_build_headless_flag_overrides_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.launch.sys.stdin.isatty", lambda: True)
    req = build_launch_request(target="x.py", module=None, extra_args=[], name="a-b-c-d", headless=True, tags=None)
    assert req.launch_mode == LaunchMode.headless
