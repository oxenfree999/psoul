"""Tests for the REPL engine: completeness detection, eval/exec, and commands."""

import pytest

from psoul.core.repl import ReplEngine


@pytest.fixture
def engine() -> ReplEngine:
    return ReplEngine()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1 + 1", True),
        ("x = 42", True),
        ("def f():\n    return 1\n", True),
        ("def f():", False),
        ("if True:", False),
        ("print(", False),
        ('x = """hello', False),
        ("def", None),
    ],
    ids=[
        "expression",
        "assignment",
        "multiline-complete",
        "incomplete-fn",
        "incomplete-if",
        "unclosed-paren",
        "unclosed-string",
        "syntax-error",
    ],
)
def test_is_complete(engine: ReplEngine, source: str, expected: bool | None) -> None:
    assert engine.is_complete(source) is expected


def test_execute_expression_tracks_result_history(engine: ReplEngine) -> None:
    assert engine.execute("__name__").value == "__main__"
    assert engine.execute("1 + 1").value == 2
    assert engine.execute("2 + 2").value == 4
    assert engine.namespace["_"] == 4
    assert engine.namespace["__"] == 2
    assert engine.namespace["___"] == "__main__"


def test_execute_statement_updates_namespace(engine: ReplEngine) -> None:
    result = engine.execute("x = 42")
    assert result.has_value is False
    assert result.exception is None
    assert engine.namespace["x"] == 42


def test_execute_captures_runtime_exceptions(engine: ReplEngine) -> None:
    result = engine.execute("1 / 0")
    assert isinstance(result.exception, ZeroDivisionError)


@pytest.mark.parametrize(
    ("source", "output", "quit_expected"),
    [
        (":help", "Psoul REPL", False),
        (":clear", "\033[2J\033[H", False),
        (":quit", "", True),
        (":wat", "Unknown command: :wat", False),
    ],
)
def test_execute_commands(engine: ReplEngine, source: str, output: str, quit_expected: bool) -> None:
    result = engine.execute(source)
    assert output in result.output
    assert result.quit is quit_expected
