"""Tests for the REPL CLI layer: history, completer, and validator."""

import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.validation import ValidationError
from typer.testing import CliRunner

from psoul.cli.main import cli
from psoul.cli.repl import (
    PythonCompleter,
    PythonValidator,
    SqliteHistory,
    _repl_key_bindings,
    run_repl,
)
from psoul.db import open_db
from psoul.repl import ReplEngine
from psoul.session import LaunchMode, Session, SessionState, TargetType
from psoul.store import SessionStore
from psoul.version import VERSION

runner = CliRunner()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a psoul database and return its path."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    conn = open_db(state_dir)
    conn.close()
    return state_dir / "psoul.db"


class TestSqliteHistory:
    @pytest.mark.parametrize(
        ("inputs", "expected"),
        [
            (["first", "second", "third"], ["third", "second", "first"]),
            (["alpha", "beta", "alpha"], ["alpha", "beta"]),
            ([], []),
        ],
        ids=["newest-first", "dedup", "empty"],
    )
    def test_load_order(self, db_path: Path, inputs: list[str], expected: list[str]) -> None:
        history = SqliteHistory(db_path, session_id="test-session")
        for s in inputs:
            history.store_string(s)
        assert history.load_history_strings() == expected

    def test_session_id_stored(self, db_path: Path) -> None:
        """History entries are tagged with the session ID."""
        history = SqliteHistory(db_path, session_id="my-session")
        history.store_string("x = 1")
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT session_id FROM history WHERE input = 'x = 1'").fetchone()
        conn.close()
        assert row[0] == "my-session"

    def test_load_from_background_thread(self, db_path: Path) -> None:
        """load_history_strings() works from a non-main thread (ThreadedHistory path)."""
        history = SqliteHistory(db_path, session_id="test-session")
        history.store_string("hello")
        result: list[str] = []
        error: list[BaseException] = []

        def load() -> None:
            try:
                result.extend(history.load_history_strings())
            except Exception as exc:  # noqa: BLE001
                error.append(exc)

        t = threading.Thread(target=load)
        t.start()
        t.join()
        assert not error, f"background load failed: {error[0]}"
        assert result == ["hello"]


def _complete(namespace: dict[str, object], text: str) -> list[str]:
    """Return completion texts for the given input."""
    doc = Document(text, len(text))
    completer = PythonCompleter(namespace)
    return [c.text for c in completer.get_completions(doc, CompleteEvent())]


class TestPythonCompleter:
    @pytest.mark.parametrize(
        ("namespace", "text", "expected"),
        [
            ({"foo_bar": 1, "foo_baz": 2}, "foo_", ["foo_bar", "foo_baz"]),
            ({}, "imp", "import"),
            ({}, "pri", "print"),
            ({"os": os}, "os.getc", "getcwd"),
        ],
        ids=["namespace", "keyword", "builtin", "attribute"],
    )
    def test_completions(self, namespace: dict[str, object], text: str, expected: list[str] | str) -> None:
        results = _complete(namespace, text)
        if isinstance(expected, list):
            assert results == expected
        else:
            assert expected in results

    def test_no_completions_on_empty(self) -> None:
        assert _complete({"x": 1}, "") == []


class TestPythonValidator:
    @pytest.mark.parametrize(
        ("text", "should_raise"),
        [
            ("def f():", True),
            ("1 + 1", False),
            ("def", False),
        ],
        ids=["incomplete", "complete", "syntax-error"],
    )
    def test_validate(self, text: str, should_raise: bool) -> None:
        validator = PythonValidator(ReplEngine())
        doc = Document(text, len(text))
        if should_raise:
            with pytest.raises(ValidationError):
                validator.validate(doc)
        else:
            validator.validate(doc)


def _seed_session(tmp_path: Path, session_id: str) -> Path:
    """Create a DB with one exited session and return a config file pointing to it."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    conn = open_db(state_dir)
    SessionStore(conn).create(
        Session(
            session_id=session_id,
            state=SessionState.exited,
            launch_mode=LaunchMode.attached,
            launch_time=datetime.now(UTC),
            psoul_version=VERSION,
            target_type=TargetType.repl,
        )
    )
    conn.close()
    config = tmp_path / "psoul.toml"
    config.write_text(f"[paths]\nstate_dir = '{state_dir}'\n")
    return config


class TestReplCLI:
    def test_bare_psoul_launches_repl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        config = tmp_path / "psoul.toml"
        config.write_text(f"[paths]\nstate_dir = '{state_dir}'\n")
        calls: list[tuple[str, Path]] = []
        monkeypatch.setattr("psoul.cli.main.resolve_session_id", lambda _name: "bare-repl")
        monkeypatch.setattr(
            "psoul.cli.main.run_repl", lambda session_id, conn, db_path: calls.append((session_id, db_path))
        )
        result = runner.invoke(cli, ["--config", str(config)])
        assert result.exit_code == 0
        assert calls == [("bare-repl", state_dir / "psoul.db")]

    def test_invalid_name_rejected(self) -> None:
        result = runner.invoke(cli, ["repl", "--name", "UPPER_CASE"])
        assert result.exit_code != 0
        assert "Error:" in result.output

    def test_duplicate_name_rejected(self, tmp_path: Path) -> None:
        config = _seed_session(tmp_path, "taken")
        result = runner.invoke(cli, ["--config", str(config), "repl", "--name", "taken"])
        assert result.exit_code != 0
        assert "already exists" in result.output


def _prompt_with_keys(keys: str) -> str:
    """Feed keystrokes into a real PromptSession with our key bindings."""
    engine = ReplEngine()
    completer = PythonCompleter(engine.namespace)
    with create_pipe_input() as inp:
        inp.send_text(keys)
        session = PromptSession(
            input=inp,
            output=DummyOutput(),
            key_bindings=_repl_key_bindings(engine, completer),
            completer=completer,
            validator=PythonValidator(engine),
            multiline=True,
            editing_mode=EditingMode.EMACS,
        )
        return session.prompt(">>> ")


class TestKeyBindings:
    @pytest.mark.parametrize(
        ("keys", "expected"),
        [
            ("x=5\r", "x=5"),
            ("1+1\r", "1+1"),
            ("def f():\rreturn 1\r\r", "def f():\n    return 1\n"),
            ("def f():\r\treturn 1\r\r", "def f():\n        return 1\n"),
            ("pri\t\r", "print"),
            ("x=5\x02\x02\r\x1b>\r", "x\n=5"),
        ],
        ids=[
            "submit-assignment",
            "submit-expression",
            "auto-indent-after-colon",
            "tab-indents-whitespace-line",
            "tab-completes-single-match",
            "enter-mid-buffer-inserts-newline",
        ],
    )
    def test_key_bindings(self, keys: str, expected: str) -> None:
        assert _prompt_with_keys(keys) == expected


def test_run_repl_exits_cleanly_and_records_supervisor_pid(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def prompt(self, _message: str) -> str:
            raise EOFError

    monkeypatch.setattr("psoul.cli.repl.PromptSession", FakePromptSession)
    conn = open_db(db_path.parent)
    try:
        run_repl("repl-session", conn, db_path)
        session = SessionStore(conn).get("repl-session")
        assert session is not None
        assert session.state is SessionState.exited
        assert session.supervisor_pid == os.getpid()
    finally:
        conn.close()
