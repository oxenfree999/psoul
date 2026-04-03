"""REPL CLI layer: prompt_toolkit session, history, completion, and validation."""

import builtins
import keyword
import os
import re
import sqlite3
import sys
import traceback
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.history import History, ThreadedHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.validation import ValidationError, Validator
from pygments.lexers import PythonLexer  # ty: ignore[unresolved-import]
from rich.console import Console

from psoul.repl import ReplEngine
from psoul.session import LaunchMode, Session, SessionState, TargetType
from psoul.store import SessionStore
from psoul.version import VERSION

_ATTR_PATTERN = re.compile(r"([\w.]+)\.([\w]*)$")
_BUILTIN_NAMES: list[str] = dir(builtins)


def _sort_key(name: str) -> tuple[int, str]:
    """Sort dunder names last, private names second-to-last."""
    if name.startswith("__"):
        return (2, name)
    if name.startswith("_"):
        return (1, name)
    return (0, name)


class SqliteHistory(History):
    """prompt_toolkit History adapter backed by psoul's history table.

    Each method opens a short-lived connection rather than holding one,
    because ThreadedHistory calls load_history_strings() on a background
    thread and store_string() on the main thread — a shared connection
    would trip SQLite's thread-affinity check.
    """

    def __init__(self, db_path: Path, session_id: str | None = None) -> None:
        """Store the database path for on-demand connections."""
        super().__init__()
        self._db_path = db_path
        self._session_id = session_id

    def load_history_strings(self) -> list[str]:
        """Load all history entries, newest first, deduplicated."""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute("SELECT input FROM history ORDER BY history_id DESC").fetchall()
        finally:
            conn.close()
        seen: set[str] = set()
        result: list[str] = []
        for (text,) in rows:
            if text not in seen:
                seen.add(text)
                result.append(text)
        return result

    def store_string(self, string: str) -> None:
        """Persist a history entry tagged with the current session."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO history (session_id, timestamp, input) VALUES (?, ?, ?)",
                (self._session_id, datetime.now(UTC).isoformat(), string),
            )
            conn.commit()
        finally:
            conn.close()


class PythonCompleter(Completer):
    """Tab completion from a live REPL namespace.

    Handles two cases:
    - Attribute access (e.g. ``foo.ba``) — eval the object, complete from dir().
    - Bare names — union of namespace keys, builtins, and keyword.kwlist.

    Following ptpython's pattern: dunder names sort to the end.
    """

    def __init__(self, namespace: dict[str, object]) -> None:
        """Bind to a live REPL namespace dict for completions."""
        self._namespace = namespace

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:  # noqa: ARG002
        """Yield completions for the current cursor position."""
        text = document.text_before_cursor
        attr_match = _ATTR_PATTERN.search(text)
        if attr_match:
            yield from self._complete_attributes(attr_match)
        else:
            yield from self._complete_names(document)

    def _complete_attributes(self, match: re.Match[str]) -> Iterable[Completion]:
        """Complete attributes on a dotted expression (e.g. ``os.path.jo``)."""
        expr, partial = match.groups()
        try:
            obj = eval(expr, self._namespace)  # noqa: S307
        except Exception:  # noqa: BLE001
            return
        names = sorted(dir(obj), key=_sort_key)
        for name in names:
            if name.startswith(partial):
                yield Completion(name, start_position=-len(partial))

    def _complete_names(self, document: Document) -> Iterable[Completion]:
        """Complete bare names from namespace + builtins + keywords."""
        word = document.get_word_before_cursor()
        if not word:
            return
        candidates = sorted(
            set(self._namespace.keys()) | set(_BUILTIN_NAMES) | set(keyword.kwlist),
            key=_sort_key,
        )
        for name in candidates:
            if name.startswith(word):
                yield Completion(name, start_position=-len(word))


class PythonValidator(Validator):
    """Reject incomplete Python input so prompt_toolkit continues multiline editing."""

    def __init__(self, engine: ReplEngine) -> None:
        """Bind to an engine for completeness checks."""
        self._engine = engine

    def validate(self, document: Document) -> None:
        """Raise ValidationError if the input is incomplete."""
        if self._engine.is_complete(document.text) is False:
            raise ValidationError(message="")


def run_repl(session_id: str, conn: sqlite3.Connection, db_path: Path) -> None:
    """Run an interactive REPL session with prompt_toolkit."""
    store = SessionStore(conn)
    session = Session(
        session_id=session_id,
        state=SessionState.starting,
        launch_mode=LaunchMode.attached,
        launch_time=datetime.now(UTC),
        psoul_version=VERSION,
        target_type=TargetType.repl,
    )
    store.create(session)
    store.update(session_id, state=SessionState.running, supervisor_pid=os.getpid())

    engine = ReplEngine()
    history = SqliteHistory(db_path, session_id=session_id)
    console = Console()
    failed = False

    try:
        prompt = PromptSession(
            history=ThreadedHistory(history),
            lexer=PygmentsLexer(PythonLexer),
            completer=PythonCompleter(engine.namespace),
            validator=PythonValidator(engine),
            multiline=True,
            prompt_continuation="... ",
            editing_mode=EditingMode.EMACS,
        )

        while True:
            try:
                text = prompt.prompt(">>> ")
            except KeyboardInterrupt:
                continue
            except EOFError:
                break

            result = engine.execute(text)

            if result.quit:
                break
            if result.output:
                console.print(result.output)
            if result.exception is not None:
                traceback.print_exception(result.exception, file=sys.stderr)
            elif result.has_value and result.value is not None:
                console.print(repr(result.value))
    except Exception:
        failed = True
        raise
    finally:
        store.update(session_id, state=SessionState.stopping)
        store.update(
            session_id,
            state=SessionState.failed if failed else SessionState.exited,
        )
