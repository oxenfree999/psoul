"""REPL CLI layer: prompt_toolkit session, history, completion, and validation."""

import builtins
import keyword
import os
import re
import sqlite3
import sys
import time
import traceback
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import Condition, emacs_insert_mode, has_focus, vi_insert_mode
from prompt_toolkit.history import History, ThreadedHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.validation import ValidationError, Validator
from pygments.lexers import PythonLexer  # ty: ignore[unresolved-import]
from rich.console import Console

from psoul.provenance import gather
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


def _auto_newline(buf: Buffer) -> None:
    """Insert a newline with Python-aware indentation.

    Copies leading whitespace from the current line and adds four extra
    spaces after lines ending with a colon — following ptpython's
    auto_newline pattern.
    """
    current_line = buf.document.current_line_before_cursor.rstrip()
    buf.insert_text("\n")
    for c in current_line:
        if c.isspace():
            buf.insert_text(c)
        else:
            break
    if current_line.endswith(":"):
        buf.insert_text("    ")


def _at_end(buf: Buffer) -> bool:
    """Check whether the cursor is at the effective end of the buffer."""
    text_after = buf.document.text_after_cursor
    return not text_after or (text_after.isspace() and "\n" not in text_after)


def _enter_handler(event: KeyPressEvent, engine: ReplEngine) -> None:
    """Handle Enter: accept completion, auto-indent, or submit."""
    buf = event.current_buffer

    if buf.complete_state:
        if buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
        buf.complete_state = None
        return

    at_end = _at_end(buf)

    # On a blank/whitespace-only line at the end, try submitting
    # the rstripped text.  codeop needs a trailing newline (not
    # indentation) to recognise a complete block, so we strip
    # trailing whitespace and append "\n" for the check.
    current_line = buf.document.current_line_before_cursor
    if at_end and (not current_line or current_line.isspace()):
        stripped = buf.text.rstrip()
        if stripped and engine.is_complete(stripped + "\n") is not False:
            submit_text = stripped + "\n"
            buf.document = Document(text=submit_text, cursor_position=len(submit_text))
            buf.validate_and_handle()
            return

    status = engine.is_complete(buf.text)

    if status is False:
        _auto_newline(buf)
    elif at_end:
        buf.validate_and_handle()
    else:
        buf.insert_text("\n")


def _repl_key_bindings(engine: ReplEngine, completer: PythonCompleter) -> KeyBindings:
    """Key bindings for the REPL prompt.

    Tab: insert 4 spaces on whitespace-only lines, otherwise complete.
    Single-match Tab auto-applies; multiple matches open the menu.

    Enter: accept completion if the menu is active; auto-indent if
    incomplete; submit if complete and cursor is at the end; otherwise
    insert a plain newline.
    """
    bindings = KeyBindings()

    @Condition
    def _tab_should_indent() -> bool:
        buf = get_app().current_buffer
        if buf.complete_state:
            return False
        before = buf.document.current_line_before_cursor
        return bool(buf.text and (not before or before.isspace()))

    @bindings.add("tab", filter=has_focus(DEFAULT_BUFFER) & _tab_should_indent)
    def _handle_tab_indent(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("    ")

    @bindings.add("tab", filter=has_focus(DEFAULT_BUFFER) & ~_tab_should_indent)
    def _handle_tab_complete(event: KeyPressEvent) -> None:
        buf = event.current_buffer
        if buf.complete_state:
            buf.complete_next()
            return
        completions = list(completer.get_completions(buf.document, CompleteEvent()))
        if len(completions) == 1:
            buf.apply_completion(completions[0])
        elif completions:
            buf.start_completion(select_first=True)

    @bindings.add(
        "enter",
        filter=has_focus(DEFAULT_BUFFER) & (vi_insert_mode | emacs_insert_mode),
    )
    def _handle_enter(event: KeyPressEvent) -> None:
        _enter_handler(event, engine)

    return bindings


def run_repl(session_id: str, conn: sqlite3.Connection, db_path: Path) -> None:
    """Run an interactive REPL session with prompt_toolkit."""
    store = SessionStore(conn)
    provenance = gather(TargetType.repl, None, Path.cwd())
    session = Session(
        session_id=session_id,
        state=SessionState.starting,
        launch_mode=LaunchMode.attached,
        launch_time=datetime.now(UTC),
        psoul_version=VERSION,
        target_type=TargetType.repl,
        **provenance,
    )
    store.create(session)
    store.update(session_id, state=SessionState.running, supervisor_pid=os.getpid())

    engine = ReplEngine()
    history = SqliteHistory(db_path, session_id=session_id)
    console = Console()
    failed = False
    start = time.monotonic()

    try:
        completer = PythonCompleter(engine.namespace)
        prompt = PromptSession(
            history=ThreadedHistory(history),
            lexer=PygmentsLexer(PythonLexer),
            completer=completer,
            validator=PythonValidator(engine),
            key_bindings=_repl_key_bindings(engine, completer),
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
        duration = time.monotonic() - start
        final_state = SessionState.failed if failed else SessionState.exited
        try:
            store.record_result(
                session_id=session_id,
                outcome="failed" if failed else "exited",
                exit_code=None,
                end_time=datetime.now(UTC),
                duration_seconds=duration,
            )
        finally:
            store.update(session_id, state=SessionState.stopping)
            store.update(session_id, state=final_state)
