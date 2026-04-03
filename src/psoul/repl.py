"""In-process REPL engine: eval/exec dispatch, namespace, and commands."""

import codeop
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Result of executing input in the REPL engine.

    value / has_value: expression result (has_value distinguishes None from no result).
    exception: the exception if execution raised.
    output: text produced by :commands.
    quit: signals the REPL loop should exit.
    """

    value: object = None
    has_value: bool = False
    exception: BaseException | None = None
    output: str = ""
    quit: bool = False


class ReplEngine:
    """REPL engine: namespace management, eval/exec dispatch, and :commands."""

    def __init__(self) -> None:
        """Initialize with a fresh namespace and command registry."""
        self._namespace: dict[str, object] = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "__doc__": None,
            "__package__": None,
            "__loader__": None,
            "__spec__": None,
        }
        self._compiler = codeop.CommandCompiler()
        self._commands = {
            "quit": self._cmd_quit,
            "clear": self._cmd_clear,
            "help": self._cmd_help,
        }

    @property
    def namespace(self) -> dict[str, object]:
        """The live REPL namespace dict."""
        return self._namespace

    def is_complete(self, source: str) -> bool | None:
        """Check whether source is a complete Python statement.

        Returns True if the source compiles successfully (ready to execute),
        False if incomplete (needs more input lines), or None if the source
        has a definite syntax error.

        Uses codeop.CommandCompiler which handles incomplete-input detection
        without brittle SyntaxError message matching.  On Python 3.13+ it
        uses PyCF_ALLOW_INCOMPLETE_INPUT; on 3.12 it uses the stdlib's
        repr-comparison fallback.
        """
        try:
            code = self._compiler(source, "<input>", "single")
        except (SyntaxError, OverflowError, ValueError):
            return None
        return code is not None

    def execute(self, source: str) -> ExecResult:
        """Execute source in the REPL namespace and return the result.

        :commands are dispatched before eval/exec.  For Python code,
        eval mode is attempted first so expression results are captured.
        The exec fallback lives outside the except block so that
        sys.exc_info() reports the correct error if exec itself raises.
        """
        stripped = source.strip()
        if stripped.startswith(":"):
            return self._dispatch_command(stripped)

        try:
            return self._eval_or_exec(source)
        except KeyboardInterrupt:
            return ExecResult(exception=KeyboardInterrupt())
        except Exception as exc:  # noqa: BLE001
            return ExecResult(exception=exc)

    def _eval_or_exec(self, source: str) -> ExecResult:
        """Try eval first for the return value, fall back to exec."""
        is_eval = True
        try:
            code = compile(source, "<input>", "eval")
        except SyntaxError:
            is_eval = False

        if is_eval:
            result = eval(code, self._namespace)  # noqa: S307
            self._store_result(result)
            return ExecResult(value=result, has_value=True)

        code = compile(source, "<input>", "exec")
        exec(code, self._namespace)  # noqa: S102
        return ExecResult()

    def _store_result(self, value: object) -> None:
        """Track the last three expression results as _, __, ___."""
        self._namespace["___"] = self._namespace.get("__")
        self._namespace["__"] = self._namespace.get("_")
        self._namespace["_"] = value

    def _dispatch_command(self, text: str) -> ExecResult:
        """Parse and run a :command, or report that it's unknown."""
        name, _, args = text[1:].strip().partition(" ")
        handler = self._commands.get(name)
        if handler is None:
            return ExecResult(output=f"Unknown command: :{name}")
        return handler(args.strip())

    def _cmd_quit(self, _args: str) -> ExecResult:
        return ExecResult(quit=True)

    def _cmd_clear(self, _args: str) -> ExecResult:
        return ExecResult(output="\033[2J\033[H")

    def _cmd_help(self, _args: str) -> ExecResult:
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        lines = [
            f"Psoul REPL (Python {version})",
            "",
            "Commands:",
            "  :help   Show this help",
            "  :clear  Clear the screen",
            "  :quit   Exit the REPL",
            "",
            "Keyboard:",
            "  Ctrl+D  Exit (on empty line)",
        ]
        return ExecResult(output="\n".join(lines))
