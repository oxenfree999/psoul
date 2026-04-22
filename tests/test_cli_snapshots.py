"""Snapshot tests for CLI help and version output.

These tests lock the exact text of every help screen and the version
string.  When the CLI surface changes intentionally, run:

    just snap

to accept the new output.
"""

import typer
from inline_snapshot import snapshot
from typer.testing import CliRunner

from psoul.cli.main import cli

runner = CliRunner()


def test_main_help() -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul [OPTIONS] COMMAND [ARGS]...                                       \n\
                                                                                \n\
 A CLI-first foundation for managed Python sessions.                            \n\
                                                                                \n\
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --verbose             -v      INTEGER              Increase output detail    │
│                                                    (-v, -vv).                │
│                                                    [default: 0]              │
│ --quiet               -q                           Suppress non-essential    │
│                                                    output.                   │
│ --color                       [auto|always|never]  Color mode.               │
│                                                    [default: auto]           │
│ --config                      PATH                 Override config file      │
│                                                    location.                 │
│ --version             -V                           Show version.             │
│ --install-completion                               Install completion for    │
│                                                    the current shell.        │
│ --show-completion                                  Show completion for the   │
│                                                    current shell, to copy it │
│                                                    or customize the          │
│                                                    installation.             │
│ --help                -h                           Show this message and     │
│                                                    exit.                     │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ repl       Start an interactive REPL session.                                │
│ doctor     Check psoul environment and report status.                        │
│ run        Launch a Python target as a managed session.                      │
│ stop       Stop a running or suspended headless session. Escalates to        │
│            SIGKILL after the stop_timeout grace period.                      │
│ kill       Kill a running or suspended headless session immediately (no      │
│            grace period).                                                    │
│ pause      Freeze a running headless session (SIGSTOP).                      │
│ resume     Unfreeze a suspended headless session (SIGCONT).                  │
│ signal     Send a named POSIX signal to a running, suspended, or orphaned    │
│            headless session's process group.                                 │
│ ps         List sessions.                                                    │
│ status     Show session detail.                                              │
│ logs       Print captured stdout/stderr for a session.                       │
│ events     Print the event log for a session.                                │
│ stats      Show current resource usage (CPU, memory, disk, GPU if            │
│            available).                                                       │
│ artifacts  List files produced by a session (plots, checkpoints, exports).   │
│ version    Show psoul version.                                               │
│ config     Show and manage configuration.                                    │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_version_help() -> None:
    result = runner.invoke(cli, ["version", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul version [OPTIONS]                                                 \n\
                                                                                \n\
 Show psoul version.                                                            \n\
                                                                                \n\
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_doctor_help() -> None:
    result = runner.invoke(cli, ["doctor", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul doctor [OPTIONS]                                                  \n\
                                                                                \n\
 Check psoul environment and report status.                                     \n\
                                                                                \n\
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --json            Output JSON instead of text.                               │
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_events_help() -> None:
    result = runner.invoke(cli, ["events", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul events [OPTIONS] SESSION_ID                                       \n\
                                                                                \n\
 Print the event log for a session.                                             \n\
                                                                                \n\
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    session_id      TEXT  Session ID or unique prefix. [required]           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --follow  -f        Stream events live until the session exits.              │
│ --json              Output JSON instead of text, as an array by default, or  │
│                     NDJSON with --follow.                                    │
│ --help    -h        Show this message and exit.                              │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_stop_help() -> None:
    result = runner.invoke(cli, ["stop", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul stop [OPTIONS] SESSION_ID                                         \n\
                                                                                \n\
 Stop a running or suspended headless session. Escalates to SIGKILL after the   \n\
 stop_timeout grace period.                                                     \n\
                                                                                \n\
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    session_id      TEXT  Session ID or unique prefix. [required]           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_kill_help() -> None:
    result = runner.invoke(cli, ["kill", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul kill [OPTIONS] SESSION_ID                                         \n\
                                                                                \n\
 Kill a running or suspended headless session immediately (no grace period).    \n\
                                                                                \n\
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    session_id      TEXT  Session ID or unique prefix. [required]           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_pause_help() -> None:
    result = runner.invoke(cli, ["pause", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul pause [OPTIONS] SESSION_ID                                        \n\
                                                                                \n\
 Freeze a running headless session (SIGSTOP).                                   \n\
                                                                                \n\
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    session_id      TEXT  Session ID or unique prefix. [required]           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_resume_help() -> None:
    result = runner.invoke(cli, ["resume", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul resume [OPTIONS] SESSION_ID                                       \n\
                                                                                \n\
 Unfreeze a suspended headless session (SIGCONT).                               \n\
                                                                                \n\
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    session_id      TEXT  Session ID or unique prefix. [required]           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_signal_help() -> None:
    result = runner.invoke(cli, ["signal", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul signal [OPTIONS] SESSION_ID SIGNAL_NAME                           \n\
                                                                                \n\
 Send a named POSIX signal to a running, suspended, or orphaned headless        \n\
 session's process group.                                                       \n\
                                                                                \n\
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    session_id       TEXT  Session ID or unique prefix. [required]          │
│ *    signal_name      TEXT  Signal name (e.g., TERM, USR1, SIGUSR1).         │
│                             [required]                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_version_output_snapshot() -> None:
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert result.output == snapshot("psoul 0.0.3\n")
