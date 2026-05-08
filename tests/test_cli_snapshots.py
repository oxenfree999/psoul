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
│ doctor     Check psoul environment and report status.                        │
│ env        Show a curated Python environment summary.                        │
│ run        Launch a Python target as a managed session.                      │
│ stop       Stop a running or suspended headless session. Escalates to        │
│            SIGKILL after the stop_timeout grace period.                      │
│ kill       Kill a running or suspended headless session immediately (no      │
│            grace period).                                                    │
│ restart    Stop and relaunch a running or suspended headless session. Same   │
│            session ID, new generation.                                       │
│ pause      Freeze a running headless session (SIGSTOP).                      │
│ resume     Unfreeze a suspended headless session (SIGCONT).                  │
│ signal     Send a named POSIX signal to a running, suspended, or orphaned    │
│            headless session's process group.                                 │
│ attach     Attach interactively to a running headless session. Detach with   │
│            Ctrl-].                                                           │
│ ps         List sessions.                                                    │
│ status     Show session detail.                                              │
│ logs       Print captured stdout/stderr for a session.                       │
│ events     Print the event log for a session.                                │
│ stats      Show current resource usage (CPU, memory, disk, GPU if            │
│            available).                                                       │
│ artifacts  List files produced by a session (plots, checkpoints, exports).   │
│ prune      Remove completed sessions and their data by age, state, or tags.  │
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


def test_env_help() -> None:
    result = runner.invoke(cli, ["env", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul env [OPTIONS] [SESSION_ID]                                        \n\
                                                                                \n\
 Show a curated Python environment summary.                                     \n\
                                                                                \n\
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   session_id      [SESSION_ID]  Session ID or unique prefix.                 │
╰──────────────────────────────────────────────────────────────────────────────╯
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


def test_restart_help() -> None:
    result = runner.invoke(cli, ["restart", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul restart [OPTIONS] SESSION_ID                                      \n\
                                                                                \n\
 Stop and relaunch a running or suspended headless session. Same session ID,    \n\
 new generation.                                                                \n\
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


def test_attach_help() -> None:
    result = runner.invoke(cli, ["attach", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul attach [OPTIONS] SESSION_ID                                       \n\
                                                                                \n\
 Attach interactively to a running headless session. Detach with Ctrl-].        \n\
                                                                                \n\
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    session_id      TEXT  Session ID or unique prefix. [required]           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_prune_help() -> None:
    result = runner.invoke(cli, ["prune", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul prune [OPTIONS]                                                   \n\
                                                                                \n\
 Remove completed sessions and their data by age, state, or tags.               \n\
                                                                                \n\
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --older-than          TEXT             Only prune sessions older than this   │
│                                        duration (e.g. 1h30m).                │
│ --state               [exited|failed]  Restrict to exited or failed          │
│                                        sessions.                             │
│ --tag                 TEXT             Restrict to sessions matching         │
│                                        key=value (repeatable, AND logic).    │
│ --all                                  Match every session row (mutually     │
│                                        exclusive with                        │
│                                        --older-than/--state/--tag).          │
│ --force                                Allow non-terminal sessions in the    │
│                                        match set.                            │
│ --json                                 Emit a JSON object instead of         │
│                                        human-readable lines.                 │
│ --help        -h                       Show this message and exit.           │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_run_help() -> None:
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert typer.unstyle(result.output) == snapshot("""\
                                                                                \n\
 Usage: psoul run [OPTIONS]                                                     \n\
                                                                                \n\
 Launch a Python target as a managed session.                                   \n\
                                                                                \n\
╭─ Options ────────────────────────────────────────────────────────────────────╮
│             -m      TEXT  Module to run.                                     │
│ --name              TEXT  Session ID.                                        │
│ --headless                Launch in background.                              │
│ --tag               TEXT  Tag as key=value (repeatable).                     │
│ --record    -r            Save this session so `psoul ps` and other commands │
│                           can find it.                                       │
│ --help      -h            Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯

""")


def test_version_output_snapshot() -> None:
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert result.output == snapshot("psoul 0.0.4\n")
