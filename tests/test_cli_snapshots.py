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
│ repl     Start an interactive REPL session.                                  │
│ doctor   Check psoul environment and report status.                          │
│ run      Launch a Python target as a managed session.                        │
│ ps       List sessions.                                                      │
│ status   Show session detail.                                                │
│ version  Show psoul version.                                                 │
│ config   Show and manage configuration.                                      │
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


def test_version_output_snapshot() -> None:
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert result.output == snapshot("psoul 0.0.1\n")
