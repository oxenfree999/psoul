# Changelog

## [Unreleased]

Sessions are now ephemeral by default. Pass `--record` to opt in to persistence.

Native support is now Linux and macOS. Windows users should use WSL.

### Changed

- `psoul run` and `psoul repl` no longer record to disk by default. Pass `--record`, set `[session] record = true` in config, or launch with `--headless` to opt in.
- Read-side commands (`ps`, `status`, `events`, `logs`, `stats`, `artifacts`, `env <session>`, `prune`) exit 0 with a single-line stderr message on a fresh home with no recorded sessions.
- Control-side commands (`stop`, `kill`, `pause`, `resume`, `restart`, `signal`, `attach`) exit with a usage error and the same stderr message on a fresh home.
- Existing session data from v0.0.4 stays in place. Use `psoul prune --all` to start fresh.

### Added

- `--record/-r` flag on `psoul run` and `psoul repl`.
- `[session] record` config option.

### Removed

#### Windows Support

The following are removed:

- The `Operating System :: Microsoft :: Windows` classifier in `pyproject.toml`.
- `windows-latest` in CI.
- The `[windows]` test recipe in the `justfile`.
- The Windows helper named-pipe adapter (`WindowsHelperPipeAdapter`).
- Windows-only branches on `stop`, `kill`, `pause`, `resume`, `restart`, `signal`, or `attach`.
- The `_HELPER_SUPPORTED` gate in `psoul.core.launch`.
- `PureWindowsPath` provenance handling.
- The `COMSPEC` shell fallback.

#### CLI REPL

The following are removed:

- The `psoul repl` Typer command.
- The bare-`psoul` REPL launch in `src/psoul/cli/main.py`.
- The `psoul.cli.repl` and `psoul.core.repl` modules.
- The `prompt-toolkit>=3.0` and `pygments>=2.0` base dependencies in `pyproject.toml`.
- The `tests/test_repl.py` and `tests/test_repl_cli.py` test files.

## [0.0.4] - 2026-04-29

Control plane and interactive takeover. Stop, kill, pause, resume, restart, or signal a session, attach to a running headless session, prune old data, and inspect the active Python environment.

### Added

- `psoul stop <session>` sends SIGTERM and escalates to SIGKILL after the configured timeout.
- `psoul kill <session>` sends SIGKILL immediately.
- `psoul pause <session>` suspends a session via SIGSTOP.
- `psoul resume <session>` resumes a suspended session via SIGCONT.
- `psoul restart <session>` terminates the current run and respawns the session, preserving its history.
- `psoul signal <session> <name>` delivers an arbitrary signal by name (for example, `SIGUSR1`).
- `psoul attach <session>` connects an interactive PTY to a running headless session — type at the child, see its output, detach with `Ctrl-]`.
- `psoul prune` deletes finished sessions and their artifacts. Filter with `--all`, `--older-than`, `--state`, or `--tag`. Pass `--force` to also delete active sessions.
- `psoul env` prints a curated Python environment summary — the running process by default, or a recorded session's launch-time snapshot when given a session ID (`psoul env <session>`).

### Fixed

- `psoul script.py | grep foo` no longer auto-promotes to headless mode when stdin is not a terminal — it now mirrors `python script.py | grep foo`.
- The resource sampler no longer crashes when a session exits fast enough to enter zombie state before its first sample.

## [0.0.3] - 2026-04-17

Core observability and GPU telemetry. See what a session is doing — live output, the full event log, CPU/memory/GPU usage, and registered artifacts.

### Added

- `psoul logs <session>` prints captured stdout and stderr. Use `--stdout` or `--stderr` to filter, `--generation N` to look at a specific run, and `--follow` / `-f` to stream live output.
- `psoul events <session>` prints the full event log as tab-separated text or JSON. `--follow` streams live events (NDJSON under `--json --follow`).
- `psoul stats <session>` shows current CPU, memory, disk, and NVIDIA GPU usage for a running session.
- `psoul artifacts <session>` lists files produced by a session (plots, checkpoints, exports) with their paths, sizes, and MIME types.
- NVIDIA GPU metrics (utilization, memory, temperature, power) via the new `psoul[gpu]` optional install.

### Fixed

- Unknown commands (like `psoul psx`) now error cleanly instead of creating a bogus session.
- Concurrent use of psoul no longer surfaces raw SQLite tracebacks.
- Config values for `python_path`, session tags, and launch mode now actually take effect at runtime.

## [0.0.2] - 2026-04-07

Launch and session lifecycle. Start a REPL or run a script, see what's running, and keep a record of what happened.

### Added

- `psoul` starts an interactive REPL session with syntax highlighting, multiline editing, and persistent history.
- `psoul script.py` runs a Python script as a supervised session.
- `psoul ps` lists your sessions, with `--state` and `--tag` filters.
- `psoul status <id>` shows a session's tags, provenance, and result.
- All sessions are stored in a local SQLite database.
- Sessions capture a complete snapshot of the launch environment and exit result, including git state, Python version, platform info, lockfile hash, script hash, duration, and exit code.
- Crashed sessions are detected and cleaned up automatically the next time `psoul` is run.
- `psoul config init` writes a starter `psoul.toml` config file; pass `--pyproject` to inject `[tool.psoul]` into an existing `pyproject.toml` instead.

## [0.0.1] - 2026-03-27

Initial release. Project foundation and core CLI.

### Added

- `psoul version` — display installed version.
- `psoul doctor` — environment health check with tool detection.
- Global flags: `--verbose`, `--quiet`, `--color`, `--format`, `--json`, and `--config`.
- Structured logging via structlog with the `PSOUL_LOG` environment variable.
- Inline snapshot tests for CLI output.
- CI on Ubuntu, macOS, and Windows across Python 3.12, 3.13, and 3.14.

[0.0.4]: https://github.com/oxenfree999/psoul/releases/tag/v0.0.4
[0.0.3]: https://github.com/oxenfree999/psoul/releases/tag/v0.0.3
[0.0.2]: https://github.com/oxenfree999/psoul/releases/tag/v0.0.2
[0.0.1]: https://github.com/oxenfree999/psoul/releases/tag/v0.0.1
