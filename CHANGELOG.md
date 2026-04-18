# Changelog

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

[0.0.3]: https://github.com/oxenfree999/psoul/releases/tag/v0.0.3
[0.0.2]: https://github.com/oxenfree999/psoul/releases/tag/v0.0.2
[0.0.1]: https://github.com/oxenfree999/psoul/releases/tag/v0.0.1
