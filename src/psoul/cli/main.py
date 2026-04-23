"""psoul CLI entry point."""

import dataclasses
import json
import os
import signal
import sqlite3
import sys
import time
import tomllib
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Annotated, cast

import click
import psutil
import tomlkit.exceptions
import typer
from typer.core import TyperGroup

from psoul.cli.doctor import format_text, get_system_info
from psoul.cli.logging import configure_logging, resolve_log_level
from psoul.cli.repl import run_repl
from psoul.cli.state import ColorMode, ExitCode, GlobalState, OutputFormat, resolve_color
from psoul.core.artifacts import ArtifactStore
from psoul.core.config import PsoulConfig, find_config_file, generate_config, inject_pyproject_config, load_config
from psoul.core.db import DB_NAME, open_db, resolve_state_dir
from psoul.core.duration import parse_duration
from psoul.core.events import EVENT_RUNTIME_STDERR, EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.launch import build_launch_request, launch_attached, launch_headless, resolve_session_id
from psoul.core.recovery import recover_sessions
from psoul.core.session import TERMINAL_STATES, LaunchMode, Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

_SIGNAL_ACCEPT_STATES: frozenset[SessionState] = frozenset(
    {SessionState.running, SessionState.suspended, SessionState.orphaned}
)


class DefaultRunGroup(TyperGroup):
    """Route bare file arguments to ``run`` for shorthand invocation.

    Allows ``psoul script.py`` as shorthand for ``psoul run script.py``,
    but only when *script.py* actually exists on disk. Unknown commands
    that happen to look like filenames (``psoul psx``) and unknown flags
    (``psoul --bad``) propagate the original ``UsageError`` so Click's
    normal error handling applies.
    """

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve *args* to a command, routing existing files to ``run``."""
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if args and not args[0].startswith("-") and Path(args[0]).exists():
                return super().resolve_command(ctx, ["run", *args])
            raise


cli = typer.Typer(
    name="psoul",
    cls=DefaultRunGroup,
    help="A CLI-first foundation for managed Python sessions.",
    invoke_without_command=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)


def _load_resolved_config(config_override: Path | None) -> PsoulConfig:
    """Find and load config, exiting on error."""
    try:
        config_file = find_config_file(config_override)
        return load_config(config_file)
    except (FileNotFoundError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from exc


def _open_db_or_exit(state_dir: Path) -> sqlite3.Connection:
    """Open the database, translating OperationalError to a clean CLI error."""
    try:
        return open_db(state_dir)
    except sqlite3.OperationalError as exc:
        print(f"Error: database is busy or locked: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from exc


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is passed."""
    if value:
        print(f"psoul {VERSION}")
        raise typer.Exit(ExitCode.SUCCESS)


def _resolve_session_selector(store: SessionStore, selector: str) -> Session:
    """Look up a session by exact ID or unique prefix."""
    session = store.get(selector)
    if session is not None:
        return session
    matches = [s for s in store.list() if s.session_id.startswith(selector)]
    if not matches:
        msg = f"session not found: {selector}"
        raise ValueError(msg)
    if len(matches) > 1:
        ids = ", ".join(sorted(s.session_id for s in matches))
        msg = f"ambiguous session selector: {selector} matches {ids}"
        raise ValueError(msg)
    return matches[0]


def parse_tags(raw: list[str] | None, defaults: dict[str, str] | None = None) -> dict[str, str] | None:
    """Parse ``--tag key=value`` CLI arguments into a tag dict.

    Each string is split on the first ``=``.  Duplicate keys keep the last
    value.  When *defaults* is provided, parsed CLI tags merge on top of
    them; CLI tags win on conflict.

    Args:
        raw (list[str] | None): Raw ``--tag`` values from Typer, or ``None``.
        defaults (dict[str, str] | None): Defaults the parsed CLI tags merge into.

    Returns:
        dict[str, str] | None: Parsed tags merged with *defaults*, or ``None``
            when both are empty.

    Raises:
        typer.BadParameter: A tag is missing ``=`` or has an empty key.

    Examples:
        >>> parse_tags(["env=dev", "team=platform"])
        {'env': 'dev', 'team': 'platform'}
        >>> parse_tags(["key=a=b"])  # value can contain '='
        {'key': 'a=b'}
        >>> parse_tags(["env=prod"], defaults={"env": "dev"})  # CLI wins
        {'env': 'prod'}
        >>> parse_tags(None)  # returns None

    """
    if not raw:
        return defaults or None
    tags: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            msg = f"invalid tag (expected key=value): {item!r}"
            raise typer.BadParameter(msg)
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            msg = f"invalid tag (empty key): {item!r}"
            raise typer.BadParameter(msg)
        tags[key] = value
    return (defaults or {}) | tags


@cli.callback()
def _main(
    ctx: typer.Context,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase output detail (-v, -vv).")] = 0,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-essential output.")] = False,
    color: Annotated[ColorMode, typer.Option("--color", help="Color mode.", case_sensitive=False)] = ColorMode.auto,
    config: Annotated[Path | None, typer.Option("--config", help="Override config file location.")] = None,
    version: Annotated[  # noqa: ARG001 — Typer requires the param; work happens in callback
        bool | None, typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="Show version.")
    ] = None,
) -> None:
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet cannot be used together.")

    log_level = resolve_log_level(verbose, quiet)

    ctx.obj = GlobalState(
        verbose=verbose,
        quiet=quiet,
        color=color,
        color_enabled=resolve_color(color),
        log_level=log_level,
        config_override=config,
    )

    configure_logging(log_level, OutputFormat.text)

    if ctx.invoked_subcommand is None:
        _launch_repl(ctx)


def _launch_repl(ctx: typer.Context, name: str | None = None, tag: list[str] | None = None) -> None:
    """Shared REPL launch logic for bare `psoul` and `psoul repl`."""
    state: GlobalState = ctx.obj
    cfg = _load_resolved_config(state.config_override)
    try:
        session_id = resolve_session_id(name)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE) from exc
    state_dir = resolve_state_dir(cfg.paths.state_dir)
    conn = _open_db_or_exit(state_dir)
    try:
        recover_sessions(conn)
        if SessionStore(conn).get(session_id) is not None:
            print(f"Error: session ID already exists: {session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR)
        run_repl(session_id, conn, db_path=state_dir / DB_NAME, tags=parse_tags(tag, defaults=cfg.session.tags))
    except sqlite3.IntegrityError:
        print(f"Error: session ID already exists: {session_id}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    finally:
        conn.close()


@cli.command()
def repl(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Option("--name", help="Session ID.")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Tag as key=value (repeatable).")] = None,
) -> None:
    """Start an interactive REPL session."""
    _launch_repl(ctx, name=name, tag=tag)


@cli.command()
def doctor(
    json_flag: Annotated[bool, typer.Option("--json", help="Output JSON instead of text.")] = False,
) -> None:
    """Check psoul environment and report status."""
    info = get_system_info()
    if json_flag:
        print(json.dumps(info, indent=2))
    else:
        print(format_text(info))


config_app = typer.Typer(name="config", help="Show and manage configuration.")
cli.add_typer(config_app, name="config")


@config_app.callback(invoke_without_command=True)
def config_cmd(
    ctx: typer.Context,
    default: Annotated[bool, typer.Option("--default", help="Show default configuration.")] = False,
    json_flag: Annotated[bool, typer.Option("--json", help="Output JSON instead of text.")] = False,
) -> None:
    """Show and manage configuration."""
    if ctx.invoked_subcommand is not None:
        return
    state: GlobalState = ctx.obj
    cfg = PsoulConfig() if default else _load_resolved_config(state.config_override)
    data = dataclasses.asdict(cfg)
    if json_flag:
        print(json.dumps(data, indent=2, default=str))
    else:
        for section, values in data.items():
            for key, value in values.items():
                print(f"{section}.{key} = {value!r}")


@config_app.command()
def init(
    ctx: typer.Context,
    pyproject: Annotated[bool, typer.Option("--pyproject", help="Inject [tool.psoul] into pyproject.toml.")] = False,
) -> None:
    """Write default psoul config to the current directory."""
    state: GlobalState = ctx.obj
    if pyproject:
        if state.config_override is not None:
            print("Error: --config cannot be used with --pyproject.", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        try:
            inject_pyproject_config(Path("pyproject.toml"))
        except (tomlkit.exceptions.TOMLKitError, TypeError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR) from exc
        if not state.quiet:
            print("Added [tool.psoul] section to pyproject.toml")
        return
    dest = Path("psoul.toml")
    if dest.exists():
        print(f"Error: {dest} already exists.", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR)
    dest.write_text(generate_config())
    if not state.quiet:
        print(f"Wrote {dest}")


@cli.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False, "ignore_unknown_options": True},
)
def run(
    ctx: typer.Context,
    module: Annotated[str | None, typer.Option("-m", help="Module to run.")] = None,
    name: Annotated[str | None, typer.Option("--name", help="Session ID.")] = None,
    headless: Annotated[bool, typer.Option("--headless", help="Launch in background.")] = False,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Tag as key=value (repeatable).")] = None,
) -> None:
    """Launch a Python target as a managed session."""
    state: GlobalState = ctx.obj
    cfg = _load_resolved_config(state.config_override)
    target = ctx.args[0] if not module and ctx.args else None
    extra_args = ctx.args[1:] if target else ctx.args
    try:
        request = build_launch_request(
            target=target,
            module=module,
            extra_args=extra_args,
            name=name,
            headless=headless,
            tags=parse_tags(tag, defaults=cfg.session.tags),
            python_path=cfg.python.python_path or Path(sys.executable),
            default_mode=LaunchMode(cfg.launch.mode),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE) from exc
    state_dir = resolve_state_dir(cfg.paths.state_dir)
    conn = _open_db_or_exit(state_dir)
    try:
        recover_sessions(conn)
        store = SessionStore(conn)
        if store.get(request.session_id) is not None:
            print(f"Error: session ID already exists: {request.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR)
        if request.launch_mode == LaunchMode.headless:
            stop_timeout = parse_duration(cfg.process.stop_timeout).total_seconds()
            session, supervisor_pid = launch_headless(request, store, state_dir, stop_timeout_seconds=stop_timeout)
            print(
                json.dumps(
                    {"session_id": session.session_id, "state": session.state.value, "supervisor_pid": supervisor_pid}
                )
            )
        else:
            launch_attached(request, store)
    except sqlite3.IntegrityError:
        print(f"Error: session ID already exists: {request.session_id}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    except NotImplementedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    finally:
        conn.close()


def _deliver_control_signal(ctx: typer.Context, selector: str, *, signame: str, verb: str) -> None:
    """Resolve *selector*, validate state, and send the signal named *signame* to the supervisor.

    *signame* is resolved to a `signal.Signals` value only after the platform
    check, since names like ``SIGUSR1`` and ``SIGUSR2`` do not exist on Windows.
    """
    if sys.platform == "win32":
        print(f"Error: {verb} is Unix-only (macOS / Linux). Windows support deferred.", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE)
    sig = getattr(signal, signame)
    state: GlobalState = ctx.obj
    cfg = _load_resolved_config(state.config_override)
    with closing(_open_db_or_exit(resolve_state_dir(cfg.paths.state_dir))) as conn:
        recover_sessions(conn)
        try:
            session = _resolve_session_selector(SessionStore(conn), selector)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE) from None
        if session.launch_mode != LaunchMode.headless:
            msg = f"{verb} requires a headless session (this session is attached): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.state == SessionState.orphaned:
            print(f"Error: session is orphaned: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.state == SessionState.stopping:
            print(f"Error: session is already stopping: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.state not in {SessionState.running, SessionState.suspended}:
            msg = f"session is not running or suspended (state: {session.state}): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.supervisor_pid is None:
            print(f"Error: session has no supervisor: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR)
        try:
            os.kill(session.supervisor_pid, sig)
        except ProcessLookupError:
            print(f"Error: supervisor process is not running: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR) from None
        except PermissionError:
            msg = f"permission denied signalling supervisor (PID={session.supervisor_pid}): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR) from None
        print(f"Sent {verb} to {session.session_id}.")


@cli.command()
def stop(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
) -> None:
    """Stop a running or suspended headless session. Escalates to SIGKILL after the stop_timeout grace period."""
    _deliver_control_signal(ctx, session_id, signame="SIGUSR1", verb="stop")


@cli.command()
def kill(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
) -> None:
    """Kill a running or suspended headless session immediately (no grace period)."""
    _deliver_control_signal(ctx, session_id, signame="SIGUSR2", verb="kill")


@cli.command()
def restart(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
) -> None:
    """Stop and relaunch a running or suspended headless session. Same session ID, new generation."""
    _deliver_control_signal(ctx, session_id, signame="SIGHUP", verb="restart")


def _resolve_child_pid(supervisor_pid: int, session_id: str) -> int:
    """Return the single managed child's PID, or exit with a clean error message."""
    try:
        children = psutil.Process(supervisor_pid).children()
    except psutil.NoSuchProcess:
        print(f"Error: supervisor process is not running: {session_id}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    if len(children) != 1:
        print(f"Error: supervisor has no managed child: {session_id}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR)
    return children[0].pid


def _deliver_child_pgroup_signal(
    ctx: typer.Context,
    selector: str,
    *,
    signame: str,
    verb: str,
    accept_state: SessionState,
) -> None:
    """Resolve *selector*, validate state, and send the signal named *signame* to the child's process group.

    *signame* is resolved to a `signal.Signals` value only after the platform
    check, since names like ``SIGSTOP`` and ``SIGCONT`` do not exist on Windows.
    """
    if sys.platform == "win32":
        print(f"Error: {verb} is Unix-only (macOS / Linux). Windows support deferred.", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE)
    sig = getattr(signal, signame)
    state: GlobalState = ctx.obj
    cfg = _load_resolved_config(state.config_override)
    with closing(_open_db_or_exit(resolve_state_dir(cfg.paths.state_dir))) as conn:
        recover_sessions(conn)
        try:
            session = _resolve_session_selector(SessionStore(conn), selector)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE) from None
        if session.launch_mode != LaunchMode.headless:
            msg = f"{verb} requires a headless session (this session is attached): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.state == SessionState.orphaned:
            print(f"Error: session is orphaned: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.state == SessionState.stopping:
            print(f"Error: session is already stopping: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.state != accept_state:
            msg = f"session is not {accept_state.value} (state: {session.state}): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.supervisor_pid is None:
            print(f"Error: session has no supervisor: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR)
        child_pid = _resolve_child_pid(session.supervisor_pid, session.session_id)
        try:
            os.killpg(child_pid, sig)
        except ProcessLookupError:
            print(f"session's child has already exited: {session.session_id}")
            return
        except PermissionError:
            msg = f"permission denied signalling child (PID={child_pid}): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR) from None
        print(f"Sent {verb} to {session.session_id}.")


@cli.command()
def pause(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
) -> None:
    """Freeze a running headless session (SIGSTOP)."""
    _deliver_child_pgroup_signal(ctx, session_id, signame="SIGSTOP", verb="pause", accept_state=SessionState.running)


@cli.command()
def resume(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
) -> None:
    """Unfreeze a suspended headless session (SIGCONT)."""
    _deliver_child_pgroup_signal(ctx, session_id, signame="SIGCONT", verb="resume", accept_state=SessionState.suspended)


def _resolve_signal_name(raw: str) -> signal.Signals:
    """Return the ``signal.Signals`` member for *raw*, or exit with a usage error.

    Accepts short (``TERM``) and full (``SIGTERM``) forms, case-insensitive.
    """
    name = raw.strip().upper()
    if not name.startswith("SIG"):
        name = f"SIG{name}"
    try:
        return signal.Signals[name]
    except KeyError:
        print(f"Error: unknown signal: {raw}", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE) from None


def _deliver_named_signal_to_pgroup(ctx: typer.Context, selector: str, raw_signal: str) -> None:
    """Resolve *selector*, validate, and send the user-named signal to the child's process group.

    Accepts running, suspended, and orphaned sessions. Orphaned sessions pass
    validation but typically fail at ``_resolve_child_pid`` because the
    supervisor is dead by definition.
    """
    if sys.platform == "win32":
        print("Error: signal is Unix-only (macOS / Linux). Windows support deferred.", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE)
    sig = _resolve_signal_name(raw_signal)
    state: GlobalState = ctx.obj
    cfg = _load_resolved_config(state.config_override)
    with closing(_open_db_or_exit(resolve_state_dir(cfg.paths.state_dir))) as conn:
        recover_sessions(conn)
        try:
            session = _resolve_session_selector(SessionStore(conn), selector)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE) from None
        if session.launch_mode != LaunchMode.headless:
            msg = f"signal requires a headless session (this session is attached): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.state == SessionState.stopping:
            print(f"Error: session is already stopping: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.state not in _SIGNAL_ACCEPT_STATES:
            msg = f"session is not running, suspended, or orphaned (state: {session.state}): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE)
        if session.supervisor_pid is None:
            print(f"Error: session has no supervisor: {session.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR)
        child_pid = _resolve_child_pid(session.supervisor_pid, session.session_id)
        try:
            os.killpg(child_pid, sig)
        except ProcessLookupError:
            print(f"session's child has already exited: {session.session_id}")
            return
        except PermissionError:
            msg = f"permission denied signalling child (PID={child_pid}): {session.session_id}"
            print(f"Error: {msg}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR) from None
        print(f"Sent {sig.name} to {session.session_id}.")


@cli.command(name="signal")
def send_signal(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
    signal_name: Annotated[str, typer.Argument(help="Signal name (e.g., TERM, USR1, SIGUSR1).")],
) -> None:
    """Send a named POSIX signal to a running, suspended, or orphaned headless session's process group."""
    _deliver_named_signal_to_pgroup(ctx, session_id, signal_name)


@cli.command()
def ps(
    ctx: typer.Context,
    state: Annotated[SessionState | None, typer.Option("--state", help="Filter by session state.")] = None,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Filter by tag key=value (repeatable, AND logic).")
    ] = None,
    json_flag: Annotated[bool, typer.Option("--json", help="Output JSON instead of text.")] = False,
) -> None:
    """List sessions."""
    gs: GlobalState = ctx.obj
    cfg = _load_resolved_config(gs.config_override)
    conn = _open_db_or_exit(resolve_state_dir(cfg.paths.state_dir))
    try:
        recover_sessions(conn)
        sessions = SessionStore(conn).list(state=state, tags=parse_tags(tag))
    finally:
        conn.close()
    if json_flag:
        print(json.dumps([dataclasses.asdict(s) for s in sessions], default=str))
        return
    for s in sessions:
        print(f"{s.session_id}  {s.state}  {s.target or 'repl'}  {s.launch_time.isoformat()}")


@cli.command()
def status(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID to inspect.")],
    json_flag: Annotated[bool, typer.Option("--json", help="Output JSON instead of text.")] = False,
) -> None:
    """Show session detail."""
    gs: GlobalState = ctx.obj
    cfg = _load_resolved_config(gs.config_override)
    conn = _open_db_or_exit(resolve_state_dir(cfg.paths.state_dir))
    try:
        recover_sessions(conn)
        session = _resolve_session_selector(SessionStore(conn), session_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    finally:
        conn.close()
    if json_flag:
        print(json.dumps(dataclasses.asdict(session), default=str, indent=2))
        return
    for key, value in dataclasses.asdict(session).items():
        if value is not None:
            print(f"{key}: {value}")


@cli.command()
def logs(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
    stdout: Annotated[bool, typer.Option("--stdout", help="Show only captured stdout.")] = False,
    stderr: Annotated[bool, typer.Option("--stderr", help="Show only captured stderr.")] = False,
    generation: Annotated[int | None, typer.Option("--generation", help="Filter to a session generation.")] = None,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Stream output live until the session exits.")] = False,
) -> None:
    """Print captured stdout/stderr for a session."""
    if stdout and stderr:
        raise typer.BadParameter("--stdout and --stderr cannot be used together.")
    gs: GlobalState = ctx.obj
    cfg = _load_resolved_config(gs.config_override)
    state_dir = resolve_state_dir(cfg.paths.state_dir)
    try:
        with closing(_open_db_or_exit(state_dir)) as conn:
            recover_sessions(conn)
            session = _resolve_session_selector(SessionStore(conn), session_id)
            events = EventStore(conn).list(session.session_id, generation=generation)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    wanted: set[str] = set()
    if not stderr:
        wanted.add(EVENT_RUNTIME_STDOUT)
    if not stdout:
        wanted.add(EVENT_RUNTIME_STDERR)

    def _emit(event: dict[str, object]) -> None:
        if event["event_type"] in wanted:
            payload = cast("dict[str, str]", event["payload"])
            print(payload["text"], end="", flush=follow)

    for event in events:
        _emit(event)
    if not follow:
        return
    last_seq = cast("int", events[-1]["sequence"]) if events else -1
    if session.state in TERMINAL_STATES and session.supervisor_pid is None:
        return
    _follow_events(session.session_id, state_dir, _emit, last_seq, generation=generation)


_FOLLOW_POLL_INTERVAL = 0.25  # seconds between polls in events --follow


def _print_event(row: dict[str, object], json_flag: bool) -> None:
    """Emit one event as tab-separated text, or as NDJSON when json_flag is set, flushing each line."""
    if json_flag:
        print(json.dumps(row), flush=True)
        return
    print(
        row["sequence"],
        row["generation"],
        row["timestamp"],
        row["event_type"],
        json.dumps(row["payload"]),
        sep="\t",
        flush=True,
    )


def _follow_events(
    session_id: str,
    state_dir: Path,
    on_row: Callable[[dict[str, object]], None],
    last_seq: int,
    *,
    generation: int | None = None,
) -> None:
    """Poll for new events until the session reaches a terminal state with supervisor_pid cleared."""
    try:
        while True:
            time.sleep(_FOLLOW_POLL_INTERVAL)
            with closing(_open_db_or_exit(state_dir)) as conn:
                recover_sessions(conn)
                session = SessionStore(conn).get(session_id)
                rows = EventStore(conn).list(session_id, after_sequence=last_seq, generation=generation)
            for row in rows:
                on_row(row)
            if rows:
                last_seq = cast("int", rows[-1]["sequence"])
            if session is None or (session.state in TERMINAL_STATES and not rows and session.supervisor_pid is None):
                return
    except KeyboardInterrupt:
        return


@cli.command()
def events(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Stream events live until the session exits.")] = False,
    json_flag: Annotated[
        bool,
        typer.Option("--json", help="Output JSON instead of text, as an array by default, or NDJSON with --follow."),
    ] = False,
) -> None:
    """Print the event log for a session."""
    gs: GlobalState = ctx.obj
    cfg = _load_resolved_config(gs.config_override)
    state_dir = resolve_state_dir(cfg.paths.state_dir)
    try:
        with closing(_open_db_or_exit(state_dir)) as conn:
            recover_sessions(conn)
            session = _resolve_session_selector(SessionStore(conn), session_id)
            rows = EventStore(conn).list(session.session_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    if follow:
        for row in rows:
            _print_event(row, json_flag)
        last_seq = cast("int", rows[-1]["sequence"]) if rows else -1
        if session.state in TERMINAL_STATES and session.supervisor_pid is None:
            return
        _follow_events(session.session_id, state_dir, lambda row: _print_event(row, json_flag), last_seq)
        return
    if json_flag:
        print(json.dumps(rows))
        return
    for row in rows:
        print(
            row["sequence"],
            row["generation"],
            row["timestamp"],
            row["event_type"],
            json.dumps(row["payload"]),
            sep="\t",
        )


@cli.command()
def stats(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
    json_flag: Annotated[bool, typer.Option("--json", help="Output JSON instead of text.")] = False,
) -> None:
    """Show current resource usage (CPU, memory, disk, GPU if available)."""
    gs: GlobalState = ctx.obj
    cfg = _load_resolved_config(gs.config_override)
    conn = _open_db_or_exit(resolve_state_dir(cfg.paths.state_dir))
    try:
        recover_sessions(conn)
        session = _resolve_session_selector(SessionStore(conn), session_id)
        cursor = conn.execute(
            "SELECT generation, timestamp, cpu_percent, memory_rss_mb, memory_vms_mb,"
            "       disk_read_mb, disk_write_mb,"
            "       gpu_utilization_pct, gpu_memory_used_mb, gpu_memory_total_mb,"
            "       gpu_temperature_c, gpu_power_watts"
            " FROM resource_samples WHERE session_id = ?"
            " ORDER BY generation DESC, timestamp DESC LIMIT 1",
            (session.session_id,),
        )
        row = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    finally:
        conn.close()
    if row is None:
        print("Error: no resource samples found.", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR)
    sample = dict(zip(columns, row, strict=True))
    if json_flag:
        print(json.dumps(sample))
        return
    for key, value in sample.items():
        if value is not None:
            print(f"{key}: {value}")


@cli.command()
def artifacts(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix.")],
) -> None:
    """List files produced by a session (plots, checkpoints, exports)."""
    gs: GlobalState = ctx.obj
    cfg = _load_resolved_config(gs.config_override)
    state_dir = resolve_state_dir(cfg.paths.state_dir)
    try:
        with closing(_open_db_or_exit(state_dir)) as conn:
            recover_sessions(conn)
            session = _resolve_session_selector(SessionStore(conn), session_id)
            rows = ArtifactStore(conn).list(session.session_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from None
    for row in rows:
        mime_type = "" if row["mime_type"] is None else row["mime_type"]
        size_bytes = "" if row["size_bytes"] is None else row["size_bytes"]
        print(
            row["registered_at"],
            row["name"],
            row["path"],
            mime_type,
            size_bytes,
            row["source"],
            row["retention_class"],
            sep="\t",
        )


@cli.command()
def version() -> None:
    """Show psoul version."""
    print(f"psoul {VERSION}")
