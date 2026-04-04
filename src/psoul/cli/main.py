"""psoul CLI entry point."""

import dataclasses
import json
import sqlite3
import sys
import tomllib
from pathlib import Path
from typing import Annotated

import typer

from psoul.cli.doctor import format_text, get_system_info
from psoul.cli.logging import configure_logging, resolve_log_level
from psoul.cli.repl import run_repl
from psoul.cli.state import ColorMode, ExitCode, GlobalState, OutputFormat, resolve_color
from psoul.config import PsoulConfig, find_config_file, generate_config, load_config
from psoul.db import DB_NAME, open_db, resolve_state_dir
from psoul.launch import build_launch_request, launch_attached, launch_headless, resolve_session_id
from psoul.session import LaunchMode, Session, SessionState
from psoul.store import SessionStore
from psoul.version import VERSION

cli = typer.Typer(
    name="psoul",
    help="A CLI and TUI Python session supervisor with batteries included.",
    invoke_without_command=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)


def _load_resolved_config(config_override: Path | None) -> PsoulConfig:
    try:
        config_file = find_config_file(config_override)
        return load_config(config_file)
    except (FileNotFoundError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR) from exc


def _version_callback(value: bool) -> None:
    if value:
        print(f"psoul {VERSION}")
        raise typer.Exit(ExitCode.SUCCESS)


def _resolve_session_selector(store: SessionStore, selector: str) -> Session:
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


def parse_tags(raw: list[str] | None) -> dict[str, str] | None:
    """Parse --tag key=value CLI arguments into a tag dict.

    Returns None when *raw* is None or empty.  Splits each item on the
    first ``=`` so values may contain additional ``=`` characters.  Keys
    and values are stripped of surrounding whitespace.  Empty values
    (``key=``) are allowed; missing ``=`` or an empty key (after
    stripping) are rejected with ``typer.BadParameter``.  Duplicate keys
    use last-wins semantics (natural dict assignment order).
    """
    if not raw:
        return None
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
    return tags


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


def _launch_repl(ctx: typer.Context, name: str | None = None, tags: dict[str, str] | None = None) -> None:
    """Shared REPL launch logic for bare `psoul` and `psoul repl`."""
    state: GlobalState = ctx.obj
    cfg = _load_resolved_config(state.config_override)
    try:
        session_id = resolve_session_id(name)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE) from exc
    state_dir = resolve_state_dir(cfg.paths.state_dir)
    conn = open_db(state_dir)
    try:
        if SessionStore(conn).get(session_id) is not None:
            print(f"Error: session ID already exists: {session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR)
        run_repl(session_id, conn, db_path=state_dir / DB_NAME, tags=tags)
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
    _launch_repl(ctx, name=name, tags=parse_tags(tag))


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
def init(ctx: typer.Context) -> None:
    """Write a default psoul.toml to the current directory."""
    dest = Path("psoul.toml")
    if dest.exists():
        print(f"Error: {dest} already exists.", file=sys.stderr)
        raise typer.Exit(ExitCode.ERROR)
    dest.write_text(generate_config())
    state: GlobalState = ctx.obj
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
            tags=parse_tags(tag),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE) from exc
    state_dir = resolve_state_dir(cfg.paths.state_dir)
    conn = open_db(state_dir)
    try:
        store = SessionStore(conn)
        if store.get(request.session_id) is not None:
            print(f"Error: session ID already exists: {request.session_id}", file=sys.stderr)
            raise typer.Exit(ExitCode.ERROR)
        if request.launch_mode == LaunchMode.headless:
            session, supervisor_pid = launch_headless(request, store, state_dir)
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
    conn = open_db(resolve_state_dir(cfg.paths.state_dir))
    try:
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
    conn = open_db(resolve_state_dir(cfg.paths.state_dir))
    try:
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
def version() -> None:
    """Show psoul version."""
    print(f"psoul {VERSION}")
