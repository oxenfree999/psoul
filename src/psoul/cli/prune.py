"""``psoul prune``: remove completed sessions and their data by filters."""

import json
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import typer

from psoul.cli.state import ExitCode
from psoul.core.duration import parse_duration
from psoul.core.session import TERMINAL_STATES, SessionState
from psoul.core.store import SessionStore


class PruneState(StrEnum):
    """States that ``psoul prune --state`` accepts.

    The frozen CLI surface restricts ``--state`` to the two terminal
    values.  To prune sessions in any other state, combine ``--all``,
    ``--older-than``, or ``--tag`` with ``--force``.
    """

    exited = "exited"
    failed = "failed"


def run_prune(  # noqa: C901, PLR0912 — flag validation and filter resolution earn the branch count
    store: SessionStore,
    conn: sqlite3.Connection,
    state_dir: Path,
    *,
    older_than: str | None,
    state: PruneState | None,
    tag: list[str] | None,
    all_flag: bool,
    force: bool,
    json_flag: bool,
) -> None:
    """Remove completed sessions and their data by filters.

    Filters compose with AND across ``--older-than``, ``--state``, and
    ``--tag``.  ``--all`` selects every row and is mutually exclusive
    with the per-filter flags.  An empty filter set with no ``--all``
    is a usage error so a bare ``psoul prune`` cannot accidentally
    wipe the database.

    Args:
        store (SessionStore): Open store wrapping ``conn``.
        conn (sqlite3.Connection): Open psoul database connection.
            Used directly for the ``results.end_time`` lookup that
            ``--older-than`` keys off.
        state_dir (Path): Resolved state directory.  The artifact
            tree at ``state_dir / "artifacts" / <session_id>/`` is
            removed for each pruned session.
        older_than (str | None): Duration string (e.g. ``"1h30m"``)
            parsed via ``parse_duration``.  Selects sessions whose
            most recent ``results.end_time`` is older than the cutoff
            for terminal sessions, or whose ``launch_time`` is older
            for active sessions reachable via ``--force``.
        state (PruneState | None): Restrict to ``exited`` or ``failed``.
        tag (list[str] | None): ``key=value`` pairs that all must match.
        all_flag (bool): Match every session row, ignoring the other
            filters.  Mutually exclusive with ``--older-than`` /
            ``--state`` / ``--tag``.
        force (bool): Allow non-terminal sessions in the match set.
            Without it, any active match raises a usage error.
        json_flag (bool): Emit a JSON object on stdout instead of
            human-readable lines.

    """
    has_filter = older_than is not None or state is not None or bool(tag)
    if all_flag and has_filter:
        print(
            "Error: --all cannot be combined with --older-than, --state, or --tag.",
            file=sys.stderr,
        )
        raise typer.Exit(ExitCode.USAGE)
    if not all_flag and not has_filter:
        print(
            "Error: psoul prune requires at least one of --all, --older-than, --state, or --tag.",
            file=sys.stderr,
        )
        raise typer.Exit(ExitCode.USAGE)

    cutoff: datetime | None = None
    if older_than is not None:
        try:
            cutoff = datetime.now(UTC) - parse_duration(older_than)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(ExitCode.USAGE) from None

    tags: dict[str, str] | None = None
    if tag:
        tags = {}
        for item in tag:
            key, _, value = item.partition("=")
            if "=" not in item or not key.strip():
                print(f"Error: invalid tag (expected key=value): {item!r}", file=sys.stderr)
                raise typer.Exit(ExitCode.USAGE)
            tags[key.strip()] = value.strip()

    state_filter = SessionState(state.value) if state is not None else None
    sessions = store.list() if all_flag else store.list(state=state_filter, tags=tags)
    if cutoff is not None:
        end_times = {
            str(session_id): datetime.fromisoformat(str(end_time))
            for session_id, end_time in conn.execute(
                "SELECT session_id, MAX(end_time) FROM results GROUP BY session_id"
            ).fetchall()
            if end_time is not None
        }
        sessions = [
            s
            for s in sessions
            if (end_times[s.session_id] if s.state in TERMINAL_STATES and s.session_id in end_times else s.launch_time)
            < cutoff
        ]

    active = [s for s in sessions if s.state not in TERMINAL_STATES]
    if active and not force:
        ids = ", ".join(s.session_id for s in active)
        print(f"Error: cannot prune active session(s) without --force: {ids}", file=sys.stderr)
        raise typer.Exit(ExitCode.USAGE)

    pruned: list[str] = []
    for s in sessions:
        store.delete(s.session_id)
        try:
            shutil.rmtree(state_dir / "artifacts" / s.session_id)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Warning: failed to remove artifact directory for {s.session_id}: {exc}", file=sys.stderr)
        pruned.append(s.session_id)

    if json_flag:
        print(json.dumps({"pruned": pruned, "count": len(pruned)}))
        return
    for session_id in pruned:
        print(session_id)
    print(f"Pruned {len(pruned)} session{'s' if len(pruned) != 1 else ''}.")
