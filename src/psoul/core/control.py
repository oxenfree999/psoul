"""Supervisor-side control: signal-triggered stop and kill for the managed child."""

import os
import signal
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from types import FrameType
from typing import Protocol

from psoul.core.events import EVENT_SESSION_CONTROLLER_ACQUIRED, EVENT_SESSION_CONTROLLER_RELEASED, EventStore
from psoul.core.recovery import ProcessStatus, check_pid
from psoul.core.session import SessionState
from psoul.core.store import SessionStore


class SupervisedProc(Protocol):
    """Structural protocol for the bits of ``subprocess.Popen`` the control handlers use."""

    @property
    def pid(self) -> int:
        """The child's process ID, used as the pgroup identifier for ``os.killpg``."""

    def poll(self) -> int | None:
        """Return the child's exit code if it has terminated, else ``None``."""


COMMAND_ACCEPTED = "command.accepted"
COMMAND_COMPLETED = "command.completed"
COMMAND_FAILED = "command.failed"
SESSION_RESTARTED = "session.restarted"

STOP_COMMAND = "stop"
KILL_COMMAND = "kill"
PAUSE_COMMAND = "pause"
RESUME_COMMAND = "resume"
RESTART_COMMAND = "restart"

OUTCOME_OK = "ok"
OUTCOME_ESCALATED = "escalated"
OUTCOME_NOOP = "noop"


@dataclass(slots=True)
class PendingCommand:
    """Bookkeeping for a stop or kill that is in flight and awaiting its terminal event."""

    command: str
    message_id: str
    start_monotonic: float


@dataclass(slots=True)
class ControlState:
    """Signal-driven control flags and pending-command bookkeeping for the supervisor.

    Signal handlers flip the request flags and nothing else. The main
    supervise loop reads the flags each tick and calls ``handle_stop`` /
    ``handle_kill``, which allocate the ``PendingCommand`` at dispatch time.
    This keeps the handler async-signal-safe and avoids pre-allocated slots
    that could be overwritten by overlapping signals.
    """

    stop_requested: bool = False
    kill_requested: bool = False
    restart_requested: bool = False
    stopping: bool = False
    escalation_deadline: float | None = None
    escalation_fired: bool = False
    pending: PendingCommand | None = None
    restart_pending: PendingCommand | None = None
    completion_emitted: bool = False


def _new_message_id() -> str:
    """Return a UUID v4 string for a supervisor-generated ``message_id``."""
    return str(uuid.uuid4())


def _killpg_if_alive(pid: int, sig: int) -> bool:
    """Deliver *sig* to process group *pid*. Return ``False`` when the pgroup is already gone.

    Closes the race between the caller's ``proc.poll()`` guard and the signal send:
    the child (and its group) can disappear between the two syscalls, which
    surfaces as ``ProcessLookupError``. Callers translate ``False`` into a
    ``command.completed(outcome="noop")`` close so a matched-pair event log
    remains the invariant.
    """
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return False
    return True


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string for ``sent_at`` payloads."""
    return datetime.now(UTC).isoformat()


def install_handlers(state: ControlState) -> None:
    """Install SIGUSR1, SIGUSR2, and SIGHUP handlers that flip request flags on *state*.

    Must be called on the main thread. Each handler performs one
    attribute write and nothing else: no locks, no I/O, no event
    emission. The main supervise loop reads the flags each tick and
    dispatches via ``handle_stop`` / ``handle_kill`` / ``handle_restart``.
    """

    def _on_sigusr1(_signum: int, _frame: FrameType | None) -> None:
        state.stop_requested = True

    def _on_sigusr2(_signum: int, _frame: FrameType | None) -> None:
        state.kill_requested = True

    def _on_sighup(_signum: int, _frame: FrameType | None) -> None:
        state.restart_requested = True

    signal.signal(signal.SIGUSR1, _on_sigusr1)
    signal.signal(signal.SIGUSR2, _on_sigusr2)
    signal.signal(signal.SIGHUP, _on_sighup)


def _emit_accepted(
    event_store: EventStore,
    session_id: str,
    generation: int,
    command: str,
    message_id: str,
    sent_at: str,
) -> None:
    """Append a ``command.accepted`` event to the session's event log."""
    payload: dict[str, object] = {
        "command": command,
        "message_id": message_id,
        "session_id": session_id,
        "sent_at": sent_at,
    }
    event_store.append(session_id=session_id, event_type=COMMAND_ACCEPTED, payload=payload, generation=generation)


def _emit_completed(
    event_store: EventStore,
    session_id: str,
    generation: int,
    command: str,
    message_id: str,
    outcome: str,
    duration_ms: int,
) -> None:
    """Append a ``command.completed`` event to the session's event log."""
    payload: dict[str, object] = {
        "command": command,
        "message_id": message_id,
        "outcome": outcome,
        "duration_ms": duration_ms,
    }
    event_store.append(session_id=session_id, event_type=COMMAND_COMPLETED, payload=payload, generation=generation)


def _emit_noop_pair(
    event_store: EventStore,
    session_id: str,
    generation: int,
    command: str,
    message_id: str,
    sent_at: str,
) -> None:
    """Emit paired ``accepted + completed(noop, duration_ms=0)`` for a command that cannot act."""
    _emit_accepted(event_store, session_id, generation, command, message_id, sent_at)
    _emit_completed(event_store, session_id, generation, command, message_id, OUTCOME_NOOP, 0)


def _emit_failed(
    event_store: EventStore,
    session_id: str,
    generation: int,
    command: str,
    message_id: str,
    error_kind: str,
    error_message: str,
) -> None:
    """Append a ``command.failed`` event to the session's event log."""
    payload: dict[str, object] = {
        "command": command,
        "message_id": message_id,
        "error": {"kind": error_kind, "message": error_message},
    }
    event_store.append(session_id=session_id, event_type=COMMAND_FAILED, payload=payload, generation=generation)


def _emit_session_restarted(event_store: EventStore, session_id: str, generation: int) -> None:
    """Append a ``session.restarted`` event tagged on the new generation."""
    payload: dict[str, object] = {"session_id": session_id, "generation": generation}
    event_store.append(session_id=session_id, event_type=SESSION_RESTARTED, payload=payload, generation=generation)


def advance_restart_on_spawn_success(
    state: ControlState,
    store: SessionStore,
    event_store: EventStore,
    session_id: str,
    new_generation: int,
    new_control_epoch: int,
) -> None:
    """Advance the row across a successful restart boundary and emit the paired events.

    Transitions the row ``restarting → starting`` (bumping ``generation`` and
    ``control_epoch`` atomically), emits ``session.restarted`` tagged on the
    new generation, transitions ``starting → running``, emits
    ``command.completed(restart, ok)`` tagged on the new generation, and
    clears the restart bookkeeping on *state*.
    """
    pending = state.restart_pending
    if pending is None:
        msg = "advance_restart_on_spawn_success called with no restart_pending"
        raise RuntimeError(msg)
    store.update(session_id, generation=new_generation, control_epoch=new_control_epoch, state=SessionState.starting)
    _emit_session_restarted(event_store, session_id, new_generation)
    store.update(session_id, state=SessionState.running)
    duration_ms = int((time.monotonic() - pending.start_monotonic) * 1000)
    _emit_completed(
        event_store, session_id, new_generation, RESTART_COMMAND, pending.message_id, OUTCOME_OK, duration_ms
    )
    state.restart_pending = None
    state.escalation_deadline = None
    state.escalation_fired = False


def advance_restart_on_spawn_failure(
    state: ControlState,
    store: SessionStore,
    event_store: EventStore,
    session_id: str,
    generation: int,
    error_kind: str,
    error_message: str,
) -> None:
    """Close a restart whose respawn exhausted its retry budget.

    Emits ``command.failed(restart, error=...)`` tagged on *generation* (the
    pre-bump generation, since the new generation never existed), transitions
    the row ``restarting → failed``, and clears the restart bookkeeping on
    *state*.
    """
    pending = state.restart_pending
    if pending is None:
        msg = "advance_restart_on_spawn_failure called with no restart_pending"
        raise RuntimeError(msg)
    _emit_failed(event_store, session_id, generation, RESTART_COMMAND, pending.message_id, error_kind, error_message)
    store.update(session_id, state=SessionState.failed)
    state.restart_pending = None
    state.escalation_deadline = None
    state.escalation_fired = False


def handle_stop(
    state: ControlState,
    proc: SupervisedProc,
    event_store: EventStore,
    session_id: str,
    generation: int,
    stop_timeout_seconds: float,
    store: SessionStore,
) -> None:
    """Dispatch a ``stop_requested`` flag: emit events, signal the child's pgroup, arm the escalation deadline.

    Noop paths emit a paired ``accepted + completed(noop)`` with no signal
    delivery. Three conditions trigger the noop path:

    1. The child has already exited.
    2. A prior stop or kill is still in flight.
    3. The session row has moved out of ``running`` between CLI validation
       and handler dispatch (TOCTOU).

    Happy path: transition the session row ``running → stopping``, record
    pending-command bookkeeping on *state*, emit ``command.accepted``, send
    SIGTERM to the child's process group, and set ``escalation_deadline``
    so the main loop escalates to SIGKILL if the child has not exited by then.
    """
    mid = _new_message_id()
    sent_at = _now_iso()
    now_mono = time.monotonic()
    if proc.poll() is not None or state.stopping or state.restart_pending is not None:
        _emit_noop_pair(event_store, session_id, generation, STOP_COMMAND, mid, sent_at)
        state.stop_requested = False
        return
    current = store.get(session_id)
    if current is None or current.state not in {SessionState.running, SessionState.suspended}:
        _emit_noop_pair(event_store, session_id, generation, STOP_COMMAND, mid, sent_at)
        state.stop_requested = False
        return
    store.update(session_id, state=SessionState.stopping)
    state.pending = PendingCommand(command=STOP_COMMAND, message_id=mid, start_monotonic=now_mono)
    state.stopping = True
    state.escalation_deadline = now_mono + stop_timeout_seconds
    state.escalation_fired = False
    state.completion_emitted = False
    _emit_accepted(event_store, session_id, generation, STOP_COMMAND, mid, sent_at)
    if not _killpg_if_alive(proc.pid, signal.SIGTERM):
        _emit_completed(event_store, session_id, generation, STOP_COMMAND, mid, OUTCOME_NOOP, 0)
        state.completion_emitted = True
        state.escalation_deadline = None
    state.stop_requested = False


def handle_kill(
    state: ControlState,
    proc: SupervisedProc,
    event_store: EventStore,
    session_id: str,
    generation: int,
    store: SessionStore,
) -> None:
    """Dispatch a ``kill_requested`` flag: emit events, signal the child's pgroup with SIGKILL.

    Six branches cover the reachable states:

    1. Child already exited:
           emit paired noop and return.
    2. Duplicate kill while ``state.stopping`` and pending is already kill:
           emit paired noop for the duplicate and return.
    3. Kill during a pending stop that has already escalated (``escalation_fired``):
           emit paired noop for this kill and leave the pending stop alone,
           so its terminal event reports ``outcome="escalated"``.
    4. Kill during a pre-escalation pending stop:
           close the stop with ``completed(noop, duration_ms=<elapsed>)``,
           open a fresh kill (new ``message_id``, new pending bookkeeping,
           clear ``escalation_deadline``), emit ``accepted(kill)``, and
           SIGKILL the group.
    5. TOCTOU (row has moved out of ``running``):
           emit paired noop.
    6. First kill, child alive, state is running:
           transition row to ``stopping``, set pending to kill,
           emit ``accepted(kill)``, and SIGKILL the group.
    """
    mid = _new_message_id()
    sent_at = _now_iso()
    now_mono = time.monotonic()
    if proc.poll() is not None or state.restart_pending is not None:
        _emit_noop_pair(event_store, session_id, generation, KILL_COMMAND, mid, sent_at)
        state.kill_requested = False
        return
    pending = state.pending
    if pending is not None and pending.command == KILL_COMMAND:
        _emit_noop_pair(event_store, session_id, generation, KILL_COMMAND, mid, sent_at)
        state.kill_requested = False
        return
    if pending is not None and pending.command == STOP_COMMAND and state.escalation_fired:
        _emit_noop_pair(event_store, session_id, generation, KILL_COMMAND, mid, sent_at)
        state.kill_requested = False
        return
    if pending is not None and pending.command == STOP_COMMAND and not state.escalation_fired:
        elapsed_ms = int((now_mono - pending.start_monotonic) * 1000)
        _emit_completed(event_store, session_id, generation, STOP_COMMAND, pending.message_id, OUTCOME_NOOP, elapsed_ms)
        state.pending = PendingCommand(command=KILL_COMMAND, message_id=mid, start_monotonic=now_mono)
        state.escalation_deadline = None
        state.escalation_fired = False
        state.completion_emitted = False
        _emit_accepted(event_store, session_id, generation, KILL_COMMAND, mid, sent_at)
        if not _killpg_if_alive(proc.pid, signal.SIGKILL):
            _emit_completed(event_store, session_id, generation, KILL_COMMAND, mid, OUTCOME_NOOP, 0)
            state.completion_emitted = True
        state.kill_requested = False
        return
    current = store.get(session_id)
    if current is None or current.state not in {SessionState.running, SessionState.suspended}:
        _emit_noop_pair(event_store, session_id, generation, KILL_COMMAND, mid, sent_at)
        state.kill_requested = False
        return
    store.update(session_id, state=SessionState.stopping)
    state.pending = PendingCommand(command=KILL_COMMAND, message_id=mid, start_monotonic=now_mono)
    state.stopping = True
    state.escalation_deadline = None
    state.escalation_fired = False
    state.completion_emitted = False
    _emit_accepted(event_store, session_id, generation, KILL_COMMAND, mid, sent_at)
    if not _killpg_if_alive(proc.pid, signal.SIGKILL):
        _emit_completed(event_store, session_id, generation, KILL_COMMAND, mid, OUTCOME_NOOP, 0)
        state.completion_emitted = True
    state.kill_requested = False


def handle_restart(
    state: ControlState,
    proc: SupervisedProc,
    event_store: EventStore,
    session_id: str,
    generation: int,
    stop_timeout_seconds: float,
    store: SessionStore,
) -> None:
    """Dispatch a ``restart_requested`` flag: record the pending restart, SIGTERM the child, emit accepted.

    Does not transition the session row. The outer generation loop in
    ``_supervise`` moves the row to ``restarting`` only after the old
    child has exited, matching ``restarting`` as the post-exit /
    pre-spawn window. ``command.completed(restart, ok)`` fires from that
    same outer loop after the new generation reaches ``running``.
    """
    mid = _new_message_id()
    sent_at = _now_iso()
    now_mono = time.monotonic()
    if proc.poll() is not None or state.stopping or state.restart_pending is not None:
        _emit_noop_pair(event_store, session_id, generation, RESTART_COMMAND, mid, sent_at)
        state.restart_requested = False
        return
    current = store.get(session_id)
    if current is None or current.state not in {SessionState.running, SessionState.suspended}:
        _emit_noop_pair(event_store, session_id, generation, RESTART_COMMAND, mid, sent_at)
        state.restart_requested = False
        return
    state.restart_pending = PendingCommand(command=RESTART_COMMAND, message_id=mid, start_monotonic=now_mono)
    state.escalation_deadline = now_mono + stop_timeout_seconds
    state.escalation_fired = False
    _emit_accepted(event_store, session_id, generation, RESTART_COMMAND, mid, sent_at)
    _killpg_if_alive(proc.pid, signal.SIGTERM)
    state.restart_requested = False


def handle_pause_observed(
    event_store: EventStore,
    session_id: str,
    generation: int,
    store: SessionStore,
) -> None:
    """Transition ``running`` to ``suspended`` and emit paired pause events.

    When the row is not ``running``, emits nothing so a concurrent
    ``handle_stop`` or ``handle_kill`` can own the terminal path.
    """
    current = store.get(session_id)
    if current is None or current.state != SessionState.running:
        return
    store.update(session_id, state=SessionState.suspended)
    mid = _new_message_id()
    sent_at = _now_iso()
    _emit_accepted(event_store, session_id, generation, PAUSE_COMMAND, mid, sent_at)
    _emit_completed(event_store, session_id, generation, PAUSE_COMMAND, mid, OUTCOME_OK, 0)


def handle_resume_observed(
    event_store: EventStore,
    session_id: str,
    generation: int,
    store: SessionStore,
) -> None:
    """Transition ``suspended`` to ``running`` and emit paired resume events.

    When the row is not ``suspended``, emits nothing.
    """
    current = store.get(session_id)
    if current is None or current.state != SessionState.suspended:
        return
    store.update(session_id, state=SessionState.running)
    mid = _new_message_id()
    sent_at = _now_iso()
    _emit_accepted(event_store, session_id, generation, RESUME_COMMAND, mid, sent_at)
    _emit_completed(event_store, session_id, generation, RESUME_COMMAND, mid, OUTCOME_OK, 0)


def _try_clean_acquire_controller(
    store: SessionStore,
    event_store: EventStore,
    session_id: str,
    generation: int,
    client_pid: int,
) -> int | None:
    """Atomic clean-acquire UPDATE plus ``session.controller_acquired`` event INSERT.

    Returns the new ``control_epoch`` on success, ``None`` when the row was
    not in the ``controller_pid IS NULL`` state. Used twice by
    ``handle_controller_acquired`` (initial attempt and post-clear retry).
    """
    cursor = store.conn.execute(
        "UPDATE sessions SET controller_pid = ?, control_acquired_at = ?, "
        "control_epoch = control_epoch + 1 "
        "WHERE session_id = ? AND controller_pid IS NULL "
        "RETURNING control_epoch",
        [client_pid, _now_iso(), session_id],
    )
    row = cursor.fetchone()
    if row is None:
        store.conn.commit()
        return None
    new_epoch = row[0]
    event_store.append(
        session_id=session_id,
        event_type=EVENT_SESSION_CONTROLLER_ACQUIRED,
        payload={"session_id": session_id, "control_epoch": new_epoch, "controller_pid": client_pid},
        generation=generation,
        commit=False,
    )
    store.conn.commit()
    return new_epoch


def handle_controller_acquired(
    store: SessionStore,
    event_store: EventStore,
    session_id: str,
    generation: int,
    client_pid: int,
) -> int | None:
    """Acquire the human-controller slot for *session_id*, reclaiming if the prior controller is dead.

    Returns the new ``control_epoch`` when *client_pid* becomes the
    controller, ``None`` when another live client already holds it
    (caller surfaces this as a usage error).

    Each row-mutating step pairs its UPDATE with the matching event INSERT
    inside one SQLite transaction, so other actors cannot observe a row
    change without the corresponding event already in the log:

    1. Try clean-acquire (``controller_pid IS NULL`` → take it). If we win,
       return the new epoch.
    2. Otherwise read the row. If the existing controller is alive (or the
       row is missing or already cleared), return ``None`` for contention.
    3. If the existing controller is dead, atomically clear the row and
       emit ``session.controller_released`` for the dead prior, then
       re-attempt clean-acquire. Return the new epoch on success or
       ``None`` if another acquirer raced in after our clear.
    """
    new_epoch = _try_clean_acquire_controller(store, event_store, session_id, generation, client_pid)
    if new_epoch is not None:
        return new_epoch

    current = store.get(session_id)
    if current is None or current.controller_pid is None:
        return None
    if check_pid(current.controller_pid) is not ProcessStatus.dead:
        return None

    old_pid = current.controller_pid
    cursor = store.conn.execute(
        "UPDATE sessions SET controller_pid = NULL, control_acquired_at = NULL "
        "WHERE session_id = ? AND controller_pid = ?",
        [session_id, old_pid],
    )
    if cursor.rowcount == 1:
        event_store.append(
            session_id=session_id,
            event_type=EVENT_SESSION_CONTROLLER_RELEASED,
            payload={"session_id": session_id},
            generation=generation,
            commit=False,
        )
    store.conn.commit()

    return _try_clean_acquire_controller(store, event_store, session_id, generation, client_pid)


def handle_controller_released(
    store: SessionStore,
    event_store: EventStore,
    session_id: str,
    generation: int,
    controller_pid: int | None = None,
) -> bool:
    """Atomically clear the controller row and emit ``session.controller_released``.

    Idempotent. The UPDATE and the event INSERT commit in a single SQLite
    transaction so other actors cannot observe ``controller_pid IS NULL``
    without the release event also being visible in the event log. When
    *controller_pid* is provided, the clear matches that specific value, so a
    stale disconnect from a prior controller cannot release a successor's row.
    Returns ``True`` when this call performed the release, ``False`` when the
    row had no matching controller (someone else already cleared it, or a
    successor took over).
    """
    if controller_pid is None:
        cursor = store.conn.execute(
            "UPDATE sessions SET controller_pid = NULL, control_acquired_at = NULL "
            "WHERE session_id = ? AND controller_pid IS NOT NULL",
            [session_id],
        )
    else:
        cursor = store.conn.execute(
            "UPDATE sessions SET controller_pid = NULL, control_acquired_at = NULL "
            "WHERE session_id = ? AND controller_pid = ?",
            [session_id, controller_pid],
        )
    released = cursor.rowcount == 1
    if released:
        event_store.append(
            session_id=session_id,
            event_type=EVENT_SESSION_CONTROLLER_RELEASED,
            payload={"session_id": session_id},
            generation=generation,
            commit=False,
        )
    store.conn.commit()
    return released


def check_escalation(state: ControlState, proc: SupervisedProc) -> None:
    """Fire SIGKILL on the child's pgroup if the stop-escalation deadline has passed.

    Called once per main-loop tick. Three conditions must all hold for this
    function to fire:

    1. ``escalation_deadline`` is armed (not None).
    2. The monotonic clock has reached or passed the deadline.
    3. ``proc.poll()`` shows the child still alive.

    When it fires, it emits no events. The terminal ``command.completed``
    with ``outcome="escalated"`` is emitted post-loop by
    ``emit_terminal_events`` after the child has actually exited.
    """
    if state.escalation_deadline is None:
        return
    if time.monotonic() < state.escalation_deadline:
        return
    if proc.poll() is not None:
        state.escalation_deadline = None
        return
    if _killpg_if_alive(proc.pid, signal.SIGKILL):
        state.escalation_fired = True
    state.escalation_deadline = None


def emit_terminal_events(
    state: ControlState,
    event_store: EventStore,
    session_id: str,
    generation: int,
) -> None:
    """Emit terminal ``command.completed`` events once the supervise loop has exited.

    Two duties:

    1. Close the pending stop or kill:
           if a command is pending (``state.stopping`` and
           ``completion_emitted == False``), emit
           ``completed(outcome="escalated" if escalation_fired else "ok")``
           with the elapsed duration in milliseconds.
    2. Close late-arrival signals:
           if ``stop_requested``, ``kill_requested``, or ``restart_requested``
           is still set, a signal arrived after the loop exited and was
           never dispatched. Emit a paired ``accepted + completed(noop)``
           for each set flag. ``restart_pending`` is left alone here because
           the outer generation loop in ``_supervise`` owns the in-flight
           restart's terminal event.
    """
    now_mono = time.monotonic()
    pending = state.pending
    if pending is not None and state.stopping and not state.completion_emitted:
        duration_ms = int((now_mono - pending.start_monotonic) * 1000)
        outcome = OUTCOME_ESCALATED if state.escalation_fired else OUTCOME_OK
        _emit_completed(event_store, session_id, generation, pending.command, pending.message_id, outcome, duration_ms)
        state.completion_emitted = True
    if state.stop_requested:
        mid = _new_message_id()
        sent_at = _now_iso()
        _emit_noop_pair(event_store, session_id, generation, STOP_COMMAND, mid, sent_at)
        state.stop_requested = False
    if state.kill_requested:
        mid = _new_message_id()
        sent_at = _now_iso()
        _emit_noop_pair(event_store, session_id, generation, KILL_COMMAND, mid, sent_at)
        state.kill_requested = False
    if state.restart_requested:
        mid = _new_message_id()
        sent_at = _now_iso()
        _emit_noop_pair(event_store, session_id, generation, RESTART_COMMAND, mid, sent_at)
        state.restart_requested = False
