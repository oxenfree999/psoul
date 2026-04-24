"""Tests for pty_spawn: ManagedChild, poll observers, spawn, drain, and respawn backoff."""

import os
import selectors
import signal
import sys
import time
from collections.abc import Callable
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ptyprocess import PtyProcess, PtyProcessError

from psoul.core.db import open_db
from psoul.core.events import EVENT_RUNTIME_STDOUT, EventStore
from psoul.core.launch import LaunchMode, build_launch_request
from psoul.core.pty_spawn import (
    _RESPAWN_BACKOFFS,
    ManagedChild,
    _drain_tick,
    _finalize_exit,
    _poll_child_status,
    _respawn_with_backoff,
    _spawn_generation,
)
from psoul.core.session import Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION


def _spawn_pty_child(argv: list[str]) -> ManagedChild:
    """Spawn a child on a fresh PTY and wrap it in a ``ManagedChild`` for tests."""
    pty_proc = PtyProcess.spawn(argv)
    return ManagedChild(pid=pty_proc.pid, main_fd=pty_proc.fd, pty_process=pty_proc)


def _cleanup_pty_child(pty_child: ManagedChild) -> None:
    """Best-effort teardown for a test ``ManagedChild``. Idempotent on already-dead children."""
    with suppress(ProcessLookupError):
        os.kill(pty_child.pid, signal.SIGKILL)
    with suppress(ChildProcessError):
        os.waitpid(pty_child.pid, 0)
    with suppress(OSError):
        os.close(pty_child.main_fd)


def _poll_until(pty_child: ManagedChild, expected: str, timeout: float = 2.0) -> None:
    """Call ``_poll_child_status`` until it returns *expected* or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _poll_child_status(pty_child)
        if result == expected:
            return
        if result is not None:
            msg = f"expected {expected!r}, got {result!r}"
            raise AssertionError(msg)
        time.sleep(0.02)
    msg = f"did not observe {expected!r} within {timeout}s"
    raise AssertionError(msg)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_poll_child_status_returns_none_when_no_change() -> None:
    pty_child = _spawn_pty_child([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _poll_child_status(pty_child) is None
    finally:
        _cleanup_pty_child(pty_child)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_poll_child_status_returns_stopped_then_continued() -> None:
    pty_child = _spawn_pty_child([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        os.kill(pty_child.pid, signal.SIGSTOP)
        _poll_until(pty_child, "stopped")
        os.kill(pty_child.pid, signal.SIGCONT)
        _poll_until(pty_child, "continued")
    finally:
        _cleanup_pty_child(pty_child)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_poll_child_status_syncs_returncode_on_exit() -> None:
    pty_child = _spawn_pty_child([sys.executable, "-c", "import sys; sys.exit(42)"])
    try:
        _poll_until(pty_child, "exited")
        assert pty_child.returncode == 42
    finally:
        _cleanup_pty_child(pty_child)


@pytest.mark.parametrize(
    ("exc_factory", "expected_exc"),
    [
        (lambda: OSError("boom"), OSError),
        (lambda: PtyProcessError("boom"), PtyProcessError),
    ],
    ids=["oserror", "ptyprocesserror"],
)
def test_respawn_with_backoff_exhausts_and_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc_factory: Callable[[], Exception],
    expected_exc: type[Exception],
) -> None:
    """After every attempt raises, _respawn_with_backoff re-raises and sleeps between attempts.

    Covers both exception classes ``PtyProcess.spawn`` can raise: ``OSError``
    from the underlying fork/exec, and ``PtyProcessError`` from
    ptyprocess's own diagnostic path.
    """
    sleeps: list[float] = []
    monkeypatch.setattr("psoul.core.pty_spawn.time.sleep", sleeps.append)

    def _always_raise(*_args: object, **_kwargs: object) -> None:
        raise exc_factory()

    monkeypatch.setattr("psoul.core.pty_spawn._spawn_generation", _always_raise)
    request = build_launch_request(
        target="noop.py",
        module=None,
        extra_args=[],
        name="sesh-respawn",
        headless=True,
        tags=None,
        python_path=Path(sys.executable),
        default_mode=LaunchMode.attached,
    )
    with pytest.raises(expected_exc, match="boom"):
        _respawn_with_backoff(
            argv=request.target.as_cmd(),
            cwd=request.cwd,
            state_dir=tmp_path,
            session_id=request.session_id,
            new_generation=1,
        )
    assert sleeps == list(_RESPAWN_BACKOFFS)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_managed_child_poll_returns_none_when_alive() -> None:
    pty_child = _spawn_pty_child([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pty_child.poll() is None
        assert pty_child.returncode is None
    finally:
        _cleanup_pty_child(pty_child)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.parametrize(
    ("argv", "external_signal", "expected_returncode"),
    [
        ([sys.executable, "-c", "import sys; sys.exit(17)"], None, 17),
        ([sys.executable, "-c", "import time; time.sleep(30)"], signal.SIGKILL, -signal.SIGKILL),
    ],
    ids=["clean-exit-17", "signaled-sigkill"],
)
def test_managed_child_poll_captures_and_caches_returncode(
    argv: list[str],
    external_signal: int | None,
    expected_returncode: int,
) -> None:
    pty_child = _spawn_pty_child(argv)
    try:
        if external_signal is not None:
            os.kill(pty_child.pid, external_signal)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and pty_child.poll() is None:
            time.sleep(0.02)
        assert pty_child.returncode == expected_returncode
        # Second poll hits the cached short-circuit path.
        assert pty_child.poll() == expected_returncode
    finally:
        _cleanup_pty_child(pty_child)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_spawn_generation_returns_managed_child_with_valid_fds(tmp_path: Path) -> None:
    pty_child, sampler, sampler_thread = _spawn_generation(
        argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=Path.cwd(),
        state_dir=tmp_path,
        session_id="spawn-direct",
        generation=0,
    )
    try:
        assert pty_child.pid > 0
        assert pty_child.main_fd > 0
        assert pty_child.returncode is None
        assert os.isatty(pty_child.main_fd)
        assert sampler is not None
        assert sampler_thread is not None
    finally:
        _cleanup_pty_child(pty_child)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.parametrize(
    ("exit_code", "expected_state"),
    [
        (0, SessionState.exited),
        (1, SessionState.failed),
    ],
    ids=["clean-exit", "non-zero-exit"],
)
def test_finalize_exit_transitions_session_state(tmp_path: Path, exit_code: int, expected_state: SessionState) -> None:
    with closing(open_db(tmp_path)) as conn:
        store = SessionStore(conn)
        store.create(
            Session(
                session_id="finalize-direct",
                state=SessionState.starting,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
            )
        )
        store.update("finalize-direct", state=SessionState.running)
        pty_child = _spawn_pty_child([sys.executable, "-c", f"import sys; sys.exit({exit_code})"])
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and pty_child.poll() is None:
                time.sleep(0.02)
            assert pty_child.returncode == exit_code
            final = _finalize_exit(pty_child, "finalize-direct", store, start_monotonic=time.monotonic() - 0.5)
            assert final is not None
            assert final.state == expected_state
        finally:
            _cleanup_pty_child(pty_child)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_drain_tick_reads_pty_main_fd_into_events(tmp_path: Path) -> None:
    with closing(open_db(tmp_path)) as conn:
        SessionStore(conn).create(
            Session(
                session_id="drain-direct",
                state=SessionState.starting,
                launch_mode=LaunchMode.headless,
                launch_time=datetime.now(UTC),
                psoul_version=VERSION,
            )
        )
        event_store = EventStore(conn)
        pty_child = _spawn_pty_child(
            [sys.executable, "-c", "import sys; sys.stdout.write('hello'); sys.stdout.flush()"]
        )
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(pty_child.main_fd, selectors.EVENT_READ, EVENT_RUNTIME_STDOUT)
                deadline = time.monotonic() + 2.0
                saw_chunk = False
                while time.monotonic() < deadline and not saw_chunk:
                    saw_chunk = _drain_tick(sel, event_store, "drain-direct", 0, timeout=0.1)
            assert saw_chunk
            events = event_store.list("drain-direct", event_type=EVENT_RUNTIME_STDOUT)
            assert any("hello" in str(e["payload"]) for e in events)
        finally:
            _cleanup_pty_child(pty_child)
