"""Tests for pty_spawn: ManagedChild, poll observers, and respawn backoff."""

import os
import signal
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import pytest
from ptyprocess import PtyProcess, PtyProcessError

from psoul.core.launch import LaunchMode, build_launch_request
from psoul.core.pty_spawn import (
    _RESPAWN_BACKOFFS,
    ManagedChild,
    _poll_child_status,
    _respawn_with_backoff,
)


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
    _poll_until(pty_child, "exited")
    assert pty_child.returncode == 42
    with suppress(OSError):
        os.close(pty_child.main_fd)


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
