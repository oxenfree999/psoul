"""Linux-only integration test for PR_SET_CHILD_SUBREAPER reparent and supervisor reap."""

import os
import signal
import sys
import time
from contextlib import closing, suppress
from pathlib import Path

import psutil
import pytest

from psoul.core.db import open_db
from psoul.core.launch import LaunchRequest, LaunchTarget, launch_headless
from psoul.core.session import LaunchMode, TargetType
from psoul.core.store import SessionStore

_FIXTURE_SCRIPT = (
    "import os, pathlib, sys, time\n"
    "grandchild_pid_file = pathlib.Path(sys.argv[1])\n"
    "new_ppid_file = pathlib.Path(sys.argv[2])\n"
    "parent_pid = os.getpid()\n"
    "if os.fork() == 0:\n"
    "    grandchild_pid_file.write_text(str(os.getpid()))\n"
    "    deadline = time.monotonic() + 5.0\n"
    "    while time.monotonic() < deadline and os.getppid() == parent_pid:\n"
    "        time.sleep(0.01)\n"
    "    new_ppid_file.write_text(str(os.getppid()))\n"
    "    while True:\n"
    "        time.sleep(0.1)\n"
    "sys.exit(0)\n"
)


def _wait_for_path(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    msg = f"path {path} did not appear within {timeout}s"
    raise RuntimeError(msg)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_supervisor_reparents_and_reaps_orphan_grandchild(tmp_path: Path) -> None:
    grandchild_pid_file = tmp_path / "grandchild.pid"
    new_ppid_file = tmp_path / "new_ppid"
    grandchild_pid: int | None = None
    supervisor_pid: int | None = None
    try:
        with closing(open_db(tmp_path)) as conn:
            store = SessionStore(conn)
            req = LaunchRequest(
                session_id="subreaper",
                launch_mode=LaunchMode.headless,
                target=LaunchTarget(
                    target_type=TargetType.script,
                    target="-c",
                    target_args=(_FIXTURE_SCRIPT, str(grandchild_pid_file), str(new_ppid_file)),
                    python_path=Path(sys.executable),
                ),
                cwd=tmp_path,
            )
            _, supervisor_pid = launch_headless(req, store, tmp_path)
        _wait_for_path(grandchild_pid_file, 5.0)
        grandchild_pid = int(grandchild_pid_file.read_text())
        _wait_for_path(new_ppid_file, 5.0)
        assert int(new_ppid_file.read_text()) == supervisor_pid
        os.kill(grandchild_pid, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                children = psutil.Process(supervisor_pid).children(recursive=True)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pytest.fail("supervisor exited before grandchild reap was observable")
            if grandchild_pid not in {c.pid for c in children}:
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"supervisor did not reap grandchild {grandchild_pid} within 5s")
    finally:
        if grandchild_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(grandchild_pid, signal.SIGKILL)
        if supervisor_pid is not None:
            with suppress(ChildProcessError):
                os.waitpid(supervisor_pid, 0)
