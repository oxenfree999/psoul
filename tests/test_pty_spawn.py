"""Tests for pty_spawn: ManagedChild, poll observers, spawn, drain, and respawn backoff."""

import errno
import fcntl
import os
import selectors
import signal
import socket
import sqlite3
import sys
import tempfile
import termios
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ptyprocess import PtyProcess, PtyProcessError

from psoul.core.db import open_db
from psoul.core.events import EVENT_RUNTIME_STDOUT, EVENT_SESSION_CONTROLLER_RELEASED, EventStore
from psoul.core.launch import LaunchMode, build_launch_request
from psoul.core.pty_spawn import (
    _FRAME_DATA,
    _FRAME_HELLO,
    _FRAME_WINSIZE,
    _HELLO_PAYLOAD,
    _RESPAWN_BACKOFFS,
    _TAG_PTY_MAIN,
    _WINSIZE_STRUCT,
    ManagedChild,
    _AttachClient,
    _authenticate_attach_client,
    _cleanup_listen_socket,
    _create_listen_socket,
    _decode_frames,
    _disconnect_attach_client,
    _drain_tick,
    _encode_frame,
    _finalize_exit,
    _handle_attach_client_read,
    _poll_child_status,
    _respawn_with_backoff,
    _send_replay,
    _spawn_generation,
)
from psoul.core.session import Session, SessionState
from psoul.core.store import SessionStore
from psoul.version import VERSION

_TEST_RECV_BUFSIZE = 1024  # bytes per recv() in attach-client tests, sized for small fixtures
_TEST_WS = (50, 80, 0, 0)  # rows, cols, xpixel, ypixel for winsize round-trip tests
_TEST_REPLAY_FRAME_CAP = 10  # patched ``_MAX_FRAME_PAYLOAD`` ceiling for replay-chunking tests


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


def _seed_session(conn: sqlite3.Connection, session_id: str, controller_pid: int | None = None) -> None:
    """Insert a starting-state session row, optionally with a controller_pid set."""
    SessionStore(conn).create(
        Session(
            session_id=session_id,
            state=SessionState.starting,
            launch_mode=LaunchMode.headless,
            launch_time=datetime.now(UTC),
            psoul_version=VERSION,
            controller_pid=controller_pid,
        )
    )


@contextmanager
def _attach_socketpair() -> Iterator[tuple[socket.socket, _AttachClient]]:
    """Yield ``(test_sender, supervisor_side_AttachClient)`` over an ``AF_UNIX`` socketpair. Closes both on exit."""
    sender, recv = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sender.setblocking(False)
    recv.setblocking(False)
    with sender, recv:
        yield sender, _AttachClient(sock=recv)


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
                sel.register(pty_child.main_fd, selectors.EVENT_READ, _TAG_PTY_MAIN)
                deadline = time.monotonic() + 2.0
                saw_chunk = False
                while time.monotonic() < deadline and not saw_chunk:
                    saw_chunk = _drain_tick(
                        sel,
                        event_store,
                        "drain-direct",
                        generation=0,
                        timeout=0.1,
                        clients={},
                        store=SessionStore(conn),
                    )
            assert saw_chunk
            events = event_store.list("drain-direct", event_type=EVENT_RUNTIME_STDOUT)
            assert any("hello" in str(e["payload"]) for e in events)
        finally:
            _cleanup_pty_child(pty_child)


@pytest.mark.parametrize("preexisting", [False, True], ids=["fresh", "stale-file-exists"])
def test_create_listen_socket_binds_at_short_tmp_path_with_user_private_mode(preexisting: bool) -> None:
    session_id = "create-listen-test"
    expected_path = Path(f"/tmp/psoul-{os.getuid()}-{session_id}.sock")
    if preexisting:
        expected_path.touch()
    sock, path = _create_listen_socket(session_id)
    try:
        assert sock is not None
        assert path is not None
        assert path == expected_path
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600
        assert sock.family == socket.AF_UNIX
        assert sock.type == socket.SOCK_STREAM
    finally:
        _cleanup_listen_socket(sock, path)


def test_create_listen_socket_returns_none_on_bind_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def bind_fails(self: socket.socket, _addr: object) -> None:
        raise OSError(errno.ENAMETOOLONG, "AF_UNIX path too long")

    monkeypatch.setattr("socket.socket.bind", bind_fails)
    sock, path = _create_listen_socket("bind-fail-test")
    assert sock is None
    assert path is None
    expected_path = Path(f"/tmp/psoul-{os.getuid()}-bind-fail-test.sock")
    assert not expected_path.exists()


def test_cleanup_listen_socket_handles_none_inputs() -> None:
    _cleanup_listen_socket(None, None)


def test_cleanup_listen_socket_closes_socket_and_unlinks_path(tmp_path: Path) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    path = tmp_path / "stand-in-for-socket"
    path.touch()
    _cleanup_listen_socket(sock, path)
    assert not path.exists()
    assert sock.fileno() == -1


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (_FRAME_DATA, b"hello world"),
        (_FRAME_WINSIZE, b"\x00\x18\x00\x50\x00\x00\x00\x00"),
        (_FRAME_HELLO, b"\x00\x00\x00\x42"),
        (_FRAME_DATA, b""),
    ],
    ids=["data-text", "winsize-8b", "hello-pid", "empty-data"],
)
def test_encode_decode_frame_round_trips(kind: int, payload: bytes) -> None:
    buffer = bytearray(_encode_frame(kind, payload))
    frames = _decode_frames(buffer)
    assert frames == [(kind, payload)]
    assert len(buffer) == 0


def test_decode_frames_consumes_complete_and_leaves_partial() -> None:
    full = _encode_frame(_FRAME_DATA, b"complete") + _encode_frame(_FRAME_HELLO, b"\x00\x00\x00\x05")
    buffer = bytearray(full[:-2])
    frames = _decode_frames(buffer)
    assert frames == [(_FRAME_DATA, b"complete")]
    assert len(buffer) > 0
    buffer.extend(full[-2:])
    frames = _decode_frames(buffer)
    assert frames == [(_FRAME_HELLO, b"\x00\x00\x00\x05")]
    assert len(buffer) == 0


@pytest.mark.parametrize(
    ("kind", "payload", "expected_ok"),
    [
        (_FRAME_HELLO, _HELLO_PAYLOAD.pack(12345), True),
        (_FRAME_HELLO, _HELLO_PAYLOAD.pack(99999), False),
        (_FRAME_DATA, _HELLO_PAYLOAD.pack(12345), False),
        (_FRAME_HELLO, b"\x00\x00\x00", False),
    ],
    ids=["match", "pid-mismatch", "wrong-frame-kind", "bad-payload-length"],
)
def test_authenticate_attach_client(tmp_path: Path, kind: int, payload: bytes, expected_ok: bool) -> None:
    session_id = "auth"
    with closing(open_db(tmp_path)) as conn, _attach_socketpair() as (sender, client):
        _seed_session(conn, session_id, controller_pid=12345)
        event_store = EventStore(conn)
        event_store.append(
            session_id=session_id, event_type=EVENT_RUNTIME_STDOUT, payload={"text": "hello"}, generation=0
        )
        ok = _authenticate_attach_client(client, kind, payload, SessionStore(conn), session_id, event_store)
        assert ok is expected_ok
        assert client.authenticated is expected_ok
        if expected_ok:
            frames = _decode_frames(bytearray(sender.recv(_TEST_RECV_BUFSIZE)))
            assert any(k == _FRAME_DATA and b"hello" in p for k, p in frames)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_handle_attach_client_read_routes_data_and_winsize_to_pty(tmp_path: Path) -> None:
    session_id = "route"
    with closing(open_db(tmp_path)) as conn, _attach_socketpair() as (sender, client):
        _seed_session(conn, session_id, controller_pid=os.getpid())
        client.authenticated = True
        pty_child = _spawn_pty_child([sys.executable, "-c", "import sys; sys.stdin.readline()"])
        try:
            sender.sendall(_encode_frame(_FRAME_WINSIZE, _WINSIZE_STRUCT.pack(*_TEST_WS)))
            sender.sendall(_encode_frame(_FRAME_DATA, b"hi\n"))
            ok = _handle_attach_client_read(client, pty_child.main_fd, SessionStore(conn), session_id, EventStore(conn))
            assert ok is True
            actual = _WINSIZE_STRUCT.unpack(
                fcntl.ioctl(pty_child.main_fd, termios.TIOCGWINSZ, bytes(_WINSIZE_STRUCT.size))
            )
            assert actual == _TEST_WS
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and pty_child.poll() is None:
                time.sleep(0.05)
            assert pty_child.returncode == 0
        finally:
            _cleanup_pty_child(pty_child)


@pytest.mark.parametrize(
    "trigger",
    [
        lambda s: s.sendall(_encode_frame(_FRAME_WINSIZE, b"x")),
        lambda s: s.close(),
    ],
    ids=["bad-winsize-payload", "client-eof"],
)
def test_handle_attach_client_read_drops(
    tmp_path: Path,
    trigger: Callable[[socket.socket], None],
) -> None:
    with (
        closing(open_db(tmp_path)) as conn,
        _attach_socketpair() as (sender, client),
        tempfile.TemporaryFile() as pty_main,
    ):
        client.authenticated = True
        trigger(sender)
        assert (
            _handle_attach_client_read(client, pty_main.fileno(), SessionStore(conn), "drop", EventStore(conn)) is False
        )


def test_send_replay_chunks_long_text_into_multiple_data_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("psoul.core.pty_spawn._MAX_FRAME_PAYLOAD", _TEST_REPLAY_FRAME_CAP)
    session_id = "replay"
    text = "a" * (_TEST_REPLAY_FRAME_CAP + 1)
    with closing(open_db(tmp_path)) as conn, _attach_socketpair() as (sender, client):
        _seed_session(conn, session_id)
        event_store = EventStore(conn)
        event_store.append(session_id=session_id, event_type=EVENT_RUNTIME_STDOUT, payload={"text": text}, generation=0)
        assert _send_replay(client, event_store, session_id) is True
        frames = _decode_frames(bytearray(sender.recv(_TEST_RECV_BUFSIZE)))
        assert len(frames) == 2
        assert all(k == _FRAME_DATA for k, _ in frames)
        assert b"".join(p for _, p in frames) == text.encode()


@pytest.mark.parametrize(
    ("authenticated", "row_pid_after_auth", "expected_release_count"),
    [(True, 12345, 1), (False, 12345, 0), (True, 99999, 0)],
    ids=["authenticated-releases", "unauthenticated-no-release", "stale-authenticated-no-release"],
)
def test_disconnect_attach_client_releases(
    tmp_path: Path,
    authenticated: bool,
    row_pid_after_auth: int,
    expected_release_count: int,
) -> None:
    session_id = "release"
    initial_pid = 12345
    with closing(open_db(tmp_path)) as conn, _attach_socketpair() as (_sender, client):
        _seed_session(conn, session_id, controller_pid=initial_pid)
        store = SessionStore(conn)
        event_store = EventStore(conn)
        client.authenticated = authenticated
        if authenticated:
            client.client_pid = initial_pid
        if row_pid_after_auth != initial_pid:
            store.update(session_id, controller_pid=row_pid_after_auth)
        with selectors.DefaultSelector() as sel:
            _disconnect_attach_client(client, sel, {client.sock.fileno(): client}, store, event_store, session_id, 0)
        events = event_store.list(session_id, event_type=EVENT_SESSION_CONTROLLER_RELEASED)
        assert len(events) == expected_release_count
        row = store.get(session_id)
        assert row is not None
        assert row.controller_pid == (None if expected_release_count == 1 else row_pid_after_auth)
