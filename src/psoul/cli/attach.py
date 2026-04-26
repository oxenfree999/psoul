"""``psoul attach``: interactive proxy between a local TTY and a headless supervisor's PTY."""

import fcntl
import os
import selectors
import signal
import socket
import sys
import termios
import tty
from contextlib import suppress
from pathlib import Path
from types import FrameType

from psoul.core.pty_spawn import (
    _FRAME_DATA,
    _FRAME_HELLO,
    _FRAME_WINSIZE,
    _HELLO_PAYLOAD,
    _WINSIZE_STRUCT,
    _decode_frames,
    _encode_frame,
)

_DETACH_ESCAPE = bytes([0x1D])  # Ctrl-], hardcoded detach trigger
_RECV_BUFSIZE = 8192  # bytes per recv()/read() in the I/O loop
_LOOP_TIMEOUT = 0.1  # selector tick, also the SIGWINCH service interval


def _read_local_winsize() -> bytes:
    """Read the CLI's stdout terminal winsize as the 8-byte ``struct winsize`` payload."""
    return fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, bytes(_WINSIZE_STRUCT.size))


def _handle_socket_read(sock: socket.socket, decode_buffer: bytearray) -> bool:
    """Drain readable bytes from *sock*, write DATA payloads to stdout. Returns ``False`` on EOF or error."""
    try:
        data = sock.recv(_RECV_BUFSIZE)
    except OSError:
        return False
    if not data:
        return False
    decode_buffer.extend(data)
    for kind, payload in _decode_frames(decode_buffer):
        if kind == _FRAME_DATA:
            os.write(sys.stdout.fileno(), payload)
    return True


def _handle_stdin_read(sock: socket.socket) -> bool:
    """Read stdin, send pre-escape bytes, return ``True`` on ``Ctrl-]`` detach or stdin EOF."""
    data = os.read(sys.stdin.fileno(), _RECV_BUFSIZE)
    if not data:
        return True
    head, sep, _tail = data.partition(_DETACH_ESCAPE)
    if head:
        with suppress(OSError):
            sock.sendall(_encode_frame(_FRAME_DATA, head))
    return bool(sep)


def _io_loop(sock: socket.socket) -> None:
    """Run the selector-driven I/O loop until detach or supervisor disconnect.

    Sets the local TTY to raw mode for the duration. Installs a SIGWINCH
    handler that flips a pending flag, serviced once per tick to forward
    the new winsize. The signal handler does no I/O so it stays
    async-signal-safe.
    """
    stdin_fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(stdin_fd)
    tty.setraw(stdin_fd)
    winch_pending = [True]

    def _on_winch(_signum: int, _frame: FrameType | None) -> None:
        winch_pending[0] = True

    prior_winch = signal.signal(signal.SIGWINCH, _on_winch)
    decode_buffer = bytearray()
    try:
        with selectors.DefaultSelector() as sel:
            sel.register(sock.fileno(), selectors.EVENT_READ)
            sel.register(stdin_fd, selectors.EVENT_READ)
            while True:
                if winch_pending[0]:
                    winch_pending[0] = False
                    with suppress(OSError):
                        sock.sendall(_encode_frame(_FRAME_WINSIZE, _read_local_winsize()))
                for key, _mask in sel.select(timeout=_LOOP_TIMEOUT):
                    if key.fd == sock.fileno():
                        if not _handle_socket_read(sock, decode_buffer):
                            return
                    elif _handle_stdin_read(sock):
                        return
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_termios)
        signal.signal(signal.SIGWINCH, prior_winch)


def run_attach_loop(socket_path: Path, client_pid: int) -> None:
    """Connect to the supervisor's listen socket, send HELLO, and run the I/O loop until detach or disconnect."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(socket_path))
        sock.setblocking(False)
        sock.sendall(_encode_frame(_FRAME_HELLO, _HELLO_PAYLOAD.pack(client_pid)))
        _io_loop(sock)
    finally:
        sock.close()
