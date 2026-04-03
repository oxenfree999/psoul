"""Provenance capture: git info, file hashes, and platform metadata."""

import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import TypedDict

from psoul.session import TargetType

_LOCKFILE_CANDIDATES = ("uv.lock", "poetry.lock", "pdm.lock", "Pipfile.lock", "requirements.txt")


class SessionProvenance(TypedDict):
    """Session fields populated from launch-time provenance."""

    git_sha: str | None
    git_dirty: bool | None
    lockfile_hash: str | None
    script_hash: str | None
    python_version: str
    python_path: Path
    host: str
    os: str
    arch: str


def _is_absolute_target(target: str) -> bool:
    """Return True for absolute paths on the current OS or Windows."""
    return Path(target).is_absolute() or PureWindowsPath(target).is_absolute()


def git_sha(cwd: Path) -> str | None:
    """Return the full SHA of HEAD, or None if unavailable."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 — git path comes from shutil.which
            [git, "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def git_dirty(cwd: Path) -> bool | None:
    """Return True if the working tree has changes including untracked files, or None if unavailable."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 — git path comes from shutil.which
            [git, "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, OSError):
        return None


def file_hash(path: Path) -> str | None:
    """Return ``sha256:<hex>`` digest of a file, or None if unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    digest = hashlib.sha256(data, usedforsecurity=False).hexdigest()
    return f"sha256:{digest}"


def find_lockfile_hash(cwd: Path) -> str | None:
    """Hash the first lockfile found in *cwd*, checked in priority order.

    True lockfiles (uv.lock, poetry.lock, pdm.lock, Pipfile.lock) are
    preferred.  ``requirements.txt`` is a last-resort fallback — it may be
    a ``pip freeze`` snapshot or a hand-written list, but either way a
    content change signals a dependency shift worth recording.
    """
    for name in _LOCKFILE_CANDIDATES:
        result = file_hash(cwd / name)
        if result is not None:
            return result
    return None


def script_hash(target_type: TargetType, target: str | None, cwd: Path) -> str | None:
    """Hash the script file if target is a real file, otherwise None.

    Returns None for non-script targets, pseudo-targets like ``-c`` and
    ``-`` (stdin), and files that don't exist on disk.
    """
    if target_type != TargetType.script or target is None:
        return None
    if target in {"-c", "-"}:
        return None
    path = Path(target) if _is_absolute_target(target) else cwd / target
    return file_hash(path)


def gather(target_type: TargetType, target: str | None, cwd: Path) -> SessionProvenance:
    """Collect all provenance fields as a dict compatible with Session kwargs."""
    return {
        "git_sha": git_sha(cwd),
        "git_dirty": git_dirty(cwd),
        "script_hash": script_hash(target_type, target, cwd),
        "lockfile_hash": find_lockfile_hash(cwd),
        "python_version": platform.python_version(),
        "python_path": Path(sys.executable),
        "host": platform.node(),
        "os": sys.platform,
        "arch": platform.machine(),
    }
