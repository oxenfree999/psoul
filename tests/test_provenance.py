"""Tests for provenance capture: git info, file hashes, and platform metadata."""

import hashlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from psoul.core.provenance import (
    file_hash,
    find_lockfile_hash,
    gather,
    git_dirty,
    git_sha,
    script_hash,
)
from psoul.core.session import TargetType

FAKE_SHA = "a" * 40


def _mock_subprocess(result: object) -> Callable[..., object]:
    """Return a callable that either returns *result* or raises it."""

    def mock(*_args: object, **_kwargs: object) -> object:
        if isinstance(result, BaseException):
            raise result
        return result

    return mock


@pytest.mark.parametrize(
    ("func", "which_result", "run_result", "expected"),
    [
        (git_sha, "/usr/bin/git", subprocess.CompletedProcess([], 0, stdout=FAKE_SHA + "\n"), FAKE_SHA),
        (git_sha, None, None, None),
        (git_sha, "/usr/bin/git", subprocess.CalledProcessError(128, "git"), None),
        (git_sha, "/usr/bin/git", subprocess.TimeoutExpired("git", 5), None),
        (git_dirty, "/usr/bin/git", subprocess.CompletedProcess([], 0, stdout=""), False),
        (git_dirty, "/usr/bin/git", subprocess.CompletedProcess([], 0, stdout=" M dirty.py\n?? new.py\n"), True),
        (git_dirty, None, None, None),
        (git_dirty, "/usr/bin/git", subprocess.CalledProcessError(128, "git"), None),
        (git_dirty, "/usr/bin/git", subprocess.TimeoutExpired("git", 5), None),
    ],
    ids=[
        "sha-ok",
        "sha-no-git",
        "sha-no-repo",
        "sha-timeout",
        "dirty-clean",
        "dirty-dirty",
        "dirty-no-git",
        "dirty-no-repo",
        "dirty-timeout",
    ],
)
def test_git_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    func: Callable[..., object],
    which_result: object,
    run_result: object,
    expected: object,
) -> None:
    monkeypatch.setattr("psoul.core.provenance.shutil.which", lambda _: which_result)
    if run_result is not None:
        monkeypatch.setattr("psoul.core.provenance.subprocess.run", _mock_subprocess(run_result))
    assert func(tmp_path) == expected


def test_git_subprocess_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both git helpers pass timeout=5 to subprocess.run."""
    captured: list[dict[str, object]] = []

    def spy(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(kwargs)
        return subprocess.CompletedProcess([], 0, stdout=FAKE_SHA + "\n")

    monkeypatch.setattr("psoul.core.provenance.shutil.which", lambda _: "/usr/bin/git")
    monkeypatch.setattr("psoul.core.provenance.subprocess.run", spy)
    git_sha(tmp_path)
    git_dirty(tmp_path)
    assert len(captured) == 2
    assert captured[0]["timeout"] == 5  # git_sha
    assert captured[1]["timeout"] == 5  # git_dirty


_HELLO_HASH = f"sha256:{hashlib.sha256(b'hello world', usedforsecurity=False).hexdigest()}"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"hello world", _HELLO_HASH),
        (None, None),
    ],
    ids=["real-file", "missing"],
)
def test_file_hash(tmp_path: Path, content: bytes | None, expected: str | None) -> None:
    f = tmp_path / "test.txt"
    if content is not None:
        f.write_bytes(content)
    assert file_hash(f) == expected


@pytest.mark.parametrize(
    ("files", "expected_name"),
    [
        ({"uv.lock": b"uv", "requirements.txt": b"req"}, "uv.lock"),
        ({"poetry.lock": b"poetry", "pdm.lock": b"pdm"}, "poetry.lock"),
        ({"pdm.lock": b"pdm", "Pipfile.lock": b"pipenv"}, "pdm.lock"),
        ({"Pipfile.lock": b"pipenv", "requirements.txt": b"req"}, "Pipfile.lock"),
        ({"requirements.txt": b"req"}, "requirements.txt"),
        ({}, None),
    ],
    ids=[
        "prefers-uv-lock",
        "prefers-poetry-over-pdm",
        "prefers-pdm-over-pipfile",
        "prefers-pipfile-over-requirements",
        "falls-back-to-requirements",
        "no-lockfile",
    ],
)
def test_find_lockfile_hash(tmp_path: Path, files: dict[str, bytes], expected_name: str | None) -> None:
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    result = find_lockfile_hash(tmp_path)
    if expected_name is None:
        assert result is None
    else:
        assert result == file_hash(tmp_path / expected_name)


@pytest.mark.parametrize(
    ("target_type", "target", "write_file", "expect_hash"),
    [
        (TargetType.script, "train.py", True, True),
        (TargetType.module, "http.server", False, False),
        (TargetType.script, "-c", False, False),
        (TargetType.script, "-", False, False),
        (TargetType.script, "missing.py", False, False),
    ],
    ids=["real-script", "module", "dash-c", "stdin", "missing-file"],
)
def test_script_hash(
    tmp_path: Path, target_type: TargetType, target: str | None, write_file: bool, expect_hash: bool
) -> None:
    if write_file and target is not None:
        (tmp_path / target).write_bytes(b"print('hi')")
    result = script_hash(target_type, target, tmp_path)
    if expect_hash:
        assert result is not None
        assert result.startswith("sha256:")
    else:
        assert result is None


def test_gather(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psoul.core.provenance.shutil.which", lambda _: "/usr/bin/git")

    def mock_run(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess([], 0, stdout=FAKE_SHA + "\n")
        return subprocess.CompletedProcess([], 0, stdout=" M dirty.py\n")

    monkeypatch.setattr("psoul.core.provenance.subprocess.run", mock_run)
    monkeypatch.setattr("psoul.core.provenance.platform.python_version", lambda: "3.14.0")
    monkeypatch.setattr("psoul.core.provenance.platform.node", lambda: "testhost")
    monkeypatch.setattr("psoul.core.provenance.platform.machine", lambda: "aarch64")
    monkeypatch.setattr(sys, "platform", "linux")

    script = tmp_path / "train.py"
    script.write_bytes(b"print('hi')")
    (tmp_path / "uv.lock").write_bytes(b"lockfile content")

    result = gather(TargetType.script, "train.py", tmp_path, Path("/usr/bin/python3"))

    assert result["git_sha"] == FAKE_SHA
    assert result["git_dirty"] is True
    assert result["script_hash"] == file_hash(script)
    assert result["lockfile_hash"] == file_hash(tmp_path / "uv.lock")
    assert result["python_version"] == "3.14.0"
    assert result["python_path"] == Path("/usr/bin/python3")
    assert result["host"] == "testhost"
    assert result["os"] == "linux"
    assert result["arch"] == "aarch64"
