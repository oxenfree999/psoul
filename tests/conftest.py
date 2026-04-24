import sys
from collections.abc import Iterator

import pytest
import structlog

collect_ignore_glob: list[str] = []
if sys.platform == "win32":
    # ptyprocess is a Unix-only dependency. Skip collecting pty_spawn tests on Windows.
    collect_ignore_glob.append("test_pty_spawn.py")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("CLICOLOR", raising=False)
    monkeypatch.delenv("CLICOLOR_FORCE", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("PSOUL_LOG", raising=False)
    monkeypatch.setenv("COLUMNS", "80")
    # Prevent Rich from reducing width by 1 on Windows legacy consoles.
    # Rich's own test suite does the equivalent via legacy_windows=False.
    monkeypatch.setattr("rich.console.WINDOWS", False)


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()
