import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from psoul.names import generate_session_id
from psoul.session import LaunchMode, TargetType, validate_session_id


@dataclass(frozen=True, slots=True)
class LaunchTarget:
    """Validated target to run: script path or module name plus arguments."""

    target_type: TargetType
    target: str
    target_args: tuple[str, ...]

    def as_cmd(self) -> list[str]:
        """Build the subprocess command list."""
        prefix = (
            [sys.executable, "-m", self.target]
            if self.target_type == TargetType.module
            else [sys.executable, self.target]
        )
        return [*prefix, *self.target_args]


def parse_launch_target(*, target: str | None, module: str | None, extra_args: Sequence[str]) -> LaunchTarget:
    """Build a LaunchTarget from mutually exclusive CLI inputs."""
    if target is not None and module is not None:
        raise ValueError("choose either a script target or -m module")
    if module is not None:
        return LaunchTarget(target_type=TargetType.module, target=module, target_args=tuple(extra_args))
    if target is None:
        raise ValueError("launch target is required")
    return LaunchTarget(target_type=TargetType.script, target=target, target_args=tuple(extra_args))


def resolve_session_id(name: str | None) -> str:
    """Return a validated session ID from --name, or generate one."""
    if name is not None:
        return validate_session_id(name)
    return generate_session_id()


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    """Frozen snapshot of everything needed to create a session."""

    session_id: str
    launch_mode: LaunchMode
    target: LaunchTarget
    cwd: Path
    tags: Mapping[str, str] | None = None


def build_launch_request(
    *,
    target: str | None,
    module: str | None,
    extra_args: Sequence[str],
    name: str | None,
    headless: bool,
    tags: dict[str, str] | None,
) -> LaunchRequest:
    """Assemble a frozen LaunchRequest from CLI inputs."""
    return LaunchRequest(
        session_id=resolve_session_id(name),
        launch_mode=LaunchMode.headless if headless or not sys.stdin.isatty() else LaunchMode.attached,
        target=parse_launch_target(target=target, module=module, extra_args=extra_args),
        cwd=Path.cwd(),
        tags=MappingProxyType(dict(tags)) if tags is not None else None,
    )
