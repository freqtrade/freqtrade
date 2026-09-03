"""Lifecycle boundaries for platform-managed workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class PlatformLifecycleState(str, Enum):
    """High-level lifecycle states for the platform layer."""

    CREATED = "created"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(slots=True)
class PlatformLifecycle:
    """Minimal lifecycle state holder for platform-managed components."""

    state: PlatformLifecycleState = PlatformLifecycleState.CREATED
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def start(self) -> None:
        """Transition to the starting state."""
        self.state = PlatformLifecycleState.STARTING
        self.started_at = datetime.now(timezone.utc)

    def stop(self) -> None:
        """Transition to the stopped state."""
        self.state = PlatformLifecycleState.STOPPED
        self.stopped_at = datetime.now(timezone.utc)
