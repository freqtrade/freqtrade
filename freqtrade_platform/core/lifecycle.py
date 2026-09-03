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

    def mark_ready(self) -> None:
        """Move the lifecycle to a ready-but-not-running state."""
        if self.state is not PlatformLifecycleState.CREATED:
            raise ValueError(f"Invalid Transition from {self.state.value} to ready")
        self.state = PlatformLifecycleState.READY

    def start(self) -> None:
        """Transition from ready or stopped into a running state."""
        if self.state in {PlatformLifecycleState.CREATED, PlatformLifecycleState.READY, PlatformLifecycleState.STOPPED}:
            self.state = PlatformLifecycleState.RUNNING
            if self.started_at is None:
                self.started_at = datetime.now(timezone.utc)
            self.stopped_at = None
            return
        raise ValueError(f"Invalid Transition from {self.state.value} to start")

    def pause(self) -> None:
        """Pause a running platform lifecycle."""
        if self.state is not PlatformLifecycleState.RUNNING:
            raise ValueError(f"Invalid Transition from {self.state.value} to pause")
        self.state = PlatformLifecycleState.PAUSED

    def resume(self) -> None:
        """Resume a paused platform lifecycle."""
        if self.state is not PlatformLifecycleState.PAUSED:
            raise ValueError(f"Invalid Transition from {self.state.value} to resume")
        self.state = PlatformLifecycleState.RUNNING

    def stop(self) -> None:
        """Transition to the stopped state."""
        if self.state not in {
            PlatformLifecycleState.CREATED,
            PlatformLifecycleState.READY,
            PlatformLifecycleState.STARTING,
            PlatformLifecycleState.RUNNING,
            PlatformLifecycleState.PAUSED,
        }:
            raise ValueError(f"Invalid Transition from {self.state.value} to stop")
        self.state = PlatformLifecycleState.STOPPED
        self.stopped_at = datetime.now(timezone.utc)
