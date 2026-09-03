"""Lifecycle boundaries for platform-managed workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class PlatformLifecycleState(str, Enum):
    """Coherent platform lifecycle states."""

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class PlatformLifecycle:
    """Minimal lifecycle state holder for platform-managed components."""

    state: PlatformLifecycleState = PlatformLifecycleState.CREATED
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def initialize(self) -> None:
        """Transition from created to initializing."""
        if self.state is not PlatformLifecycleState.CREATED:
            raise ValueError(f"Invalid Transition from {self.state.value} to initializing")
        self.state = PlatformLifecycleState.INITIALIZING

    def mark_ready(self) -> None:
        """Move the lifecycle to a ready-but-not-running state."""
        if self.state not in {PlatformLifecycleState.CREATED, PlatformLifecycleState.INITIALIZING}:
            raise ValueError(f"Invalid Transition from {self.state.value} to ready")
        self.state = PlatformLifecycleState.READY

    def start(self) -> None:
        """Transition from ready or stopped into a running state."""
        if self.state in {PlatformLifecycleState.READY, PlatformLifecycleState.STOPPED}:
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
        """Transition directly to the stopped state without a transient stopping state."""
        if self.state not in {PlatformLifecycleState.READY, PlatformLifecycleState.RUNNING, PlatformLifecycleState.PAUSED}:
            raise ValueError(f"Invalid Transition from {self.state.value} to stop")
        self.state = PlatformLifecycleState.STOPPED
        self.stopped_at = datetime.now(timezone.utc)

    def fail(self) -> None:
        """Transition to a failed lifecycle state."""
        if self.state in {PlatformLifecycleState.CREATED, PlatformLifecycleState.INITIALIZING, PlatformLifecycleState.READY, PlatformLifecycleState.RUNNING, PlatformLifecycleState.PAUSED}:
            self.state = PlatformLifecycleState.FAILED
            return
        raise ValueError(f"Invalid Transition from {self.state.value} to failed")
