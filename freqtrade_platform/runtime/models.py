"""Domain models and state definitions for Strategy Runtimes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


class RuntimeState(str, Enum):
    """Lifecycle states for a strategy runtime instance."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    READY = "READY"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class RuntimeMode(str, Enum):
    """Runtime execution modes."""

    BACKTEST = "BACKTEST"
    DRY_RUN = "DRY_RUN"
    LIVE = "LIVE"


class MarketType(str, Enum):
    """Market types supported by runtimes."""

    SPOT = "SPOT"
    FUTURES = "FUTURES"


# Allowed state transitions for StrategyRuntimeInstance
_VALID_TRANSITIONS: dict[RuntimeState, set[RuntimeState]] = {
    RuntimeState.CREATED: {RuntimeState.VALIDATING, RuntimeState.FAILED},
    RuntimeState.VALIDATING: {RuntimeState.READY, RuntimeState.FAILED},
    RuntimeState.READY: {RuntimeState.STARTING, RuntimeState.STOPPED, RuntimeState.FAILED},
    RuntimeState.STARTING: {RuntimeState.RUNNING, RuntimeState.FAILED, RuntimeState.STOPPING},
    RuntimeState.RUNNING: {RuntimeState.STOPPING, RuntimeState.FAILED},
    RuntimeState.STOPPING: {RuntimeState.STOPPED, RuntimeState.FAILED},
    RuntimeState.STOPPED: {RuntimeState.STARTING, RuntimeState.CREATED},
    RuntimeState.FAILED: {RuntimeState.CREATED, RuntimeState.STOPPED},
}


def calculate_source_hash(source_code: str) -> str:
    """Calculate deterministic SHA-256 hash of strategy source code."""
    if source_code is None:
        raise PlatformValidationError("source_code cannot be None")
    return hashlib.sha256(source_code.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class StrategyRuntimeInstance:
    """Represents a single isolated Freqtrade strategy runtime instance."""

    runtime_id: str
    profile_id: str
    strategy_id: str
    strategy_source_hash: str
    mode: RuntimeMode = RuntimeMode.DRY_RUN
    market_type: MarketType = MarketType.SPOT
    state: RuntimeState = RuntimeState.CREATED
    workspace_path: str = ""
    process_id: int | None = None
    created_at: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.runtime_id or not self.runtime_id.strip():
            raise PlatformValidationError("runtime_id is required")
        if not self.profile_id or not self.profile_id.strip():
            raise PlatformValidationError("profile_id is required")
        if not self.strategy_id or not self.strategy_id.strip():
            raise PlatformValidationError("strategy_id is required")
        if not self.strategy_source_hash or not self.strategy_source_hash.strip():
            raise PlatformValidationError("strategy_source_hash is required")

    def transition_to(self, new_state: RuntimeState, error_message: str | None = None) -> None:
        """Transition runtime to new state if transition is valid."""
        if not isinstance(new_state, RuntimeState):
            new_state = RuntimeState(new_state)

        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise PlatformValidationError(
                f"Invalid runtime state transition from {self.state.value} to {new_state.value}"
            )

        self.state = new_state
        if error_message:
            self.last_error = error_message
