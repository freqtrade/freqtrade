"""Real and simulated account domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class RealAccountSnapshot:
    """Represents actual exchange/account state and never carries simulation fields."""

    timestamp: str
    exchange: str
    market_type: str
    available_balance: float
    total_balance: float
    equity: float
    positions: dict[str, float] = field(default_factory=dict)
    raw_source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp or not self.timestamp.strip():
            raise PlatformValidationError("timestamp is required")
        if not self.exchange or not self.exchange.strip():
            raise PlatformValidationError("exchange is required")
        if not self.market_type or not self.market_type.strip():
            raise PlatformValidationError("market_type is required")
        if self.available_balance < 0 or self.total_balance < 0 or self.equity < 0:
            raise PlatformValidationError("account balances cannot be negative")


@dataclass(slots=True)
class SimulationBootstrap:
    """Defines the capital baseline used to initialize a simulation."""

    timestamp: str
    exchange: str
    market_type: str
    starting_balance: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp or not self.timestamp.strip():
            raise PlatformValidationError("timestamp is required")
        if not self.exchange or not self.exchange.strip():
            raise PlatformValidationError("exchange is required")
        if not self.market_type or not self.market_type.strip():
            raise PlatformValidationError("market_type is required")
        if self.starting_balance < 0:
            raise PlatformValidationError("starting_balance cannot be negative")


@dataclass(slots=True)
class SimulationAccount:
    """Represents a simulation-only view of account state and never aliases real data."""

    timestamp: str
    exchange: str
    market_type: str
    starting_balance: float
    available_balance: float
    total_balance: float
    equity: float
    positions: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    bootstrap: SimulationBootstrap | None = None

    def __post_init__(self) -> None:
        if not self.timestamp or not self.timestamp.strip():
            raise PlatformValidationError("timestamp is required")
        if not self.exchange or not self.exchange.strip():
            raise PlatformValidationError("exchange is required")
        if not self.market_type or not self.market_type.strip():
            raise PlatformValidationError("market_type is required")
        if self.starting_balance < 0 or self.available_balance < 0 or self.total_balance < 0 or self.equity < 0:
            raise PlatformValidationError("simulation balances cannot be negative")

