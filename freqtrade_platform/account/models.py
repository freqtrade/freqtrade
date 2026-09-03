"""Account balance snapshot domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class AccountSnapshot:
    """Represents a point-in-time account view separated from simulated balances."""

    timestamp: str
    exchange: str
    market_type: str
    available_balance: float
    total_balance: float
    equity: float
    positions: dict[str, float] = field(default_factory=dict)
    raw_source_metadata: dict[str, Any] = field(default_factory=dict)
    simulated_balance: float | None = None
    simulated_equity: float | None = None

    def __post_init__(self) -> None:
        if not self.timestamp or not self.timestamp.strip():
            raise PlatformValidationError("timestamp is required")
        if not self.exchange or not self.exchange.strip():
            raise PlatformValidationError("exchange is required")
        if not self.market_type or not self.market_type.strip():
            raise PlatformValidationError("market_type is required")
        if self.available_balance < 0 or self.total_balance < 0 or self.equity < 0:
            raise PlatformValidationError("account balances cannot be negative")
