"""Trading profile domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class TradingProfile:
    """Definition of how a profile should allocate and execute within a platform scope."""

    profile_id: str
    name: str
    exchange: str
    market_type: str
    symbol_scope: list[str] = field(default_factory=list)
    primary_timeframe: str | None = None
    informative_timeframes: list[str] = field(default_factory=list)
    assigned_strategies: list[str] = field(default_factory=list)
    regime_policy: str | None = None
    risk_configuration: dict[str, Any] = field(default_factory=dict)
    execution_configuration: dict[str, Any] = field(default_factory=dict)
    capital_allocation: float | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.profile_id or not self.profile_id.strip():
            raise PlatformValidationError("profile_id is required")
        if not self.name or not self.name.strip():
            raise PlatformValidationError("name is required")
        if not self.exchange or not self.exchange.strip():
            raise PlatformValidationError("exchange is required")
        if not self.market_type or not self.market_type.strip():
            raise PlatformValidationError("market_type is required")
        if self.capital_allocation is not None and not 0 <= float(self.capital_allocation) <= 100:
            raise PlatformValidationError("capital_allocation must be between 0 and 100")


def build_default_profile() -> TradingProfile:
    """Return a default, valid profile used for integration testing."""
    return TradingProfile(
        profile_id="default-profile",
        name="Default Profile",
        exchange="binance",
        market_type="spot",
        symbol_scope=["BTC/USDT"],
    )
