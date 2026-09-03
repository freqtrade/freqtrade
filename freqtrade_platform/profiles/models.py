"""Trading profile domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class TradingProfile:
    """Profile-specific trading configuration.

    ``TradingUniverse`` defines the base eligible asset set for the platform. A profile
    references that universe through ``universe_id`` and may add an optional narrower
    ``symbol_scope`` constraint. The final eligible symbols are the universe eligibility
    intersected with the profile scope when the profile scope is non-empty.
    """

    profile_id: str
    name: str
    exchange: str
    market_type: str
    universe_id: str | None = None
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
        self.symbol_scope = self._normalize_symbols(self.symbol_scope)

    @staticmethod
    def _normalize_symbols(symbols: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if symbol is None:
                continue
            cleaned = str(symbol).strip()
            if not cleaned:
                continue
            canonical = cleaned.upper()
            if canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)
        return normalized

    def resolve_symbols(self, universe_symbols: list[str], *, universe_enabled: bool = True) -> list[str]:
        """Return the final eligible symbols for this profile.

        The base universe is authoritative. The profile scope can only narrow it.
        """
        if not universe_enabled:
            return []

        universe_set = set(self._normalize_symbols(universe_symbols))
        if not self.symbol_scope:
            return sorted(universe_set)

        narrowed = set(self._normalize_symbols(self.symbol_scope))
        return sorted(universe_set & narrowed)

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
        universe_id="default-universe",
        symbol_scope=["BTC/USDT"],
    )
