"""Market regime domain objects and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from freqtrade_platform.core.exceptions import PlatformValidationError


class MarketRegimeType(str, Enum):
    """Supported regime names for later detection and strategy compatibility checks."""

    STRONG_UPTREND = "STRONG_UPTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    QUIET_RANGE = "QUIET_RANGE"
    VOLATILE_RANGE = "VOLATILE_RANGE"
    BREAKOUT = "BREAKOUT"
    TRANSITION = "TRANSITION"
    EXTREME = "EXTREME"
    NO_TRADE = "NO_TRADE"


@dataclass(slots=True)
class MarketRegime:
    """Domain model for market regime classification metadata."""

    name: str
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    compatible_strategies: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.name or not self.name.strip():
            raise PlatformValidationError("name is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise PlatformValidationError("confidence must be between 0 and 1")
