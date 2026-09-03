"""Market regime domain objects and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MarketRegimeType(str, Enum):
    """Canonical typed vocabulary for regime classification."""

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
class MarketObservation:
    """Timeframe-bound observation used for future multi-timeframe detection."""

    timeframe: str
    signal: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MarketRegimeResult:
    """Result object for a regime classification without strategy selection logic."""

    regime: MarketRegimeType
    confidence: float
    timeframe: str
    timestamp: str
    evidence: dict[str, object] = field(default_factory=dict)
    observations: list[MarketObservation] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe is required")
        if not self.timestamp or not self.timestamp.strip():
            raise ValueError("timestamp is required")
        if not isinstance(self.regime, MarketRegimeType):
            self.regime = MarketRegimeType(self.regime)
        if self.timestamp.endswith("Z"):
            try:
                datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("timestamp must be ISO-8601") from exc
