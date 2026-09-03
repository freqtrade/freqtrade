"""Canonical market regime detector contract."""

from __future__ import annotations

from typing import Protocol

from freqtrade_platform.regimes.models import MarketObservation, MarketRegimeResult


class MarketRegimeDetector(Protocol):
    """Protocol for market regime detection based on typed multi-timeframe observations."""

    def detect(self, observations: list[MarketObservation] | tuple[MarketObservation, ...]) -> MarketRegimeResult:
        """Return the dominant market regime for the supplied observation set."""
        ...
