"""Interface boundary for future market regime detection services."""

from __future__ import annotations

from abc import ABC, abstractmethod

from freqtrade_platform.regimes.models import MarketRegime


class RegimeDetector(ABC):
    """Interface for future regime evaluation engines.

    This phase intentionally avoids implementation. The detector is a stable boundary for later
    integration with Freqtrade data providers and strategy compatibility checks.
    """

    @abstractmethod
    def detect(self, market_data: object) -> MarketRegime:
        """Return the currently dominant market regime for the supplied context."""
