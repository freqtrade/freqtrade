"""Market regime domain namespace."""

from freqtrade_platform.regimes.interface import RegimeDetector
from freqtrade_platform.regimes.models import MarketRegime, MarketRegimeType

__all__ = ["MarketRegime", "MarketRegimeType", "RegimeDetector"]
