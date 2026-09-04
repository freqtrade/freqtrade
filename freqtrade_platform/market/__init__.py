"""Market data, validation, features, and regime detection for the platform layer."""

from freqtrade_platform.market.data import CanonicalMarketSeries, DataProviderMarketAdapter
from freqtrade_platform.market.detector import MarketRegimeDetector
from freqtrade_platform.market.features import MarketFeatureExtractor
from freqtrade_platform.market.validator import MarketDataValidator

__all__ = [
    "CanonicalMarketSeries",
    "DataProviderMarketAdapter",
    "MarketDataValidator",
    "MarketFeatureExtractor",
    "MarketRegimeDetector",
]
