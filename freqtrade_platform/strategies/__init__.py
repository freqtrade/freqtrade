"""Strategy metadata and registry namespace."""

from freqtrade_platform.strategies.manager import StrategyManager
from freqtrade_platform.strategies.models import Strategy, StrategyMetadata
from freqtrade_platform.strategies.registry import StrategyRegistry

__all__ = ["Strategy", "StrategyMetadata", "StrategyManager", "StrategyRegistry"]
