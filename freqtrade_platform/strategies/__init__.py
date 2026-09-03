"""Strategy metadata and registry namespace."""

from freqtrade_platform.strategies.manager import StrategyManager
from freqtrade_platform.strategies.models import StrategyDefinition
from freqtrade_platform.strategies.registry import StrategyRegistry

__all__ = ["StrategyDefinition", "StrategyManager", "StrategyRegistry"]
