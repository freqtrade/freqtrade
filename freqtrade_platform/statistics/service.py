"""Statistics service boundary for future analytics orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod

from freqtrade_platform.statistics.models import StrategyPerformance


class PerformanceTracker(ABC):
    """Abstract tracker for future strategy performance calculations."""

    @abstractmethod
    def record(self, performance: StrategyPerformance) -> StrategyPerformance:
        """Persist or stream performance data for later scoring."""


class StatisticsService:
    """Placeholder service for future strategy performance aggregation."""

    def __init__(self) -> None:
        self._records: dict[str, StrategyPerformance] = {}

    def record(self, performance: StrategyPerformance) -> StrategyPerformance:
        self._records[performance.strategy_id] = performance
        return performance

    def get(self, strategy_id: str) -> StrategyPerformance | None:
        return self._records.get(strategy_id)
