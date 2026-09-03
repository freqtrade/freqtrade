"""Strategy registry metadata boundary.

This registry does not replace Freqtrade's StrategyResolver. It is a platform-level metadata
registry for discovered strategies and their enablement state.
"""

from __future__ import annotations

from freqtrade_platform.strategies.models import Strategy, StrategyMetadata


class StrategyRegistry:
    """Track platform-managed strategy metadata and lifecycle state."""

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyMetadata] = {}

    def register(self, strategy: StrategyMetadata) -> StrategyMetadata:
        strategy_id = strategy.strategy_id
        if strategy_id in self._strategies:
            raise ValueError(f"duplicate strategy id: {strategy_id}")
        self._strategies[strategy_id] = strategy
        return strategy

    def get(self, strategy_id: str) -> StrategyMetadata | None:
        return self._strategies.get(strategy_id)

    def list(self) -> list[StrategyMetadata]:
        return list(self._strategies.values())

    def update(self, strategy_id: str, **changes: object) -> StrategyMetadata:
        strategy = self._strategies[strategy_id]
        for key, value in changes.items():
            setattr(strategy, key, value)
        strategy._validate()
        return strategy

    def enable(self, strategy_id: str) -> StrategyMetadata:
        strategy = self._strategies[strategy_id]
        strategy.enabled = True
        return strategy

    def disable(self, strategy_id: str) -> StrategyMetadata:
        strategy = self._strategies[strategy_id]
        strategy.enabled = False
        return strategy

    def remove(self, strategy_id: str) -> None:
        self._strategies.pop(strategy_id, None)
