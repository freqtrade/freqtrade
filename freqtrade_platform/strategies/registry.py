"""Strategy registry metadata boundary.

This registry does not replace Freqtrade's StrategyResolver. It is a platform-level metadata
registry for discovered strategies and their enablement state.
"""

from __future__ import annotations

from typing import Any

from freqtrade_platform.strategies.models import StrategyDefinition


class StrategyRegistry:
    """Track platform-managed strategy metadata and lifecycle state."""

    _ALLOWED_UPDATE_FIELDS = {
        "strategy_id",
        "name",
        "market_type",
        "description",
        "version",
        "enabled",
        "compatible_regimes",
        "config",
    }

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyDefinition] = {}

    def register(self, strategy: StrategyDefinition) -> StrategyDefinition:
        strategy_id = strategy.strategy_id
        if strategy_id in self._strategies:
            raise ValueError(f"duplicate strategy id: {strategy_id}")
        self._strategies[strategy_id] = strategy
        return strategy

    def get(self, strategy_id: str) -> StrategyDefinition | None:
        return self._strategies.get(strategy_id)

    def list(self) -> list[StrategyDefinition]:
        return list(self._strategies.values())

    def update(self, strategy_id: str, **changes: Any) -> StrategyDefinition:
        if strategy_id not in self._strategies:
            raise KeyError(strategy_id)

        strategy = self._strategies[strategy_id]
        for key, value in changes.items():
            if key not in self._ALLOWED_UPDATE_FIELDS:
                raise ValueError(f"unknown strategy field: {key}")
            if key == "strategy_id":
                new_id = str(value)
                if not new_id or not new_id.strip():
                    raise ValueError("strategy_id is required")
                if new_id != strategy_id and new_id in self._strategies:
                    raise ValueError(f"duplicate strategy id: {new_id}")
                self._strategies.pop(strategy_id)
                strategy.strategy_id = new_id
                strategy_id = new_id
            elif key == "name":
                strategy.name = str(value)
            elif key == "market_type":
                strategy.market_type = str(value)
            elif key == "description":
                strategy.description = value if value is None else str(value)
            elif key == "version":
                strategy.version = str(value)
            elif key == "enabled":
                strategy.enabled = bool(value)
            elif key == "compatible_regimes":
                strategy.compatible_regimes = list(value)
            elif key == "config":
                strategy.config = dict(value)

        strategy.validate()
        self._strategies[strategy_id] = strategy
        return strategy

    def enable(self, strategy_id: str) -> StrategyDefinition:
        strategy = self._strategies[strategy_id]
        strategy.enabled = True
        return strategy

    def disable(self, strategy_id: str) -> StrategyDefinition:
        strategy = self._strategies[strategy_id]
        strategy.enabled = False
        return strategy

    def remove(self, strategy_id: str) -> None:
        self._strategies.pop(strategy_id, None)
