"""Manager boundary for strategy operations."""

from __future__ import annotations

from freqtrade_platform.strategies.models import StrategyDefinition
from freqtrade_platform.strategies.registry import StrategyRegistry


class StrategyManager:
    """Responsible for validated strategy lifecycle operations.

    This is intentionally thin and does not trigger code execution or strategy loading.
    """

    def __init__(self, registry: StrategyRegistry | None = None) -> None:
        self.registry = registry or StrategyRegistry()

    def validate(self, strategy: StrategyDefinition) -> StrategyDefinition:
        strategy.validate()
        return strategy

    def add(self, strategy: StrategyDefinition) -> StrategyDefinition:
        self.validate(strategy)
        self.registry.register(strategy)
        return strategy

    def update(self, strategy_id: str, **changes: object) -> StrategyDefinition:
        current = self.get(strategy_id)
        if current is None:
            raise KeyError(strategy_id)
        updated = self.registry.update(strategy_id, **changes)
        self.validate(updated)
        return updated

    def delete(self, strategy_id: str) -> None:
        self.registry.remove(strategy_id)

    def get(self, strategy_id: str) -> StrategyDefinition | None:
        return self.registry.get(strategy_id)

    def list(self) -> list[StrategyDefinition]:
        return list(self.registry.list())

    def activate(self, strategy_id: str) -> StrategyDefinition:
        return self.registry.enable(strategy_id)

    def deactivate(self, strategy_id: str) -> StrategyDefinition:
        return self.registry.disable(strategy_id)
