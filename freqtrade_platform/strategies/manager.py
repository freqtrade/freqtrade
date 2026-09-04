"""Manager boundary for strategy operations."""

from __future__ import annotations

from freqtrade_platform.storage.repositories import PlatformStrategyRepository, PlatformStrategySourceRepository
from freqtrade_platform.strategies.models import StrategyDefinition
from freqtrade_platform.strategies.registry import StrategyRegistry


class StrategyManager:
    """Responsible for validated strategy lifecycle operations.

    This is intentionally thin and does not trigger code execution or strategy loading.
    """

    def __init__(
        self,
        registry: StrategyRegistry | None = None,
        source_repository: PlatformStrategySourceRepository | None = None,
        strategy_repository: PlatformStrategyRepository | None = None,
    ) -> None:
        self.registry = registry or StrategyRegistry()
        self.source_repository = source_repository
        self.strategy_repository = strategy_repository
        self.reload_from_repository()

    def reload_from_repository(self) -> None:
        if self.source_repository:
            sources = self.source_repository.list()
            for source in sources:
                if self.registry.get(source.strategy_id) is None:
                    enabled = True
                    market_type = "SPOT"
                    if self.strategy_repository:
                        rec = self.strategy_repository.get(source.strategy_id)
                        if rec:
                            enabled = rec.enabled
                            market_type = rec.market_type
                    strat_def = StrategyDefinition(
                        strategy_id=source.strategy_id,
                        name=source.name,
                        market_type=market_type,
                        enabled=enabled,
                    )
                    self.registry.register(strat_def)

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
