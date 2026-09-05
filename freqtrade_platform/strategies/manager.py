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
                enabled = True
                market_type = "SPOT"
                version = "1.0.0"
                compatible_regimes = []
                config = {}

                if self.strategy_repository:
                    rec = self.strategy_repository.get(source.strategy_id)
                    if rec:
                        enabled = rec.enabled
                        market_type = rec.market_type
                        if hasattr(rec, "version") and rec.version:
                            version = rec.version
                        if hasattr(rec, "compatible_regimes") and rec.compatible_regimes:
                            compatible_regimes = rec.compatible_regimes
                        if hasattr(rec, "config") and rec.config:
                            config = rec.config

                existing = self.registry.get(source.strategy_id)
                if existing is None:
                    strat_def = StrategyDefinition(
                        strategy_id=source.strategy_id,
                        name=source.name,
                        market_type=market_type,
                        enabled=enabled,
                        version=version,
                        compatible_regimes=compatible_regimes,
                        config=config,
                    )
                    self.registry.register(strat_def)
                else:
                    existing.name = source.name
                    existing.market_type = market_type
                    existing.enabled = enabled
                    existing.version = version
                    existing.compatible_regimes = compatible_regimes
                    existing.config = config

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
