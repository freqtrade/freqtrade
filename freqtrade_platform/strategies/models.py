"""Strategy domain model and metadata definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class StrategyMetadata:
    """Metadata for a reusable strategy component managed by the platform."""

    strategy_id: str
    name: str
    market_type: str
    description: str | None = None
    version: str = "1.0.0"
    enabled: bool = True
    compatible_regimes: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.strategy_id or not self.strategy_id.strip():
            raise PlatformValidationError("strategy_id is required")
        if not self.name or not self.name.strip():
            raise PlatformValidationError("name is required")
        if not self.market_type or not self.market_type.strip():
            raise PlatformValidationError("market_type is required")


@dataclass(slots=True)
class StrategyMetadata:
    """Canonical domain model for platform-managed strategies.

    The platform layer keeps one strategy definition model. Actual strategy execution remains
    delegated to Freqtrade's strategy implementation and resolver.
    """

    strategy_id: str
    name: str
    market_type: str
    description: str | None = None
    version: str = "1.0.0"
    enabled: bool = True
    compatible_regimes: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.strategy_id or not self.strategy_id.strip():
            raise PlatformValidationError("strategy_id is required")
        if not self.name or not self.name.strip():
            raise PlatformValidationError("name is required")
        if not self.market_type or not self.market_type.strip():
            raise PlatformValidationError("market_type is required")

    def validate(self) -> None:
        self._validate()


Strategy = StrategyMetadata
