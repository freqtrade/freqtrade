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
class Strategy:
    """Abstract platform strategy boundary.

    This is intentionally metadata-first. Actual execution remains delegated to Freqtrade's
    StrategyResolver and the strategy implementation loaded by Freqtrade itself.
    """

    strategy_id: str
    name: str
    market_type: str
    description: str | None = None
    compatible_regimes: list[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.strategy_id.strip():
            raise PlatformValidationError("strategy_id is required")
        if not self.name.strip():
            raise PlatformValidationError("name is required")
        if not self.market_type.strip():
            raise PlatformValidationError("market_type is required")
