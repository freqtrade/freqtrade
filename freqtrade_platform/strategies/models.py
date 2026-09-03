"""Canonical strategy definition domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class StrategyDefinition:
    """Platform-level metadata for a strategy definition.

    This remains a domain model only. It does not carry execution logic or Freqtrade runtime
    behavior.
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
        self.validate()

    def validate(self) -> None:
        if not self.strategy_id or not self.strategy_id.strip():
            raise PlatformValidationError("strategy_id is required")
        if not self.name or not self.name.strip():
            raise PlatformValidationError("name is required")
        if not self.market_type or not self.market_type.strip():
            raise PlatformValidationError("market_type is required")

        if self.version is not None and not str(self.version).strip():
            raise PlatformValidationError("version is required")
