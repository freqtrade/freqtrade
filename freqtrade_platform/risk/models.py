"""Strategy-level risk configuration domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class StrategyRiskConfig:
    """Risk settings owned by a strategy or profile combination."""

    risk_per_trade: float = 0.01
    max_position_size: float | None = None
    stoploss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None
    leverage: float | None = None
    max_position_adjustments: int = 0
    dca_constraints: dict[str, Any] = field(default_factory=dict)
    exposure_constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.risk_per_trade < 0:
            raise PlatformValidationError("risk_per_trade cannot be negative")
        if self.max_position_size is not None and self.max_position_size < 0:
            raise PlatformValidationError("max_position_size cannot be negative")


@dataclass(slots=True)
class SafetyGuardPolicy:
    """Account-level safety policy that complements strategy-level risk controls."""

    emergency_stop: bool = False
    max_total_exposure: float | None = None
    max_daily_loss: float | None = None
    max_drawdown: float | None = None
    max_simultaneous_positions: int | None = None
    global_leverage_cap: float | None = None
    max_notional: float | None = None
