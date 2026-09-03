"""Statistics domain models and interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class StrategyPerformance:
    """Performance metrics attached to a strategy and later populated by calculations."""

    strategy_id: str
    total_pnl: float = 0.0
    return_percent: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    average_r: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    mfe: float | None = None
    mae: float | None = None
    trade_count: int = 0
    average_trade_duration: float | None = None
    fees: float = 0.0
    funding: float = 0.0
    slippage: float = 0.0
    performance_by_regime: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy_id or not self.strategy_id.strip():
            raise PlatformValidationError("strategy_id is required")


@dataclass(slots=True)
class StrategyPerformanceScore:
    """Aggregate score from later calculations and regime analysis."""

    strategy_id: str
    score: float = 0.0
    reason: str | None = None
