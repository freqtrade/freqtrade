"""Compatibility evaluation between a strategy and a market regime."""

from __future__ import annotations

from dataclasses import dataclass

from freqtrade_platform.regimes.models import MarketRegimeResult, MarketRegimeType
from freqtrade_platform.strategies.models import StrategyDefinition


@dataclass(slots=True)
class CompatibilityResult:
    """Explicit result object for compatibility evaluation."""

    strategy_id: str
    compatible: bool
    reason: str = ""
    regime: MarketRegimeType | None = None
    market_type: str | None = None


class StrategyCompatibilityEvaluator:
    """Decoupled compatibility evaluation without performance scoring."""

    def evaluate(
        self,
        strategy: StrategyDefinition,
        regime_result: MarketRegimeResult,
        *,
        market_type: str,
    ) -> CompatibilityResult:
        if not strategy.enabled:
            return CompatibilityResult(
                strategy_id=strategy.strategy_id,
                compatible=False,
                reason="strategy is disabled",
                regime=regime_result.regime,
                market_type=market_type,
            )

        if market_type and strategy.market_type and market_type != strategy.market_type:
            return CompatibilityResult(
                strategy_id=strategy.strategy_id,
                compatible=False,
                reason=f"market type mismatch: expected {strategy.market_type}, got {market_type}",
                regime=regime_result.regime,
                market_type=market_type,
            )

        if strategy.compatible_regimes and regime_result.regime not in strategy.compatible_regimes:
            return CompatibilityResult(
                strategy_id=strategy.strategy_id,
                compatible=False,
                reason=f"strategy does not support regime {regime_result.regime.value}",
                regime=regime_result.regime,
                market_type=market_type,
            )

        return CompatibilityResult(
            strategy_id=strategy.strategy_id,
            compatible=True,
            reason="strategy matches enabled market and regime constraints",
            regime=regime_result.regime,
            market_type=market_type,
        )
