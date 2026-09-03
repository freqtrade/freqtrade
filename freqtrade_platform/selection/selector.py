"""Deterministic strategy selection for platform-managed decision making."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from freqtrade_platform.compatibility.evaluator import StrategyCompatibilityEvaluator
from freqtrade_platform.profiles.models import TradingProfile
from freqtrade_platform.regimes.models import MarketRegimeResult
from freqtrade_platform.strategies.models import StrategyDefinition


class SelectionDecision(str, Enum):
    """Decision outcome emitted by the selector."""

    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


@dataclass(slots=True)
class StrategySelectionResult:
    """Explicit output produced by the strategy selector."""

    selected_strategy_id: str | None
    decision: SelectionDecision
    regime: str
    reason: str
    candidate_strategies: list[str] = field(default_factory=list)
    rejected_candidates: list[str] = field(default_factory=list)
    timestamp: str | None = None


class StrategySelector:
    """Select the best compatible strategy based on regime and profile constraints."""

    def __init__(self) -> None:
        self._compatibility = StrategyCompatibilityEvaluator()

    def select(
        self,
        *,
        profile: TradingProfile,
        regime_result: MarketRegimeResult,
        strategies: list[StrategyDefinition],
    ) -> StrategySelectionResult:
        candidate_ids: list[str] = []
        rejected: list[str] = []

        assigned_ids = [strategy_id.strip() for strategy_id in profile.assigned_strategies if strategy_id and str(strategy_id).strip()]
        if not assigned_ids:
            return StrategySelectionResult(
                selected_strategy_id=None,
                decision=SelectionDecision.NO_TRADE,
                regime=regime_result.regime.value,
                reason="no assigned strategies configured",
                candidate_strategies=[],
                rejected_candidates=["profile: no assigned strategies"],
                timestamp=regime_result.timestamp,
            )

        strategy_map = {strategy.strategy_id: strategy for strategy in strategies}
        for strategy_id in assigned_ids:
            strategy = strategy_map.get(strategy_id)
            if strategy is None:
                rejected.append(f"unknown strategy: {strategy_id}")
                continue
            if not strategy.enabled:
                rejected.append(f"{strategy.strategy_id}: disabled")
                continue

            compatibility = self._compatibility.evaluate(
                strategy,
                regime_result,
                market_type=profile.market_type,
            )
            if compatibility.compatible:
                candidate_ids.append(strategy.strategy_id)
            else:
                rejected.append(f"{strategy.strategy_id}: {compatibility.reason}")

        ordered = self._order_candidates(profile, candidate_ids)
        if not ordered:
            return StrategySelectionResult(
                selected_strategy_id=None,
                decision=SelectionDecision.NO_TRADE,
                regime=regime_result.regime.value,
                reason=f"no compatible assigned strategy for regime {regime_result.regime.value}",
                candidate_strategies=[],
                rejected_candidates=rejected,
                timestamp=regime_result.timestamp,
            )

        selected_id = ordered[0]
        return StrategySelectionResult(
            selected_strategy_id=selected_id,
            decision=SelectionDecision.TRADE,
            regime=regime_result.regime.value,
            reason=f"selected {selected_id} for regime {regime_result.regime.value}",
            candidate_strategies=ordered,
            rejected_candidates=rejected,
            timestamp=regime_result.timestamp,
        )

    @staticmethod
    def _order_candidates(profile: TradingProfile, candidate_ids: list[str]) -> list[str]:
        priority = {strategy_id: index for index, strategy_id in enumerate(profile.assigned_strategies)}
        return sorted(set(candidate_ids), key=lambda strategy_id: (priority.get(strategy_id, len(priority)), strategy_id))
