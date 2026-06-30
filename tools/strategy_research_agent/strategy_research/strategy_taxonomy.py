#!/usr/bin/env python3
"""Canonical market-regime strategy taxonomy for the research agent."""

from __future__ import annotations

import json
from typing import Any


STRATEGY_TAXONOMY: dict[str, dict[str, Any]] = {
    "downtrend_failed_bounce_short": {
        "code": "A1",
        "name": "Downtrend failed-bounce short",
        "direction": "short",
        "earns_from": "Downside continuation after weak rebound failure.",
        "allowed_regimes": ["downtrend", "weak_rebound_failure", "bear_continuation"],
        "disabled_regimes": ["bull_trend", "strong_short_squeeze", "range_chop", "high_volatility_reversal"],
        "entry_intent": "Sell after a weak bounce fails and downside structure resumes.",
        "failure_mode": "Mistakes bull-trend pullbacks or squeeze setups for short continuation.",
    },
    "uptrend_failed_pullback_long": {
        "code": "A2",
        "name": "Uptrend failed-pullback long",
        "direction": "long",
        "earns_from": "Upside continuation after shallow pullback failure.",
        "allowed_regimes": ["uptrend", "weak_pullback_failure", "bull_continuation"],
        "disabled_regimes": ["bear_trend", "liquidation_cascade", "range_chop", "high_volatility_reversal"],
        "entry_intent": "Buy after a weak pullback fails and upside structure resumes.",
        "failure_mode": "Mistakes bear-market relief bounces for real trend continuation.",
    },
    "range_upper_reversion_short": {
        "code": "B1",
        "name": "Range upper-bound mean-reversion short",
        "direction": "short",
        "earns_from": "Reversion from range resistance back toward fair value.",
        "allowed_regimes": ["range_bound", "low_to_mid_volatility", "clear_upper_boundary"],
        "disabled_regimes": ["trend_breakout", "volatility_expansion", "news_impulse"],
        "entry_intent": "Sell exhaustion near a validated range upper boundary.",
        "failure_mode": "Shorts true upside breakouts as if they were range fades.",
    },
    "range_lower_reversion_long": {
        "code": "B2",
        "name": "Range lower-bound mean-reversion long",
        "direction": "long",
        "earns_from": "Reversion from range support back toward fair value.",
        "allowed_regimes": ["range_bound", "low_to_mid_volatility", "clear_lower_boundary"],
        "disabled_regimes": ["trend_breakdown", "volatility_expansion", "news_impulse"],
        "entry_intent": "Buy exhaustion near a validated range lower boundary.",
        "failure_mode": "Buys true downside breakdowns as if they were range fades.",
    },
    "downtrend_pullback_short": {
        "code": "C1",
        "name": "Downtrend pullback short",
        "direction": "short",
        "earns_from": "Trend-following short after controlled pullback into resistance.",
        "allowed_regimes": ["downtrend", "controlled_pullback", "lower_high_structure"],
        "disabled_regimes": ["trend_exhaustion", "range_chop", "sharp_reversal"],
        "entry_intent": "Sell a pullback into resistance when downside trend resumes.",
        "failure_mode": "Keeps shorting after the downtrend has exhausted or reversed.",
    },
    "uptrend_pullback_long": {
        "code": "C2",
        "name": "Uptrend pullback long",
        "direction": "long",
        "earns_from": "Trend-following long after controlled pullback into support.",
        "allowed_regimes": ["uptrend", "controlled_pullback", "higher_low_structure"],
        "disabled_regimes": ["trend_exhaustion", "range_chop", "sharp_reversal"],
        "entry_intent": "Buy a pullback into support when upside trend resumes.",
        "failure_mode": "Keeps buying after the uptrend has exhausted or reversed.",
    },
    "downside_breakout_continuation_short": {
        "code": "D1",
        "name": "Downside breakout continuation short",
        "direction": "short",
        "earns_from": "Continuation after confirmed downside range or structure break.",
        "allowed_regimes": ["volatility_expansion", "confirmed_breakdown", "trend_acceleration"],
        "disabled_regimes": ["false_breakout_cluster", "illiquid_spike", "mean_reverting_range"],
        "entry_intent": "Sell confirmed downside break and continuation, not the first wick.",
        "failure_mode": "Chases false breakdowns and gets trapped in violent reclaim candles.",
    },
    "upside_breakout_continuation_long": {
        "code": "D2",
        "name": "Upside breakout continuation long",
        "direction": "long",
        "earns_from": "Continuation after confirmed upside range or structure break.",
        "allowed_regimes": ["volatility_expansion", "confirmed_breakout", "trend_acceleration"],
        "disabled_regimes": ["false_breakout_cluster", "illiquid_spike", "mean_reverting_range"],
        "entry_intent": "Buy confirmed upside break and continuation, not the first wick.",
        "failure_mode": "Chases false breakouts and gets trapped in violent rejection candles.",
    },
    "volatility_compression_breakout": {
        "code": "E",
        "name": "Volatility compression then directional expansion",
        "direction": "long_or_short",
        "earns_from": "First directional expansion after volatility compression.",
        "allowed_regimes": ["volatility_compression", "range_squeeze", "post_compression_directional_release"],
        "disabled_regimes": ["already_extended_trend", "random_high_volatility", "thin_liquidity"],
        "entry_intent": "Wait for compression, then trade confirmed directional release.",
        "failure_mode": "Treats noisy high volatility as useful expansion without prior compression.",
    },
    "defense_no_trade": {
        "code": "F",
        "name": "Defense / no-trade regime",
        "direction": "no_trade",
        "earns_from": "Avoiding trades when edge is not available.",
        "allowed_regimes": ["unknown_regime", "hostile_regime", "data_gap", "cost_or_bias_unverified"],
        "disabled_regimes": [],
        "entry_intent": "Block entries, reduce exposure, or require manual review.",
        "failure_mode": "Fails open instead of preserving capital during hostile regimes.",
    },
}


REQUIRED_TAXONOMY_IDS = {
    "downtrend_failed_bounce_short",
    "uptrend_failed_pullback_long",
    "range_upper_reversion_short",
    "range_lower_reversion_long",
    "downtrend_pullback_short",
    "uptrend_pullback_long",
    "downside_breakout_continuation_short",
    "upside_breakout_continuation_long",
    "volatility_compression_breakout",
    "defense_no_trade",
}


def taxonomy_summary() -> list[dict[str, Any]]:
    return [
        {"id": family_id, **family}
        for family_id, family in sorted(STRATEGY_TAXONOMY.items(), key=lambda item: item[1]["code"])
    ]


def family_contract(family_id: str) -> dict[str, Any]:
    family = STRATEGY_TAXONOMY.get(family_id) or STRATEGY_TAXONOMY["defense_no_trade"]
    return {
        "strategy_family": family_id,
        "family_code": family["code"],
        "family_name": family["name"],
        "direction": family["direction"],
        "allowed_regimes": list(family["allowed_regimes"]),
        "disabled_regimes": list(family["disabled_regimes"]),
        "entry_intent": family["entry_intent"],
        "failure_mode": family["failure_mode"],
        "contract_required_for_generation": True,
        "contract_required_for_attribution": True,
        "contract_required_for_promotion": True,
    }


def classify_strategy_family(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    normalized = text.replace("_", " ").replace("-", " ")

    if any(term in normalized for term in ["no trade", "abstention", "defense", "kill switch", "bias check"]):
        return "defense_no_trade"
    if any(term in normalized for term in ["compression", "squeeze", "atr compression", "volatility squeeze"]):
        return "volatility_compression_breakout"

    is_short = any(term in normalized for term in ["short", "breakdown", "downside", "bear", "sell"])
    is_long = any(term in normalized for term in ["long", "breakout", "upside", "bull", "buy"])
    is_range = any(term in normalized for term in ["range", "mean reversion", "reversion", "upper", "lower"])
    is_pullback = any(term in normalized for term in ["pullback", "回调", "retest", "resume"])
    is_failed_bounce = any(term in normalized for term in ["failed bounce", "weak rebound", "second leg", "body not too red"])
    is_breakout = any(term in normalized for term in ["breakout", "breakdown", "structure break", "break continuation"])

    if is_range and is_short:
        return "range_upper_reversion_short"
    if is_range and is_long:
        return "range_lower_reversion_long"
    if is_failed_bounce or ("second leg" in normalized and is_short):
        return "downtrend_failed_bounce_short"
    if "failed pullback" in normalized or ("second leg" in normalized and is_long):
        return "uptrend_failed_pullback_long"
    if is_breakout and is_short:
        return "downside_breakout_continuation_short"
    if is_breakout and is_long:
        return "upside_breakout_continuation_long"
    if is_pullback and is_short:
        return "downtrend_pullback_short"
    if is_pullback and is_long:
        return "uptrend_pullback_long"
    if is_short:
        return "downtrend_failed_bounce_short"
    if is_long:
        return "uptrend_failed_pullback_long"
    return "defense_no_trade"


def infer_family_from_card(card: dict[str, Any]) -> str:
    translation = card.get("freqtrade_translation") or {}
    explicit = translation.get("strategy_family")
    if explicit in STRATEGY_TAXONOMY:
        return explicit
    return classify_strategy_family(
        explicit,
        card.get("title"),
        card.get("category"),
        card.get("strategy_hypothesis"),
        json.dumps(card.get("concepts", []), ensure_ascii=False),
        json.dumps(translation, ensure_ascii=False),
    )
