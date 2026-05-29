from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from freqtrade_ext.bot_factory.strategy_suitability import (
    SELECTOR_ELIGIBLE_DECISION,
    validate_strategy_suitability_matrix_for_selector,
)


SELECTOR_MATCHING_DECISION_SCHEMA_VERSION = "selector_matching_decision_v1"
NO_TRADE_SCORECARD_SCHEMA_VERSION = "no_trade_scorecard_v1"
SELECTOR_MATCHING_REPORT_SCHEMA_VERSION = "selector_matching_report_v1"


def build_selector_matching_decision(
    *,
    current_market_state: Mapping[str, Any],
    strategy_suitability_matrix: Mapping[str, Any],
    selector_state: Mapping[str, Any] | None = None,
    selector_version: str = "offline_state_selector_v1",
    generated_at: str | None = None,
    decision_id: str | None = None,
    min_state_confidence: float = 0.5,
    max_out_of_distribution_score: float = 0.8,
    cooldown_observations: int = 0,
    hysteresis_margin: float = 0.1,
    source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    decision_id = decision_id or "selector_decision_" + _compact_timestamp(generated_at)
    current = _current_market_state(current_market_state)
    selector_state = dict(selector_state or {})
    matrix_validation = validate_strategy_suitability_matrix_for_selector(
        strategy_suitability_matrix
    )
    current_scopes = _current_state_scopes(current)
    preflight_reasons = _market_state_no_trade_reasons(
        current,
        min_state_confidence=min_state_confidence,
        max_out_of_distribution_score=max_out_of_distribution_score,
    )
    if not matrix_validation["ok"]:
        preflight_reasons.append("strategy_suitability_matrix_validation_failed")
    if strategy_suitability_matrix.get("cost_model_stale") is True:
        preflight_reasons.append("cost_model_stale")

    eligible_rows = _eligible_rows_for_current_state(
        strategy_suitability_matrix.get("rows", []),
        current=current,
        current_scopes=current_scopes,
        current_horizon_profile_id=str(current.get("horizon_profile_id") or ""),
    )
    comparison_rows = _comparison_rows(
        strategy_suitability_matrix.get("rows", []),
        current=current,
        current_scopes=current_scopes,
        current_horizon_profile_id=str(current.get("horizon_profile_id") or ""),
        selector_state=selector_state,
    )

    if preflight_reasons:
        return _decision_payload(
            decision_id=decision_id,
            generated_at=generated_at,
            current=current,
            selector_version=selector_version,
            selected_row=None,
            selected_action="no_trade",
            no_trade_reason=preflight_reasons[0],
            reason_codes=sorted(set(preflight_reasons)),
            comparison_rows=comparison_rows,
            rejected_rows=eligible_rows,
            selector_state=selector_state,
            source_artifacts=source_artifacts,
        )

    if not eligible_rows:
        return _decision_payload(
            decision_id=decision_id,
            generated_at=generated_at,
            current=current,
            selector_version=selector_version,
            selected_row=None,
            selected_action="no_trade",
            no_trade_reason="no_selector_eligible_strategy_for_current_state",
            reason_codes=["no_selector_eligible_strategy_for_current_state"],
            comparison_rows=comparison_rows,
            rejected_rows=[],
            selector_state=selector_state,
            source_artifacts=source_artifacts,
        )

    ranked = sorted(eligible_rows, key=_rank_key, reverse=True)
    selected = ranked[0]
    reason_codes = ["selected_by_stress_cost_utility"]
    previous = _previous_row(ranked, selector_state)
    if previous is not None and previous.get("candidate_id") != selected.get("candidate_id"):
        observations_since_switch = int(selector_state.get("observations_since_switch") or 0)
        if observations_since_switch < cooldown_observations:
            return _decision_payload(
                decision_id=decision_id,
                generated_at=generated_at,
                current=current,
                selector_version=selector_version,
                selected_row=None,
                selected_action="no_trade",
                no_trade_reason="selector_cooldown_blocks_switching",
                reason_codes=["selector_cooldown_blocks_switching"],
                comparison_rows=comparison_rows,
                rejected_rows=ranked,
                selector_state=selector_state,
                source_artifacts=source_artifacts,
            )
        if _rank_score(selected) - _rank_score(previous) < hysteresis_margin:
            selected = previous
            reason_codes.append("selector_hysteresis_kept_previous_candidate")

    rejected = [row for row in ranked if row.get("candidate_id") != selected.get("candidate_id")]
    return _decision_payload(
        decision_id=decision_id,
        generated_at=generated_at,
        current=current,
        selector_version=selector_version,
        selected_row=selected,
        selected_action="select_strategy",
        no_trade_reason=None,
        reason_codes=reason_codes,
        comparison_rows=comparison_rows,
        rejected_rows=rejected,
        selector_state=selector_state,
        source_artifacts=source_artifacts,
    )


def build_no_trade_scorecard(
    *,
    current_market_state: Mapping[str, Any],
    strategy_suitability_matrix: Mapping[str, Any],
    selector_decision: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    current = _current_market_state(current_market_state)
    current_scope_keys = {
        (scope["state_id"], str(current.get("horizon_profile_id") or ""))
        for scope in _current_state_scopes(current)
    }
    rows = []
    matrix_rows = [
        row for row in strategy_suitability_matrix.get("rows", []) if isinstance(row, Mapping)
    ]
    state_keys = sorted(
        {
            (str(row.get("state_id") or ""), str(row.get("horizon_profile_id") or ""))
            for row in matrix_rows
            if row.get("state_id") and row.get("horizon_profile_id")
        }
        | current_scope_keys
    )
    for state_id, horizon_profile_id in state_keys:
        strategy_rows = [
            row
            for row in matrix_rows
            if row.get("row_type") == "strategy"
            and row.get("state_id") == state_id
            and row.get("horizon_profile_id") == horizon_profile_id
            and row.get("decision") == SELECTOR_ELIGIBLE_DECISION
        ]
        best_utility = max((_rank_score(row) for row in strategy_rows), default=0.0)
        best_drawdown = max((_number(row.get("max_drawdown"), 0.0) for row in strategy_rows), default=0.0)
        current_state = (state_id, horizon_profile_id) in current_scope_keys
        rows.append(
            {
                "state_id": state_id,
                "horizon_profile_id": horizon_profile_id,
                "current_state": current_state,
                "avoided_drawdown": round(best_drawdown, 6)
                if _state_is_uncertain(current)
                else 0.0,
                "avoided_negative_expectancy": 0.0,
                "opportunity_cost_vs_hold": round(max(best_utility, 0.0), 6),
                "opportunity_cost_vs_incumbent": None,
                "opportunity_cost_vs_best_selector_eligible_strategy": round(
                    max(best_utility, 0.0), 6
                ),
                "uncertainty_reduction_value": round(_number(current.get("uncertainty"), 0.0), 6)
                if current_state and _state_is_uncertain(current)
                else 0.0,
                "state_confidence": current.get("state_confidence") if current_state else None,
                "assessment": _no_trade_assessment(
                    current=current,
                    current_state=current_state,
                    best_utility=best_utility,
                ),
                "reason_codes": _no_trade_reason_codes(
                    current=current,
                    current_state=current_state,
                    best_utility=best_utility,
                ),
            }
        )
    return {
        "factory": "no_trade_scorecard",
        "schema_version": NO_TRADE_SCORECARD_SCHEMA_VERSION,
        "run_id": run_id or "no_trade_" + _compact_timestamp(generated_at),
        "generated_at": generated_at,
        "selector_decision_id": (selector_decision or {}).get("decision_id"),
        "rows": rows,
        "reason_codes": ["no_trade_policy_evaluated_without_hindsight_reward"],
        "safety_scope": _safety_scope(),
    }


def write_selector_matching_artifacts(
    decision: Mapping[str, Any],
    *,
    output_root: Path,
    no_trade_scorecard: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    run_id = _safe_component(str(decision.get("decision_id") or "selector_matching"))
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "selector_matching_decision": out_dir / "selector_matching_decision.json",
        "selector_matching_report": out_dir / "selector_matching_report.md",
    }
    paths["selector_matching_decision"].write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["selector_matching_report"].write_text(
        render_selector_matching_report(decision),
        encoding="utf-8",
    )
    if no_trade_scorecard is not None:
        paths["no_trade_scorecard"] = out_dir / "no_trade_scorecard.json"
        paths["no_trade_scorecard"].write_text(
            json.dumps(no_trade_scorecard, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return paths


def render_selector_matching_report(decision: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Matching Report",
        "",
        f"- Decision ID: `{decision.get('decision_id')}`",
        f"- Selected action: `{decision.get('selected_action')}`",
        f"- Selected candidate: `{decision.get('selected_candidate_id')}`",
        f"- No-trade reason: `{decision.get('no_trade_reason') or 'not_applicable'}`",
        f"- Reason codes: `{', '.join(decision.get('reason_codes', []))}`",
        "",
        "## Rejected Alternatives",
        "",
        "| candidate | action | rank score | reasons |",
        "| --- | --- | ---: | --- |",
    ]
    for row in decision.get("rejected_alternatives", []):
        lines.append(
            "| {candidate} | {action} | {score} | {reasons} |".format(
                candidate=row.get("candidate_id"),
                action=row.get("matching_action"),
                score=row.get("rank_score"),
                reasons=", ".join(row.get("reason_codes", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Why Not Trade?",
            "",
            "- `no_trade` is selected for stale, unknown, mixed, transition, out-of-distribution, weak-evidence, identity-mismatched, or cooldown-blocked states.",
            "- A selected strategy means local selector simulation only. It does not permit paper, dry-run, live trading, or order placement.",
            "",
            "## What Would Need To Become True Before Selection?",
            "",
            "- Current local market state must be fresh, confident, and covered by a checked selector-eligible state scorecard row.",
            "- Strategy suitability rows must pass identity, evidence, cost, and state-scope checks.",
        ]
    )
    return "\n".join(lines) + "\n"


def _decision_payload(
    *,
    decision_id: str,
    generated_at: str,
    current: Mapping[str, Any],
    selector_version: str,
    selected_row: Mapping[str, Any] | None,
    selected_action: str,
    no_trade_reason: str | None,
    reason_codes: Sequence[str],
    comparison_rows: Sequence[Mapping[str, Any]],
    rejected_rows: Sequence[Mapping[str, Any]],
    selector_state: Mapping[str, Any],
    source_artifacts: Mapping[str, str] | None,
) -> dict[str, Any]:
    selected_candidate_id = selected_row.get("candidate_id") if selected_row else None
    next_state = {
        "last_selected_candidate_id": selected_candidate_id
        if selected_action == "select_strategy"
        else selector_state.get("last_selected_candidate_id"),
        "last_selected_state_id": selected_row.get("state_id") if selected_row else selector_state.get("last_selected_state_id"),
        "observations_since_switch": _next_observations_since_switch(
            selector_state,
            selected_candidate_id=selected_candidate_id,
            selected_action=selected_action,
        ),
    }
    return {
        "factory": "selector_matching",
        "schema_version": SELECTOR_MATCHING_DECISION_SCHEMA_VERSION,
        "decision_id": decision_id,
        "generated_at": generated_at,
        "data_asof": current.get("data_asof"),
        "selected_action": selected_action,
        "selected_strategy_id": selected_row.get("strategy_id") if selected_row else "no_trade",
        "selected_candidate_id": selected_candidate_id,
        "selected_state_id": selected_row.get("state_id") if selected_row else None,
        "selected_horizon_profile_id": selected_row.get("horizon_profile_id")
        if selected_row
        else current.get("horizon_profile_id"),
        "no_trade_reason": no_trade_reason,
        "selector_version": selector_version,
        "state_encoder_version": current.get("state_encoder_version"),
        "evidence_unit": selected_row.get("strategy_identity_unit") if selected_row else None,
        "confidence": current.get("state_confidence"),
        "uncertainty": current.get("uncertainty"),
        "reason_codes": sorted(set(reason_codes)),
        "comparison_set": [_comparison_summary(row) for row in comparison_rows],
        "rejected_alternatives": [_comparison_summary(row) for row in rejected_rows],
        "selected_row": dict(selected_row) if selected_row else None,
        "selector_state": dict(selector_state),
        "next_selector_state": next_state,
        "source_artifacts": dict(source_artifacts or {}),
        "safety_scope": _safety_scope(),
    }


def _current_market_state(current: Mapping[str, Any]) -> Mapping[str, Any]:
    if current.get("schema_version") == "market_state_snapshot_v1":
        return {
            "schema_version": "current_market_state_v1",
            "snapshot_run_id": current.get("run_id"),
            "generated_at": current.get("generated_at"),
            "data_asof": current.get("data_asof"),
            "latest_local_candle_at": current.get("latest_local_candle_at"),
            "pair": current.get("pair"),
            "pair_group": current.get("pair_group"),
            "base_timeframe": current.get("base_timeframe"),
            "cost_model_id": current.get("cost_model_id"),
            "aggregate_label": current.get("aggregate_label"),
            "horizon_profile_id": current.get("horizon_profile_id"),
            "state_encoder_version": current.get("state_encoder_version"),
            "state_confidence": current.get("state_confidence"),
            "uncertainty": current.get("uncertainty"),
            "out_of_distribution_score": current.get("out_of_distribution_score"),
            "stale_data": bool((current.get("data_quality_summary") or {}).get("stale_data")),
            "no_trade_default": bool(current.get("no_trade_default")),
            "horizon_conflict": current.get("horizon_conflict"),
            "feature_quality_summary": current.get("feature_quality_summary"),
            "horizons": [
                {
                    "horizon": row.get("horizon"),
                    "horizon_group": row.get("horizon_group"),
                    "state_id": row.get("state_id"),
                    "label": row.get("label"),
                    "confidence": row.get("confidence"),
                    "uncertainty": row.get("uncertainty"),
                    "out_of_distribution_score": row.get("out_of_distribution_score"),
                    "reason_codes": row.get("reason_codes", []),
                }
                for row in current.get("horizons", [])
            ],
            "reason_codes": list(current.get("reason_codes", [])),
            "safety_scope": current.get("safety_scope", {}),
        }
    return current


def _market_state_no_trade_reasons(
    current: Mapping[str, Any],
    *,
    min_state_confidence: float,
    max_out_of_distribution_score: float,
) -> list[str]:
    reasons: list[str] = []
    if current.get("stale_data") is True:
        reasons.append("stale_local_data")
    if current.get("no_trade_default") is True:
        reasons.append("market_state_no_trade_default")
    if current.get("aggregate_label") in {
        "unknown",
        "mixed",
        "transition",
        "out_of_distribution",
    }:
        reasons.append(f"{current.get('aggregate_label')}_state_no_trade")
    if _number(current.get("state_confidence"), 0.0) < min_state_confidence:
        reasons.append("state_confidence_below_threshold")
    if _number(current.get("out_of_distribution_score"), 0.0) >= max_out_of_distribution_score:
        reasons.append("out_of_distribution_state")
    if current.get("cost_model_stale") is True:
        reasons.append("cost_model_stale")
    cost_model = current.get("cost_model") or {}
    if isinstance(cost_model, Mapping) and cost_model.get("stale") is True:
        reasons.append("cost_model_stale")
    feature_quality = current.get("feature_quality_summary") or {}
    if feature_quality.get("feature_quality_pass") is False:
        reasons.append("feature_quality_failed")
    conflict = current.get("horizon_conflict") or {}
    if conflict.get("conflict_detected") is True:
        reasons.append("horizon_conflict")
    return sorted(set(reasons))


def _current_state_scopes(current: Mapping[str, Any]) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for row in current.get("horizons", []):
        state_id = str(row.get("state_id") or "")
        if not state_id:
            continue
        scopes.append(
            {
                "state_id": state_id,
                "state_label": row.get("label"),
                "horizon": row.get("horizon"),
                "horizon_group": row.get("horizon_group"),
            }
        )
    return scopes


def _eligible_rows_for_current_state(
    rows: Any,
    *,
    current: Mapping[str, Any],
    current_scopes: Sequence[Mapping[str, Any]],
    current_horizon_profile_id: str,
) -> list[Mapping[str, Any]]:
    state_ids = {scope["state_id"] for scope in current_scopes}
    if not isinstance(rows, list):
        return []
    eligible = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("row_type") != "strategy":
            continue
        if row.get("decision") != SELECTOR_ELIGIBLE_DECISION:
            continue
        if row.get("state_id") not in state_ids:
            continue
        if str(row.get("horizon_profile_id") or "") != current_horizon_profile_id:
            continue
        if not _row_matches_current_market_identity(row, current):
            continue
        if row.get("identity_mismatch") is True:
            continue
        if row.get("data_quality_pass") is False or row.get("feature_quality_pass") is False:
            continue
        eligible.append(row)
    return eligible


def _comparison_rows(
    rows: Any,
    *,
    current: Mapping[str, Any],
    current_scopes: Sequence[Mapping[str, Any]],
    current_horizon_profile_id: str,
    selector_state: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list):
        return []
    state_ids = {scope["state_id"] for scope in current_scopes}
    previous_candidate_id = selector_state.get("last_selected_candidate_id")
    comparison = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        same_state = (
            row.get("state_id") in state_ids
            and str(row.get("horizon_profile_id") or "") == current_horizon_profile_id
            and _row_matches_current_market_identity(row, current)
        )
        incumbent = previous_candidate_id and row.get("candidate_id") == previous_candidate_id
        if same_state or incumbent:
            comparison.append(row)
    return comparison


def _row_matches_current_market_identity(
    row: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    row_pair = _identity_value(row.get("pair"))
    current_pair = _identity_value(current.get("pair"))
    row_timeframe = _identity_value(row.get("timeframe"))
    current_timeframe = _current_timeframe(current)
    row_cost_model = _identity_value(row.get("cost_model_id"))
    current_cost_model = _identity_value(current.get("cost_model_id"))
    return (
        bool(
            row_pair
            and current_pair
            and row_timeframe
            and current_timeframe
            and row_cost_model
            and current_cost_model
        )
        and row_pair == current_pair
        and row_timeframe == current_timeframe
        and row_cost_model == current_cost_model
    )


def _current_timeframe(current: Mapping[str, Any]) -> str:
    return _identity_value(current.get("base_timeframe") or current.get("timeframe"))


def _identity_value(value: Any) -> str:
    return str(value or "")


def _previous_row(
    rows: Sequence[Mapping[str, Any]], selector_state: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    previous_candidate_id = selector_state.get("last_selected_candidate_id")
    if not previous_candidate_id:
        return None
    for row in rows:
        if row.get("candidate_id") == previous_candidate_id:
            return row
    return None


def _rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        _number(row.get("stress_cost_utility"), 0.0),
        _number(row.get("expected_utility_after_cost"), 0.0),
        _number(row.get("lower_confidence_bound"), 0.0),
        -_number(row.get("max_drawdown"), 0.0),
    )


def _rank_score(row: Mapping[str, Any]) -> float:
    return _number(row.get("stress_cost_utility"), _number(row.get("rank_score"), 0.0))


def _comparison_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_type": row.get("row_type"),
        "strategy_id": row.get("strategy_id"),
        "candidate_id": row.get("candidate_id"),
        "state_id": row.get("state_id"),
        "horizon_profile_id": row.get("horizon_profile_id"),
        "matching_action": row.get("matching_action"),
        "decision": row.get("decision"),
        "rank_score": row.get("rank_score"),
        "stress_cost_utility": row.get("stress_cost_utility"),
        "expected_utility_after_cost": row.get("expected_utility_after_cost"),
        "reason_codes": list(row.get("reason_codes", [])),
    }


def _next_observations_since_switch(
    selector_state: Mapping[str, Any],
    *,
    selected_candidate_id: str | None,
    selected_action: str,
) -> int:
    if selected_action != "select_strategy" or selected_candidate_id is None:
        return int(selector_state.get("observations_since_switch") or 0) + 1
    if selected_candidate_id == selector_state.get("last_selected_candidate_id"):
        return int(selector_state.get("observations_since_switch") or 0) + 1
    return 0


def _state_is_uncertain(current: Mapping[str, Any]) -> bool:
    return bool(current.get("no_trade_default")) or current.get("aggregate_label") in {
        "unknown",
        "mixed",
        "transition",
        "out_of_distribution",
        "high_volatility",
    }


def _no_trade_assessment(
    *,
    current: Mapping[str, Any],
    current_state: bool,
    best_utility: float,
) -> str:
    if not current_state:
        return "not_current_state"
    if _state_is_uncertain(current):
        return "acceptable_uncertain_or_ood_state"
    if best_utility > 0:
        return "costly_supported_state"
    return "acceptable_no_supported_positive_edge"


def _no_trade_reason_codes(
    *,
    current: Mapping[str, Any],
    current_state: bool,
    best_utility: float,
) -> list[str]:
    reasons = ["no_hindsight_profit_credit"]
    if not current_state:
        return reasons + ["not_current_state"]
    if _state_is_uncertain(current):
        reasons.append("uncertain_state_no_trade_value")
    if best_utility > 0:
        reasons.append("opportunity_cost_present")
    if current.get("aggregate_label") == "high_volatility":
        reasons.append("high_volatility_safety_value")
    return sorted(set(reasons))


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safety_scope() -> dict[str, bool]:
    return {
        "local_artifacts_source_of_truth": True,
        "historical_evaluation_only": True,
        "selector_simulation_only": True,
        "freqtrade_trade_started": False,
        "paper_trading_started": False,
        "dry_run_trading_started": False,
        "live_trading_started": False,
        "exchange_order_placement": False,
        "uses_api_keys_or_secrets": False,
        "metadata_contains_secrets": False,
        "process_control": False,
        "leverage_above_one": False,
        "shorting": False,
        "promotion_authorized_by_this_artifact": False,
    }


def _safe_component(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return clean.strip("._") or "selector_matching"


def _compact_timestamp(value: str) -> str:
    return _safe_component(value.replace("+00:00", "Z").replace(":", "").replace("-", ""))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
