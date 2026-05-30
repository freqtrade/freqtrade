from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


STATE_CONDITIONED_SCORECARD_SCHEMA_VERSION = "state_conditioned_scorecard_v1"
STATE_CONDITIONED_SCORECARD_REPORT_SCHEMA_VERSION = "state_conditioned_scorecard_report_v1"
REQUIRED_SELECTOR_ROW_FIELDS = (
    "state_id",
    "horizon_profile_id",
    "state_encoder_version",
    "strategy_version",
    "signal_version",
    "risk_policy_version",
    "cost_model_id",
    "pair",
    "timeframe",
)
SOURCE_OBSERVATION_STATE_FIELDS = (
    "state_id",
    "horizon_profile_id",
    "state_encoder_version",
    "state_window_id",
    "feature_cutoff_timestamp",
    "label_cutoff_timestamp",
    "decision_window_start",
    "decision_window_end",
    "future_data_used",
)
STATE_CONDITIONED_AGGREGATION_KEYS = (
    "strategy_version",
    "signal_version",
    "risk_policy_version",
    "state_id",
    "horizon_profile_id",
    "state_encoder_version",
    "cost_model_id",
    "pair_group",
    "timeframe",
)
HARD_ROW_BLOCKERS = {
    "state_fields_missing_from_source_observation",
    "future_data_used",
    "strategy_identity_mismatch",
    "cost_model_mismatch",
    "state_encoder_mismatch",
    "pair_outside_candidate_identity",
    "timeframe_outside_candidate_identity",
}


def build_state_conditioned_scorecard(
    *,
    regime_scorecard: Mapping[str, Any],
    market_state_snapshot: Mapping[str, Any],
    run_id: str | None = None,
    generated_at: str | None = None,
    source_artifacts: Mapping[str, str] | None = None,
    proxy_evidence: bool = False,
    relaxed_thresholds_used: bool = False,
    require_walk_forward_evidence: bool = True,
    reviewer_notes: Sequence[str] = (),
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    candidate_identity = dict(regime_scorecard.get("candidate_identity") or {})
    candidate_id = str(candidate_identity.get("candidate_id") or regime_scorecard.get("candidate_id") or "unknown")
    run_id = run_id or f"{candidate_id}_state_scorecard"
    state_by_label = _state_rows_by_label(market_state_snapshot)
    source_strict = _source_scorecard_strict(regime_scorecard)
    historical_gate_passed = regime_scorecard.get("decision") in {
        "GLOBAL_SELECTOR_ELIGIBLE",
        "REGIME_SCOPED_SELECTOR_ELIGIBLE",
    }
    walk_forward_gate_passed = _walk_forward_gate_passed(regime_scorecard)
    raw_rows = [
        _scorecard_row(
            row,
            candidate_identity=candidate_identity,
            state_row=state_by_label.get(str(row.get("market_regime") or "")),
            cost_model_id=str(
                regime_scorecard.get("cost_model_id")
                or candidate_identity.get("cost_model_id")
                or market_state_snapshot.get("cost_model_id")
                or ""
            ),
        )
        for row in regime_scorecard.get("scorecard_by_regime", [])
    ]
    rows = _aggregate_state_conditioned_rows(raw_rows)
    baseline_comparisons = _baseline_comparisons(
        regime_scorecard,
        market_state_snapshot=market_state_snapshot,
        state_by_label=state_by_label,
    )
    blockers = _blockers(
        source_strict=source_strict,
        historical_gate_passed=historical_gate_passed,
        walk_forward_gate_passed=walk_forward_gate_passed,
        proxy_evidence=proxy_evidence,
        relaxed_thresholds_used=relaxed_thresholds_used,
        rows=rows,
    )
    diagnostic_only = bool(blockers)
    selector_allowed = (
        not diagnostic_only
        and any(row["decision"] == "STATE_SELECTOR_ELIGIBLE" for row in rows)
    )
    reason_codes = ["strict_state_conditioned_scorecard"] if selector_allowed else blockers
    return {
        "factory": "state_conditioned_scorecard",
        "schema_version": STATE_CONDITIONED_SCORECARD_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "candidate_id": candidate_id,
        "candidate_identity": candidate_identity,
        "candidate_identity_schema_version": "strategy_candidate_identity_v1",
        "source_artifacts": dict(source_artifacts or {}),
        "source_regime_scorecard_schema_version": regime_scorecard.get("schema_version"),
        "source_market_state_schema_version": market_state_snapshot.get("schema_version"),
        "state_encoder_version": market_state_snapshot.get("state_encoder_version"),
        "horizon_profile_ids": sorted(
            {
                str(row.get("horizon_profile_id"))
                for row in rows
                if row.get("horizon_profile_id")
            }
        ),
        "cost_model_id": (
            candidate_identity.get("cost_model_id")
            or market_state_snapshot.get("cost_model_id")
        ),
        "evidence_eligibility": (
            "selector_eligible_candidate" if selector_allowed else "diagnostic_only"
        ),
        "diagnostic_only": diagnostic_only,
        "proxy_evidence": bool(proxy_evidence),
        "relaxed_thresholds_used": bool(relaxed_thresholds_used),
        "actual_strategy_backtest_required": True,
        "historical_gate_passed": historical_gate_passed,
        "walk_forward_gate_required": bool(require_walk_forward_evidence),
        "walk_forward_gate_passed": walk_forward_gate_passed,
        "selector_candidate_creation_allowed": selector_allowed,
        "paper_readiness_input_allowed": selector_allowed,
        "rows": rows,
        "baseline_comparisons": baseline_comparisons,
        "summary_decision": _summary_decision(rows, diagnostic_only=diagnostic_only),
        "blockers": blockers,
        "reason_codes": reason_codes,
        "reviewer_notes": list(reviewer_notes),
        "safety_scope": _safety_scope(),
    }


def validate_state_conditioned_scorecard_for_selector(
    scorecard: Mapping[str, Any],
) -> dict[str, Any]:
    selector_rows = [
        row
        for row in scorecard.get("rows", [])
        if isinstance(row, Mapping) and row.get("decision") == "STATE_SELECTOR_ELIGIBLE"
    ]
    checks = [
        _check(
            "state_conditioned_scorecard_schema",
            scorecard.get("factory") == "state_conditioned_scorecard"
            and scorecard.get("schema_version") == STATE_CONDITIONED_SCORECARD_SCHEMA_VERSION,
            {
                "factory": scorecard.get("factory"),
                "schema_version": scorecard.get("schema_version"),
            },
        ),
        _check(
            "state_conditioned_scorecard_not_diagnostic_only",
            scorecard.get("diagnostic_only") is False,
            {"diagnostic_only": scorecard.get("diagnostic_only")},
        ),
        _check(
            "state_conditioned_scorecard_selector_creation_allowed",
            scorecard.get("selector_candidate_creation_allowed") is True,
            {
                "selector_candidate_creation_allowed": scorecard.get(
                    "selector_candidate_creation_allowed"
                )
            },
        ),
        _check(
            "state_conditioned_scorecard_no_proxy_evidence",
            scorecard.get("proxy_evidence") is False,
            {"proxy_evidence": scorecard.get("proxy_evidence")},
        ),
        _check(
            "state_conditioned_scorecard_no_relaxed_thresholds",
            scorecard.get("relaxed_thresholds_used") is False,
            {"relaxed_thresholds_used": scorecard.get("relaxed_thresholds_used")},
        ),
        _check(
            "state_conditioned_scorecard_walk_forward_gate_passed",
            scorecard.get("walk_forward_gate_passed") is True,
            {"walk_forward_gate_passed": scorecard.get("walk_forward_gate_passed")},
        ),
        _check(
            "state_conditioned_scorecard_identity_present",
            isinstance(scorecard.get("candidate_identity"), Mapping)
            and bool((scorecard.get("candidate_identity") or {}).get("candidate_id")),
            {
                "candidate_identity_type": type(scorecard.get("candidate_identity")).__name__,
                "candidate_id": (scorecard.get("candidate_identity") or {}).get("candidate_id")
                if isinstance(scorecard.get("candidate_identity"), Mapping)
                else None,
            },
        ),
        _check(
            "state_conditioned_scorecard_rows_present",
            isinstance(scorecard.get("rows"), list) and bool(scorecard.get("rows")),
            {
                "rows_type": type(scorecard.get("rows")).__name__,
                "row_count": len(scorecard.get("rows", []))
                if isinstance(scorecard.get("rows"), list)
                else 0,
            },
        ),
        _check(
            "state_conditioned_scorecard_selector_rows_present",
            bool(selector_rows),
            {"selector_row_count": len(selector_rows)},
        ),
        _check(
            "state_conditioned_scorecard_selector_rows_complete",
            not _selector_row_missing_fields(selector_rows),
            {"missing_by_row": _selector_row_missing_fields(selector_rows)},
        ),
        _check(
            "state_conditioned_scorecard_selector_rows_source_scope_complete",
            not _selector_row_missing_source_state_fields(selector_rows)
            and all(row.get("future_data_used") is False for row in selector_rows),
            {
                "missing_by_row": _selector_row_missing_source_state_fields(
                    selector_rows
                ),
                "future_data_used_rows": [
                    row.get("candidate_id")
                    for row in selector_rows
                    if row.get("future_data_used") is not False
                ],
            },
        ),
        _check(
            "state_conditioned_scorecard_baselines_present",
            not _missing_baseline_scopes(scorecard, selector_rows),
            {"missing_baselines": _missing_baseline_scopes(scorecard, selector_rows)},
        ),
        _check(
            "state_conditioned_scorecard_paper_readiness_allowed",
            scorecard.get("paper_readiness_input_allowed") is True,
            {
                "paper_readiness_input_allowed": scorecard.get(
                    "paper_readiness_input_allowed"
                )
            },
        ),
    ]
    ok = all(check["passed"] for check in checks)
    return {
        "factory": "state_conditioned_scorecard_selector_validation",
        "schema_version": STATE_CONDITIONED_SCORECARD_SCHEMA_VERSION,
        "ok": ok,
        "checks": checks,
        "reason_codes": ["state_conditioned_scorecard_selector_valid"]
        if ok
        else [check["name"] for check in checks if not check["passed"]],
        "safety_scope": _safety_scope(),
    }


def write_state_conditioned_scorecard_artifacts(
    scorecard: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Path]:
    candidate_id = _safe_component(str(scorecard.get("candidate_id") or "unknown"))
    run_id = _safe_component(str(scorecard.get("run_id") or "state_scorecard"))
    out_dir = output_root / candidate_id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_conditioned_scorecard": out_dir / "state_conditioned_scorecard.json",
        "state_conditioned_scorecard_report": out_dir / "state_conditioned_scorecard_report.md",
    }
    paths["state_conditioned_scorecard"].write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["state_conditioned_scorecard_report"].write_text(
        render_state_conditioned_scorecard_report(scorecard),
        encoding="utf-8",
    )
    return paths


def render_state_conditioned_scorecard_report(scorecard: Mapping[str, Any]) -> str:
    lines = [
        "# State-Conditioned Scorecard",
        "",
        f"- Candidate: `{scorecard.get('candidate_id')}`",
        f"- Decision: `{scorecard.get('summary_decision')}`",
        f"- Evidence eligibility: `{scorecard.get('evidence_eligibility')}`",
        f"- Diagnostic only: `{scorecard.get('diagnostic_only')}`",
        f"- Selector candidate creation allowed: `{scorecard.get('selector_candidate_creation_allowed')}`",
        f"- Paper readiness input allowed: `{scorecard.get('paper_readiness_input_allowed')}`",
        f"- Reason codes: `{', '.join(scorecard.get('reason_codes', []))}`",
        "",
        "## Rows",
        "",
        "| state | horizon profile | decision | trades | stress edge | blockers |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in scorecard.get("rows", []):
        lines.append(
            "| {state} | {profile} | {decision} | {trades} | {stress} | {blockers} |".format(
                state=row.get("state_id"),
                profile=row.get("horizon_profile_id"),
                decision=row.get("decision"),
                trades=row.get("trade_count", 0),
                stress=row.get("net_return_stress_cost", 0.0),
                blockers=", ".join(row.get("blockers", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- This artifact is local-only evidence review.",
            "- It does not start paper, dry-run, live trading, `freqtrade trade`, process control, or exchange order placement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _scorecard_row(
    row: Mapping[str, Any],
    *,
    candidate_identity: Mapping[str, Any],
    state_row: Mapping[str, Any] | None,
    cost_model_id: str,
) -> dict[str, Any]:
    state_row = state_row or {}
    decision = _state_decision(str(row.get("decision") or ""))
    pair = _first_present(
        row.get("pair"),
        state_row.get("pair"),
        _first_or_unknown(candidate_identity.get("allowed_pairs")),
    )
    timeframe = _first_present(
        row.get("timeframe"),
        state_row.get("timeframe"),
        _first_or_unknown(candidate_identity.get("allowed_timeframes")),
    )
    row_cost_model_id = _first_present(row.get("cost_model_id"), cost_model_id)
    blockers = _row_blockers(
        row,
        state_row=state_row,
        candidate_identity=candidate_identity,
        row_pair=pair,
        row_timeframe=timeframe,
        row_cost_model_id=row_cost_model_id,
    )
    if decision == "STATE_SELECTOR_ELIGIBLE" and any(
        blocker in HARD_ROW_BLOCKERS for blocker in blockers
    ):
        decision = "STATE_DIAGNOSTIC_ONLY"
    elif blockers and decision == "STATE_SELECTOR_ELIGIBLE":
        decision = "STATE_SHADOW_ONLY"
    net_stress = _number(row.get("net_pnl_stress_cost"), 0.0)
    lcb = _number(row.get("lower_confidence_bound"), 0.0)
    max_drawdown = _number(row.get("max_drawdown"), 0.0)
    return {
        "candidate_id": candidate_identity.get("candidate_id"),
        "strategy_id": candidate_identity.get("strategy_id"),
        "strategy_version": candidate_identity.get("strategy_version"),
        "signal_version": candidate_identity.get("signal_version"),
        "risk_policy_version": candidate_identity.get("risk_policy_version"),
        "source_regime_decision": row.get("decision"),
        "source_state_observation_scope_complete": row.get(
            "source_state_observation_scope_complete"
        ),
        "source_state_observation_scope_reason_codes": list(
            row.get("source_state_observation_scope_reason_codes") or []
        ),
        "state_id": row.get("state_id"),
        "state_label": row.get("market_regime"),
        "horizon_profile_id": row.get("horizon_profile_id"),
        "state_encoder_version": row.get("state_encoder_version"),
        "state_window_id": row.get("state_window_id"),
        "state_window_ids": _state_window_ids(row),
        "feature_cutoff_timestamp": row.get("feature_cutoff_timestamp"),
        "label_cutoff_timestamp": row.get("label_cutoff_timestamp"),
        "feature_cutoff_range": _range_from_row(
            row,
            "feature_cutoff_range",
            "feature_cutoff_timestamp",
        ),
        "label_cutoff_range": _range_from_row(
            row,
            "label_cutoff_range",
            "label_cutoff_timestamp",
        ),
        "decision_window_start": row.get("decision_window_start"),
        "decision_window_end": row.get("decision_window_end"),
        "decision_windows": _decision_windows(row),
        "source_observation_count": _source_observation_count(row),
        "future_data_used": row.get("future_data_used"),
        "diagnostic_snapshot_state_id": state_row.get("state_id"),
        "diagnostic_snapshot_horizon_profile_id": state_row.get("horizon_profile_id"),
        "pair": pair,
        "pair_group": _first_present(row.get("pair_group"), row.get("pair_group_id"), pair),
        "timeframe": timeframe,
        "cost_model_id": row_cost_model_id,
        "sample_days": _number(row.get("sample_days"), 0.0),
        "independent_window_count": int(_number(row.get("window_count"), 0.0)),
        "non_overlapping_window_count": int(_number(row.get("window_count"), 0.0)),
        "trade_count": int(_number(row.get("trade_count"), 0.0)),
        "exposure_ratio": _number(row.get("exposure_ratio"), 0.0),
        "average_holding_time": None,
        "gross_return": _number(row.get("gross_return"), 0.0),
        "net_return_normal_cost": _number(row.get("net_pnl_normal_cost"), 0.0),
        "net_return_stress_cost": net_stress,
        "expected_utility_after_cost": net_stress,
        "risk_adjusted_score": round(lcb - (max_drawdown * 0.1), 6),
        "stress_cost_utility": round(net_stress + lcb - (max_drawdown * 0.05), 6),
        "uncertainty": _number(state_row.get("uncertainty"), 1.0),
        "expectancy": _number(row.get("expectancy"), 0.0),
        "profit_factor": row.get("profit_factor"),
        "win_rate": _number(row.get("win_rate"), 0.0),
        "max_drawdown": max_drawdown,
        "downside_deviation": _number(row.get("downside_deviation"), 0.0),
        "turnover": None,
        "cost_burden": None,
        "no_trade_delta": _number(row.get("no_trade_baseline_delta"), 0.0),
        "hold_delta": _number(row.get("hold_baseline_delta"), 0.0),
        "incumbent_delta": row.get("incumbent_delta"),
        "lower_confidence_bound": lcb,
        "pair_concentration": _number(row.get("pair_concentration"), 0.0),
        "calendar_concentration": _number(row.get("calendar_concentration"), 0.0),
        "state_sample_count": int(_number(row.get("window_count"), 0.0)),
        "state_cluster_stability": None,
        "data_quality_pass": bool(row.get("data_quality_pass")),
        "feature_quality_pass": True,
        "evidence_quality": "checked" if decision == "STATE_SELECTOR_ELIGIBLE" else "weak",
        "decision": decision,
        "blockers": blockers,
        "reason_codes": list(row.get("reason_codes", [])),
    }


def _aggregate_state_conditioned_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in STATE_CONDITIONED_AGGREGATION_KEYS)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    return [_aggregate_state_conditioned_row(grouped[key]) for key in order]


def _aggregate_state_conditioned_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = dict(rows[0])
    source_count = sum(_source_observation_count(row) for row in rows)
    state_window_ids = _unique_strings(
        item
        for row in rows
        for item in _state_window_ids(row)
    )
    decision_windows = _unique_decision_windows(
        item
        for row in rows
        for item in _decision_windows(row)
    )
    feature_cutoffs = [
        value
        for row in rows
        for value in _range_values(
            _range_from_row(row, "feature_cutoff_range", "feature_cutoff_timestamp")
        )
    ]
    label_cutoffs = [
        value
        for row in rows
        for value in _range_values(
            _range_from_row(row, "label_cutoff_range", "label_cutoff_timestamp")
        )
    ]
    decision_starts = [item["start"] for item in decision_windows if item.get("start")]
    decision_ends = [item["end"] for item in decision_windows if item.get("end")]
    blockers = sorted(
        {
            str(blocker)
            for row in rows
            for blocker in row.get("blockers", [])
            if blocker not in (None, "")
        }
    )
    reason_codes = sorted(
        {
            str(code)
            for row in rows
            for code in row.get("reason_codes", [])
            if code not in (None, "")
        }
    )
    first.update(
        {
            "state_window_id": state_window_ids[0] if state_window_ids else first.get("state_window_id"),
            "state_window_ids": state_window_ids,
            "feature_cutoff_timestamp": max(feature_cutoffs)
            if feature_cutoffs
            else first.get("feature_cutoff_timestamp"),
            "label_cutoff_timestamp": max(label_cutoffs)
            if label_cutoffs
            else first.get("label_cutoff_timestamp"),
            "feature_cutoff_range": _value_range(feature_cutoffs),
            "label_cutoff_range": _value_range(label_cutoffs),
            "decision_window_start": min(decision_starts)
            if decision_starts
            else first.get("decision_window_start"),
            "decision_window_end": max(decision_ends)
            if decision_ends
            else first.get("decision_window_end"),
            "decision_windows": decision_windows,
            "source_observation_count": source_count,
            "sample_days": round(sum(_number(row.get("sample_days"), 0.0) for row in rows), 6),
            "independent_window_count": int(
                sum(_number(row.get("independent_window_count"), 0.0) for row in rows)
            ),
            "non_overlapping_window_count": int(
                sum(_number(row.get("non_overlapping_window_count"), 0.0) for row in rows)
            ),
            "trade_count": int(sum(_number(row.get("trade_count"), 0.0) for row in rows)),
            "gross_return": round(sum(_number(row.get("gross_return"), 0.0) for row in rows), 6),
            "net_return_normal_cost": round(
                sum(_number(row.get("net_return_normal_cost"), 0.0) for row in rows),
                6,
            ),
            "net_return_stress_cost": round(
                sum(_number(row.get("net_return_stress_cost"), 0.0) for row in rows),
                6,
            ),
            "expected_utility_after_cost": round(
                sum(_number(row.get("expected_utility_after_cost"), 0.0) for row in rows),
                6,
            ),
            "no_trade_delta": round(
                sum(_number(row.get("no_trade_delta"), 0.0) for row in rows),
                6,
            ),
            "hold_delta": round(
                sum(_number(row.get("hold_delta"), 0.0) for row in rows),
                6,
            ),
            "lower_confidence_bound": min(
                (_number(row.get("lower_confidence_bound"), 0.0) for row in rows),
                default=0.0,
            ),
            "max_drawdown": max(
                (_number(row.get("max_drawdown"), 0.0) for row in rows),
                default=0.0,
            ),
            "downside_deviation": max(
                (_number(row.get("downside_deviation"), 0.0) for row in rows),
                default=0.0,
            ),
            "exposure_ratio": _weighted_mean(rows, "exposure_ratio"),
            "profit_factor": _weighted_mean(rows, "profit_factor"),
            "win_rate": _weighted_mean(rows, "win_rate"),
            "pair_concentration": max(
                (_number(row.get("pair_concentration"), 0.0) for row in rows),
                default=0.0,
            ),
            "calendar_concentration": max(
                (_number(row.get("calendar_concentration"), 0.0) for row in rows),
                default=0.0,
            ),
            "state_sample_count": int(
                sum(_number(row.get("state_sample_count"), 0.0) for row in rows)
            ),
            "blockers": blockers,
            "reason_codes": reason_codes,
        }
    )
    first["risk_adjusted_score"] = round(
        _number(first.get("lower_confidence_bound"), 0.0)
        - (_number(first.get("max_drawdown"), 0.0) * 0.1),
        6,
    )
    first["stress_cost_utility"] = round(
        _number(first.get("net_return_stress_cost"), 0.0)
        + _number(first.get("lower_confidence_bound"), 0.0)
        - (_number(first.get("max_drawdown"), 0.0) * 0.05),
        6,
    )
    first["decision"] = _aggregate_row_decision(rows, blockers)
    return first


def _aggregate_row_decision(
    rows: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
) -> str:
    decisions = {str(row.get("decision") or "") for row in rows}
    if any(blocker in HARD_ROW_BLOCKERS for blocker in blockers):
        return "STATE_DIAGNOSTIC_ONLY"
    if blockers:
        return "STATE_SHADOW_ONLY"
    if decisions == {"STATE_SELECTOR_ELIGIBLE"}:
        return "STATE_SELECTOR_ELIGIBLE"
    if "STATE_UNSAFE" in decisions:
        return "STATE_UNSAFE"
    if "STATE_NO_TRADE_POLICY" in decisions:
        return "STATE_NO_TRADE_POLICY"
    if "STATE_DIAGNOSTIC_ONLY" in decisions:
        return "STATE_DIAGNOSTIC_ONLY"
    return "STATE_INSUFFICIENT_EVIDENCE"


def _weighted_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    numerator = 0.0
    denominator = 0
    for row in rows:
        weight = max(_source_observation_count(row), 1)
        numerator += _number(row.get(field), 0.0) * weight
        denominator += weight
    return round(numerator / denominator, 6) if denominator else 0.0


def _row_identity_mismatches_candidate(
    row: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
) -> bool:
    fields = (
        "candidate_id",
        "strategy_id",
        "strategy_version",
        "signal_version",
        "risk_policy_version",
    )
    return any(
        row.get(field) not in (None, "")
        and candidate_identity.get(field) not in (None, "")
        and str(row.get(field)) != str(candidate_identity.get(field))
        for field in fields
    )


def _state_window_ids(row: Mapping[str, Any]) -> list[str]:
    values = row.get("state_window_ids")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return _unique_strings(values)
    return _unique_strings([row.get("state_window_id")])


def _decision_windows(row: Mapping[str, Any]) -> list[dict[str, str]]:
    windows = row.get("decision_windows")
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
        result: list[dict[str, str]] = []
        for item in windows:
            if not isinstance(item, Mapping):
                continue
            start = item.get("start")
            end = item.get("end")
            if start in (None, "") or end in (None, ""):
                continue
            result.append({"start": str(start), "end": str(end)})
        return _unique_decision_windows(result)
    start = row.get("decision_window_start")
    end = row.get("decision_window_end")
    if start in (None, "") or end in (None, ""):
        return []
    return [{"start": str(start), "end": str(end)}]


def _range_from_row(
    row: Mapping[str, Any],
    range_field: str,
    value_field: str,
) -> dict[str, str | None]:
    value_range = row.get(range_field)
    if isinstance(value_range, Mapping):
        return {
            "start": str(value_range.get("start"))
            if value_range.get("start") not in (None, "")
            else None,
            "end": str(value_range.get("end"))
            if value_range.get("end") not in (None, "")
            else None,
        }
    value = row.get(value_field)
    text = str(value) if value not in (None, "") else None
    return {"start": text, "end": text}


def _range_values(value_range: Mapping[str, Any]) -> list[str]:
    return _unique_strings([value_range.get("start"), value_range.get("end")])


def _value_range(values: Sequence[str]) -> dict[str, str | None]:
    clean = [str(value) for value in values if value not in (None, "")]
    return {
        "start": min(clean) if clean else None,
        "end": max(clean) if clean else None,
    }


def _unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _unique_decision_windows(
    windows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for item in windows:
        start = item.get("start")
        end = item.get("end")
        if start in (None, "") or end in (None, ""):
            continue
        key = (str(start), str(end))
        if key in seen:
            continue
        seen.add(key)
        result.append({"start": key[0], "end": key[1]})
    return result


def _source_observation_count(row: Mapping[str, Any]) -> int:
    return max(int(_number(row.get("source_observation_count"), 1.0)), 1)


def _state_rows_by_label(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in snapshot.get("horizons", []):
        label = str(row.get("label") or "")
        if label and label not in result:
            enriched = dict(row)
            enriched["horizon_profile_id"] = snapshot.get("horizon_profile_id")
            enriched["pair"] = snapshot.get("pair")
            enriched["timeframe"] = snapshot.get("base_timeframe")
            enriched["cost_model_id"] = snapshot.get("cost_model_id")
            result[label] = enriched
    return result


def _state_decision(regime_decision: str) -> str:
    return {
        "REGIME_SCOPED_SELECTOR_ELIGIBLE": "STATE_SELECTOR_ELIGIBLE",
        "GLOBAL_SELECTOR_ELIGIBLE": "STATE_SELECTOR_ELIGIBLE",
        "SHADOW_ONLY": "STATE_SHADOW_ONLY",
        "INSUFFICIENT_EVIDENCE": "STATE_INSUFFICIENT_EVIDENCE",
        "REJECT": "STATE_UNSAFE",
        "NO_TRADE_POLICY": "STATE_NO_TRADE_POLICY",
    }.get(regime_decision, "STATE_DIAGNOSTIC_ONLY")


def _row_blockers(
    row: Mapping[str, Any],
    *,
    state_row: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
    row_pair: str,
    row_timeframe: str,
    row_cost_model_id: str,
) -> list[str]:
    blockers: list[str] = []
    source_eligible = row.get("decision") in {
        "REGIME_SCOPED_SELECTOR_ELIGIBLE",
        "GLOBAL_SELECTOR_ELIGIBLE",
    }
    if source_eligible and _source_state_missing_fields(row):
        blockers.append("state_fields_missing_from_source_observation")
    if source_eligible and row.get("future_data_used") is not False:
        blockers.append("future_data_used")
    if source_eligible and _row_identity_mismatches_candidate(row, candidate_identity):
        blockers.append("strategy_identity_mismatch")
    identity_cost_model_id = str(candidate_identity.get("cost_model_id") or "")
    if (
        source_eligible
        and identity_cost_model_id
        and row_cost_model_id
        and row_cost_model_id != identity_cost_model_id
    ):
        blockers.append("cost_model_mismatch")
    snapshot_state_encoder = str(state_row.get("state_encoder_version") or "")
    row_state_encoder = str(row.get("state_encoder_version") or "")
    if (
        source_eligible
        and snapshot_state_encoder
        and row_state_encoder
        and row_state_encoder != snapshot_state_encoder
    ):
        blockers.append("state_encoder_mismatch")
    allowed_pairs = {str(item) for item in candidate_identity.get("allowed_pairs") or []}
    if source_eligible and allowed_pairs and row_pair not in allowed_pairs:
        blockers.append("pair_outside_candidate_identity")
    allowed_timeframes = {
        str(item) for item in candidate_identity.get("allowed_timeframes") or []
    }
    if source_eligible and allowed_timeframes and row_timeframe not in allowed_timeframes:
        blockers.append("timeframe_outside_candidate_identity")
    if int(_number(row.get("window_count"), 0.0)) < 2 and source_eligible:
        blockers.append("insufficient_windows")
    if int(_number(row.get("trade_count"), 0.0)) <= 0 and source_eligible:
        blockers.append("insufficient_trades")
    if _number(row.get("net_pnl_stress_cost"), 0.0) <= 0 and source_eligible:
        blockers.append("negative_stress_cost_edge")
    if _number(row.get("lower_confidence_bound"), 0.0) <= 0 and source_eligible:
        blockers.append("lower_confidence_bound_not_positive")
    if _number(row.get("pair_concentration"), 0.0) > 0.8 and source_eligible:
        blockers.append("pair_concentration_too_high")
    if _number(row.get("calendar_concentration"), 0.0) > 0.8 and source_eligible:
        blockers.append("calendar_concentration_too_high")
    if row.get("data_quality_pass") is False:
        blockers.append("data_quality_failed")
    return blockers


def _selector_row_missing_fields(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_missing = [
            field
            for field in REQUIRED_SELECTOR_ROW_FIELDS
            if row.get(field) in (None, "")
        ]
        if row_missing:
            missing.append({"row_index": index, "missing_fields": row_missing})
    return missing


def _selector_row_missing_source_state_fields(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_missing = _source_state_missing_fields(row)
        if row_missing:
            missing.append({"row_index": index, "missing_fields": row_missing})
    return missing


def _source_state_missing_fields(row: Mapping[str, Any]) -> list[str]:
    return [
        field
        for field in SOURCE_OBSERVATION_STATE_FIELDS
        if row.get(field) in (None, "")
    ]


def _missing_baseline_scopes(
    scorecard: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    baseline_keys = {
        (
            str(item.get("state_id") or ""),
            str(item.get("horizon_profile_id") or ""),
            str(item.get("baseline_id") or ""),
        )
        for item in scorecard.get("baseline_comparisons", [])
        if isinstance(item, Mapping)
    }
    missing: list[dict[str, Any]] = []
    for row in rows:
        state_id = str(row.get("state_id") or "")
        horizon_profile_id = str(row.get("horizon_profile_id") or "")
        missing_baselines = [
            baseline_id
            for baseline_id in ("no_trade", "hold")
            if (state_id, horizon_profile_id, baseline_id) not in baseline_keys
        ]
        if missing_baselines:
            missing.append(
                {
                    "state_id": state_id,
                    "horizon_profile_id": horizon_profile_id,
                    "missing_baseline_ids": missing_baselines,
                }
            )
    return missing


def _baseline_comparisons(
    regime_scorecard: Mapping[str, Any],
    *,
    market_state_snapshot: Mapping[str, Any],
    state_by_label: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_state_by_regime = {
        str(row.get("market_regime") or ""): row
        for row in regime_scorecard.get("scorecard_by_regime", [])
        if isinstance(row, Mapping) and not _source_state_missing_fields(row)
    }
    for item in (regime_scorecard.get("baseline_comparison") or {}).get("by_regime", []):
        regime = str(item.get("market_regime") or "")
        source_state_row = source_state_by_regime.get(regime, {})
        state_row = source_state_row or state_by_label.get(regime, {})
        common = {
            "state_id": state_row.get("state_id") or f"missing:{regime}",
            "horizon_profile_id": state_row.get("horizon_profile_id")
            or market_state_snapshot.get("horizon_profile_id"),
            "pair": None,
            "timeframe": None,
        }
        rows.append(
            {
                **common,
                "baseline_id": "no_trade",
                "net_return_delta": _number(item.get("no_trade_delta"), 0.0),
                "drawdown_delta": None,
                "exposure_delta": None,
                "opportunity_cost": max(_number(item.get("candidate_return"), 0.0), 0.0),
                "reason_codes": ["no_trade_baseline_delta"],
            }
        )
        rows.append(
            {
                **common,
                "baseline_id": "hold",
                "net_return_delta": _number(item.get("hold_delta"), 0.0),
                "drawdown_delta": None,
                "exposure_delta": None,
                "opportunity_cost": None,
                "reason_codes": ["hold_baseline_delta"],
            }
        )
    return rows


def _source_scorecard_strict(scorecard: Mapping[str, Any]) -> bool:
    return (
        scorecard.get("factory") == "regime_fitness_scorecard"
        and scorecard.get("schema_version") == "regime_fitness_scorecard_v1"
        and scorecard.get("manual_review_only") is False
        and scorecard.get("candidate_identity") is not None
        and scorecard.get("raw_aggregate_pnl_promotion_allowed") is False
        and scorecard.get("promotion_authorized_by_this_command") is False
        and scorecard.get("phase3_readiness_required_after_scorecard") is True
    )


def _walk_forward_gate_passed(scorecard: Mapping[str, Any]) -> bool:
    eligible_rows = [
        row
        for row in scorecard.get("scorecard_by_regime", [])
        if row.get("decision") == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
    ]
    return bool(eligible_rows) and all(
        _number(row.get("walk_forward_pass_rate"), 0.0) >= 1.0
        for row in eligible_rows
    )


def _blockers(
    *,
    source_strict: bool,
    historical_gate_passed: bool,
    walk_forward_gate_passed: bool,
    proxy_evidence: bool,
    relaxed_thresholds_used: bool,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if not source_strict:
        blockers.append("source_regime_scorecard_not_strict")
    if not historical_gate_passed:
        blockers.append("historical_gate_not_selector_eligible")
    if proxy_evidence:
        blockers.append("proxy_evidence")
    if relaxed_thresholds_used:
        blockers.append("relaxed_thresholds_used")
    if not walk_forward_gate_passed:
        blockers.append("missing_walk_forward_evidence")
    if not any(row.get("decision") == "STATE_SELECTOR_ELIGIBLE" for row in rows):
        blockers.append("no_state_selector_eligible_rows")
    row_blockers = sorted(
        {
            item
            for row in rows
            if row.get("source_regime_decision")
            in {"REGIME_SCOPED_SELECTOR_ELIGIBLE", "GLOBAL_SELECTOR_ELIGIBLE"}
            for item in row.get("blockers", [])
        }
    )
    blockers.extend(row_blockers)
    return sorted(set(blockers))


def _summary_decision(rows: Sequence[Mapping[str, Any]], *, diagnostic_only: bool) -> str:
    if diagnostic_only:
        return "STATE_DIAGNOSTIC_ONLY"
    if any(row.get("decision") == "STATE_SELECTOR_ELIGIBLE" for row in rows):
        return "STATE_SELECTOR_ELIGIBLE"
    if any(row.get("decision") == "STATE_UNSAFE" for row in rows):
        return "STATE_UNSAFE"
    return "STATE_INSUFFICIENT_EVIDENCE"


def _safety_scope() -> dict[str, bool]:
    return {
        "local_artifacts_source_of_truth": True,
        "historical_evaluation_only": True,
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
        "phase3_readiness_required_after_scorecard": True,
    }


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _first_or_unknown(values: Any) -> str:
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
        return str(values[0])
    return "unknown"


def _first_present(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_component(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return clean.strip("._") or "state_scorecard"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
