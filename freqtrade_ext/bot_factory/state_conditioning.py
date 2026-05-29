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
    rows = [
        _scorecard_row(
            row,
            candidate_identity=candidate_identity,
            state_row=state_by_label.get(str(row.get("market_regime") or "")),
            fallback_horizon_profile_id=str(market_state_snapshot.get("horizon_profile_id") or ""),
            cost_model_id=str(
                regime_scorecard.get("cost_model_id")
                or candidate_identity.get("cost_model_id")
                or market_state_snapshot.get("cost_model_id")
                or ""
            ),
            state_encoder_version=str(market_state_snapshot.get("state_encoder_version") or ""),
        )
        for row in regime_scorecard.get("scorecard_by_regime", [])
    ]
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
    fallback_horizon_profile_id: str,
    cost_model_id: str,
    state_encoder_version: str,
) -> dict[str, Any]:
    state_row = state_row or {}
    decision = _state_decision(str(row.get("decision") or ""))
    blockers = _row_blockers(row, state_row=state_row)
    if blockers and decision == "STATE_SELECTOR_ELIGIBLE":
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
        "state_id": state_row.get("state_id")
        or f"{state_encoder_version}:missing:{row.get('market_regime')}:low",
        "state_label": state_row.get("label") or row.get("market_regime"),
        "horizon_profile_id": state_row.get("horizon_profile_id")
        or fallback_horizon_profile_id,
        "state_encoder_version": state_encoder_version,
        "pair": _first_or_unknown(candidate_identity.get("allowed_pairs")),
        "timeframe": _first_or_unknown(candidate_identity.get("allowed_timeframes")),
        "cost_model_id": cost_model_id,
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


def _state_rows_by_label(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in snapshot.get("horizons", []):
        label = str(row.get("label") or "")
        if label and label not in result:
            enriched = dict(row)
            enriched["horizon_profile_id"] = snapshot.get("horizon_profile_id")
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


def _row_blockers(row: Mapping[str, Any], *, state_row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    source_eligible = row.get("decision") in {
        "REGIME_SCOPED_SELECTOR_ELIGIBLE",
        "GLOBAL_SELECTOR_ELIGIBLE",
    }
    if not state_row and row.get("decision") in {
        "REGIME_SCOPED_SELECTOR_ELIGIBLE",
        "GLOBAL_SELECTOR_ELIGIBLE",
    }:
        blockers.append("state_id_missing_for_regime")
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
    for item in (regime_scorecard.get("baseline_comparison") or {}).get("by_regime", []):
        regime = str(item.get("market_regime") or "")
        state_row = state_by_label.get(regime, {})
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
            if row.get("decision") == "STATE_SELECTOR_ELIGIBLE"
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
