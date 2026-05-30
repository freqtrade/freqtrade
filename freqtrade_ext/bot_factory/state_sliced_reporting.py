from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from freqtrade_ext.bot_factory.backtest_results import STYLE_GATE_PROFILES


STATE_SLICED_EVALUATION_SCHEMA_VERSION = "state_sliced_strategy_evaluation_v1"
STATE_SLICED_EVALUATION_REPORT_SCHEMA_VERSION = "state_sliced_strategy_evaluation_report_v1"


def build_state_sliced_evaluation_report(
    *,
    state_scorecard: Mapping[str, Any],
    historical_metrics: Mapping[str, Any] | None = None,
    walk_forward_metrics: Mapping[str, Any] | None = None,
    expected_state_ids: Sequence[str] = (),
    candidate_style: str = "scalp",
    incumbent_baseline_by_state: Mapping[str, float] | None = None,
    style_baseline_by_state: Mapping[str, float] | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
    max_state_drawdown_pct: float | None = None,
    source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    run_id = run_id or "state_sliced_" + _compact_timestamp(generated_at)
    rows = [row for row in state_scorecard.get("rows", []) if isinstance(row, Mapping)]
    expected_states = sorted(set(str(item) for item in expected_state_ids if item))
    observed_states = sorted({str(row.get("state_id") or "") for row in rows if row.get("state_id")})
    all_states = sorted(set(expected_states) | set(observed_states))
    incumbent_baseline_by_state = incumbent_baseline_by_state or {}
    style_baseline_by_state = style_baseline_by_state or {}
    backtest_slices = [
        _backtest_slice(
            row,
            incumbent_baseline_by_state=incumbent_baseline_by_state,
            style_baseline_by_state=style_baseline_by_state,
        )
        for row in rows
    ]
    walk_forward_slices = [
        _walk_forward_slice(row, walk_forward_metrics=walk_forward_metrics)
        for row in rows
    ]
    coverage = _coverage(rows, all_states)
    style_gates = [
        _style_state_gate(
            row,
            candidate_style=candidate_style,
            style_baseline_by_state=style_baseline_by_state,
            max_state_drawdown_pct=max_state_drawdown_pct,
        )
        for row in rows
    ]
    baseline_deltas = _baseline_deltas(
        state_scorecard,
        rows=rows,
        incumbent_baseline_by_state=incumbent_baseline_by_state,
        style_baseline_by_state=style_baseline_by_state,
    )
    crash_states = _state_specific_crashes(
        rows,
        historical_metrics=historical_metrics,
        max_state_drawdown_pct=max_state_drawdown_pct,
    )
    summary_decision = _summary_decision(
        coverage=coverage,
        style_gates=style_gates,
        crash_states=crash_states,
    )
    reason_codes = _reason_codes(
        coverage=coverage,
        style_gates=style_gates,
        crash_states=crash_states,
    )
    return {
        "factory": "state_sliced_strategy_evaluation",
        "schema_version": STATE_SLICED_EVALUATION_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "candidate_id": state_scorecard.get("candidate_id"),
        "candidate_identity": dict(state_scorecard.get("candidate_identity") or {}),
        "candidate_style": candidate_style,
        "source_state_scorecard_run_id": state_scorecard.get("run_id"),
        "historical_metrics_summary": dict(historical_metrics or {}),
        "walk_forward_metrics_summary": dict(walk_forward_metrics or {}),
        "state_coverage": coverage,
        "backtest_state_slices": backtest_slices,
        "walk_forward_state_slices": walk_forward_slices,
        "baseline_deltas_by_state": baseline_deltas,
        "style_specific_state_gates": style_gates,
        "state_specific_crashes": crash_states,
        "summary_decision": summary_decision,
        "reason_codes": reason_codes,
        "source_artifacts": dict(source_artifacts or {}),
        "safety_scope": _safety_scope(),
    }


def write_state_sliced_evaluation_artifacts(
    evaluation: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Path]:
    candidate_id = _safe_component(str(evaluation.get("candidate_id") or "unknown"))
    run_id = _safe_component(str(evaluation.get("run_id") or "state_sliced"))
    out_dir = output_root / candidate_id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_sliced_evaluation": out_dir / "state_sliced_evaluation.json",
        "state_sliced_evaluation_report": out_dir / "state_sliced_evaluation_report.md",
    }
    paths["state_sliced_evaluation"].write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["state_sliced_evaluation_report"].write_text(
        render_state_sliced_evaluation_report(evaluation),
        encoding="utf-8",
    )
    return paths


def render_state_sliced_evaluation_report(evaluation: Mapping[str, Any]) -> str:
    coverage = evaluation.get("state_coverage") or {}
    lines = [
        "# State-Sliced Strategy Evaluation",
        "",
        f"- Candidate: `{evaluation.get('candidate_id')}`",
        f"- Candidate style: `{evaluation.get('candidate_style')}`",
        f"- Decision: `{evaluation.get('summary_decision')}`",
        f"- Coverage: `{coverage.get('covered_state_count')}/{coverage.get('expected_state_count')}`",
        f"- Missing states: `{', '.join(coverage.get('missing_state_ids', [])) or 'none'}`",
        f"- Reason codes: `{', '.join(evaluation.get('reason_codes', []))}`",
        "",
        "## Backtest State Slices",
        "",
        "| state | decision | trades | normal edge | stress edge | drawdown | no_trade delta | hold delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in evaluation.get("backtest_state_slices", []):
        lines.append(
            "| {state} | {decision} | {trades} | {normal} | {stress} | {drawdown} | {no_trade} | {hold} |".format(
                state=row.get("state_id"),
                decision=row.get("state_decision"),
                trades=row.get("trade_count"),
                normal=row.get("net_return_normal_cost"),
                stress=row.get("net_return_stress_cost"),
                drawdown=row.get("max_drawdown"),
                no_trade=row.get("baseline_deltas", {}).get("no_trade"),
                hold=row.get("baseline_deltas", {}).get("hold"),
            )
        )
    lines.extend(
        [
            "",
            "## Walk-Forward State Slices",
            "",
            "| state | windows | non-overlap | source observations | walk-forward decision |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in evaluation.get("walk_forward_state_slices", []):
        lines.append(
            "| {state} | {windows} | {non_overlap} | {source_count} | {decision} |".format(
                state=row.get("state_id"),
                windows=row.get("independent_window_count"),
                non_overlap=row.get("non_overlapping_window_count"),
                source_count=row.get("source_observation_count"),
                decision=row.get("walk_forward_state_decision"),
            )
        )
    lines.extend(
        [
            "",
            "## Style-Specific State Gates",
            "",
            "| state | gate | failed checks |",
            "| --- | --- | --- |",
        ]
    )
    for row in evaluation.get("style_specific_state_gates", []):
        lines.append(
            "| {state} | {gate} | {failed} |".format(
                state=row.get("state_id"),
                gate=row.get("recommendation"),
                failed=", ".join(
                    check.get("name", "")
                    for check in row.get("checks", [])
                    if check.get("passed") is False
                )
                or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- This is local historical state-sliced evaluation only.",
            "- It does not start paper, dry-run, live trading, `freqtrade trade`, process control, or exchange order placement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _backtest_slice(
    row: Mapping[str, Any],
    *,
    incumbent_baseline_by_state: Mapping[str, float],
    style_baseline_by_state: Mapping[str, float],
) -> dict[str, Any]:
    state_id = str(row.get("state_id") or "")
    normal = _number(row.get("net_return_normal_cost"), 0.0)
    return {
        "state_id": state_id,
        "horizon_profile_id": row.get("horizon_profile_id"),
        "state_decision": row.get("decision"),
        "trade_count": int(_number(row.get("trade_count"), 0.0)),
        "sample_days": _number(row.get("sample_days"), 0.0),
        "net_return_normal_cost": normal,
        "net_return_stress_cost": _number(row.get("net_return_stress_cost"), 0.0),
        "max_drawdown": _number(row.get("max_drawdown"), 0.0),
        "lower_confidence_bound": _number(row.get("lower_confidence_bound"), 0.0),
        "baseline_deltas": {
            "no_trade": _number(row.get("no_trade_delta"), normal),
            "hold": _number(row.get("hold_delta"), 0.0),
            "incumbent": normal - _number(incumbent_baseline_by_state.get(state_id), 0.0)
            if state_id in incumbent_baseline_by_state
            else None,
            "style_specific": normal - _number(style_baseline_by_state.get(state_id), 0.0)
            if state_id in style_baseline_by_state
            else None,
        },
        "reason_codes": list(row.get("reason_codes", [])),
        "blockers": list(row.get("blockers", [])),
    }


def _walk_forward_slice(
    row: Mapping[str, Any],
    *,
    walk_forward_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    independent = int(_number(row.get("independent_window_count"), 0.0))
    non_overlap = int(_number(row.get("non_overlapping_window_count"), 0.0))
    state_decision = str(row.get("decision") or "")
    return {
        "state_id": row.get("state_id"),
        "horizon_profile_id": row.get("horizon_profile_id"),
        "independent_window_count": independent,
        "non_overlapping_window_count": non_overlap,
        "source_observation_count": int(_number(row.get("source_observation_count"), independent)),
        "walk_forward_pass_rate": (walk_forward_metrics or {}).get("summary", {}).get("pass_rate"),
        "walk_forward_state_decision": "STATE_WALK_FORWARD_PASS"
        if state_decision == "STATE_SELECTOR_ELIGIBLE" and independent > 0
        else "STATE_WALK_FORWARD_INSUFFICIENT",
        "state_window_ids": list(row.get("state_window_ids") or []),
        "decision_windows": list(row.get("decision_windows") or []),
    }


def _coverage(rows: Sequence[Mapping[str, Any]], state_ids: Sequence[str]) -> dict[str, Any]:
    covered = sorted(
        {
            str(row.get("state_id") or "")
            for row in rows
            if row.get("state_id") and row.get("decision") == "STATE_SELECTOR_ELIGIBLE"
        }
    )
    unsupported = sorted(
        {
            str(row.get("state_id") or "")
            for row in rows
            if row.get("state_id") and row.get("decision") != "STATE_SELECTOR_ELIGIBLE"
        }
    )
    expected = sorted(set(state_ids) | set(covered) | set(unsupported))
    missing = sorted(set(expected) - set(covered) - set(unsupported))
    return {
        "expected_state_ids": expected,
        "covered_state_ids": covered,
        "unsupported_state_ids": unsupported,
        "missing_state_ids": missing,
        "expected_state_count": len(expected),
        "covered_state_count": len(covered),
        "unsupported_state_count": len(unsupported),
        "missing_state_count": len(missing),
        "coverage_ratio": round(len(covered) / len(expected), 8) if expected else 0.0,
        "missingness_ratio": round(len(missing) / len(expected), 8) if expected else 0.0,
    }


def _style_state_gate(
    row: Mapping[str, Any],
    *,
    candidate_style: str,
    style_baseline_by_state: Mapping[str, float],
    max_state_drawdown_pct: float | None,
) -> dict[str, Any]:
    profile = STYLE_GATE_PROFILES.get(candidate_style, STYLE_GATE_PROFILES["scalp"])
    state_id = str(row.get("state_id") or "")
    normal = _number(row.get("net_return_normal_cost"), 0.0)
    style_baseline = style_baseline_by_state.get(state_id)
    checks = [
        _gate_check(
            "state_min_trades",
            int(_number(row.get("trade_count"), 0.0)) >= profile.min_trades,
            int(_number(row.get("trade_count"), 0.0)),
            f">= {profile.min_trades}",
        ),
        _gate_check(
            "state_positive_stress_edge",
            _number(row.get("net_return_stress_cost"), 0.0) > 0,
            _number(row.get("net_return_stress_cost"), 0.0),
            "> 0",
        ),
        _gate_check(
            "state_lower_confidence_positive",
            _number(row.get("lower_confidence_bound"), 0.0) > 0,
            _number(row.get("lower_confidence_bound"), 0.0),
            "> 0",
        ),
        _gate_check(
            "state_max_drawdown",
            _number(row.get("max_drawdown"), 0.0)
            <= (max_state_drawdown_pct or profile.max_drawdown_pct),
            _number(row.get("max_drawdown"), 0.0),
            f"<= {max_state_drawdown_pct or profile.max_drawdown_pct}",
        ),
    ]
    if profile.require_hold_baseline:
        checks.append(
            _gate_check(
                "state_hold_baseline_delta_positive",
                _number(row.get("hold_delta"), 0.0) > 0,
                _number(row.get("hold_delta"), 0.0),
                "> 0",
            )
        )
    if style_baseline is not None:
        checks.append(
            _gate_check(
                "state_style_baseline_delta_positive",
                normal - _number(style_baseline, 0.0) > 0,
                normal - _number(style_baseline, 0.0),
                "> 0",
            )
        )
    return {
        "state_id": state_id,
        "candidate_style": candidate_style,
        "recommendation": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
        "reason_codes": ["state_style_gate_passed"]
        if all(check["passed"] for check in checks)
        else [check["name"] for check in checks if not check["passed"]],
    }


def _baseline_deltas(
    scorecard: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    incumbent_baseline_by_state: Mapping[str, float],
    style_baseline_by_state: Mapping[str, float],
) -> list[dict[str, Any]]:
    output = [
        {
            "state_id": item.get("state_id"),
            "horizon_profile_id": item.get("horizon_profile_id"),
            "baseline_id": item.get("baseline_id"),
            "net_return_delta": item.get("net_return_delta"),
            "opportunity_cost": item.get("opportunity_cost"),
            "reason_codes": list(item.get("reason_codes", [])),
        }
        for item in scorecard.get("baseline_comparisons", [])
        if isinstance(item, Mapping)
    ]
    for row in rows:
        state_id = str(row.get("state_id") or "")
        normal = _number(row.get("net_return_normal_cost"), 0.0)
        if state_id in incumbent_baseline_by_state:
            output.append(
                {
                    "state_id": state_id,
                    "horizon_profile_id": row.get("horizon_profile_id"),
                    "baseline_id": "incumbent",
                    "net_return_delta": round(
                        normal - _number(incumbent_baseline_by_state.get(state_id), 0.0),
                        8,
                    ),
                    "opportunity_cost": None,
                    "reason_codes": ["incumbent_baseline_delta"],
                }
            )
        if state_id in style_baseline_by_state:
            output.append(
                {
                    "state_id": state_id,
                    "horizon_profile_id": row.get("horizon_profile_id"),
                    "baseline_id": "style_specific",
                    "net_return_delta": round(
                        normal - _number(style_baseline_by_state.get(state_id), 0.0),
                        8,
                    ),
                    "opportunity_cost": None,
                    "reason_codes": ["style_specific_baseline_delta"],
                }
            )
    return output


def _state_specific_crashes(
    rows: Sequence[Mapping[str, Any]],
    *,
    historical_metrics: Mapping[str, Any] | None,
    max_state_drawdown_pct: float | None,
) -> list[dict[str, Any]]:
    global_positive = _number((historical_metrics or {}).get("total_return_pct"), 0.0) > 0
    if not global_positive:
        return []
    threshold = max_state_drawdown_pct or STYLE_GATE_PROFILES["scalp"].max_drawdown_pct
    crashes = []
    for row in rows:
        reasons = []
        if _number(row.get("net_return_stress_cost"), 0.0) < 0:
            reasons.append("negative_state_stress_edge")
        if _number(row.get("max_drawdown"), 0.0) > threshold:
            reasons.append("state_drawdown_beyond_threshold")
        if reasons:
            crashes.append(
                {
                    "state_id": row.get("state_id"),
                    "horizon_profile_id": row.get("horizon_profile_id"),
                    "net_return_stress_cost": row.get("net_return_stress_cost"),
                    "max_drawdown": row.get("max_drawdown"),
                    "reason_codes": reasons,
                }
            )
    return crashes


def _summary_decision(
    *,
    coverage: Mapping[str, Any],
    style_gates: Sequence[Mapping[str, Any]],
    crash_states: Sequence[Mapping[str, Any]],
) -> str:
    if crash_states:
        return "STATE_SLICED_FAIL"
    if any(gate.get("recommendation") == "fail" for gate in style_gates):
        return "STATE_SLICED_REVIEW"
    if coverage.get("missing_state_count", 0) or coverage.get("unsupported_state_count", 0):
        return "STATE_SLICED_REVIEW"
    return "STATE_SLICED_PASS"


def _reason_codes(
    *,
    coverage: Mapping[str, Any],
    style_gates: Sequence[Mapping[str, Any]],
    crash_states: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons = ["state_sliced_evaluation_completed"]
    if crash_states:
        reasons.append("positive_global_result_hides_state_crash")
    if coverage.get("missing_state_count", 0):
        reasons.append("state_coverage_missing_states")
    if coverage.get("unsupported_state_count", 0):
        reasons.append("state_coverage_unsupported_states")
    if any(gate.get("recommendation") == "fail" for gate in style_gates):
        reasons.append("style_specific_state_gate_failed")
    return sorted(set(reasons))


def _gate_check(name: str, passed: bool, actual: Any, rule: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "actual": actual, "rule": rule}


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
    return clean.strip("._") or "state_sliced"


def _compact_timestamp(value: str) -> str:
    return _safe_component(value.replace("+00:00", "Z").replace(":", "").replace("-", ""))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
