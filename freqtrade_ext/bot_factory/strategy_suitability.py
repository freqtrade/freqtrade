from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from freqtrade_ext.bot_factory.state_conditioning import (
    validate_state_conditioned_scorecard_for_selector,
)


STRATEGY_SUITABILITY_MATRIX_SCHEMA_VERSION = "strategy_suitability_matrix_v1"
STRATEGY_SUITABILITY_REPORT_SCHEMA_VERSION = "strategy_suitability_report_v1"
SELECTOR_ELIGIBLE_DECISION = "SELECTOR_ELIGIBLE"
NO_TRADE_DECISIONS = {
    "NO_TRADE_POLICY",
    "NO_SUPPORTED_STRATEGY",
    "UNKNOWN_NO_TRADE",
    "OUT_OF_DISTRIBUTION_NO_TRADE",
    "UNSAFE_NO_TRADE",
}


def build_strategy_suitability_matrix(
    *,
    state_scorecards: Sequence[Mapping[str, Any]],
    market_state_snapshot: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
    source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    run_id = run_id or "strategy_suitability_" + _compact_timestamp(generated_at)
    scorecard_validations: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for scorecard in state_scorecards:
        validation = validate_state_conditioned_scorecard_for_selector(scorecard)
        scorecard_validations.append(
            {
                "candidate_id": scorecard.get("candidate_id"),
                "run_id": scorecard.get("run_id"),
                "ok": validation["ok"],
                "reason_codes": validation["reason_codes"],
                "validation": validation,
            }
        )
        identity = dict(scorecard.get("candidate_identity") or {})
        for row in scorecard.get("rows", []):
            if isinstance(row, Mapping):
                rows.append(
                    _strategy_matrix_row(
                        row,
                        scorecard=scorecard,
                        identity=identity,
                        scorecard_validation_ok=validation["ok"],
                    )
                )

    state_scopes = _known_state_scopes(rows, market_state_snapshot)
    covered_states = {
        (row["state_id"], row["horizon_profile_id"])
        for row in rows
        if row.get("decision") == SELECTOR_ELIGIBLE_DECISION
    }
    for scope in state_scopes:
        rows.append(_no_trade_policy_row(scope))
        if (scope["state_id"], scope["horizon_profile_id"]) not in covered_states:
            rows.append(_missing_state_row(scope))

    rows = sorted(
        rows,
        key=lambda item: (
            str(item.get("state_id") or ""),
            str(item.get("horizon_profile_id") or ""),
            str(item.get("row_type") or ""),
            str(item.get("candidate_id") or ""),
        ),
    )
    selector_row_count = sum(1 for row in rows if row.get("decision") == SELECTOR_ELIGIBLE_DECISION)
    return {
        "factory": "strategy_suitability_matrix",
        "schema_version": STRATEGY_SUITABILITY_MATRIX_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "source_artifacts": dict(source_artifacts or {}),
        "source_state_scorecard_count": len(state_scorecards),
        "source_market_state_schema_version": (
            market_state_snapshot or {}
        ).get("schema_version"),
        "scorecard_validations": scorecard_validations,
        "selector_row_count": selector_row_count,
        "state_count": len(state_scopes),
        "rows": rows,
        "summary": {
            "selector_eligible_rows": selector_row_count,
            "no_trade_rows": sum(1 for row in rows if row.get("row_type") == "no_trade"),
            "missing_state_rows": sum(
                1 for row in rows if row.get("row_type") == "missing_state"
            ),
            "diagnostic_rows": sum(
                1 for row in rows if row.get("decision") == "DIAGNOSTIC_ONLY"
            ),
        },
        "reason_codes": ["strategy_suitability_matrix_built_from_state_scorecards"],
        "safety_scope": _safety_scope(),
    }


def validate_strategy_suitability_matrix_for_selector(
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    rows = matrix.get("rows", [])
    strategy_rows = [row for row in rows if isinstance(row, Mapping) and row.get("row_type") == "strategy"]
    selector_rows = [
        row for row in strategy_rows if row.get("decision") == SELECTOR_ELIGIBLE_DECISION
    ]
    no_trade_rows = [row for row in rows if isinstance(row, Mapping) and row.get("row_type") == "no_trade"]
    checks = [
        _check(
            "strategy_suitability_matrix_schema",
            matrix.get("factory") == "strategy_suitability_matrix"
            and matrix.get("schema_version") == STRATEGY_SUITABILITY_MATRIX_SCHEMA_VERSION,
            {
                "factory": matrix.get("factory"),
                "schema_version": matrix.get("schema_version"),
            },
        ),
        _check(
            "strategy_suitability_rows_present",
            isinstance(rows, list) and bool(rows),
            {"row_count": len(rows) if isinstance(rows, list) else 0},
        ),
        _check(
            "strategy_suitability_no_trade_rows_present",
            bool(no_trade_rows),
            {"no_trade_row_count": len(no_trade_rows)},
        ),
        _check(
            "strategy_suitability_selector_rows_present",
            bool(selector_rows),
            {"selector_row_count": len(selector_rows)},
        ),
        _check(
            "strategy_suitability_selector_rows_have_checked_source",
            all(row.get("source_scorecard_selector_validation_ok") is True for row in selector_rows),
            {
                "selector_row_count": len(selector_rows),
                "unchecked_selector_rows": [
                    row.get("candidate_id")
                    for row in selector_rows
                    if row.get("source_scorecard_selector_validation_ok") is not True
                ],
            },
        ),
        _check(
            "strategy_suitability_selector_rows_identity_checked",
            all(not row.get("identity_mismatch") for row in selector_rows),
            {
                "identity_mismatches": [
                    row.get("candidate_id")
                    for row in selector_rows
                    if row.get("identity_mismatch")
                ]
            },
        ),
    ]
    ok = all(check["passed"] for check in checks)
    return {
        "factory": "strategy_suitability_matrix_selector_validation",
        "schema_version": STRATEGY_SUITABILITY_MATRIX_SCHEMA_VERSION,
        "ok": ok,
        "checks": checks,
        "reason_codes": ["strategy_suitability_matrix_selector_valid"]
        if ok
        else [check["name"] for check in checks if not check["passed"]],
        "safety_scope": _safety_scope(),
    }


def diff_strategy_suitability_matrices(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    previous_rows = _row_key_map(previous.get("rows", []))
    current_rows = _row_key_map(current.get("rows", []))
    added = sorted(set(current_rows) - set(previous_rows))
    removed = sorted(set(previous_rows) - set(current_rows))
    changed = sorted(
        key
        for key in set(previous_rows) & set(current_rows)
        if previous_rows[key].get("decision") != current_rows[key].get("decision")
        or previous_rows[key].get("rank_score") != current_rows[key].get("rank_score")
    )
    return {
        "factory": "strategy_suitability_matrix_diff",
        "schema_version": "strategy_suitability_matrix_diff_v1",
        "previous_run_id": previous.get("run_id"),
        "current_run_id": current.get("run_id"),
        "added_rows": added,
        "removed_rows": removed,
        "changed_rows": changed,
        "safety_scope": _safety_scope(),
    }


def write_strategy_suitability_artifacts(
    matrix: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Path]:
    run_id = _safe_component(str(matrix.get("run_id") or "strategy_suitability"))
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "strategy_suitability_matrix": out_dir / "strategy_state_suitability_matrix.json",
        "strategy_suitability_report": out_dir / "strategy_suitability_report.md",
    }
    paths["strategy_suitability_matrix"].write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["strategy_suitability_report"].write_text(
        render_strategy_suitability_report(matrix),
        encoding="utf-8",
    )
    return paths


def render_strategy_suitability_report(matrix: Mapping[str, Any]) -> str:
    lines = [
        "# Strategy Suitability Matrix",
        "",
        f"- Run ID: `{matrix.get('run_id')}`",
        f"- Selector-eligible rows: `{matrix.get('selector_row_count')}`",
        f"- State count: `{matrix.get('state_count')}`",
        "- This matrix permits local selector simulation only.",
        "- It does not permit paper, dry-run, live trading, process control, or exchange orders.",
        "",
        "## Rows",
        "",
        "| row type | candidate | state | horizon profile | decision | rank score | reason codes |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in matrix.get("rows", []):
        lines.append(
            "| {row_type} | {candidate} | {state} | {profile} | {decision} | {score} | {reasons} |".format(
                row_type=row.get("row_type"),
                candidate=row.get("candidate_id") or row.get("strategy_id"),
                state=row.get("state_id"),
                profile=row.get("horizon_profile_id"),
                decision=row.get("decision"),
                score=row.get("rank_score"),
                reasons=", ".join(row.get("reason_codes", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Why Not Trade?",
            "",
            "- States without checked selector-eligible strategy rows are represented by explicit no-trade rows.",
            "- Unknown, stale, mixed, weak-evidence, and out-of-distribution states must remain no-trade in selector matching.",
        ]
    )
    return "\n".join(lines) + "\n"


def _strategy_matrix_row(
    row: Mapping[str, Any],
    *,
    scorecard: Mapping[str, Any],
    identity: Mapping[str, Any],
    scorecard_validation_ok: bool,
) -> dict[str, Any]:
    mismatch_fields = _identity_mismatch_fields(row, identity)
    decision = _matrix_decision(row, scorecard_validation_ok=scorecard_validation_ok)
    if mismatch_fields and decision == SELECTOR_ELIGIBLE_DECISION:
        decision = "IDENTITY_MISMATCH"
    blockers = list(row.get("blockers", []))
    if mismatch_fields:
        blockers.append("strategy_identity_mismatch")
    expected_utility = _number(
        row.get("expected_utility_after_cost"),
        _number(row.get("net_return_stress_cost"), 0.0),
    )
    lower_confidence = _number(row.get("lower_confidence_bound"), 0.0)
    max_drawdown = _number(row.get("max_drawdown"), 0.0)
    stress_utility = _number(
        row.get("stress_cost_utility"),
        expected_utility + lower_confidence - (max_drawdown * 0.05),
    )
    return {
        "row_type": "strategy",
        "strategy_identity_unit": _strategy_identity_unit(identity),
        "strategy_id": identity.get("strategy_id") or row.get("strategy_id"),
        "strategy_class_name": identity.get("strategy_class_name"),
        "candidate_id": identity.get("candidate_id") or row.get("candidate_id"),
        "source_state_scorecard_run_id": scorecard.get("run_id"),
        "source_state_scorecard_schema_version": scorecard.get("schema_version"),
        "source_scorecard_selector_validation_ok": bool(scorecard_validation_ok),
        "state_id": row.get("state_id"),
        "state_label": row.get("state_label"),
        "horizon_profile_id": row.get("horizon_profile_id"),
        "pair_group": row.get("pair_group") or "single_pair",
        "pair": row.get("pair"),
        "timeframe": row.get("timeframe"),
        "cost_model_id": row.get("cost_model_id"),
        "state_encoder_version": row.get("state_encoder_version"),
        "decision": decision,
        "matching_action": "select_strategy"
        if decision == SELECTOR_ELIGIBLE_DECISION
        else "no_trade",
        "selector_eligible": decision == SELECTOR_ELIGIBLE_DECISION,
        "evidence_quality": row.get("evidence_quality") or "checked",
        "expected_utility_after_cost": round(expected_utility, 6),
        "risk_adjusted_score": round(_number(row.get("risk_adjusted_score"), 0.0), 6),
        "stress_cost_utility": round(stress_utility, 6),
        "rank_score": round(stress_utility, 6),
        "uncertainty": _number(row.get("uncertainty"), 1.0),
        "no_trade_delta": _number(row.get("no_trade_delta"), 0.0),
        "hold_delta": _number(row.get("hold_delta"), 0.0),
        "incumbent_delta": row.get("incumbent_delta"),
        "lower_confidence_bound": lower_confidence,
        "max_drawdown": max_drawdown,
        "data_quality_pass": bool(row.get("data_quality_pass")),
        "feature_quality_pass": bool(row.get("feature_quality_pass", True)),
        "identity_mismatch": bool(mismatch_fields),
        "identity_mismatch_fields": mismatch_fields,
        "blockers": sorted(set(blockers)),
        "reason_codes": sorted(set(list(row.get("reason_codes", [])) + [_decision_reason(decision)])),
    }


def _matrix_decision(
    row: Mapping[str, Any], *, scorecard_validation_ok: bool
) -> str:
    if not scorecard_validation_ok:
        return "DIAGNOSTIC_ONLY"
    state_decision = row.get("decision")
    if state_decision == "STATE_SELECTOR_ELIGIBLE":
        return SELECTOR_ELIGIBLE_DECISION
    if state_decision == "STATE_SHADOW_ONLY":
        return "SHADOW_ONLY"
    if state_decision == "STATE_UNSAFE":
        return "UNSAFE_NO_TRADE"
    if state_decision == "STATE_INSUFFICIENT_EVIDENCE":
        return "UNKNOWN_NO_TRADE"
    if state_decision == "STATE_NO_TRADE_POLICY":
        return "NO_TRADE_POLICY"
    return "DIAGNOSTIC_ONLY"


def _known_state_scopes(
    rows: Sequence[Mapping[str, Any]],
    market_state_snapshot: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    scopes: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        state_id = str(row.get("state_id") or "")
        horizon_profile_id = str(row.get("horizon_profile_id") or "")
        if not state_id or not horizon_profile_id:
            continue
        scopes.setdefault(
            (state_id, horizon_profile_id),
            {
                "state_id": state_id,
                "state_label": row.get("state_label"),
                "horizon_profile_id": horizon_profile_id,
                "pair_group": row.get("pair_group") or "single_pair",
                "pair": row.get("pair"),
                "timeframe": row.get("timeframe"),
                "cost_model_id": row.get("cost_model_id"),
                "state_encoder_version": row.get("state_encoder_version"),
                "out_of_distribution_score": row.get("out_of_distribution_score"),
            },
        )
    snapshot = market_state_snapshot or {}
    for state in snapshot.get("horizons", []):
        state_id = str(state.get("state_id") or "")
        horizon_profile_id = str(snapshot.get("horizon_profile_id") or "")
        if not state_id or not horizon_profile_id:
            continue
        scopes.setdefault(
            (state_id, horizon_profile_id),
            {
                "state_id": state_id,
                "state_label": state.get("label"),
                "horizon_profile_id": horizon_profile_id,
                "pair_group": snapshot.get("pair_group") or "single_pair",
                "pair": snapshot.get("pair"),
                "timeframe": state.get("timeframe") or state.get("horizon"),
                "cost_model_id": snapshot.get("cost_model_id"),
                "state_encoder_version": snapshot.get("state_encoder_version"),
                "out_of_distribution_score": state.get("out_of_distribution_score"),
            },
        )
    return list(scopes.values())


def _no_trade_policy_row(scope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_type": "no_trade",
        "strategy_identity_unit": {"policy_id": "no_trade"},
        "strategy_id": "no_trade",
        "candidate_id": "no_trade",
        "state_id": scope.get("state_id"),
        "state_label": scope.get("state_label"),
        "horizon_profile_id": scope.get("horizon_profile_id"),
        "pair_group": scope.get("pair_group"),
        "pair": scope.get("pair"),
        "timeframe": scope.get("timeframe"),
        "cost_model_id": scope.get("cost_model_id"),
        "state_encoder_version": scope.get("state_encoder_version"),
        "decision": "NO_TRADE_POLICY",
        "matching_action": "no_trade",
        "selector_eligible": False,
        "evidence_quality": "policy",
        "expected_utility_after_cost": 0.0,
        "risk_adjusted_score": 0.0,
        "stress_cost_utility": 0.0,
        "rank_score": 0.0,
        "uncertainty": 0.0,
        "no_trade_delta": 0.0,
        "hold_delta": 0.0,
        "incumbent_delta": None,
        "blockers": [],
        "reason_codes": ["first_class_no_trade_policy"],
    }


def _missing_state_row(scope: Mapping[str, Any]) -> dict[str, Any]:
    decision = _missing_state_decision(scope)
    return {
        "row_type": "missing_state",
        "strategy_identity_unit": {"policy_id": "missing_state"},
        "strategy_id": "missing_state",
        "candidate_id": "missing_state",
        "state_id": scope.get("state_id"),
        "state_label": scope.get("state_label"),
        "horizon_profile_id": scope.get("horizon_profile_id"),
        "pair_group": scope.get("pair_group"),
        "pair": scope.get("pair"),
        "timeframe": scope.get("timeframe"),
        "cost_model_id": scope.get("cost_model_id"),
        "state_encoder_version": scope.get("state_encoder_version"),
        "decision": decision,
        "matching_action": "no_trade",
        "selector_eligible": False,
        "evidence_quality": "missing",
        "expected_utility_after_cost": 0.0,
        "risk_adjusted_score": 0.0,
        "stress_cost_utility": 0.0,
        "rank_score": 0.0,
        "uncertainty": 1.0,
        "no_trade_delta": 0.0,
        "hold_delta": 0.0,
        "incumbent_delta": None,
        "blockers": ["no_selector_eligible_strategy_for_state"],
        "reason_codes": [_decision_reason(decision)],
    }


def _missing_state_decision(scope: Mapping[str, Any]) -> str:
    label = str(scope.get("state_label") or "")
    ood_score = _number(scope.get("out_of_distribution_score"), 0.0)
    if label == "out_of_distribution" or ood_score >= 0.8:
        return "OUT_OF_DISTRIBUTION_NO_TRADE"
    if label in {"unknown", "mixed", "transition", ""}:
        return "UNKNOWN_NO_TRADE"
    return "NO_SUPPORTED_STRATEGY"


def _strategy_identity_unit(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": identity.get("candidate_id"),
        "strategy_id": identity.get("strategy_id"),
        "strategy_class_name": identity.get("strategy_class_name"),
        "strategy_source_path": identity.get("strategy_source_path"),
        "strategy_version": identity.get("strategy_version"),
        "signal_version": identity.get("signal_version"),
        "risk_policy_version": identity.get("risk_policy_version"),
        "regime_classifier_version": identity.get("regime_classifier_version"),
        "cost_model_id": identity.get("cost_model_id"),
        "allowed_pairs": list(identity.get("allowed_pairs") or []),
        "allowed_timeframes": list(identity.get("allowed_timeframes") or []),
        "created_at": identity.get("created_at"),
        "source_artifacts": dict(identity.get("source_artifacts") or {}),
    }


def _identity_mismatch_fields(
    row: Mapping[str, Any], identity: Mapping[str, Any]
) -> list[str]:
    fields = (
        "candidate_id",
        "strategy_id",
        "strategy_version",
        "signal_version",
        "risk_policy_version",
        "cost_model_id",
    )
    return [
        field
        for field in fields
        if row.get(field) not in (None, "")
        and identity.get(field) not in (None, "")
        and str(row.get(field)) != str(identity.get(field))
    ]


def _row_key_map(rows: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = "|".join(
            str(row.get(field) or "")
            for field in ("row_type", "candidate_id", "state_id", "horizon_profile_id")
        )
        result[key] = row
    return result


def _decision_reason(decision: str) -> str:
    return {
        SELECTOR_ELIGIBLE_DECISION: "selector_eligible_checked_state_evidence",
        "NO_TRADE_POLICY": "first_class_no_trade_policy",
        "NO_SUPPORTED_STRATEGY": "no_supported_strategy_for_state",
        "UNKNOWN_NO_TRADE": "unknown_or_weak_state_no_trade",
        "OUT_OF_DISTRIBUTION_NO_TRADE": "out_of_distribution_no_trade",
        "UNSAFE_NO_TRADE": "unsafe_state_no_trade",
        "SHADOW_ONLY": "shadow_only_not_selector_input",
        "DIAGNOSTIC_ONLY": "diagnostic_only_not_selector_input",
        "IDENTITY_MISMATCH": "strategy_identity_mismatch",
    }.get(decision, "strategy_suitability_decision_recorded")


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


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
    return clean.strip("._") or "strategy_suitability"


def _compact_timestamp(value: str) -> str:
    return _safe_component(value.replace("+00:00", "Z").replace(":", "").replace("-", ""))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
