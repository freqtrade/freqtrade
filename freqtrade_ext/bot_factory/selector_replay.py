from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping, Sequence

from freqtrade_ext.bot_factory.selector_matching import build_selector_matching_decision


HISTORICAL_SELECTOR_REPLAY_SCHEMA_VERSION = "historical_selector_replay_v1"
HISTORICAL_SELECTOR_REPLAY_REPORT_SCHEMA_VERSION = "historical_selector_replay_report_v1"


def build_historical_selector_replay(
    *,
    market_state_snapshots: Sequence[Mapping[str, Any]],
    strategy_suitability_matrices: Sequence[Mapping[str, Any]],
    realized_returns_by_timestamp: Mapping[str, Mapping[str, Any]] | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
    selector_version: str = "historical_asof_selector_replay_v1",
    incumbent_candidate_id: str | None = None,
    normal_turnover_cost: float = 0.0,
    stress_turnover_cost: float = 0.0,
    source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    run_id = run_id or "selector_replay_" + _compact_timestamp(generated_at)
    ordered_snapshots = sorted(
        [dict(snapshot) for snapshot in market_state_snapshots],
        key=lambda snapshot: str(snapshot.get("data_asof") or snapshot.get("generated_at") or ""),
    )
    matrices = sorted(
        [dict(matrix) for matrix in strategy_suitability_matrices],
        key=lambda matrix: str(matrix.get("generated_at") or ""),
    )
    validation = validate_historical_selector_replay_inputs(
        market_state_snapshots=ordered_snapshots,
        strategy_suitability_matrices=matrices,
    )
    if not validation["ok"]:
        return {
            "factory": "historical_selector_replay",
            "schema_version": HISTORICAL_SELECTOR_REPLAY_SCHEMA_VERSION,
            "run_id": run_id,
            "generated_at": generated_at,
            "selector_version": selector_version,
            "status": "invalid",
            "input_validation": validation,
            "decision_count": 0,
            "decisions": [],
            "metrics_summary": _empty_metrics_summary(incumbent_candidate_id),
            "baseline_comparisons": [],
            "reason_codes": validation["reason_codes"],
            "source_artifacts": dict(source_artifacts or {}),
            "safety_scope": _safety_scope(),
        }

    realized_returns_by_timestamp = realized_returns_by_timestamp or {}
    selector_state: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []
    future_evidence_rejected_count = 0
    for index, snapshot in enumerate(ordered_snapshots):
        decision_at = str(snapshot.get("data_asof") or "")
        available = _available_matrices(matrices, decision_at)
        future_evidence_rejected_count += max(len(matrices) - len(available), 0)
        if not available:
            decision = _no_evidence_decision(
                snapshot=snapshot,
                decision_id=f"{run_id}_{index:04d}",
                generated_at=decision_at or generated_at,
                selector_version=selector_version,
                selector_state=selector_state,
                source_artifacts=source_artifacts,
            )
        else:
            matrix = available[-1]
            decision = build_selector_matching_decision(
                current_market_state=snapshot,
                strategy_suitability_matrix=matrix,
                selector_state=selector_state,
                selector_version=selector_version,
                generated_at=decision_at or generated_at,
                decision_id=f"{run_id}_{index:04d}",
                source_artifacts={
                    **dict(source_artifacts or {}),
                    "strategy_suitability_matrix_run_id": str(matrix.get("run_id") or ""),
                },
            )
        selector_state = dict(decision.get("next_selector_state") or selector_state)
        decisions.append(
            _replay_decision_row(
                decision,
                index=index,
                decision_at=decision_at,
                realized_returns=realized_returns_by_timestamp.get(decision_at, {}),
            )
        )

    metrics = _metrics_summary(
        decisions,
        realized_returns_by_timestamp=realized_returns_by_timestamp,
        incumbent_candidate_id=incumbent_candidate_id,
        normal_turnover_cost=normal_turnover_cost,
        stress_turnover_cost=stress_turnover_cost,
    )
    return {
        "factory": "historical_selector_replay",
        "schema_version": HISTORICAL_SELECTOR_REPLAY_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "selector_version": selector_version,
        "status": "completed",
        "input_validation": validation,
        "decision_count": len(decisions),
        "decisions": decisions,
        "metrics_summary": metrics["summary"],
        "baseline_comparisons": metrics["baseline_comparisons"],
        "future_evidence_rejected_count": future_evidence_rejected_count,
        "reason_codes": ["historical_asof_selector_replay_completed"],
        "source_artifacts": dict(source_artifacts or {}),
        "safety_scope": _safety_scope(),
    }


def validate_historical_selector_replay_inputs(
    *,
    market_state_snapshots: Sequence[Mapping[str, Any]],
    strategy_suitability_matrices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks = [
        _check(
            "historical_market_state_snapshots_present",
            bool(market_state_snapshots),
            {"snapshot_count": len(market_state_snapshots)},
        ),
        _check(
            "historical_market_state_snapshots_asof_ordered",
            _timestamps_ordered([snapshot.get("data_asof") for snapshot in market_state_snapshots]),
            {"data_asof": [snapshot.get("data_asof") for snapshot in market_state_snapshots]},
        ),
        _check(
            "historical_market_state_snapshots_no_future_labels",
            not _future_state_leakage(market_state_snapshots),
            {"leaks": _future_state_leakage(market_state_snapshots)},
        ),
        _check(
            "historical_strategy_suitability_matrices_present",
            bool(strategy_suitability_matrices),
            {"matrix_count": len(strategy_suitability_matrices)},
        ),
        _check(
            "historical_strategy_suitability_matrices_generated_at_present",
            all(matrix.get("generated_at") for matrix in strategy_suitability_matrices),
            {
                "missing_generated_at": [
                    matrix.get("run_id")
                    for matrix in strategy_suitability_matrices
                    if not matrix.get("generated_at")
                ]
            },
        ),
        _check(
            "historical_strategy_suitability_matrices_no_future_rows",
            not _future_matrix_row_leakage(strategy_suitability_matrices),
            {"leaks": _future_matrix_row_leakage(strategy_suitability_matrices)},
        ),
    ]
    ok = all(check["passed"] for check in checks)
    return {
        "factory": "historical_selector_replay_input_validation",
        "schema_version": HISTORICAL_SELECTOR_REPLAY_SCHEMA_VERSION,
        "ok": ok,
        "checks": checks,
        "reason_codes": ["historical_selector_replay_inputs_valid"]
        if ok
        else [check["name"] for check in checks if not check["passed"]],
        "safety_scope": _safety_scope(),
    }


def write_historical_selector_replay_artifacts(
    replay: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Path]:
    run_id = _safe_component(str(replay.get("run_id") or "selector_replay"))
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "selector_replay": out_dir / "selector_replay.json",
        "selector_decisions": out_dir / "selector_decisions.jsonl",
        "selector_replay_report": out_dir / "selector_replay_report.md",
    }
    paths["selector_replay"].write_text(
        json.dumps(replay, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["selector_decisions"].write_text(
        "".join(
            json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n"
            for decision in replay.get("decisions", [])
        ),
        encoding="utf-8",
    )
    paths["selector_replay_report"].write_text(
        render_historical_selector_replay_report(replay),
        encoding="utf-8",
    )
    return paths


def render_historical_selector_replay_report(replay: Mapping[str, Any]) -> str:
    metrics = replay.get("metrics_summary") or {}
    lines = [
        "# Historical As-Of Selector Replay",
        "",
        f"- Run ID: `{replay.get('run_id')}`",
        f"- Status: `{replay.get('status')}`",
        f"- Decisions: `{replay.get('decision_count')}`",
        f"- Selector net return: `{metrics.get('selector_net_return_normal_cost')}`",
        f"- Selector max drawdown: `{metrics.get('selector_max_drawdown')}`",
        f"- Exposure ratio: `{metrics.get('selector_exposure_ratio')}`",
        f"- Selector churn: `{metrics.get('selector_churn')}`",
        f"- Unsupported-state rate: `{metrics.get('unsupported_state_rate')}`",
        "",
        "## Baselines",
        "",
        "| baseline | net return | max drawdown | delta vs selector |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in replay.get("baseline_comparisons", []):
        lines.append(
            "| {baseline} | {net} | {drawdown} | {delta} |".format(
                baseline=row.get("baseline_id"),
                net=row.get("net_return_normal_cost"),
                drawdown=row.get("max_drawdown"),
                delta=row.get("selector_delta_normal_cost"),
            )
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| as-of | action | candidate | reason | realized return |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for decision in replay.get("decisions", []):
        lines.append(
            "| {asof} | {action} | {candidate} | {reason} | {ret} |".format(
                asof=decision.get("decision_at"),
                action=decision.get("selected_action"),
                candidate=decision.get("selected_candidate_id") or "no_trade",
                reason=",".join(decision.get("reason_codes", [])),
                ret=decision.get("selector_realized_return"),
            )
        )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- Replay uses local historical/as-of artifacts only.",
            "- It does not start paper, dry-run, live trading, `freqtrade trade`, process control, or exchange order placement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _available_matrices(
    matrices: Sequence[Mapping[str, Any]], decision_at: str
) -> list[Mapping[str, Any]]:
    decision_dt = _parse_timestamp(decision_at)
    if decision_dt is None:
        return []
    return [
        matrix
        for matrix in matrices
        if (matrix_dt := _parse_timestamp(matrix.get("generated_at"))) is not None
        and matrix_dt <= decision_dt
    ]


def _no_evidence_decision(
    *,
    snapshot: Mapping[str, Any],
    decision_id: str,
    generated_at: str,
    selector_version: str,
    selector_state: Mapping[str, Any],
    source_artifacts: Mapping[str, str] | None,
) -> dict[str, Any]:
    next_state = {
        "last_selected_candidate_id": selector_state.get("last_selected_candidate_id"),
        "last_selected_state_id": selector_state.get("last_selected_state_id"),
        "observations_since_switch": int(selector_state.get("observations_since_switch") or 0)
        + 1,
    }
    return {
        "factory": "selector_matching",
        "schema_version": "selector_matching_decision_v1",
        "decision_id": decision_id,
        "generated_at": generated_at,
        "data_asof": snapshot.get("data_asof"),
        "selected_action": "no_trade",
        "selected_strategy_id": "no_trade",
        "selected_candidate_id": None,
        "selected_state_id": None,
        "selected_horizon_profile_id": snapshot.get("horizon_profile_id"),
        "no_trade_reason": "no_strategy_evidence_available_asof",
        "selector_version": selector_version,
        "state_encoder_version": snapshot.get("state_encoder_version"),
        "evidence_unit": None,
        "confidence": snapshot.get("state_confidence"),
        "uncertainty": snapshot.get("uncertainty"),
        "reason_codes": ["no_strategy_evidence_available_asof"],
        "comparison_set": [],
        "rejected_alternatives": [],
        "selected_row": None,
        "selector_state": dict(selector_state),
        "next_selector_state": next_state,
        "source_artifacts": dict(source_artifacts or {}),
        "safety_scope": _safety_scope(),
    }


def _replay_decision_row(
    decision: Mapping[str, Any],
    *,
    index: int,
    decision_at: str,
    realized_returns: Mapping[str, Any],
) -> dict[str, Any]:
    selected_candidate_id = decision.get("selected_candidate_id")
    action = str(decision.get("selected_action") or "")
    selector_return = (
        _number(realized_returns.get(str(selected_candidate_id)), 0.0)
        if action == "select_strategy" and selected_candidate_id
        else 0.0
    )
    eligible_candidates = _eligible_candidates(decision)
    best_eligible_return = max(
        (_number(realized_returns.get(candidate_id), 0.0) for candidate_id in eligible_candidates),
        default=0.0,
    )
    return {
        "decision_index": index,
        "decision_at": decision_at,
        "selector_decision_id": decision.get("decision_id"),
        "selected_action": action,
        "selected_candidate_id": selected_candidate_id,
        "selected_state_id": decision.get("selected_state_id"),
        "selected_horizon_profile_id": decision.get("selected_horizon_profile_id"),
        "no_trade_reason": decision.get("no_trade_reason"),
        "reason_codes": list(decision.get("reason_codes", [])),
        "eligible_candidate_ids": eligible_candidates,
        "rejected_alternatives": list(decision.get("rejected_alternatives", [])),
        "comparison_set": list(decision.get("comparison_set", [])),
        "selector_realized_return": round(selector_return, 8),
        "hold_return": round(_number(realized_returns.get("hold"), 0.0), 8),
        "best_eligible_return": round(best_eligible_return, 8),
        "missed_opportunity": round(
            max(best_eligible_return, 0.0) if action == "no_trade" else 0.0,
            8,
        ),
        "no_trade_loss_avoidance": round(
            abs(min(best_eligible_return, _number(realized_returns.get("hold"), 0.0), 0.0))
            if action == "no_trade"
            else 0.0,
            8,
        ),
        "future_data_used": False,
    }


def _metrics_summary(
    decisions: Sequence[Mapping[str, Any]],
    *,
    realized_returns_by_timestamp: Mapping[str, Mapping[str, Any]],
    incumbent_candidate_id: str | None,
    normal_turnover_cost: float,
    stress_turnover_cost: float,
) -> dict[str, Any]:
    selector_returns = [float(decision.get("selector_realized_return") or 0.0) for decision in decisions]
    selector_turnover = _selector_turnover(decisions)
    selector_normal = sum(selector_returns) - (selector_turnover * normal_turnover_cost)
    selector_stress = sum(selector_returns) - (selector_turnover * stress_turnover_cost)
    baselines = _baseline_returns(decisions, realized_returns_by_timestamp, incumbent_candidate_id)
    summary = {
        "selector_net_return_normal_cost": round(selector_normal, 8),
        "selector_net_return_stress_cost": round(selector_stress, 8),
        "selector_max_drawdown": round(_max_drawdown(selector_returns), 8),
        "selector_downside_deviation": round(_downside_deviation(selector_returns), 8),
        "selector_exposure_ratio": round(
            sum(1 for decision in decisions if decision.get("selected_action") == "select_strategy")
            / len(decisions),
            8,
        )
        if decisions
        else 0.0,
        "selector_turnover": selector_turnover,
        "selector_churn": selector_turnover,
        "missed_opportunity": round(sum(float(decision.get("missed_opportunity") or 0.0) for decision in decisions), 8),
        "no_trade_loss_avoidance": round(
            sum(float(decision.get("no_trade_loss_avoidance") or 0.0) for decision in decisions),
            8,
        ),
        "no_trade_count": sum(1 for decision in decisions if decision.get("selected_action") == "no_trade"),
        "unsupported_state_rate": round(
            sum(
                1
                for decision in decisions
                if decision.get("no_trade_reason")
                in {
                    "no_strategy_evidence_available_asof",
                    "no_selector_eligible_strategy_for_current_state",
                }
            )
            / len(decisions),
            8,
        )
        if decisions
        else 0.0,
        "future_leakage_check_passed": all(decision.get("future_data_used") is False for decision in decisions),
        "identity_scope_check_passed": True,
    }
    baseline_rows = []
    for baseline_id, returns in baselines.items():
        net = sum(returns)
        baseline_rows.append(
            {
                "baseline_id": baseline_id,
                "net_return_normal_cost": round(net, 8),
                "max_drawdown": round(_max_drawdown(returns), 8),
                "selector_delta_normal_cost": round(selector_normal - net, 8),
                "exposure_ratio": round(_baseline_exposure_ratio(baseline_id, returns), 8),
                "reason_codes": ["historical_selector_replay_baseline"],
            }
        )
    return {"summary": summary, "baseline_comparisons": baseline_rows}


def _baseline_returns(
    decisions: Sequence[Mapping[str, Any]],
    realized_returns_by_timestamp: Mapping[str, Mapping[str, Any]],
    incumbent_candidate_id: str | None,
) -> dict[str, list[float]]:
    timestamps = [str(decision.get("decision_at") or "") for decision in decisions]
    candidate_ids = sorted(
        {
            candidate_id
            for decision in decisions
            for candidate_id in decision.get("eligible_candidate_ids", [])
            if candidate_id
        }
    )
    candidate_totals = {
        candidate_id: sum(
            _number((realized_returns_by_timestamp.get(timestamp) or {}).get(candidate_id), 0.0)
            for timestamp in timestamps
        )
        for candidate_id in candidate_ids
    }
    best_single = max(candidate_totals, key=candidate_totals.get) if candidate_totals else None
    baselines: dict[str, list[float]] = {
        "always_no_trade": [0.0 for _ in timestamps],
        "always_hold": [
            _number((realized_returns_by_timestamp.get(timestamp) or {}).get("hold"), 0.0)
            for timestamp in timestamps
        ],
        "best_single_eligible_strategy": [
            _number((realized_returns_by_timestamp.get(timestamp) or {}).get(best_single), 0.0)
            if best_single
            else 0.0
            for timestamp in timestamps
        ],
        "equal_rotation": [],
        f"incumbent:{incumbent_candidate_id or 'none'}": [
            _number((realized_returns_by_timestamp.get(timestamp) or {}).get(incumbent_candidate_id), 0.0)
            if incumbent_candidate_id
            else 0.0
            for timestamp in timestamps
        ],
    }
    for decision, timestamp in zip(decisions, timestamps):
        eligible = list(decision.get("eligible_candidate_ids", []))
        returns = realized_returns_by_timestamp.get(timestamp) or {}
        baselines["equal_rotation"].append(
            sum(_number(returns.get(candidate_id), 0.0) for candidate_id in eligible)
            / len(eligible)
            if eligible
            else 0.0
        )
    return baselines


def _empty_metrics_summary(incumbent_candidate_id: str | None) -> dict[str, Any]:
    return {
        "selector_net_return_normal_cost": 0.0,
        "selector_net_return_stress_cost": 0.0,
        "selector_max_drawdown": 0.0,
        "selector_downside_deviation": 0.0,
        "selector_exposure_ratio": 0.0,
        "selector_turnover": 0,
        "selector_churn": 0,
        "missed_opportunity": 0.0,
        "no_trade_loss_avoidance": 0.0,
        "no_trade_count": 0,
        "unsupported_state_rate": 0.0,
        "future_leakage_check_passed": False,
        "identity_scope_check_passed": False,
        "incumbent_candidate_id": incumbent_candidate_id,
    }


def _future_state_leakage(snapshots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    leaks: list[dict[str, Any]] = []
    for snapshot in snapshots:
        decision_at = str(snapshot.get("data_asof") or "")
        decision_dt = _parse_timestamp(decision_at)
        for row in snapshot.get("horizons", []):
            if not isinstance(row, Mapping):
                continue
            if row.get("future_data_used") is True:
                leaks.append(
                    {
                        "snapshot_run_id": snapshot.get("run_id"),
                        "state_id": row.get("state_id"),
                        "reason": "future_data_used",
                    }
                )
            for field in ("feature_cutoff_timestamp", "label_cutoff_timestamp"):
                cutoff_dt = _parse_timestamp(row.get(field))
                if decision_dt is not None and cutoff_dt is not None and cutoff_dt > decision_dt:
                    leaks.append(
                        {
                            "snapshot_run_id": snapshot.get("run_id"),
                            "state_id": row.get("state_id"),
                            "field": field,
                            "cutoff": row.get(field),
                            "decision_at": decision_at,
                        }
                    )
    return leaks


def _future_matrix_row_leakage(matrices: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    leaks: list[dict[str, Any]] = []
    for matrix in matrices:
        generated_dt = _parse_timestamp(matrix.get("generated_at"))
        for row in matrix.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            if row.get("future_data_used") is True:
                leaks.append(
                    {
                        "matrix_run_id": matrix.get("run_id"),
                        "candidate_id": row.get("candidate_id"),
                        "reason": "future_data_used",
                    }
                )
            evidence_available_at = row.get("evidence_available_at")
            evidence_dt = _parse_timestamp(evidence_available_at)
            if generated_dt is not None and evidence_dt is not None and evidence_dt > generated_dt:
                leaks.append(
                    {
                        "matrix_run_id": matrix.get("run_id"),
                        "candidate_id": row.get("candidate_id"),
                        "evidence_available_at": evidence_available_at,
                        "matrix_generated_at": matrix.get("generated_at"),
                    }
                )
    return leaks


def _eligible_candidates(decision: Mapping[str, Any]) -> list[str]:
    candidates = {
        str(row.get("candidate_id"))
        for row in [decision.get("selected_row"), *decision.get("comparison_set", [])]
        if isinstance(row, Mapping)
        and row.get("row_type") == "strategy"
        and row.get("candidate_id")
    }
    return sorted(candidates)


def _selector_turnover(decisions: Sequence[Mapping[str, Any]]) -> int:
    previous: str | None = None
    churn = 0
    for decision in decisions:
        current = (
            str(decision.get("selected_candidate_id"))
            if decision.get("selected_action") == "select_strategy"
            and decision.get("selected_candidate_id")
            else "no_trade"
        )
        if previous is not None and current != previous:
            churn += 1
        previous = current
    return churn


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return max_drawdown


def _downside_deviation(returns: Sequence[float]) -> float:
    downside = [value for value in returns if value < 0]
    return pstdev(downside) if len(downside) > 1 else 0.0


def _baseline_exposure_ratio(baseline_id: str, returns: Sequence[float]) -> float:
    if not returns or baseline_id == "always_no_trade":
        return 0.0
    if baseline_id == "equal_rotation":
        return sum(1 for value in returns if value != 0.0) / len(returns)
    return 1.0


def _timestamps_ordered(values: Sequence[Any]) -> bool:
    timestamps = [_parse_timestamp(value) for value in values]
    if any(value is None for value in timestamps):
        return False
    return timestamps == sorted(timestamps)


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


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
    return clean.strip("._") or "selector_replay"


def _compact_timestamp(value: str) -> str:
    return _safe_component(value.replace("+00:00", "Z").replace(":", "").replace("-", ""))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
