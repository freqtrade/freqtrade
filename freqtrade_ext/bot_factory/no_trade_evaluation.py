from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


NO_TRADE_POLICY_EVALUATION_SCHEMA_VERSION = "no_trade_policy_evaluation_v1"
NO_TRADE_POLICY_EVALUATION_REPORT_SCHEMA_VERSION = "no_trade_policy_evaluation_report_v1"
DEFAULT_OPPORTUNITY_COST_THRESHOLDS = {
    "uncertain_or_ood": 0.03,
    "unsupported": 0.02,
    "cooldown": 0.015,
    "supported": 0.01,
}


def build_no_trade_policy_evaluation(
    *,
    selector_replay: Mapping[str, Any],
    opportunity_cost_thresholds: Mapping[str, float] | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
    source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    run_id = run_id or "no_trade_policy_" + _compact_timestamp(generated_at)
    thresholds = {
        **DEFAULT_OPPORTUNITY_COST_THRESHOLDS,
        **{str(key): float(value) for key, value in (opportunity_cost_thresholds or {}).items()},
    }
    no_trade_rows = [
        _no_trade_row(decision, thresholds=thresholds)
        for decision in selector_replay.get("decisions", [])
        if isinstance(decision, Mapping) and decision.get("selected_action") == "no_trade"
    ]
    state_rows = _state_quality_rows(no_trade_rows, thresholds=thresholds)
    summary = _summary(no_trade_rows, state_rows)
    return {
        "factory": "no_trade_policy_evaluation",
        "schema_version": NO_TRADE_POLICY_EVALUATION_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "source_selector_replay_run_id": selector_replay.get("run_id"),
        "opportunity_cost_thresholds": thresholds,
        "no_trade_decisions": no_trade_rows,
        "state_no_trade_quality": state_rows,
        "summary": summary,
        "summary_decision": summary["summary_decision"],
        "reason_codes": summary["reason_codes"],
        "source_artifacts": dict(source_artifacts or {}),
        "safety_scope": _safety_scope(),
    }


def write_no_trade_policy_evaluation_artifacts(
    evaluation: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Path]:
    run_id = _safe_component(str(evaluation.get("run_id") or "no_trade_policy"))
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "no_trade_policy_evaluation": out_dir / "no_trade_policy_evaluation.json",
        "no_trade_policy_evaluation_report": out_dir / "no_trade_policy_evaluation_report.md",
    }
    paths["no_trade_policy_evaluation"].write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["no_trade_policy_evaluation_report"].write_text(
        render_no_trade_policy_evaluation_report(evaluation),
        encoding="utf-8",
    )
    return paths


def render_no_trade_policy_evaluation_report(evaluation: Mapping[str, Any]) -> str:
    summary = evaluation.get("summary") or {}
    lines = [
        "# No-Trade Policy Evaluation",
        "",
        f"- Run ID: `{evaluation.get('run_id')}`",
        f"- Decision: `{evaluation.get('summary_decision')}`",
        f"- No-trade decisions: `{summary.get('no_trade_count')}`",
        f"- Avoided drawdown: `{summary.get('avoided_drawdown')}`",
        f"- Opportunity cost vs hold: `{summary.get('opportunity_cost_vs_hold')}`",
        f"- Opportunity cost vs best eligible: `{summary.get('opportunity_cost_vs_best')}`",
        f"- Uncertainty/OOD safety value: `{summary.get('uncertainty_ood_safety_value')}`",
        f"- Reason codes: `{', '.join(evaluation.get('reason_codes', []))}`",
        "",
        "## State Quality",
        "",
        "| state type | state | assessment | count | avoided drawdown | opportunity vs hold | opportunity vs best |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in evaluation.get("state_no_trade_quality", []):
        lines.append(
            "| {state_type} | {state} | {assessment} | {count} | {avoided} | {hold} | {best} |".format(
                state_type=row.get("state_type"),
                state=row.get("state_id"),
                assessment=row.get("assessment"),
                count=row.get("no_trade_count"),
                avoided=row.get("avoided_drawdown"),
                hold=row.get("opportunity_cost_vs_hold"),
                best=row.get("opportunity_cost_vs_best"),
            )
        )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- This evaluates historical no-trade decisions only.",
            "- It does not start paper, dry-run, live trading, `freqtrade trade`, process control, or exchange order placement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _no_trade_row(
    decision: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    state_type = _state_type(decision)
    opportunity_hold = max(_number(decision.get("hold_return"), 0.0), 0.0)
    opportunity_best = max(_number(decision.get("missed_opportunity"), 0.0), 0.0)
    avoided = max(_number(decision.get("no_trade_loss_avoidance"), 0.0), 0.0)
    threshold = thresholds.get(state_type, thresholds["supported"])
    uncertainty_value = avoided if state_type == "uncertain_or_ood" else 0.0
    return {
        "decision_at": decision.get("decision_at"),
        "state_id": decision.get("selected_state_id") or "no_state",
        "state_type": state_type,
        "no_trade_reason": decision.get("no_trade_reason"),
        "avoided_drawdown": round(avoided, 8),
        "opportunity_cost_vs_hold": round(opportunity_hold, 8),
        "opportunity_cost_vs_best": round(opportunity_best, 8),
        "uncertainty_ood_safety_value": round(uncertainty_value, 8),
        "opportunity_cost_threshold": threshold,
        "assessment": _assessment(
            avoided_drawdown=avoided,
            opportunity_cost=max(opportunity_hold, opportunity_best),
            threshold=threshold,
        ),
        "reason_codes": list(decision.get("reason_codes", [])),
    }


def _state_quality_rows(
    rows: list[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("state_type")), str(row.get("state_id")))
        grouped.setdefault(key, []).append(row)
    output = []
    for state_type, state_id in sorted(grouped):
        items = grouped[(state_type, state_id)]
        avoided = sum(_number(item.get("avoided_drawdown"), 0.0) for item in items)
        opportunity_hold = sum(_number(item.get("opportunity_cost_vs_hold"), 0.0) for item in items)
        opportunity_best = sum(_number(item.get("opportunity_cost_vs_best"), 0.0) for item in items)
        uncertainty_value = sum(
            _number(item.get("uncertainty_ood_safety_value"), 0.0) for item in items
        )
        threshold = thresholds.get(state_type, thresholds["supported"]) * len(items)
        output.append(
            {
                "state_type": state_type,
                "state_id": state_id,
                "no_trade_count": len(items),
                "avoided_drawdown": round(avoided, 8),
                "opportunity_cost_vs_hold": round(opportunity_hold, 8),
                "opportunity_cost_vs_best": round(opportunity_best, 8),
                "uncertainty_ood_safety_value": round(uncertainty_value, 8),
                "opportunity_cost_threshold": round(threshold, 8),
                "assessment": _assessment(
                    avoided_drawdown=avoided,
                    opportunity_cost=max(opportunity_hold, opportunity_best),
                    threshold=threshold,
                ),
            }
        )
    return output


def _summary(
    decision_rows: list[Mapping[str, Any]],
    state_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    assessments = [str(row.get("assessment")) for row in state_rows]
    decision = "NO_TRADE_ACCEPTABLE"
    if "overused" in assessments:
        decision = "NO_TRADE_OVERUSED"
    elif "costly" in assessments:
        decision = "NO_TRADE_COSTLY"
    elif "good" in assessments:
        decision = "NO_TRADE_GOOD"
    reasons = ["no_trade_policy_evaluated"]
    if decision == "NO_TRADE_OVERUSED":
        reasons.append("no_trade_overused_opportunity_cost")
    if decision == "NO_TRADE_COSTLY":
        reasons.append("no_trade_costly_opportunity_cost")
    if decision == "NO_TRADE_GOOD":
        reasons.append("no_trade_avoided_loss_with_acceptable_opportunity_cost")
    return {
        "summary_decision": decision,
        "no_trade_count": len(decision_rows),
        "avoided_drawdown": round(
            sum(_number(row.get("avoided_drawdown"), 0.0) for row in decision_rows),
            8,
        ),
        "opportunity_cost_vs_hold": round(
            sum(_number(row.get("opportunity_cost_vs_hold"), 0.0) for row in decision_rows),
            8,
        ),
        "opportunity_cost_vs_best": round(
            sum(_number(row.get("opportunity_cost_vs_best"), 0.0) for row in decision_rows),
            8,
        ),
        "uncertainty_ood_safety_value": round(
            sum(
                _number(row.get("uncertainty_ood_safety_value"), 0.0)
                for row in decision_rows
            ),
            8,
        ),
        "reason_codes": reasons,
    }


def _state_type(decision: Mapping[str, Any]) -> str:
    reason_text = " ".join(
        [
            str(decision.get("no_trade_reason") or ""),
            " ".join(str(code) for code in decision.get("reason_codes", [])),
        ]
    )
    if any(token in reason_text for token in ("unknown", "mixed", "ood", "out_of_distribution", "stale", "conflict")):
        return "uncertain_or_ood"
    if "cooldown" in reason_text:
        return "cooldown"
    if "no_strategy_evidence" in reason_text or "no_selector_eligible" in reason_text:
        return "unsupported"
    return "supported"


def _assessment(*, avoided_drawdown: float, opportunity_cost: float, threshold: float) -> str:
    if opportunity_cost > threshold * 2 and avoided_drawdown <= 0:
        return "overused"
    if opportunity_cost > threshold and avoided_drawdown <= 0:
        return "costly"
    if avoided_drawdown > 0 and opportunity_cost <= threshold:
        return "good"
    return "acceptable"


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
    return clean.strip("._") or "no_trade_policy"


def _compact_timestamp(value: str) -> str:
    return _safe_component(value.replace("+00:00", "Z").replace(":", "").replace("-", ""))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
