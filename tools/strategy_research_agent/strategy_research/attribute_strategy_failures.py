#!/usr/bin/env python3
"""Build cross-evidence strategy failure attribution reports."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REPORT_DIR = AGENT_ROOT / "failure_attribution"
LATEST_REPORT_JSON = REPORT_DIR / "latest_failure_attribution.json"
LATEST_REPORT_MD = REPORT_DIR / "latest_failure_attribution.md"


@dataclass
class FailureMode:
    mode: str
    severity: int
    evidence: list[str]
    recommendation: str
    linked_experiments: list[str]


@dataclass
class StrategyAttribution:
    strategy: str
    failure_modes: list[dict[str, Any]]
    top_mode: str | None
    summary: str


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def index_by(items: list[dict[str, Any]], key: str = "strategy") -> dict[str, dict[str, Any]]:
    return {item[key]: item for item in items if item.get(key)}


def group_plans(plans: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in plans:
        if item.get("strategy"):
            grouped[item["strategy"]].append(item)
    return grouped


def linked(plans: list[dict[str, Any]], *experiment_ids: str) -> list[str]:
    wanted = set(experiment_ids)
    return [item["experiment_id"] for item in plans if item.get("experiment_id") in wanted]


def add_mode(modes: list[FailureMode], mode: str, severity: int, evidence: list[str], recommendation: str, linked_experiments: list[str]) -> None:
    modes.append(
        FailureMode(
            mode=mode,
            severity=severity,
            evidence=evidence,
            recommendation=recommendation,
            linked_experiments=linked_experiments,
        )
    )


def attribute_one(
    strategy: str,
    scorecard: dict[str, Any] | None,
    promotion: dict[str, Any] | None,
    behavior: dict[str, Any] | None,
    plans: list[dict[str, Any]],
) -> StrategyAttribution:
    modes: list[FailureMode] = []
    failures = set(scorecard.get("primary_failures", []) if scorecard else [])
    blocks = set(promotion.get("blocks", []) if promotion else [])

    if "too_few_matrix_trades" in failures or "too_few_trades" in blocks:
        trades = (behavior or scorecard or {}).get("trades")
        add_mode(
            modes,
            "insufficient_sample",
            80,
            [f"trade_count={trades}", "matrix/promotion marked trade count insufficient"],
            "Increase valid trades through controlled entry redesign, then re-check behavior quality.",
            linked(plans, "entry_timing_confirmation_sweep", "stop_loss_and_invalidation_sweep"),
        )
    if "fragile_matrix" in failures or "matrix_not_robust" in blocks:
        add_mode(
            modes,
            "regime_fragility",
            78,
            [
                f"matrix_verdict={(scorecard or {}).get('matrix_verdict')}",
                f"stress_negative_runs={(scorecard or {}).get('stress_negative_runs')}",
                f"too_few_trade_runs={(scorecard or {}).get('too_few_trade_runs')}",
            ],
            "Split behavior by market regime and reject variants that only work in one slice.",
            linked(plans, "short_only_regime_split", "disable_dragging_pair_BTC_USDT_USDT"),
        )
    if "stress_cost_failure" in failures or "negative_after_cost" in failures:
        add_mode(
            modes,
            "cost_sensitivity",
            70,
            [f"adjusted_return_pct={(scorecard or {}).get('adjusted_return_pct')}"],
            "Raise per-trade edge or reduce churn before testing higher leverage.",
            linked(plans, "entry_timing_confirmation_sweep", "stop_loss_and_invalidation_sweep"),
        )
    if "lookahead_or_recursive_unverified" in failures or "bias_checks_missing" in blocks:
        add_mode(
            modes,
            "bias_unverified",
            68,
            ["recursive/lookahead checks missing from candidate evidence"],
            "Run recursive-analysis and lookahead-analysis before any dry-run promotion.",
            [],
        )
    if behavior:
        if behavior.get("stop_loss_trades", 0) > 0 and behavior.get("stop_loss_profit_abs", 0) < 0:
            add_mode(
                modes,
                "loss_exit_quality",
                76,
                [
                    f"stop_loss_trades={behavior.get('stop_loss_trades')}",
                    f"stop_loss_profit_abs={behavior.get('stop_loss_profit_abs')}",
                    f"max_consecutive_losses={behavior.get('max_consecutive_losses')}",
                ],
                "Test tighter entry confirmation, faster invalidation, and cooldown after loss clusters.",
                linked(plans, "stop_loss_and_invalidation_sweep", "cooldown_after_loss_cluster"),
            )
        if behavior.get("short_trades", 0) > behavior.get("long_trades", 0) * 3:
            add_mode(
                modes,
                "directional_concentration",
                62,
                [f"long_short={behavior.get('long_trades')}/{behavior.get('short_trades')}"],
                "Treat this as a short-only model and explicitly test bull-window failure.",
                linked(plans, "short_only_regime_split"),
            )
        pair_drag = [row for row in behavior.get("pair_breakdown", []) if row.get("profit_abs", 0) < 0]
        if pair_drag:
            add_mode(
                modes,
                "pair_drag",
                58,
                [f"{row.get('pair')} profit_abs={row.get('profit_abs')}" for row in pair_drag],
                "Run pair-disabled and pair-only diagnostics before keeping the weak lane.",
                linked(plans, "disable_dragging_pair_BTC_USDT_USDT"),
            )
    if "underperforms_market" in failures:
        add_mode(
            modes,
            "benchmark_underperformance",
            55,
            [
                f"base_return_pct={(scorecard or {}).get('base_return_pct')}",
                f"market_change_pct={(scorecard or {}).get('market_change_pct')}",
            ],
            "Keep benchmark-relative acceptance gates; do not promote low-edge strategies just because they are positive.",
            [],
        )

    modes = sorted(modes, key=lambda item: (-item.severity, item.mode))
    top = modes[0].mode if modes else None
    if top:
        summary = f"{strategy}: top failure mode is {top}; next best experiments: {', '.join(modes[0].linked_experiments) or 'run missing verification'}."
    else:
        summary = f"{strategy}: no dominant failure mode found from current evidence."
    return StrategyAttribution(strategy=strategy, failure_modes=[asdict(item) for item in modes], top_mode=top, summary=summary)


def build_payload() -> dict[str, Any]:
    assessment = load_json(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json")
    promotion = load_json(AGENT_ROOT / "promotion_reports/latest_promotion_report.json")
    behavior = load_json(AGENT_ROOT / "trade_behavior/latest_trade_behavior.json")
    experiment_plan = load_json(AGENT_ROOT / "behavior_experiments/latest_behavior_experiment_plan.json")

    scorecards = index_by(assessment.get("scorecards", []))
    promotions = index_by(promotion.get("verdicts", []))
    behaviors = index_by(behavior.get("summaries", []))
    plans_by_strategy = group_plans(experiment_plan.get("plans", []))
    strategies = sorted(set(scorecards) | set(promotions) | set(behaviors) | set(plans_by_strategy))
    attributions = [
        asdict(
            attribute_one(
                strategy,
                scorecards.get(strategy),
                promotions.get(strategy),
                behaviors.get(strategy),
                plans_by_strategy.get(strategy, []),
            )
        )
        for strategy in strategies
    ]
    mode_counts: dict[str, int] = defaultdict(int)
    for item in attributions:
        for mode in item["failure_modes"]:
            mode_counts[mode["mode"]] += 1
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "strategy_count": len(attributions),
        "attributions": attributions,
        "failure_mode_summary": [
            {"mode": mode, "count": count}
            for mode, count in sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "source_artifacts": {
            "strategy_assessment": rel(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json"),
            "promotion_report": rel(AGENT_ROOT / "promotion_reports/latest_promotion_report.json"),
            "trade_behavior": rel(AGENT_ROOT / "trade_behavior/latest_trade_behavior.json"),
            "behavior_experiment_plan": rel(AGENT_ROOT / "behavior_experiments/latest_behavior_experiment_plan.json"),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strategy Failure Attribution",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Strategy count: `{payload['strategy_count']}`",
        "",
        "## Failure Mode Summary",
        "",
        "| Mode | Count |",
        "|---|---:|",
    ]
    for item in payload["failure_mode_summary"]:
        lines.append("| {mode} | {count} |".format(**item))
    lines.extend(["", "## Strategy Attribution", ""])
    for item in payload["attributions"]:
        lines.append(f"### {item['strategy']}")
        lines.append(f"- Summary: {item['summary']}")
        lines.append("")
        lines.append("| Severity | Mode | Evidence | Recommendation | Linked Experiments |")
        lines.append("|---:|---|---|---|---|")
        for mode in item["failure_modes"]:
            lines.append(
                "| {severity} | {mode} | {evidence} | {recommendation} | {linked} |".format(
                    severity=mode["severity"],
                    mode=mode["mode"],
                    evidence="; ".join(mode.get("evidence", [])),
                    recommendation=mode["recommendation"],
                    linked=", ".join(mode.get("linked_experiments", [])),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"failure_attribution_{timestamp}.json"
    md_path = REPORT_DIR / f"failure_attribution_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_REPORT_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_REPORT_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"Strategies attributed: {payload['strategy_count']}")


def main() -> None:
    payload = build_payload()
    write_outputs(payload)


if __name__ == "__main__":
    main()
