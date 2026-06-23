#!/usr/bin/env python3
"""Turn trade behavior diagnostics into concrete follow-up experiment plans."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
BEHAVIOR_PATH = AGENT_ROOT / "trade_behavior/latest_trade_behavior.json"
REPORT_DIR = AGENT_ROOT / "behavior_experiments"
LATEST_REPORT_JSON = REPORT_DIR / "latest_behavior_experiment_plan.json"
LATEST_REPORT_MD = REPORT_DIR / "latest_behavior_experiment_plan.md"


@dataclass
class ExperimentPlan:
    priority: int
    strategy: str
    experiment_id: str
    hypothesis: str
    change_set: list[str]
    expected_effect: str
    success_gate: str
    risk_note: str


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


def pair_drag(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in summary.get("pair_breakdown", []) if row.get("profit_abs", 0) < 0]


def build_plans_for_strategy(summary: dict[str, Any]) -> list[ExperimentPlan]:
    strategy = summary["strategy"]
    plans: list[ExperimentPlan] = []
    diagnostics = " ".join(summary.get("diagnostics", []))

    if summary.get("stop_loss_trades", 0) > 0:
        plans.append(
            ExperimentPlan(
                priority=80,
                strategy=strategy,
                experiment_id="stop_loss_and_invalidation_sweep",
                hypothesis="Stop-loss and invalidation exits are cutting losers after weak entries; test tighter confirmation plus separate stop distance variants.",
                change_set=[
                    "Add stronger short-entry confirmation after pullback rejection.",
                    "Test stoploss at current, 0.8x, and 1.2x distance.",
                    "Test faster invalidation exit versus wider stop with time stop.",
                ],
                expected_effect="Reduce average loss and max losing streak without destroying ROI winners.",
                success_gate="profit_factor improves, stop_loss_profit_abs becomes less negative, and trades stay above 100 where possible.",
                risk_note="Wider stops can hide poor entries; require MAE and drawdown checks.",
            )
        )

    if summary.get("max_consecutive_losses", 0) >= 5:
        plans.append(
            ExperimentPlan(
                priority=75,
                strategy=strategy,
                experiment_id="cooldown_after_loss_cluster",
                hypothesis="Losses cluster in bad micro-regimes; cooldown after consecutive losses may improve expectancy.",
                change_set=[
                    "Add pair-level cooldown after two consecutive losses.",
                    "Block new entries for 3h, 6h, and 12h cooldown variants.",
                    "Compare cooldown variants against unchanged baseline.",
                ],
                expected_effect="Lower max consecutive losses and drawdown while preserving most ROI exits.",
                success_gate="max_consecutive_losses drops below 5 with positive adjusted return.",
                risk_note="Cooldown can skip profitable rebounds; compare missed-trade opportunity cost.",
            )
        )

    if summary.get("short_trades", 0) > summary.get("long_trades", 0) * 3:
        plans.append(
            ExperimentPlan(
                priority=70,
                strategy=strategy,
                experiment_id="short_only_regime_split",
                hypothesis="The strategy is effectively short-only; evaluate it as a dedicated short regime model instead of a general strategy.",
                change_set=[
                    "Keep short-only lane but require stronger bearish regime confirmation.",
                    "Add separate bull-window hard block.",
                    "Benchmark against ETH-only short lane and BTC-only short lane.",
                ],
                expected_effect="Reduce short trades during hostile bull/range windows.",
                success_gate="walk-forward short-only windows improve without negative stress-cost cluster.",
                risk_note="Short-only systems can look good in bear slices and fail badly in bull slices.",
            )
        )

    for row in pair_drag(summary):
        pair = row.get("pair", "")
        plans.append(
            ExperimentPlan(
                priority=65,
                strategy=strategy,
                experiment_id=f"disable_dragging_pair_{pair.replace('/', '_').replace(':', '_')}",
                hypothesis=f"{pair} is dragging total performance; test pair-specific disable or stricter filters.",
                change_set=[
                    f"Run {pair}-disabled variant.",
                    f"Run {pair}-only diagnostic variant to isolate signal quality.",
                    f"Require stronger entry confirmation for {pair}.",
                ],
                expected_effect="Improve total expectancy by removing or fixing weak pair lane.",
                success_gate="aggregate adjusted return improves and remaining pair count still supports enough trades.",
                risk_note="Pair removal can overfit to the current sample; require walk-forward confirmation.",
            )
        )

    if summary.get("avg_mae_pct") is not None and summary.get("avg_mfe_pct") is not None:
        if summary["avg_mae_pct"] > summary["avg_mfe_pct"] * 0.5 or "entry" in diagnostics.lower():
            plans.append(
                ExperimentPlan(
                    priority=60,
                    strategy=strategy,
                    experiment_id="entry_timing_confirmation_sweep",
                    hypothesis="Adverse excursion is high enough to warrant delayed or multi-timeframe entry confirmation.",
                    change_set=[
                        "Require 5m and 15m continuation after the 1m setup.",
                        "Test delayed entry by 3, 5, and 10 candles after signal.",
                        "Require price to move in favor before entry instead of entering on first signal.",
                    ],
                    expected_effect="Lower MAE and stop-loss rate at the cost of fewer trades.",
                    success_gate="avg_mae_pct declines while profit factor and adjusted return improve.",
                    risk_note="Delayed confirmation can miss fast reversals; inspect MFE loss.",
                )
            )

    if not plans:
        plans.append(
            ExperimentPlan(
                priority=30,
                strategy=strategy,
                experiment_id="baseline_retest",
                hypothesis="No dominant behavior failure was detected; retest baseline across fresh matrix slices before changing logic.",
                change_set=["Refresh matrix and walk-forward results.", "Compare against unchanged baseline."],
                expected_effect="Confirm whether the baseline remains stable.",
                success_gate="baseline keeps positive adjusted return across recent windows.",
                risk_note="Do not optimize without a clear failure mode.",
            )
        )
    return plans


def build_payload() -> dict[str, Any]:
    behavior = load_json(BEHAVIOR_PATH)
    plans: list[ExperimentPlan] = []
    for summary in behavior.get("summaries", []):
        plans.extend(build_plans_for_strategy(summary))
    sorted_plans = sorted(plans, key=lambda item: (-item.priority, item.strategy, item.experiment_id))
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_behavior_report": rel(BEHAVIOR_PATH),
        "strategy_count": behavior.get("strategy_count", 0),
        "experiment_count": len(sorted_plans),
        "plans": [asdict(item) for item in sorted_plans],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Behavior-Driven Experiment Plan",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Source behavior report: `{payload['source_behavior_report']}`",
        f"- Experiments: `{payload['experiment_count']}`",
        "",
        "| Priority | Strategy | Experiment | Hypothesis | Success Gate |",
        "|---:|---|---|---|---|",
    ]
    for item in payload["plans"]:
        lines.append(
            "| {priority} | {strategy} | {experiment_id} | {hypothesis} | {success_gate} |".format(
                **item
            )
        )
    lines.extend(["", "## Change Sets", ""])
    for item in payload["plans"]:
        lines.append(f"### {item['strategy']} / {item['experiment_id']}")
        for change in item["change_set"]:
            lines.append(f"- {change}")
        lines.append(f"- Expected effect: {item['expected_effect']}")
        lines.append(f"- Risk: {item['risk_note']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"behavior_experiment_plan_{timestamp}.json"
    md_path = REPORT_DIR / f"behavior_experiment_plan_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_REPORT_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_REPORT_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"Experiments planned: {payload['experiment_count']}")


def main() -> None:
    payload = build_payload()
    write_outputs(payload)


if __name__ == "__main__":
    main()
