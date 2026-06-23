#!/usr/bin/env python3
"""Plan memory-guided strategy hypotheses from durable research memory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_JSON = AGENT_ROOT / "experiments/memory_guided_hypothesis_plan.json"
OUTPUT_MD = AGENT_ROOT / "experiments/memory_guided_hypothesis_ledger.md"
OUTPUT_EXPERIMENT = AGENT_ROOT / "experiments/memory_guided_strategy_experiment.json"


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


def slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    return "_".join(part for part in cleaned.split("_") if part)[:64]


def risk_template(blocker: str) -> dict[str, Any]:
    if blocker in {"bias_checks_missing", "lookahead_or_recursive_unverified"}:
        return {
            "risk_change": "do_not_change_logic_until_bias_checks_pass",
            "entry_change": "hold current candidate logic; prioritize recursive/lookahead verification",
            "exit_change": "none",
            "leverage_cap": 3.0,
        }
    if blocker in {"matrix_not_robust", "fragile_matrix", "matrix_not_tested"}:
        return {
            "risk_change": "split by market regime and reject one-slice winners",
            "entry_change": "add regime-specific confirmation and disable hostile regime entries",
            "exit_change": "exit faster when regime flips against the trade",
            "leverage_cap": 3.0,
        }
    if blocker in {"too_few_trades", "too_few_matrix_trades", "insufficient_sample"}:
        return {
            "risk_change": "preserve sample size before tightening filters",
            "entry_change": "replace hard filters with scored confirmation and minimum signal quality",
            "exit_change": "keep exits simple so entry changes are isolated",
            "leverage_cap": 2.0,
        }
    if blocker in {"negative_after_cost", "cost_evidence_missing", "cost_not_estimated", "stress_cost_failure"}:
        return {
            "risk_change": "reduce churn and require larger expected move than fee/slippage/funding",
            "entry_change": "require volatility-adjusted edge before entry",
            "exit_change": "avoid tiny ROI exits that cannot survive stress costs",
            "leverage_cap": 2.0,
        }
    if blocker in {"loss_exit_quality", "weak_profit_factor", "negative_or_missing_return"}:
        return {
            "risk_change": "tighten invalidation and add cooldown after loss clusters",
            "entry_change": "require pullback plus resume confirmation instead of immediate signal entry",
            "exit_change": "cut trades when MFE fails to develop quickly",
            "leverage_cap": 2.0,
        }
    return {
        "risk_change": "isolate one blocker and keep leverage conservative",
        "entry_change": "use explicit confirmation instead of raw directional signal",
        "exit_change": "record exit reason and compare payoff before/after",
        "leverage_cap": 2.0,
    }


def success_gate(blocker: str) -> str:
    gates = {
        "bias_checks_missing": "recursive_analysis and lookahead_analysis both return ok.",
        "lookahead_or_recursive_unverified": "recursive_analysis and lookahead_analysis both return ok.",
        "matrix_not_robust": "matrix verdict becomes robust_candidate with no stress-negative cluster.",
        "fragile_matrix": "matrix verdict becomes robust_candidate across bull/bear/range/high-vol slices.",
        "too_few_trades": "full-sample and matrix windows each keep enough trades for evaluation.",
        "too_few_matrix_trades": "matrix windows each keep enough trades for evaluation.",
        "negative_after_cost": "adjusted return stays positive after funding, fees, and stress slippage.",
        "cost_evidence_missing": "latest cost estimate includes this strategy with adjusted return.",
        "loss_exit_quality": "max consecutive losses and stop-loss loss share improve without killing trade count.",
        "weak_profit_factor": "profit factor improves while total trades remain statistically reviewable.",
    }
    return gates.get(blocker, "scorecard improves without introducing a higher-severity failure mode.")


def build_hypotheses(memory: dict[str, Any], lineage: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = {item.get("name"): item for item in lineage.get("nodes", [])}
    avoid_rules = [item.get("memory_rule") for item in memory.get("avoid_patterns", []) if item.get("memory_rule")]
    hypotheses = []
    seen: set[tuple[str, str]] = set()
    for index, focus in enumerate(memory.get("next_focus", []), start=1):
        strategy = focus.get("strategy")
        blocker = focus.get("blocker") or "unknown_blocker"
        if not strategy or (strategy, blocker) in seen:
            continue
        seen.add((strategy, blocker))
        template = risk_template(blocker)
        parent = nodes.get(strategy, {})
        hypothesis_id = f"mem_{index:02d}_{slug(strategy)}_{slug(blocker)}"
        hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "strategy": strategy,
                "root": parent.get("root") or strategy,
                "blocker": blocker,
                "objective": focus.get("objective") or template["entry_change"],
                "memory_guidance": {
                    "avoid_rules": avoid_rules[:5],
                    "active_root_state": parent.get("recommended_state"),
                    "top_failure": parent.get("failure_attribution", {}).get("top_mode"),
                },
                "proposed_changes": template,
                "success_gate": focus.get("success_gate") or success_gate(blocker),
                "next_command": focus.get("next_command") or "user_data/strategy_research/start_manual_research.sh --behavior-variants",
                "risk_notes": "Research-only plan. Do not raise leverage or promote without passing scorecard, matrix, walk-forward, cost, and bias gates.",
            }
        )
    return hypotheses[:8]


def build_experiment(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "memory_guided_strategy_research",
        "description": "Memory-guided strategy hypotheses. Generate concrete strategy variants from these plans before backtesting.",
        "timeframes": ["1m"],
        "timeranges": ["20240101-20260622"],
        "fee": 0.0006,
        "strategies": [item["strategy"] for item in hypotheses],
        "hypothesis_ids": [item["hypothesis_id"] for item in hypotheses],
        "notes": [
            "This experiment file is a planning handoff, not a runnable generated-strategy registry.",
            "Concrete strategy code must be generated in an isolated research file before running Freqtrade backtesting.",
        ],
    }


def build_payload() -> dict[str, Any]:
    memory = load_json(AGENT_ROOT / "research_memory/latest_research_memory.json")
    lineage = load_json(AGENT_ROOT / "strategy_library/latest_strategy_lineage.json")
    hypotheses = build_hypotheses(memory, lineage)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "experiment": build_experiment(hypotheses),
        "source_artifacts": {
            "research_memory": rel(AGENT_ROOT / "research_memory/latest_research_memory.json") if memory else None,
            "strategy_lineage": rel(AGENT_ROOT / "strategy_library/latest_strategy_lineage.json") if lineage else None,
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Memory-Guided Hypothesis Ledger",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Hypotheses: `{payload['hypothesis_count']}`",
        "",
        "| ID | Strategy | Blocker | Objective | Entry Change | Risk Change | Success Gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in payload["hypotheses"]:
        changes = item["proposed_changes"]
        lines.append(
            "| {hypothesis_id} | {strategy} | {blocker} | {objective} | {entry_change} | {risk_change} | {success_gate} |".format(
                entry_change=changes.get("entry_change"),
                risk_change=changes.get("risk_change"),
                **item,
            )
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- This ledger plans strategy research; it does not create live-trading code.",
            "- Concrete variants must be generated into isolated research strategy files before backtesting.",
            "- Every hypothesis inherits avoid rules from research memory to reduce repeated failure loops.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_EXPERIMENT.write_text(json.dumps(payload["experiment"], indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(OUTPUT_MD, payload)
    print(f"Wrote {rel(OUTPUT_JSON)}")
    print(f"Wrote {rel(OUTPUT_MD)}")
    print(f"Wrote {rel(OUTPUT_EXPERIMENT)}")


if __name__ == "__main__":
    main()
