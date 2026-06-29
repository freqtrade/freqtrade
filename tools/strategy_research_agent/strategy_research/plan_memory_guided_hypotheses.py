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
GRAPH_CONTEXT_JSON = AGENT_ROOT / "knowledge/graph/strategy_agent_graph_context.json"


BLOCKER_TO_CONCEPTS = {
    "weak_profit_factor": ["confirmation", "pullback", "actual_risk", "signal_score"],
    "negative_or_missing_return": ["edge", "regime_router", "pullback", "invalidation"],
    "loss_exit_quality": ["stoploss", "invalidation", "actual_risk", "timebox"],
    "too_few_trades": ["signal_score", "condition_count", "entry_confirmation"],
    "too_few_matrix_trades": ["market_cycle", "regime_router", "sessionless_market"],
    "matrix_not_robust": ["market_cycle", "regime_router", "trend", "range"],
    "fragile_matrix": ["market_cycle", "regime_router", "walk_forward"],
    "negative_after_cost": ["fees", "slippage", "scalp", "funding"],
    "stress_cost_failure": ["fees", "slippage", "microstructure"],
    "cost_evidence_missing": ["fees", "funding", "fee_stress"],
    "bias_checks_missing": ["review", "out_of_sample", "falsification"],
    "lookahead_or_recursive_unverified": ["review", "out_of_sample", "falsification"],
}


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


def dedupe(values: list[Any]) -> list[Any]:
    seen = set()
    out = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def select_graph_cards(blocker: str, objective: str, graph_context: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    desired = set(BLOCKER_TO_CONCEPTS.get(blocker, []))
    text_terms = set((objective or "").lower().replace("/", " ").replace("_", " ").split())
    scored = []
    for card in graph_context.get("cards", []):
        haystack = json.dumps(card, ensure_ascii=False).lower()
        concepts = set(card.get("concepts", []))
        score = len(desired & concepts) * 6
        score += sum(1 for term in desired if term and term.lower() in haystack)
        score += sum(1 for term in text_terms if len(term) > 3 and term in haystack)
        if card.get("source_quality") == "high":
            score += 2
        if card.get("category") in {"risk", "crypto_adaptation"} and blocker in {
            "negative_after_cost",
            "stress_cost_failure",
            "loss_exit_quality",
        }:
            score += 3
        if score > 0:
            scored.append((score, card))
    scored.sort(key=lambda item: (-item[0], item[1].get("card_node") or item[1].get("title") or ""))
    selected = [item[1] for item in scored[:limit]]
    if len(selected) < limit:
        fallback = [
            card
            for card in graph_context.get("cards", [])
            if card not in selected and card.get("category") in {"entry", "risk", "crypto_adaptation"}
        ]
        selected.extend(fallback[: limit - len(selected)])
    return selected[:limit]


def knowledge_guidance(blocker: str, objective: str, graph_context: dict[str, Any]) -> dict[str, Any]:
    selected = select_graph_cards(blocker, objective, graph_context)
    return {
        "source_cards": [card.get("card_node") for card in selected if card.get("card_node")],
        "concepts": dedupe(sum((card.get("concepts", []) for card in selected), []))[:12],
        "entry_rules": dedupe(sum((card.get("entry_rules", []) for card in selected), []))[:5],
        "exit_rules": dedupe(sum((card.get("exit_rules", []) for card in selected), []))[:4],
        "risk_notes": dedupe(sum((card.get("risk_notes", []) for card in selected), []))[:5],
        "avoid_rules": dedupe(sum((card.get("avoid_rules", []) for card in selected), []))[:6],
        "required_checks": dedupe(sum((card.get("required_checks", []) for card in selected), []))
        or ["freqtrade_backtesting", "recursive_analysis", "lookahead_analysis", "regime_matrix", "fee_slippage_stress"],
    }


def build_hypotheses(memory: dict[str, Any], lineage: dict[str, Any], graph_context: dict[str, Any]) -> list[dict[str, Any]]:
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
        objective = focus.get("objective") or template["entry_change"]
        hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "strategy": strategy,
                "root": parent.get("root") or strategy,
                "blocker": blocker,
                "objective": objective,
                "memory_guidance": {
                    "avoid_rules": avoid_rules[:5],
                    "active_root_state": parent.get("recommended_state"),
                    "top_failure": parent.get("failure_attribution", {}).get("top_mode"),
                },
                "knowledge_guidance": knowledge_guidance(blocker, objective, graph_context) if graph_context else {},
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
    graph_context = load_json(GRAPH_CONTEXT_JSON)
    hypotheses = build_hypotheses(memory, lineage, graph_context)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "experiment": build_experiment(hypotheses),
        "source_artifacts": {
            "research_memory": rel(AGENT_ROOT / "research_memory/latest_research_memory.json") if memory else None,
            "strategy_lineage": rel(AGENT_ROOT / "strategy_library/latest_strategy_lineage.json") if lineage else None,
            "knowledge_graph_context": rel(GRAPH_CONTEXT_JSON) if graph_context else None,
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Memory-Guided Hypothesis Ledger",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Hypotheses: `{payload['hypothesis_count']}`",
        "",
        "| ID | Strategy | Blocker | Knowledge Cards | Objective | Entry Change | Risk Change | Success Gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in payload["hypotheses"]:
        changes = item["proposed_changes"]
        lines.append(
            "| {hypothesis_id} | {strategy} | {blocker} | {cards} | {objective} | {entry_change} | {risk_change} | {success_gate} |".format(
                entry_change=changes.get("entry_change"),
                risk_change=changes.get("risk_change"),
                cards=", ".join(item.get("knowledge_guidance", {}).get("source_cards", [])[:3]),
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
