#!/usr/bin/env python3
"""Build the solidified operating layer for the strategy research agent.

This layer converts knowledge graph context and research memory into durable,
machine-readable policy. It is intentionally research-only: it never promotes a
strategy, edits trading config, or touches exchange credentials.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_DIR = AGENT_ROOT / "consolidation"
LATEST_JSON = OUTPUT_DIR / "latest_research_consolidation.json"
LATEST_MD = OUTPUT_DIR / "latest_research_consolidation.md"
OPERATING_RULES_JSON = OUTPUT_DIR / "agent_operating_rules.json"
WORKFLOW_CONTRACT_MD = OUTPUT_DIR / "agent_workflow_contract.md"

MEMORY_JSON = AGENT_ROOT / "research_memory/latest_research_memory.json"
GRAPH_CONTEXT_JSON = AGENT_ROOT / "knowledge/graph/strategy_agent_graph_context.json"
KNOWLEDGE_PLAN_JSON = AGENT_ROOT / "experiments/knowledge_guided_hypothesis_plan.json"
MEMORY_PLAN_JSON = AGENT_ROOT / "experiments/memory_guided_hypothesis_plan.json"
PROMOTION_JSON = AGENT_ROOT / "promotion_reports/latest_promotion_report.json"
WEEKLY_KNOWLEDGE_UPDATE_JSON = AGENT_ROOT / "knowledge_updates/latest_weekly_knowledge_update.json"


REQUIRED_GATES = [
    "event_study_edge_check",
    "freqtrade_backtesting",
    "post_run_attribution",
    "recursive_analysis",
    "lookahead_analysis",
    "regime_matrix",
    "fee_slippage_stress",
    "walk_forward_validation",
    "promotion_gate",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def infer_family(card: dict[str, Any]) -> str:
    concepts = set(card.get("concepts", []))
    text = json.dumps(card, ensure_ascii=False).lower()
    if {"breakout", "breakout_test", "failed_breakout"} & concepts or "突破" in text:
        return "breakout"
    if {"pullback", "trend_bar", "counting_bars"} & concepts or "回调" in text:
        return "pullback"
    if {"scalp", "microstructure"} & concepts or "剥头皮" in text:
        return "scalp"
    if {"reversal", "wedge", "parabolic"} & concepts or "反转" in text:
        return "reversal"
    if {"risk", "actual_risk", "fees", "funding"} & concepts or "风险" in text:
        return "risk_control"
    if {"market_cycle", "regime_router"} & concepts or "震荡" in text or "趋势" in text:
        return "regime"
    return card.get("category") or "general"


def build_allowed_families(graph_context: dict[str, Any]) -> list[dict[str, Any]]:
    counts = Counter(infer_family(card) for card in graph_context.get("cards", []))
    return [{"family": key, "active_card_count": value} for key, value in counts.most_common()]


def build_solidified_rules(memory: dict[str, Any], graph_context: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule in memory.get("durable_rules", []):
        rules.append({"source": "research_memory", "type": "durable_rule", "rule": rule})
    for item in memory.get("avoid_patterns", []):
        if item.get("memory_rule"):
            rules.append(
                {
                    "source": "research_memory",
                    "type": "avoid_pattern",
                    "pattern": item.get("pattern"),
                    "evidence_count": item.get("evidence_count"),
                    "rule": item.get("memory_rule"),
                }
            )
    for rule in memory.get("knowledge_memory", {}).get("knowledge_avoid_rules", []):
        rules.append({"source": "knowledge_graph", "type": "knowledge_avoid_rule", "rule": rule})
    for card in graph_context.get("cards", []):
        for rule in card.get("avoid_rules", []):
            rules.append(
                {
                    "source": "knowledge_graph_card",
                    "type": "card_avoid_rule",
                    "card_node": card.get("card_node"),
                    "rule": rule,
                }
            )
    rules.extend(
        [
            {
                "source": "system_policy",
                "type": "hard_boundary",
                "rule": "Knowledge cards can generate hypotheses only; they cannot promote strategies into dry-run or live trading.",
            },
            {
                "source": "system_policy",
                "type": "hard_boundary",
                "rule": "Do not use quarantined, mismatched, or missing-source material as strategy fuel.",
            },
            {
                "source": "system_policy",
                "type": "hard_boundary",
                "rule": "Do not increase leverage to compensate for weak signal edge.",
            },
        ]
    )
    return dedupe(rules)


def build_blocked_patterns(memory: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = []
    for item in memory.get("avoid_patterns", []):
        blocked.append(
            {
                "pattern": item.get("pattern"),
                "evidence_count": item.get("evidence_count"),
                "allowed_use": "counterexample_or_redesign_only",
                "rule": item.get("memory_rule"),
            }
        )
    return blocked


def build_next_actions(memory: dict[str, Any], knowledge_plan: dict[str, Any], memory_plan: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    for item in memory.get("next_focus", [])[:5]:
        actions.append(
            {
                "priority": len(actions) + 1,
                "source": "research_memory",
                "action": "design_or_run_focused_experiment",
                "strategy": item.get("strategy"),
                "blocker": item.get("blocker"),
                "success_gate": item.get("success_gate"),
            }
        )
    for item in knowledge_plan.get("hypotheses", [])[:4]:
        actions.append(
            {
                "priority": len(actions) + 1,
                "source": "knowledge_guided_hypothesis",
                "action": "translate_hypothesis_to_isolated_freqtrade_variant",
                "hypothesis_id": item.get("hypothesis_id"),
                "status": item.get("status"),
                "source_cards": item.get("source_graph_nodes") or item.get("source_cards"),
            }
        )
    for item in memory_plan.get("hypotheses", [])[:4]:
        actions.append(
            {
                "priority": len(actions) + 1,
                "source": "memory_guided_hypothesis",
                "action": "generate_variant_only_if_blocker_is_changed",
                "hypothesis_id": item.get("hypothesis_id"),
                "strategy": item.get("strategy"),
                "blocker": item.get("blocker"),
            }
        )
    return actions


def build_payload() -> dict[str, Any]:
    memory = load_json(MEMORY_JSON)
    graph_context = load_json(GRAPH_CONTEXT_JSON)
    knowledge_plan = load_json(KNOWLEDGE_PLAN_JSON)
    memory_plan = load_json(MEMORY_PLAN_JSON)
    promotion = load_json(PROMOTION_JSON)
    weekly_update = load_json(WEEKLY_KNOWLEDGE_UPDATE_JSON)
    required_checks = Counter()
    for card in graph_context.get("cards", []):
        for check in card.get("required_checks", []):
            required_checks[check] += 1
    gates = dedupe(REQUIRED_GATES + list(required_checks))
    payload = {
        "generated_at_utc": now_utc(),
        "layer_version": 1,
        "research_only": True,
        "allowed_research_families": build_allowed_families(graph_context),
        "blocked_patterns": build_blocked_patterns(memory),
        "solidified_rules": build_solidified_rules(memory, graph_context),
        "required_gates": gates,
        "promotion_boundaries": [
            "No knowledge-derived hypothesis can enter candidate/watchlist pools before full evidence gates.",
            "No OHLCV or L2-inspired hypothesis can generate a strategy class before an event study shows edge_candidate, unless the run is explicitly labeled counterexample or negative-control.",
            "No backtest round can feed the next experiment queue until post-run attribution has identified signal, timing, exit, cost, risk, regime, and sample-size failure modes.",
            "No strategy reaches dry-run review without manual approval after promotion gate.",
            "Live trading is outside this agent flow.",
            "Dry-run/live config files must not be modified by this consolidation layer.",
        ],
        "next_agent_actions": build_next_actions(memory, knowledge_plan, memory_plan),
        "agent_prompt_contract": [
            "The Agent already has materials, a knowledge graph, and a self-iteration loop; frame future work as deeper integration, not as building those from zero.",
            "The Agent has two iteration loops: internal self-iteration from backtest evidence and external knowledge iteration from the weekly knowledge update layer.",
            "Load knowledge graph context, research memory, and this consolidation policy before generating strategies.",
            "Before generating a concrete strategy class, define a measurable event and run or read event-study evidence for samples, forward returns, win rate, and MFE/MAE.",
            "If no event has verdict=edge_candidate, produce event redesigns, data-collection tasks, or negative-control studies instead of another strategy class.",
            "After every backtest or strategy research round, run post-run attribution before updating research memory, mature researcher queues, or next experiments.",
            "Post-run attribution must separate signal edge, entry timing, exit quality, cost/funding drag, fixed 50x risk amplification, regime dependency, and sample validity.",
            "Use at most 1-3 active knowledge cards per hypothesis.",
            "If memory avoid rules conflict with a knowledge card, downgrade to counterexample or redesign-only.",
            "Always define measurable entry, exit, invalidation, applicable regime, hostile regime, and validation requirements.",
            "Never claim profitability from theory; only validated backtest evidence can update queues.",
        ],
        "source_artifacts": {
            "research_memory": rel(MEMORY_JSON) if memory else None,
            "knowledge_graph_context": rel(GRAPH_CONTEXT_JSON) if graph_context else None,
            "knowledge_guided_plan": rel(KNOWLEDGE_PLAN_JSON) if knowledge_plan else None,
            "memory_guided_plan": rel(MEMORY_PLAN_JSON) if memory_plan else None,
            "promotion_report": rel(PROMOTION_JSON) if promotion else None,
            "weekly_knowledge_update": rel(WEEKLY_KNOWLEDGE_UPDATE_JSON) if weekly_update else None,
            "workflow_contract": rel(WORKFLOW_CONTRACT_MD) if WORKFLOW_CONTRACT_MD.exists() else None,
        },
        "observed_counts": {
            "active_knowledge_cards": graph_context.get("active_card_count", 0),
            "knowledge_hypotheses": knowledge_plan.get("hypothesis_count", 0),
            "memory_hypotheses": memory_plan.get("hypothesis_count", 0),
            "avoid_patterns": len(memory.get("avoid_patterns", [])),
            "solidified_rules": 0,
        },
    }
    payload["observed_counts"]["solidified_rules"] = len(payload["solidified_rules"])
    return payload


def build_operating_rules(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": payload["generated_at_utc"],
        "must_load_before_research": [
            rel(GRAPH_CONTEXT_JSON),
            rel(MEMORY_JSON),
            rel(LATEST_JSON),
            rel(WORKFLOW_CONTRACT_MD),
            rel(WEEKLY_KNOWLEDGE_UPDATE_JSON),
        ],
        "research_only": True,
        "hard_boundaries": payload["promotion_boundaries"],
        "required_gates": payload["required_gates"],
        "blocked_patterns": payload["blocked_patterns"],
        "prompt_contract": payload["agent_prompt_contract"],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strategy Research Consolidation Layer",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Research-only: `{payload['research_only']}`",
        f"- Active knowledge cards: `{payload['observed_counts']['active_knowledge_cards']}`",
        f"- Solidified rules: `{payload['observed_counts']['solidified_rules']}`",
        "",
        "## Allowed Research Families",
        "",
        "| Family | Active Cards |",
        "|---|---:|",
    ]
    for item in payload["allowed_research_families"]:
        lines.append("| {family} | {active_card_count} |".format(**item))
    lines.extend(["", "## Blocked Patterns", "", "| Pattern | Evidence | Allowed Use | Rule |", "|---|---:|---|---|"])
    for item in payload["blocked_patterns"]:
        lines.append(
            "| {pattern} | {evidence_count} | {allowed_use} | {rule} |".format(
                pattern=item.get("pattern") or "",
                evidence_count=item.get("evidence_count") or 0,
                allowed_use=item.get("allowed_use") or "",
                rule=item.get("rule") or "",
            )
        )
    lines.extend(["", "## Required Gates", ""])
    for gate in payload["required_gates"]:
        lines.append(f"- {gate}")
    lines.extend(["", "## Prompt Contract", ""])
    for rule in payload["agent_prompt_contract"]:
        lines.append(f"- {rule}")
    lines.extend(["", "## Next Agent Actions", "", "| Priority | Source | Action | Target |", "|---:|---|---|---|"])
    for item in payload["next_agent_actions"]:
        target = item.get("hypothesis_id") or item.get("strategy") or item.get("blocker") or ""
        lines.append("| {priority} | {source} | {action} | {target} |".format(target=target, **item))
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- This file is a policy handoff for research only.",
            "- It does not edit Freqtrade config, dry-run config, live config, or exchange credentials.",
            "- It does not place strategies into promotion pools by itself.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = OUTPUT_DIR / f"research_consolidation_{timestamp}.json"
    md_path = OUTPUT_DIR / f"research_consolidation_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    OPERATING_RULES_JSON.write_text(
        json.dumps(build_operating_rules(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")
    print(f"Wrote {rel(OPERATING_RULES_JSON)}")


def main() -> None:
    write_outputs(build_payload())


if __name__ == "__main__":
    main()
