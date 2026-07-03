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
REGIME_WINDOWS_JSON = AGENT_ROOT / "regime_windows/latest_regime_windows.json"
REGIME_QUARANTINE_JSON = AGENT_ROOT / "regime_windows/regime_inference_quarantine.json"


REQUIRED_GATES = [
    "factor_research",
    "factor_to_strategy_plan",
    "event_study_edge_check",
    "freqtrade_backtesting",
    "post_run_attribution",
    "recursive_analysis",
    "lookahead_analysis",
    "regime_matrix",
    "fee_slippage_stress",
    "walk_forward_validation",
    "family_risk_gate",
    "promotion_gate",
    "dryrun_strategy_risk_preflight",
]
TIMEFRAME_POLICY = {
    "allowed_primary_entry_timeframes": ["3m", "5m", "15m"],
    "background_confirmation_timeframes": ["1h"],
    "forbidden_primary_entry_timeframes": ["1h", "4h", "1d"],
    "rule": "For fixed 50x futures research, 1h can only be used as background/regime/confirmation, not as the primary entry timeframe.",
}
FACTOR_RESEARCH_POLICY = {
    "same_agent_subflow": True,
    "runs_before_event_study": True,
    "runs_before_strategy_generation": True,
    "allowed_primary_timeframes": ["3m", "5m", "15m"],
    "latest_factor_report": "user_data/strategy_research/factors/latest_factor_research.json",
    "latest_factor_strategy_plan": "user_data/strategy_research/factors/latest_factor_strategy_plan.json",
    "rule": "Knowledge and memory propose research directions; factor research tests forward-return/MFE/MAE evidence; only factor edge candidates may become event-study hypotheses before strategy generation.",
}


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
            {
                "source": "system_policy",
                "type": "hard_boundary",
                "rule": "Do not generate fixed-50x strategy classes directly from knowledge cards or memory without factor or event-study evidence, unless the run is explicitly labeled negative-control.",
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
    quarantine = load_json(REGIME_QUARANTINE_JSON)
    if quarantine:
        blocked.append(
            {
                "pattern": "legacy_hardcoded_regime_window",
                "evidence_count": len(quarantine.get("entries", [])),
                "allowed_use": "raw_date_range_backtest_only",
                "rule": "Do not use legacy bull_home/range_home/bear_home/high_vol_hostile interpretations as Agent memory or promotion evidence until relabeled by the data-derived regime manifest.",
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
    regime_windows = load_json(REGIME_WINDOWS_JSON)
    regime_quarantine = load_json(REGIME_QUARANTINE_JSON)
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
            "No fixed-50x strategy class can be generated directly from external knowledge or memory; factor research must first test forward-return, MFE/MAE, sample size, and side-specific expectancy unless the run is explicitly labeled negative-control.",
            "No OHLCV or L2-inspired hypothesis can generate a strategy class before an event study shows edge_candidate, unless the run is explicitly labeled counterexample or negative-control.",
            "No backtest round can feed the next experiment queue until post-run attribution has identified signal, timing, exit, cost, risk, regime, and sample-size failure modes.",
            "Regime windows must come from user_data/strategy_research/regime_windows/latest_regime_windows.json; legacy hardcoded bull_home/range_home/bear_home/high_vol_hostile windows are quarantined and cannot fuel strategy generation or promotion.",
            "No new fixed-50x futures strategy may use 1h or higher candles as its primary entry timeframe; use 3m/5m/15m for entry and 1h only for background confirmation.",
            "No strategy reaches dry-run review without manual approval after promotion gate.",
            "Promotion gate is family-level: evaluate target-regime edge plus hostile-regime loss containment under router, cooldown, drawdown, and consecutive stop-loss circuit breakers, not naked all-regime performance alone.",
            "Live trading is outside this agent flow.",
            "Dry-run/live config files must not be modified by this consolidation layer.",
        ],
        "next_agent_actions": build_next_actions(memory, knowledge_plan, memory_plan),
        "agent_prompt_contract": [
            "The Agent already has materials, a knowledge graph, and a self-iteration loop; frame future work as deeper integration, not as building those from zero.",
            "The Agent has two iteration loops: internal self-iteration from backtest evidence and external knowledge iteration from the weekly knowledge update layer.",
            "Load knowledge graph context, research memory, and this consolidation policy before generating strategies.",
            "Load the data-derived regime manifest and regime inference quarantine before event-study planning, family-risk gates, promotion gates, or strategy generation.",
            "Do not treat legacy bull_home/range_home/bear_home/high_vol_hostile labels as market truth; old outputs are raw date-range backtests only until relabeled.",
            "Run factor research as the same Agent's front-door evidence layer before event-study planning or strategy synthesis.",
            "Convert only factor rows with sufficient sample, after-fee expectancy, win rate, and MFE/MAE evidence into factor-to-strategy event hypotheses.",
            "Before generating a concrete strategy class, define a measurable event and run or read event-study evidence for samples, forward returns, win rate, and MFE/MAE.",
            "If factor research has no edge_candidate rows, do not generate another strategy class from theory; redesign factors, run negative controls, or improve data.",
            "If no event has verdict=edge_candidate, produce event redesigns, data-collection tasks, or negative-control studies instead of another strategy class.",
            "After every backtest or strategy research round, run post-run attribution before updating research memory, mature researcher queues, or next experiments.",
            "Post-run attribution must separate signal edge, entry timing, exit quality, cost/funding drag, fixed 50x risk amplification, regime dependency, and sample validity.",
            "For promotion, evaluate every strategy family under regime-router and family/portfolio circuit breakers; do not require high-leverage crypto strategies to be all-regime holy grails.",
            "A family may be a dry-run review candidate only when its target-regime edge survives and hostile-regime losses are contained by family-level drawdown, cooldown, and consecutive stop-loss guards.",
            "For fixed 50x futures strategy generation, primary entry timeframe must be one of 3m, 5m, or 15m; 1h is background confirmation only.",
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
            "regime_windows": rel(REGIME_WINDOWS_JSON) if regime_windows else None,
            "regime_inference_quarantine": rel(REGIME_QUARANTINE_JSON) if regime_quarantine else None,
            "workflow_contract": rel(WORKFLOW_CONTRACT_MD) if WORKFLOW_CONTRACT_MD.exists() else None,
            "factor_research": FACTOR_RESEARCH_POLICY["latest_factor_report"],
            "factor_strategy_plan": FACTOR_RESEARCH_POLICY["latest_factor_strategy_plan"],
        },
        "observed_counts": {
            "active_knowledge_cards": graph_context.get("active_card_count", 0),
            "knowledge_hypotheses": knowledge_plan.get("hypothesis_count", 0),
            "memory_hypotheses": memory_plan.get("hypothesis_count", 0),
            "avoid_patterns": len(memory.get("avoid_patterns", [])),
            "solidified_rules": 0,
            "quarantined_regime_inference_entries": len(regime_quarantine.get("entries", [])) if regime_quarantine else 0,
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
            rel(REGIME_WINDOWS_JSON),
            rel(REGIME_QUARANTINE_JSON),
            rel(LATEST_JSON),
            rel(WORKFLOW_CONTRACT_MD),
            rel(WEEKLY_KNOWLEDGE_UPDATE_JSON),
        ],
        "research_only": True,
        "timeframe_policy": TIMEFRAME_POLICY,
        "factor_research_policy": FACTOR_RESEARCH_POLICY,
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
        f"- Quarantined legacy regime entries: `{payload['observed_counts'].get('quarantined_regime_inference_entries', 0)}`",
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
