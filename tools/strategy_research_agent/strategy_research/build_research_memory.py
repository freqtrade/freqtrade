#!/usr/bin/env python3
"""Build durable research memory from latest strategy evidence."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strategy_taxonomy import infer_family_from_card, taxonomy_summary


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_DIR = AGENT_ROOT / "research_memory"
LATEST_JSON = OUTPUT_DIR / "latest_research_memory.json"
LATEST_MD = OUTPUT_DIR / "latest_research_memory.md"
GRAPH_CONTEXT_JSON = AGENT_ROOT / "knowledge/graph/strategy_agent_graph_context.json"
MANUAL_LESSONS_DIR = OUTPUT_DIR / "manual_lessons"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manual_lessons() -> list[dict[str, Any]]:
    if not MANUAL_LESSONS_DIR.exists():
        return []
    lessons = []
    for path in sorted(MANUAL_LESSONS_DIR.glob("*.json")):
        item = load_json(path)
        if not item:
            continue
        item.setdefault("source_file", rel(path))
        lessons.append(item)
    return lessons


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def grouped_nodes(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in nodes:
        grouped[item.get("root") or item.get("name") or "unknown"].append(item)
    return grouped


def first_action(item: dict[str, Any]) -> str:
    actions = item.get("promotion", {}).get("next_actions", [])
    if actions:
        return actions[0]
    return item.get("failure_attribution", {}).get("recommendation") or ""


def build_active_roots(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots = []
    for root, items in grouped_nodes(nodes).items():
        candidates = [item for item in items if item.get("pool_status") == "candidate"]
        watchlist = [item for item in items if item.get("pool_status") == "watchlist"]
        top_failures = Counter(
            item.get("failure_attribution", {}).get("top_mode")
            for item in items
            if item.get("failure_attribution", {}).get("top_mode")
        )
        if not candidates and not watchlist:
            continue
        roots.append(
            {
                "root": root,
                "candidate_count": len(candidates),
                "watchlist_count": len(watchlist),
                "child_count": max(0, len(items) - 1),
                "top_failures": [{"mode": key, "count": count} for key, count in top_failures.most_common(3)],
                "recommended_state": candidates[0].get("recommended_state") if candidates else watchlist[0].get("recommended_state"),
                "next_action": first_action(candidates[0] if candidates else watchlist[0]),
            }
        )
    return sorted(roots, key=lambda item: (-item["candidate_count"], -item["watchlist_count"], item["root"]))


def build_avoid_patterns(nodes: list[dict[str, Any]], failure_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns = []
    failure_counts = Counter({item.get("mode"): item.get("count") for item in failure_summary if item.get("mode")})
    archived = [item for item in nodes if item.get("recommended_state") == "archive_or_redesign"]
    mode_counts = Counter(
        item.get("failure_attribution", {}).get("top_mode")
        for item in archived
        if item.get("failure_attribution", {}).get("top_mode")
    )
    combined = failure_counts + mode_counts
    for mode, count in combined.most_common():
        if not mode:
            continue
        patterns.append(
            {
                "pattern": mode,
                "evidence_count": count or failure_counts.get(mode),
                "memory_rule": memory_rule_for_mode(mode),
            }
        )
    return patterns


def memory_rule_for_mode(mode: str) -> str:
    rules = {
        "insufficient_sample": "Do not accept stricter entries until the strategy still produces enough trades for evaluation.",
        "bias_unverified": "Run recursive/lookahead checks before treating any promising result as reusable evidence.",
        "regime_fragility": "Require explicit bull/bear/range/high-vol split evidence before promoting a single-rule variant.",
        "loss_exit_quality": "Improve invalidation and cooldown before adding leverage or widening take profit.",
        "benchmark_underperformance": "Require benchmark-relative edge, not only positive absolute return.",
        "cost_sensitivity": "Reject variants whose edge disappears after fee, slippage, and funding adjustments.",
        "directional_concentration": "Label short-only or long-only behavior explicitly and test hostile regimes.",
        "pair_drag": "Test pair-disabled and pair-only variants before keeping the weak lane.",
    }
    return rules.get(mode, "Keep this failure mode visible in the next experiment design.")


def build_next_focus(agenda: dict[str, Any], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    focus = []
    for item in agenda.get("top_priorities", [])[:8]:
        focus.append(
            {
                "strategy": item.get("strategy"),
                "blocker": item.get("blocker"),
                "objective": item.get("objective"),
                "success_gate": item.get("success_gate"),
                "next_command": item.get("next_command"),
                "source": "research_agenda",
            }
        )
    if focus:
        return focus
    for item in nodes:
        if item.get("recommended_state") in {"research_candidate", "watchlist", "redesign"}:
            focus.append(
                {
                    "strategy": item.get("name"),
                    "blocker": item.get("failure_attribution", {}).get("top_mode"),
                    "objective": first_action(item),
                    "success_gate": item.get("success_gate") or "Improve evidence without violating safety gates.",
                    "next_command": "user_data/strategy_research/start_manual_research.sh --memory-guided-hypotheses",
                    "source": "strategy_lineage",
                }
            )
    return focus[:8]


def build_knowledge_gaps(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = Counter()
    for item in nodes:
        blocks = item.get("promotion", {}).get("blocks", [])
        for block in blocks:
            gaps[block] += 1
        failures = item.get("scorecard", {}).get("primary_failures", [])
        for failure in failures:
            gaps[failure] += 1
    return [
        {"gap": gap, "count": count, "close_by": close_gap_instruction(gap)}
        for gap, count in gaps.most_common(10)
    ]


def close_gap_instruction(gap: str) -> str:
    if gap in {"bias_checks_missing", "lookahead_or_recursive_unverified"}:
        return "Run recursive-analysis and lookahead-analysis on candidates before promotion review."
    if gap in {"too_few_trades", "too_few_matrix_trades"}:
        return "Redesign entries to preserve enough valid trades across fixed windows."
    if gap in {"matrix_not_robust", "fragile_matrix", "matrix_not_tested"}:
        return "Run market-regime and cost matrix; split logic by regime if needed."
    if gap in {"negative_after_cost", "stress_cost_failure", "cost_evidence_missing", "cost_not_estimated"}:
        return "Export trades and re-estimate fee, slippage, and funding impact."
    if gap == "weak_profit_factor":
        return "Improve entry quality or exit asymmetry before adding more variants."
    if gap in {"negative_or_missing_return", "scorecard_not_promotable"}:
        return "Rebuild the hypothesis around positive expectancy, then rerun scorecards."
    if gap == "underperforms_market":
        return "Add benchmark-relative acceptance gates to the experiment."
    return "Create a focused experiment that directly tests this gap."


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


def build_knowledge_memory(graph_context: dict[str, Any]) -> dict[str, Any]:
    cards = graph_context.get("cards", [])
    concept_counts = Counter()
    category_counts = Counter()
    quality_counts = Counter()
    required_checks = Counter()
    avoid_rules: list[str] = []
    strategy_families = Counter()
    high_risk_cards = []

    for card in cards:
        category_counts[card.get("category") or "unknown"] += 1
        quality_counts[card.get("source_quality") or "unknown"] += 1
        for concept in card.get("concepts", []):
            concept_counts[concept] += 1
        for check in card.get("required_checks", []):
            required_checks[check] += 1
        avoid_rules.extend(card.get("avoid_rules", []))
        family = infer_strategy_family(card)
        if family:
            strategy_families[family] += 1
        if card.get("category") in {"risk", "crypto_adaptation"} or any(
            keyword in json.dumps(card, ensure_ascii=False).lower()
            for keyword in ["scalp", "leverage", "funding", "slippage", "高杠杆", "滑点", "手续费"]
        ):
            high_risk_cards.append(
                {
                    "card_node": card.get("card_node"),
                    "title": card.get("title"),
                    "risk_notes": card.get("risk_notes", [])[:3],
                }
            )

    return {
        "active_card_count": graph_context.get("active_card_count") or len(cards),
        "policy": graph_context.get("policy", {}),
        "category_summary": [{"category": key, "count": value} for key, value in category_counts.most_common()],
        "source_quality_summary": [{"quality": key, "count": value} for key, value in quality_counts.most_common()],
        "top_concepts": [{"concept": key, "count": value} for key, value in concept_counts.most_common(20)],
        "strategy_family_summary": [
            {"family": key, "count": value} for key, value in strategy_families.most_common(15)
        ],
        "required_checks": [{"check": key, "count": value} for key, value in required_checks.most_common()],
        "knowledge_avoid_rules": dedupe(avoid_rules)[:30],
        "high_risk_cards": high_risk_cards[:12],
    }


def infer_strategy_family(card: dict[str, Any]) -> str:
    return infer_family_from_card(card)


def build_payload() -> dict[str, Any]:
    lineage = load_json(AGENT_ROOT / "strategy_library/latest_strategy_lineage.json")
    failure = load_json(AGENT_ROOT / "failure_attribution/latest_failure_attribution.json")
    agenda = load_json(AGENT_ROOT / "research_agendas/latest_research_agenda.json")
    assessment = load_json(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json")
    graph_context = load_json(GRAPH_CONTEXT_JSON)
    manual_lessons = load_manual_lessons()
    nodes = lineage.get("nodes", [])
    failure_summary = failure.get("failure_mode_summary", [])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "memory_version": 2,
        "strategy_count": len(nodes),
        "active_roots": build_active_roots(nodes),
        "avoid_patterns": build_avoid_patterns(nodes, failure_summary),
        "next_focus": build_next_focus(agenda, nodes),
        "knowledge_gaps": build_knowledge_gaps(nodes),
        "knowledge_memory": build_knowledge_memory(graph_context) if graph_context else {},
        "strategy_taxonomy": taxonomy_summary(),
        "manual_lessons": manual_lessons,
        "durable_rules": [
            "Never promote a strategy from a single favorable slice.",
            "Treat leverage as a risk multiplier, not a fix for weak entry timing.",
            "Prefer variants that improve trade quality and sample size together.",
            "Every futures strategy hypothesis must declare a canonical strategy family and regime contract before strategy code is generated.",
            "Post-run attribution must be grouped by strategy family as well as by individual strategy class.",
            "Keep external strategies quarantined until translated into local auditable code.",
            "Do not repeat archived variants unless the new experiment changes the diagnosed failure mode.",
            "Generate new hypotheses from active knowledge-graph cards only; quarantined cards are reference material, not strategy fuel.",
            "Any knowledge-derived strategy must remain research-only until backtest, recursive, lookahead, regime, cost, and promotion gates pass.",
        ]
        + [lesson["memory_rule"] for lesson in manual_lessons if lesson.get("memory_rule")],
        "source_artifacts": {
            "strategy_lineage": rel(AGENT_ROOT / "strategy_library/latest_strategy_lineage.json") if lineage else None,
            "failure_attribution": rel(AGENT_ROOT / "failure_attribution/latest_failure_attribution.json") if failure else None,
            "research_agenda": rel(AGENT_ROOT / "research_agendas/latest_research_agenda.json") if agenda else None,
            "strategy_assessment": rel(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json") if assessment else None,
            "knowledge_graph_context": rel(GRAPH_CONTEXT_JSON) if graph_context else None,
        },
    }
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strategy Research Memory",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Strategies observed: `{payload['strategy_count']}`",
        "",
        "## Active Roots",
        "",
        "| Root | Candidates | Watchlist | Children | Top Failures | Next Action |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in payload["active_roots"]:
        failures = ", ".join(f"{row['mode']}({row['count']})" for row in item.get("top_failures", []))
        lines.append(
            "| {root} | {candidate_count} | {watchlist_count} | {child_count} | {failures} | {next_action} |".format(
                failures=failures,
                **item,
            )
        )
    lines.extend(["", "## Avoid Patterns", "", "| Pattern | Evidence Count | Memory Rule |", "|---|---:|---|"])
    for item in payload["avoid_patterns"]:
        lines.append("| {pattern} | {evidence_count} | {memory_rule} |".format(**item))
    lines.extend(["", "## Next Focus", "", "| Strategy | Blocker | Objective | Success Gate | Source |", "|---|---|---|---|---|"])
    for item in payload["next_focus"]:
        lines.append(
            "| {strategy} | {blocker} | {objective} | {success_gate} | {source} |".format(
                strategy=item.get("strategy") or "",
                blocker=item.get("blocker") or "",
                objective=item.get("objective") or "",
                success_gate=item.get("success_gate") or "",
                source=item.get("source") or "",
            )
        )
    lines.extend(["", "## Knowledge Gaps", "", "| Gap | Count | Close By |", "|---|---:|---|"])
    for item in payload["knowledge_gaps"]:
        lines.append("| {gap} | {count} | {close_by} |".format(**item))
    knowledge = payload.get("knowledge_memory") or {}
    lines.extend(
        [
            "",
            "## Knowledge Graph Memory",
            "",
            f"- Active knowledge cards: `{knowledge.get('active_card_count', 0)}`",
            f"- Research-only: `{knowledge.get('policy', {}).get('research_only', True)}`",
            f"- Exclude quarantined cards: `{knowledge.get('policy', {}).get('exclude_quarantined_cards', True)}`",
            "",
            "| Family | Count |",
            "|---|---:|",
        ]
    )
    for item in knowledge.get("strategy_family_summary", []):
        lines.append("| {family} | {count} |".format(**item))
    lines.extend(
        [
            "",
            "## Strategy Taxonomy",
            "",
            "| Code | Family | Direction | Allowed Regimes | Disabled Regimes |",
            "|---|---|---|---|---|",
        ]
    )
    for item in payload.get("strategy_taxonomy", []):
        lines.append(
            "| {code} | {name} | {direction} | {allowed} | {disabled} |".format(
                code=item.get("code") or "",
                name=item.get("name") or item.get("id") or "",
                direction=item.get("direction") or "",
                allowed=", ".join(item.get("allowed_regimes", [])),
                disabled=", ".join(item.get("disabled_regimes", [])),
            )
        )
    lines.extend(["", "### Knowledge Avoid Rules", ""])
    for rule in knowledge.get("knowledge_avoid_rules", [])[:12]:
        lines.append(f"- {rule}")
    lines.extend(["", "### Required Checks", "", "| Check | Cards |", "|---|---:|"])
    for item in knowledge.get("required_checks", []):
        lines.append("| {check} | {count} |".format(**item))
    lines.extend(
        [
            "",
            "## Manual Evidence Lessons",
            "",
            "| Lesson | Evidence | Memory Rule | Next Test |",
            "|---|---|---|---|",
        ]
    )
    for item in payload.get("manual_lessons", []):
        evidence = ", ".join(item.get("evidence", [])[:3])
        lines.append(
            "| {lesson} | {evidence} | {memory_rule} | {next_test} |".format(
                lesson=item.get("lesson") or "",
                evidence=evidence,
                memory_rule=item.get("memory_rule") or "",
                next_test=item.get("next_test") or "",
            )
        )
    lines.extend(["", "## Durable Rules", ""])
    for rule in payload["durable_rules"]:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Research memory is advisory; it does not start dry-run or live trading.",
            "- Memory must be rebuilt from current local evidence instead of hand-edited after every run.",
            "- Promotion remains controlled by the promotion gate and manual review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = OUTPUT_DIR / f"research_memory_{timestamp}.json"
    md_path = OUTPUT_DIR / f"research_memory_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    payload = build_payload()
    json_path, md_path = write_outputs(payload)
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")


if __name__ == "__main__":
    main()
