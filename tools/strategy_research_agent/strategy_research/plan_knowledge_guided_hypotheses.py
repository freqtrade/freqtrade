#!/usr/bin/env python3
"""Plan strategy hypotheses from price-action knowledge cards and research memory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strategy_taxonomy import family_contract, infer_family_from_card, taxonomy_summary


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
KNOWLEDGE_ROOT = AGENT_ROOT / "knowledge"
CARDS_DIR = KNOWLEDGE_ROOT / "knowledge_cards"
INDEX_JSON = KNOWLEDGE_ROOT / "index/price_action_knowledge_index.json"
GRAPH_CONTEXT_JSON = KNOWLEDGE_ROOT / "graph/strategy_agent_graph_context.json"
MEMORY_JSON = AGENT_ROOT / "research_memory/latest_research_memory.json"
OUTPUT_JSON = AGENT_ROOT / "experiments/knowledge_guided_hypothesis_plan.json"
OUTPUT_MD = AGENT_ROOT / "experiments/knowledge_guided_hypothesis_ledger.md"
OUTPUT_EXPERIMENT = AGENT_ROOT / "experiments/knowledge_guided_strategy_experiment.json"


BLOCKER_TO_CONCEPTS = {
    "weak_profit_factor": ["entry_confirmation", "pullback", "actual_risk", "fees"],
    "negative_or_missing_return": ["edge", "regime_router", "fee_stress", "pullback"],
    "loss_exit_quality": ["stoploss", "invalidation", "actual_risk", "timebox"],
    "too_few_trades": ["regime_router", "entry_confirmation", "session_filter"],
    "too_few_matrix_trades": ["regime_router", "session", "time_filter"],
    "matrix_not_robust": ["market_cycle", "regime_router", "sessionless_market"],
    "fragile_matrix": ["market_cycle", "regime_router", "walk_forward"],
    "negative_after_cost": ["fees", "fee_stress", "scalp", "slippage"],
    "stress_cost_failure": ["fees", "slippage", "microstructure"],
    "cost_evidence_missing": ["fees", "funding", "fee_stress"],
    "bias_checks_missing": ["walk_forward", "out_of_sample", "falsification"],
    "lookahead_or_recursive_unverified": ["walk_forward", "out_of_sample", "falsification"],
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


def load_cards() -> list[dict[str, Any]]:
    graph_context = load_json(GRAPH_CONTEXT_JSON)
    if graph_context.get("cards"):
        return graph_context["cards"]
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(CARDS_DIR.glob("*.json"))]


def memory_avoid_rules(memory: dict[str, Any]) -> list[str]:
    return [item.get("memory_rule") for item in memory.get("avoid_patterns", []) if item.get("memory_rule")]


def active_blockers(memory: dict[str, Any]) -> list[str]:
    blockers = []
    for focus in memory.get("next_focus", []):
        blocker = focus.get("blocker")
        if blocker and blocker not in blockers:
            blockers.append(blocker)
    return blockers or ["weak_profit_factor", "matrix_not_robust", "negative_after_cost"]


def card_score(card: dict[str, Any], desired: list[str], avoid_rules: list[str]) -> tuple[int, bool]:
    concepts = set(card.get("concepts", []))
    text = json.dumps(card, ensure_ascii=False).lower()
    score = 0
    for concept in desired:
        if concept in concepts or concept.lower() in text:
            score += 5
    quality = card.get("source_quality")
    if quality == "high" or (isinstance(quality, dict) and quality.get("level") == "high"):
        score += 3
    if card.get("category") in {"risk", "crypto_adaptation"}:
        score += 1
    conflicts = []
    family = strategy_family(card)
    for rule in avoid_rules:
        rule_text = str(rule or "").lower()
        if family and family.lower() in rule_text:
            conflicts.append(rule)
        if "高杠杆" in rule_text and card.get("category") == "crypto_adaptation":
            conflicts.append(rule)
    blocked = bool(conflicts)
    if blocked:
        score -= 4
    return score, blocked


def choose_cards(cards: list[dict[str, Any]], blocker: str, avoid_rules: list[str]) -> list[dict[str, Any]]:
    desired = BLOCKER_TO_CONCEPTS.get(blocker, ["entry_confirmation", "regime_router", "risk_control"])
    scored = []
    for card in cards:
        score, blocked = card_score(card, desired, avoid_rules)
        if score > 0:
            scored.append((score, blocked, card))
    scored.sort(key=lambda item: (-item[0], item[1], card_id(item[2])))
    selected = [item[2] for item in scored[:3]]
    if len(selected) < 3:
        fallback = [card for card in cards if card not in selected and card.get("category") in {"risk", "definition", "crypto_adaptation"}]
        selected.extend(fallback[: 3 - len(selected)])
    return selected[:3]


def strategy_family(card: dict[str, Any]) -> str:
    return infer_family_from_card(card)


def card_id(card: dict[str, Any]) -> str:
    if card.get("id"):
        return card["id"]
    node = card.get("card_node") or ""
    return node.split(":", 1)[-1] if ":" in node else node


def hypothesis_from_cards(index: int, blocker: str, selected: list[dict[str, Any]], avoid_rules: list[str]) -> dict[str, Any]:
    primary = selected[0]
    family = strategy_family(primary)
    contract = family_contract(family)
    conflict = any(family.lower() in str(rule or "").lower() for rule in avoid_rules)
    status = "counterexample_or_variant_only" if conflict else "research_hypothesis"
    entry_rules = []
    exit_rules = []
    features = []
    risk_notes = []
    not_applicable = []
    concepts = []
    required_checks = []
    avoid_from_cards = []
    for card in selected:
        trans = card.get("freqtrade_translation") or {}
        entry_rules.extend(trans.get("entry_rules", []))
        entry_rules.extend(card.get("entry_rules", []))
        exit_rules.extend(trans.get("exit_rules", []))
        exit_rules.extend(card.get("exit_rules", []))
        features.extend(trans.get("features", []))
        not_applicable.extend(trans.get("not_applicable_regimes", []))
        risk_notes.extend(card.get("risk_notes", []))
        concepts.extend(card.get("concepts", []))
        required_checks.extend(card.get("required_checks", []))
        avoid_from_cards.extend(card.get("avoid_rules", []))
    return {
        "hypothesis_id": f"kg_{index:02d}_{blocker}_{family}",
        "status": status,
        "blocker_addressed": blocker,
        "source_cards": [card_id(card) for card in selected],
        "source_graph_nodes": [card.get("card_node") for card in selected if card.get("card_node")],
        "strategy_family": family,
        "strategy_family_code": contract["family_code"],
        "strategy_family_name": contract["family_name"],
        "strategy_family_direction": contract["direction"],
        "regime_contract": contract,
        "concepts": dedupe(concepts)[:12],
        "trading_idea": primary["strategy_hypothesis"],
        "quantified_entry_rules": dedupe(entry_rules)[:5],
        "quantified_exit_or_invalidation_rules": dedupe(exit_rules)[:4],
        "applicable_regimes": dedupe(
            sum(((card.get("freqtrade_translation") or {}).get("applicable_regimes", []) for card in selected), [])
            or contract["allowed_regimes"]
        )[:5],
        "not_applicable_regimes": dedupe(not_applicable or contract["disabled_regimes"])[:5],
        "freqtrade_feature_suggestions": dedupe(features)[:10],
        "backtest_requirements": dedupe(required_checks)
        or [
            "BTC/ETH only",
            "base and stress fee/slippage scenarios",
            "recursive-analysis and lookahead-analysis before any promotion",
            "regime matrix and walk-forward validation",
        ],
        "risk_notes": dedupe(risk_notes)[:5],
        "avoid_rules": dedupe(avoid_from_cards)[:6],
        "memory_guardrail": {
            "avoid_rules_considered": avoid_rules[:5],
            "decision": "downgraded_to_variant" if conflict else "allowed_for_research_only",
        },
        "graph_guardrail": {
            "uses_active_graph_cards_only": True,
            "does_not_promote_to_candidate_pool": True,
            "must_pass_required_checks": True,
            "must_declare_strategy_family_before_generation": True,
            "must_attribute_results_by_strategy_family": True,
        },
    }


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


def build_payload() -> dict[str, Any]:
    index = load_json(INDEX_JSON)
    memory = load_json(MEMORY_JSON)
    cards = load_cards()
    avoid = memory_avoid_rules(memory)
    blockers = active_blockers(memory)
    hypotheses = []
    used_primary: set[str] = set()
    for blocker in blockers:
        selected = choose_cards(cards, blocker, avoid)
        if not selected:
            continue
        if card_id(selected[0]) in used_primary and len(selected) > 1:
            selected = selected[1:] + selected[:1]
        used_primary.add(card_id(selected[0]))
        hypotheses.append(hypothesis_from_cards(len(hypotheses) + 1, blocker, selected, avoid))
        if len(hypotheses) >= 8:
            break
    return {
        "generated_at_utc": now_utc(),
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "source_artifacts": {
            "knowledge_index": rel(INDEX_JSON),
            "knowledge_graph_context": rel(GRAPH_CONTEXT_JSON) if GRAPH_CONTEXT_JSON.exists() else None,
            "research_memory": rel(MEMORY_JSON) if memory else None,
            "strategy_taxonomy": "user_data/strategy_research/strategy_taxonomy.py",
        },
        "strategy_taxonomy": taxonomy_summary(),
        "knowledge_summary": {
            "card_count": index.get("card_count"),
            "claim_count": index.get("claim_count"),
            "quality_summary": index.get("quality_summary"),
        },
        "experiment": {
            "id": "knowledge_guided_strategy_research",
            "description": "Planning handoff for graph-knowledge-guided strategy variants. Concrete Freqtrade code must be generated separately.",
            "timeframes": ["1m", "3m", "5m"],
            "timeranges": ["20260101-20260401", "20260401-20260622", "20260101-20260622"],
            "fee": 0.0006,
            "hypothesis_ids": [item["hypothesis_id"] for item in hypotheses],
            "strategy_families": sorted({item["strategy_family"] for item in hypotheses}),
            "strategy_family_contract_required": True,
            "family_attribution_required": True,
            "promotion_policy": "research_only_no_live_no_dryrun_promotion",
        },
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Knowledge-Guided Hypothesis Ledger",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Hypotheses: `{payload['hypothesis_count']}`",
        f"- Knowledge cards: `{payload['knowledge_summary'].get('card_count')}`",
        "",
        "| ID | Family | Direction | Status | Blocker | Allowed Regimes | Disabled Regimes | Trading Idea | Guardrail |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in payload["hypotheses"]:
        lines.append(
            "| {hypothesis_id} | {family} | {direction} | {status} | {blocker_addressed} | {allowed} | {disabled} | {idea} | {guardrail} |".format(
                hypothesis_id=item["hypothesis_id"],
                family=item["strategy_family"],
                direction=item["strategy_family_direction"],
                status=item["status"],
                blocker_addressed=item["blocker_addressed"],
                allowed=", ".join(item["regime_contract"]["allowed_regimes"]),
                disabled=", ".join(item["regime_contract"]["disabled_regimes"]),
                idea=item["trading_idea"],
                guardrail=item["memory_guardrail"]["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- This file only creates research hypotheses; it does not create live trading code.",
            "- Every generated strategy must first declare one canonical strategy family and regime contract.",
            "- Post-run attribution must aggregate evidence by strategy family, not only by individual strategy class.",
            "- Any generated strategy must pass backtesting, recursive-analysis, lookahead-analysis, regime matrix, fee/slippage stress, and promotion gate.",
            "- If research memory conflicts with a knowledge card family, the output is downgraded to a variant/counterexample experiment.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    write_json(OUTPUT_JSON, payload)
    write_json(OUTPUT_EXPERIMENT, payload["experiment"])
    write_markdown(payload)
    print(f"Wrote {rel(OUTPUT_JSON)}")
    print(f"Wrote {rel(OUTPUT_MD)}")
    print(f"Wrote {rel(OUTPUT_EXPERIMENT)}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
