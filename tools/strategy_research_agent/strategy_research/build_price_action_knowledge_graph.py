#!/usr/bin/env python3
"""Build a graph-structured price-action knowledge layer for the research agent."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = REPO_ROOT / "user_data/strategy_research/knowledge"
CARDS_DIR = KNOWLEDGE_ROOT / "knowledge_cards"
QUARANTINED_CARDS_DIR = KNOWLEDGE_ROOT / "knowledge_cards_quarantined"
GRAPH_DIR = KNOWLEDGE_ROOT / "graph"
NODES_JSON = GRAPH_DIR / "nodes.json"
EDGES_JSON = GRAPH_DIR / "edges.json"
GRAPH_INDEX_JSON = GRAPH_DIR / "graph_index.json"
GRAPH_MD = GRAPH_DIR / "knowledge_graph.md"
AGENT_CONTEXT_JSON = GRAPH_DIR / "strategy_agent_graph_context.json"
SOURCE_MANIFEST_JSON = KNOWLEDGE_ROOT / "raw_sources/public_web_sources_manifest.json"
BOOK_MANIFEST_JSON = KNOWLEDGE_ROOT / "raw_sources/books_extracted/xu_jiacong_price_action_manifest.json"
TRANSCRIPT_QUALITY_JSON = KNOWLEDGE_ROOT / "index/transcript_quality_report.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slug(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("|", "_")
    )


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def add_node(self, node_id: str, node_type: str, label: str, **props: Any) -> str:
        existing = self.nodes.get(node_id)
        payload = {
            "id": node_id,
            "type": node_type,
            "label": label,
            **{key: value for key, value in props.items() if value is not None},
        }
        if existing:
            existing.update({key: value for key, value in payload.items() if value not in (None, [], {})})
        else:
            self.nodes[node_id] = payload
        return node_id

    def add_edge(self, source: str, relation: str, target: str, **props: Any) -> str:
        edge_id = f"{source}|{relation}|{target}"
        self.edges[edge_id] = {
            "id": edge_id,
            "source": source,
            "relation": relation,
            "target": target,
            **{key: value for key, value in props.items() if value is not None},
        }
        return edge_id


def load_cards() -> list[dict[str, Any]]:
    cards = []
    for path in sorted(CARDS_DIR.glob("*.json")):
        card = read_json(path, {})
        card["_active"] = True
        card["_path"] = rel(path)
        cards.append(card)
    for path in sorted(QUARANTINED_CARDS_DIR.glob("*.json")):
        card = read_json(path, {})
        card["_active"] = False
        card["_path"] = rel(path)
        cards.append(card)
    return cards


def source_lookup() -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    web = read_json(SOURCE_MANIFEST_JSON, {"sources": []})
    for item in web.get("sources", []):
        sources[item["id"]] = {
            "source_kind": item.get("kind", "web"),
            "title": item.get("title", item["id"]),
            "author": item.get("author"),
            "url": item.get("url"),
            "status": item.get("fetch_status"),
            "local_text": item.get("text_extract"),
        }
    book = read_json(BOOK_MANIFEST_JSON, {})
    if book:
        sources[book["id"]] = {
            "source_kind": "book",
            "title": book.get("title"),
            "author": book.get("author"),
            "local_pdf": book.get("local_pdf"),
            "status": "user_confirmed_local_research",
        }
        for ref, topic in (book.get("usable_page_refs") or {}).items():
            page = None
            if "_p" in ref:
                try:
                    page = int(ref.rsplit("_p", 1)[1])
                except ValueError:
                    page = None
            sources[ref] = {
                "source_kind": "book_page",
                "title": f"{book.get('title', 'book')} p{page}",
                "author": book.get("author"),
                "page": page,
                "topic": topic,
                "parent_source": book["id"],
                "local_pdf": book.get("local_pdf"),
                "status": "user_confirmed_local_research",
            }
    quality = read_json(TRANSCRIPT_QUALITY_JSON, {"pages": []})
    for page in quality.get("pages", []):
        ref = f"bilibili_p{int(page['page']):03d}"
        sources[ref] = {
            "source_kind": "bilibili_transcript",
            "title": page.get("title"),
            "page": page.get("page"),
            "status": page.get("status"),
            "source_path": page.get("source_path"),
            "line_count": page.get("line_count"),
            "trading_keyword_hits": page.get("trading_keyword_hits"),
        }
    return sources


def add_card(g: GraphBuilder, card: dict[str, Any], sources: dict[str, dict[str, Any]]) -> None:
    active = bool(card.get("_active"))
    verification = card.get("verification_status") or {}
    quality = card.get("source_quality") or {}
    card_id = g.add_node(
        f"card:{card['id']}",
        "knowledge_card",
        card["title"],
        card_id=card["id"],
        category=card.get("category"),
        active=active,
        quarantined=not active or verification.get("quarantined", False),
        source_quality=quality.get("level"),
        path=card.get("_path"),
        knowledge=card.get("knowledge"),
        copyright_note=card.get("copyright_note"),
    )
    category_id = g.add_node(f"category:{card.get('category')}", "category", card.get("category", "unknown"))
    g.add_edge(card_id, "BELONGS_TO_CATEGORY", category_id)

    translation = card.get("freqtrade_translation") or {}
    family = translation.get("strategy_family", "unknown")
    family_id = g.add_node(f"strategy_family:{family}", "strategy_family", family)
    g.add_edge(card_id, "BELONGS_TO_STRATEGY_FAMILY", family_id)

    for concept in card.get("concepts", []):
        concept_id = g.add_node(f"concept:{concept}", "concept", concept)
        g.add_edge(card_id, "HAS_CONCEPT", concept_id)

    for ref in card.get("source_refs", []):
        src = sources.get(ref, {})
        source_id = g.add_node(
            f"source:{ref}",
            "source",
            src.get("title") or ref,
            source_ref=ref,
            source_kind=src.get("source_kind", "unknown"),
            status=src.get("status"),
            page=src.get("page"),
            url=src.get("url"),
            local_pdf=src.get("local_pdf"),
            source_path=src.get("source_path"),
            topic=src.get("topic"),
        )
        if src.get("parent_source"):
            parent = sources[src["parent_source"]]
            parent_id = g.add_node(
                f"source:{src['parent_source']}",
                "source",
                parent.get("title", src["parent_source"]),
                source_ref=src["parent_source"],
                source_kind=parent.get("source_kind"),
                local_pdf=parent.get("local_pdf"),
                status=parent.get("status"),
            )
            g.add_edge(source_id, "PART_OF_SOURCE", parent_id)
        g.add_edge(card_id, "DERIVED_FROM", source_id, source_quality=quality.get("level"))

    hypothesis_text = card.get("strategy_hypothesis") or ""
    hypothesis_id = g.add_node(
        f"hypothesis:{card['id']}",
        "strategy_hypothesis",
        card["title"],
        text=hypothesis_text,
        active=active,
    )
    g.add_edge(card_id, "PROPOSES_HYPOTHESIS", hypothesis_id)

    for feature in translation.get("features", []):
        feature_id = g.add_node(f"feature:{feature}", "freqtrade_feature", feature)
        g.add_edge(hypothesis_id, "USES_FEATURE", feature_id)

    for rule in translation.get("entry_rules", []):
        rule_id = g.add_node(f"entry_rule:{slug(rule)}", "entry_rule", rule)
        g.add_edge(hypothesis_id, "HAS_ENTRY_RULE", rule_id)

    for rule in translation.get("exit_rules", []):
        rule_id = g.add_node(f"exit_rule:{slug(rule)}", "exit_rule", rule)
        g.add_edge(hypothesis_id, "HAS_EXIT_RULE", rule_id)

    for regime in translation.get("applicable_regimes", []):
        regime_id = g.add_node(f"regime:{regime}", "market_regime", regime)
        g.add_edge(hypothesis_id, "APPLIES_TO_REGIME", regime_id)

    for regime in translation.get("not_applicable_regimes", []):
        regime_id = g.add_node(f"regime:{regime}", "market_regime", regime)
        g.add_edge(hypothesis_id, "NOT_APPLICABLE_IN", regime_id)

    for note in card.get("risk_notes", []):
        risk_id = g.add_node(f"risk:{slug(note)}", "risk_note", note)
        g.add_edge(card_id, "HAS_RISK_NOTE", risk_id)

    for rule in card.get("avoid_rules", []):
        avoid_id = g.add_node(f"avoid_rule:{slug(rule)}", "avoid_rule", rule)
        g.add_edge(card_id, "AVOIDS", avoid_id)

    for check in verification.get("required_checks", []):
        check_id = g.add_node(f"verification:{check}", "verification_check", check)
        g.add_edge(card_id, "MUST_VERIFY_BY", check_id)

    if not active:
        quarantine_id = g.add_node(
            "status:quarantined_weak_source",
            "status",
            "quarantined_weak_source",
            reason="Only low-confidence transcript evidence or otherwise not agent-active.",
        )
        g.add_edge(card_id, "QUARANTINED_AS", quarantine_id)


def add_semantic_edges(g: GraphBuilder) -> None:
    def exists(node_id: str) -> bool:
        return node_id in g.nodes

    semantic_pairs = [
        ("concept:pinbar", "REQUIRES_CONTEXT", "concept:key_level"),
        ("concept:false_breakout", "REQUIRES_CONFIRMATION", "concept:key_level_reclaim"),
        ("concept:false_breakout", "CAN_CREATE", "concept:trap"),
        ("concept:breakout_entry", "REQUIRES_CONFIRMATION", "concept:momentum_confirmation"),
        ("concept:limit_order", "USED_FOR", "concept:pullback"),
        ("concept:pullback", "REQUIRES_CONFIRMATION", "concept:trend_resume"),
        ("concept:scalp", "CONSTRAINED_BY", "concept:fees"),
        ("concept:scalp", "CONSTRAINED_BY", "concept:slippage"),
        ("concept:pyramiding", "REQUIRES", "concept:add_to_winner"),
        ("concept:pyramiding", "CONFLICTS_WITH", "concept:no_averaging_down"),
        ("concept:price_action", "PRIORITIZES", "concept:raw_price"),
        ("concept:trading_system", "REQUIRES", "concept:risk_process"),
        ("concept:review", "SUPPORTS", "concept:execution_audit"),
        ("concept:discipline", "PREVENTS", "concept:intuition_trading"),
    ]
    for source, relation, target in semantic_pairs:
        if exists(source) and exists(target):
            g.add_edge(source, relation, target, semantic=True)


def build_index(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    nodes_by_type: dict[str, list[str]] = defaultdict(list)
    edges_by_relation: dict[str, int] = Counter()
    adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
    reverse_adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
    concept_to_cards: dict[str, list[str]] = defaultdict(list)
    source_to_cards: dict[str, list[str]] = defaultdict(list)
    active_cards = []
    quarantined_cards = []

    node_lookup = {node["id"]: node for node in nodes}
    for node in nodes:
        nodes_by_type[node["type"]].append(node["id"])
        if node["type"] == "knowledge_card":
            if node.get("active"):
                active_cards.append(node["id"])
            else:
                quarantined_cards.append(node["id"])
    for edge in edges:
        edges_by_relation[edge["relation"]] += 1
        adjacency[edge["source"]].append({"relation": edge["relation"], "target": edge["target"]})
        reverse_adjacency[edge["target"]].append({"relation": edge["relation"], "source": edge["source"]})
        if edge["relation"] == "HAS_CONCEPT" and edge["source"].startswith("card:"):
            concept = node_lookup.get(edge["target"], {}).get("label", edge["target"])
            concept_to_cards[concept].append(edge["source"])
        if edge["relation"] == "DERIVED_FROM" and edge["source"].startswith("card:"):
            source_to_cards[edge["target"].replace("source:", "")].append(edge["source"])
    return {
        "generated_at_utc": now_utc(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "active_card_count": len(active_cards),
        "quarantined_card_count": len(quarantined_cards),
        "nodes_by_type": {key: sorted(value) for key, value in nodes_by_type.items()},
        "edges_by_relation": dict(sorted(edges_by_relation.items())),
        "concept_to_cards": {key: sorted(set(value)) for key, value in concept_to_cards.items()},
        "source_to_cards": {key: sorted(set(value)) for key, value in source_to_cards.items()},
        "adjacency": {key: value for key, value in adjacency.items()},
        "reverse_adjacency": {key: value for key, value in reverse_adjacency.items()},
        "artifacts": {
            "nodes": rel(NODES_JSON),
            "edges": rel(EDGES_JSON),
            "markdown": rel(GRAPH_MD),
            "agent_context": rel(AGENT_CONTEXT_JSON),
        },
    }


def write_markdown(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], index: dict[str, Any]) -> None:
    node_counter = Counter(node["type"] for node in nodes)
    lines = [
        "# Price Action Knowledge Graph",
        "",
        f"- Generated UTC: `{index['generated_at_utc']}`",
        f"- Nodes: `{index['node_count']}`",
        f"- Edges: `{index['edge_count']}`",
        f"- Active cards: `{index['active_card_count']}`",
        f"- Quarantined cards: `{index['quarantined_card_count']}`",
        "",
        "## Node Types",
        "",
        "| Type | Count |",
        "|---|---:|",
    ]
    for node_type, count in sorted(node_counter.items()):
        lines.append(f"| `{node_type}` | {count} |")
    lines.extend(["", "## Edge Relations", "", "| Relation | Count |", "|---|---:|"])
    for relation, count in index["edges_by_relation"].items():
        lines.append(f"| `{relation}` | {count} |")
    lines.extend(["", "## Active Knowledge Cards", "", "| Card | Family | Quality | Concepts |", "|---|---|---|---|"])
    for node in sorted(nodes, key=lambda item: item["id"]):
        if node["type"] != "knowledge_card" or not node.get("active"):
            continue
        concepts = [
            edge["target"].replace("concept:", "")
            for edge in edges
            if edge["source"] == node["id"] and edge["relation"] == "HAS_CONCEPT"
        ]
        family_edges = [edge for edge in edges if edge["source"] == node["id"] and edge["relation"] == "BELONGS_TO_STRATEGY_FAMILY"]
        family = family_edges[0]["target"].replace("strategy_family:", "") if family_edges else ""
        lines.append(f"| `{node['id']}` | `{family}` | `{node.get('source_quality')}` | {', '.join(concepts)} |")
    lines.extend(
        [
            "",
            "## Agent Boundary",
            "",
            "- Strategy Agent context uses active `knowledge_card` nodes only.",
            "- Quarantined cards remain visible in the graph for provenance but are not included in active strategy context.",
            "- Source nodes reference Bilibili transcript pages, public web snapshots, or user-confirmed local book pages; no long source text is copied into graph nodes.",
        ]
    )
    GRAPH_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_agent_context(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], index: dict[str, Any]) -> None:
    node_lookup = {node["id"]: node for node in nodes}
    active_cards = [node for node in nodes if node["type"] == "knowledge_card" and node.get("active")]
    cards = []
    for card in active_cards:
        outgoing = [edge for edge in edges if edge["source"] == card["id"]]
        hypothesis_edges = [edge for edge in outgoing if edge["relation"] == "PROPOSES_HYPOTHESIS"]
        hypothesis = node_lookup[hypothesis_edges[0]["target"]] if hypothesis_edges else {}
        cards.append(
            {
                "card_node": card["id"],
                "title": card["label"],
                "category": card.get("category"),
                "source_quality": card.get("source_quality"),
                "concepts": [
                    edge["target"].replace("concept:", "")
                    for edge in outgoing
                    if edge["relation"] == "HAS_CONCEPT"
                ],
                "source_refs": [
                    edge["target"].replace("source:", "")
                    for edge in outgoing
                    if edge["relation"] == "DERIVED_FROM"
                ],
                "knowledge": card.get("knowledge"),
                "strategy_hypothesis": hypothesis.get("text"),
                "entry_rules": [
                    node_lookup[edge["target"]]["label"]
                    for edge in edges
                    if edge["source"] == hypothesis.get("id") and edge["relation"] == "HAS_ENTRY_RULE"
                ],
                "exit_rules": [
                    node_lookup[edge["target"]]["label"]
                    for edge in edges
                    if edge["source"] == hypothesis.get("id") and edge["relation"] == "HAS_EXIT_RULE"
                ],
                "risk_notes": [
                    node_lookup[edge["target"]]["label"]
                    for edge in outgoing
                    if edge["relation"] == "HAS_RISK_NOTE"
                ],
                "avoid_rules": [
                    node_lookup[edge["target"]]["label"]
                    for edge in outgoing
                    if edge["relation"] == "AVOIDS"
                ],
                "required_checks": [
                    edge["target"].replace("verification:", "")
                    for edge in outgoing
                    if edge["relation"] == "MUST_VERIFY_BY"
                ],
            }
        )
    write_json(
        AGENT_CONTEXT_JSON,
        {
            "generated_at_utc": index["generated_at_utc"],
            "active_card_count": len(cards),
            "policy": {
                "research_only": True,
                "exclude_quarantined_cards": True,
                "must_backtest_before_promotion": True,
                "no_live_or_dryrun_config_changes": True,
            },
            "cards": cards,
        },
    )


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    cards = load_cards()
    sources = source_lookup()
    g = GraphBuilder()
    for card in cards:
        add_card(g, card, sources)
    add_semantic_edges(g)
    nodes = sorted(g.nodes.values(), key=lambda item: item["id"])
    edges = sorted(g.edges.values(), key=lambda item: item["id"])
    index = build_index(nodes, edges)
    write_json(NODES_JSON, nodes)
    write_json(EDGES_JSON, edges)
    write_json(GRAPH_INDEX_JSON, index)
    write_markdown(nodes, edges, index)
    write_agent_context(nodes, edges, index)
    print(f"Wrote {rel(NODES_JSON)}")
    print(f"Wrote {rel(EDGES_JSON)}")
    print(f"Wrote {rel(GRAPH_INDEX_JSON)}")
    print(f"Wrote {rel(GRAPH_MD)}")
    print(f"Wrote {rel(AGENT_CONTEXT_JSON)}")


if __name__ == "__main__":
    main()
