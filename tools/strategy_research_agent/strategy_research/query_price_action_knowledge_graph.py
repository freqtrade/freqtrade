#!/usr/bin/env python3
"""Query the local price-action knowledge graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = REPO_ROOT / "user_data/strategy_research/knowledge/graph"
NODES_JSON = GRAPH_DIR / "nodes.json"
EDGES_JSON = GRAPH_DIR / "edges.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Keyword query over graph labels and properties.")
    parser.add_argument("--type", dest="node_type", help="Filter node type, for example concept or knowledge_card.")
    parser.add_argument("--relation", help="Filter outgoing/incoming edges by relation.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not NODES_JSON.exists() or not EDGES_JSON.exists():
        raise SystemExit("Knowledge graph not built. Run build_price_action_knowledge_graph.py first.")
    return (
        json.loads(NODES_JSON.read_text(encoding="utf-8")),
        json.loads(EDGES_JSON.read_text(encoding="utf-8")),
    )


def node_text(node: dict[str, Any]) -> str:
    return json.dumps(node, ensure_ascii=False).lower()


def edge_text(edge: dict[str, Any], node_lookup: dict[str, dict[str, Any]]) -> str:
    payload = {
        **edge,
        "source_label": node_lookup.get(edge["source"], {}).get("label", ""),
        "target_label": node_lookup.get(edge["target"], {}).get("label", ""),
    }
    return json.dumps(payload, ensure_ascii=False).lower()


def score(value: str, terms: list[str]) -> int:
    if not terms:
        return 1
    return sum(value.count(term) for term in terms)


def main() -> None:
    args = parse_args()
    nodes, edges = load_graph()
    node_lookup = {node["id"]: node for node in nodes}
    terms = [term.lower() for term in args.query]

    node_hits = []
    for node in nodes:
        if args.node_type and node.get("type") != args.node_type:
            continue
        item_score = score(node_text(node), terms)
        if item_score:
            node_hits.append((item_score, node))
    node_hits.sort(key=lambda item: (-item[0], item[1]["id"]))

    edge_hits = []
    for edge in edges:
        if args.relation and edge.get("relation") != args.relation:
            continue
        item_score = score(edge_text(edge, node_lookup), terms)
        if item_score:
            edge_hits.append((item_score, edge))
    edge_hits.sort(key=lambda item: (-item[0], item[1]["id"]))

    payload = {
        "query": " ".join(args.query),
        "node_hits": [node for _, node in node_hits[: args.limit]],
        "edge_hits": [edge for _, edge in edge_hits[: args.limit]],
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"# Knowledge Graph Query: {' '.join(args.query) or '*'}")
    print("")
    print("## Nodes")
    print("")
    if not payload["node_hits"]:
        print("No matching nodes.")
    for node in payload["node_hits"]:
        status = []
        if node.get("active") is not None:
            status.append(f"active={node.get('active')}")
        if node.get("quarantined") is not None:
            status.append(f"quarantined={node.get('quarantined')}")
        status_text = f" ({', '.join(status)})" if status else ""
        print(f"- `{node['id']}` [{node['type']}]{status_text}: {node['label']}")
    print("")
    print("## Edges")
    print("")
    if not payload["edge_hits"]:
        print("No matching edges.")
    for edge in payload["edge_hits"]:
        source = node_lookup.get(edge["source"], {}).get("label", edge["source"])
        target = node_lookup.get(edge["target"], {}).get("label", edge["target"])
        print(f"- `{edge['relation']}`: `{edge['source']}` ({source}) -> `{edge['target']}` ({target})")


if __name__ == "__main__":
    main()
