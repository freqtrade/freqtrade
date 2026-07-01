#!/usr/bin/env python3
"""Build strategy lineage from current registry and local research evidence."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strategy_taxonomy import classify_strategy_family


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_DIR = AGENT_ROOT / "strategy_library"
LATEST_JSON = OUTPUT_DIR / "latest_strategy_lineage.json"
LATEST_MD = OUTPUT_DIR / "latest_strategy_lineage.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_by_strategy(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row.get("strategy") or row.get("name") or "": row for row in rows if row.get("strategy") or row.get("name")}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_blocks(value: Any) -> list[str]:
    if value in (None, "", "none", "None", []):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [part.strip() for part in str(value).split(";") if part.strip() and part.strip().lower() != "none"]


def current_registry() -> dict[str, Any]:
    registry = load_json(AGENT_ROOT / "strategy_registry.json")
    if registry.get("strategies"):
        return registry
    latest = load_json(LATEST_JSON)
    if latest.get("strategies"):
        return {
            "generated_at_utc": latest.get("generated_at_utc"),
            "strategies": latest.get("strategies", []),
            "profile": {},
        }
    return {"strategies": []}


def load_family_gate() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    gate = load_json(AGENT_ROOT / "family_risk_gate/latest_family_risk_gate.json")
    verdicts = {item.get("strategy"): item for item in gate.get("verdicts", []) if item.get("strategy")}
    csv_path = Path(gate.get("source_csv") or "")
    if csv_path and not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    for name, row in load_csv_by_strategy(csv_path).items():
        verdicts.setdefault(name, {}).update({k: v for k, v in row.items() if v not in (None, "")})
    return verdicts, gate


def collect_pool_cards() -> list[tuple[str, Path, dict[str, Any]]]:
    cards: list[tuple[str, Path, dict[str, Any]]] = []
    for pool in ["candidates", "watchlist", "rejected"]:
        directory = AGENT_ROOT / pool
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            data = load_json(path)
            if data:
                cards.append((pool[:-1] if pool.endswith("s") else pool, path, data))
    return cards


def node_from_current(item: dict[str, Any], gate_row: dict[str, Any]) -> dict[str, Any]:
    name = item.get("name") or item.get("strategy") or "unknown"
    family = item.get("family") or gate_row.get("strategy_family") or classify_strategy_family(name, item.get("hypothesis"))
    state = item.get("state") or gate_row.get("state") or "research_candidate"
    ready = as_bool(gate_row.get("ready_for_manual_dryrun_review")) or "dryrun_candidate" in state
    blocks = normalize_blocks(gate_row.get("blocks"))
    evidence = item.get("evidence") or {}
    evidence_paths: list[str] = []
    for value in evidence.values():
        if isinstance(value, list):
            evidence_paths.extend(str(v) for v in value)
        elif value:
            evidence_paths.append(str(value))
    return {
        "name": name,
        "generation": "current_new_risk_gate",
        "family": family,
        "source": item.get("source"),
        "parent": item.get("parent") or item.get("base_strategy"),
        "root": item.get("root") or item.get("parent") or item.get("base_strategy") or name,
        "children": [],
        "lineage_path": [item.get("parent") or item.get("base_strategy"), name] if item.get("parent") or item.get("base_strategy") else [name],
        "hypothesis": item.get("hypothesis"),
        "risk_notes": item.get("risk_notes"),
        "experiment_id": item.get("experiment_id"),
        "source_review_id": item.get("source_review_id"),
        "leverage_cap": 50,
        "regime": item.get("regime"),
        "direction": "short" if "short" in family else ("long" if "long" in family else None),
        "change_set": item.get("change_set"),
        "success_gate": item.get("success_gate"),
        "pool_status": "current",
        "metrics": {
            "target_65d_guarded_pct": as_float(gate_row.get("target_65d_guarded_pct")),
            "target_30d_guarded_pct": as_float(gate_row.get("target_30d_guarded_pct")),
            "latest5_guarded_pct": as_float(gate_row.get("latest5_guarded_pct")),
            "target_65d_trades_taken": as_int(gate_row.get("target_65d_trades_taken")),
            "walk_forward_positive": as_int(gate_row.get("walk_forward_positive")),
            "walk_forward_total": as_int(gate_row.get("walk_forward_total")),
            "walk_forward_worst_guarded_pct": as_float(gate_row.get("walk_forward_worst_guarded_pct")),
            "hostile_raw_worst_pct": as_float(gate_row.get("hostile_raw_worst_pct")),
            "hostile_guarded_worst_pct": as_float(gate_row.get("hostile_guarded_worst_pct")),
            "evidence_modes": gate_row.get("evidence_modes"),
        },
        "scorecard": {
            "tier": state,
            "score": 90 if ready and not blocks else 65,
            "primary_failures": blocks,
        },
        "promotion": {
            "verdict": state,
            "ready_for_manual_dryrun_review": ready,
            "blocks": blocks,
            "next_actions": [] if ready else ["Close promotion blocks before dry-run review."],
        },
        "behavior": {"diagnostics": []},
        "failure_attribution": {
            "top_mode": None if ready else (blocks[0] if blocks else None),
            "top_severity": None,
            "linked_experiments": [],
            "recommendation": "" if ready else "Address current family-risk gate blocks.",
        },
        "recommended_state": state,
        "evidence_paths": evidence_paths,
    }


def node_from_pool(pool: str, path: Path, card: dict[str, Any]) -> dict[str, Any]:
    name = card.get("strategy") or card.get("name") or path.stem
    family = card.get("family") or classify_strategy_family(name, card.get("hypothesis"))
    reasons = card.get("reasons") or []
    classification = card.get("classification") or pool
    top_failure = reasons[0] if reasons else ("archived_historical_evidence" if pool == "rejected" else None)
    return {
        "name": name,
        "generation": card.get("source") or "historical_pool",
        "family": family,
        "source": card.get("source"),
        "parent": card.get("base_strategy"),
        "root": card.get("base_strategy") or name,
        "children": [],
        "lineage_path": [card.get("base_strategy"), name] if card.get("base_strategy") else [name],
        "hypothesis": card.get("hypothesis"),
        "risk_notes": card.get("risk_notes"),
        "experiment_id": card.get("experiment_id"),
        "source_review_id": card.get("source_review_id"),
        "leverage_cap": card.get("leverage_cap"),
        "regime": card.get("regime") or card.get("regime_label"),
        "direction": "short" if (card.get("short_trades") or 0) > (card.get("long_trades") or 0) else None,
        "change_set": card.get("change_set"),
        "success_gate": card.get("success_gate"),
        "pool_status": pool,
        "metrics": {
            "trades": card.get("trades"),
            "total_profit_pct": card.get("total_profit_pct"),
            "adjusted_return_pct": card.get("adjusted_return_pct"),
            "market_change_pct": card.get("market_change_pct"),
            "profit_factor": card.get("profit_factor"),
            "max_drawdown_pct": card.get("max_drawdown_pct"),
            "classification": classification,
            "reasons": reasons,
        },
        "scorecard": {
            "tier": "reject_or_archive" if pool == "rejected" else pool,
            "score": 30 if pool == "rejected" else 55,
            "primary_failures": reasons,
        },
        "promotion": {
            "verdict": classification,
            "ready_for_manual_dryrun_review": False,
            "blocks": reasons,
            "next_actions": [],
        },
        "behavior": {
            "long_trades": card.get("long_trades"),
            "short_trades": card.get("short_trades"),
            "diagnostics": [],
        },
        "failure_attribution": {
            "top_mode": top_failure,
            "top_severity": None,
            "linked_experiments": [],
            "recommendation": "Historical evidence only; do not reactivate without a new hypothesis." if pool == "rejected" else "",
        },
        "recommended_state": "archive_or_redesign" if pool == "rejected" else pool,
        "evidence_paths": [rel(path)],
    }


def attach_children(nodes: list[dict[str, Any]]) -> None:
    by_root: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        parent = node.get("parent")
        if parent:
            by_root[parent].append(node["name"])
    for node in nodes:
        node["children"] = sorted(set(by_root.get(node["name"], [])))


def build_payload() -> dict[str, Any]:
    registry = current_registry()
    gate_rows, gate = load_family_gate()
    current_nodes = [node_from_current(item, gate_rows.get(item.get("name") or item.get("strategy") or "", {})) for item in registry.get("strategies", [])]
    current_names = {item["name"] for item in current_nodes}
    historical_nodes = [
        node_from_pool(pool, path, card)
        for pool, path, card in collect_pool_cards()
        if (card.get("strategy") or card.get("name") or path.stem) not in current_names
    ]
    nodes = current_nodes + historical_nodes
    attach_children(nodes)
    by_state = Counter(node.get("recommended_state") or "" for node in nodes)
    by_generation = Counter(node.get("generation") or "" for node in nodes)
    roots = Counter(node.get("root") or node.get("name") for node in nodes)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "strategy_count": len(nodes),
        "root_count": len(roots),
        "current_scope": "current registry candidates plus historical evidence pools",
        "policy": "Current registry strategies are active. Historical pool nodes are evidence only and must not be treated as active candidates.",
        "summary": {
            "by_recommended_state": dict(by_state),
            "by_generation": dict(by_generation),
            "largest_roots": [{"root": root, "count": count} for root, count in roots.most_common(10)],
        },
        "strategies": registry.get("strategies", []),
        "fixed_risk_policy": gate.get("fixed_risk_policy", {}),
        "risk_controls": gate.get("risk_controls", {}),
        "nodes": nodes,
        "source_artifacts": {
            "registry": rel(AGENT_ROOT / "strategy_registry.json"),
            "family_risk_gate": rel(AGENT_ROOT / "family_risk_gate/latest_family_risk_gate.json") if gate else None,
            "candidate_dir": rel(AGENT_ROOT / "candidates"),
            "watchlist_dir": rel(AGENT_ROOT / "watchlist"),
            "rejected_dir": rel(AGENT_ROOT / "rejected"),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strategy Library Lineage",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Strategy nodes: `{payload['strategy_count']}`",
        f"- Roots: `{payload['root_count']}`",
        f"- Scope: {payload['current_scope']}",
        "",
        "## Policy",
        "",
        f"- {payload['policy']}",
        "- Deleting old experiment outputs must not delete this lineage builder.",
        "",
        "## Nodes",
        "",
        "| Strategy | Gen | Family | Root | Parent | Children | Pool | State | Score | Top Failure |",
        "|---|---|---|---|---|---:|---|---|---:|---|",
    ]
    for item in payload["nodes"]:
        score = item.get("scorecard", {}).get("score")
        lines.append(
            "| {name} | {generation} | {family} | {root} | {parent} | {children} | {pool} | {state} | {score} | {failure} |".format(
                name=item.get("name", ""),
                generation=item.get("generation", ""),
                family=item.get("family", ""),
                root=item.get("root", ""),
                parent=item.get("parent") or "",
                children=len(item.get("children", [])),
                pool=item.get("pool_status", ""),
                state=item.get("recommended_state", ""),
                score="" if score is None else score,
                failure=item.get("failure_attribution", {}).get("top_mode") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Current Risk Contract",
            "",
            f"- Fixed risk policy: `{json.dumps(payload.get('fixed_risk_policy', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- Risk controls: `{json.dumps(payload.get('risk_controls', {}), ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    ts = payload["generated_at_utc"]
    out_json = OUTPUT_DIR / f"strategy_lineage_{ts}.json"
    out_md = OUTPUT_DIR / f"strategy_lineage_{ts}.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    shutil.copyfile(out_json, LATEST_JSON)
    shutil.copyfile(out_md, LATEST_MD)
    print(f"Wrote {rel(out_json)}")
    print(f"Wrote {rel(out_md)}")
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")


if __name__ == "__main__":
    main()
