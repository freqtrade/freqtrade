#!/usr/bin/env python3
"""Build a strategy library lineage map from research evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_DIR = AGENT_ROOT / "strategy_library"
LATEST_JSON = OUTPUT_DIR / "latest_strategy_lineage.json"
LATEST_MD = OUTPUT_DIR / "latest_strategy_lineage.md"

REGISTRIES = [
    ("manual_registry", AGENT_ROOT / "strategy_registry.json"),
    ("generated_variant", AGENT_ROOT / "experiments/generated_variant_registry.json"),
    ("source_translated", AGENT_ROOT / "experiments/source_translated_registry.json"),
    ("autonomous", AGENT_ROOT / "experiments/autonomous_strategy_registry.json"),
    ("iterative", AGENT_ROOT / "experiments/iterative_strategy_registry.json"),
    ("behavior_experiment", AGENT_ROOT / "experiments/behavior_experiment_strategy_registry.json"),
]


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


def index_by(items: list[dict[str, Any]], key: str = "strategy") -> dict[str, dict[str, Any]]:
    return {str(item[key]): item for item in items if item.get(key)}


def parent_of(item: dict[str, Any]) -> str | None:
    parent = item.get("base_strategy") or item.get("iterated_from")
    if parent and parent != item.get("name"):
        return str(parent)
    return None


def load_registry_nodes() -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for generation, path in REGISTRIES:
        payload = load_json(path)
        for item in payload.get("strategies", []):
            name = item.get("name")
            if not name:
                continue
            current = dict(item)
            current["_generation"] = generation
            current["_registry_path"] = rel(path)
            nodes[str(name)] = current
    return nodes


def load_pool_entries() -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}
    for status in ["candidates", "watchlist", "rejected"]:
        directory = AGENT_ROOT / status
        for path in sorted(directory.glob("*.json")):
            payload = load_json(path)
            strategy = payload.get("strategy") or payload.get("name")
            if not strategy:
                continue
            payload["_pool_status"] = status[:-1] if status == "candidates" else status
            payload["_pool_path"] = rel(path)
            pools[str(strategy)] = payload
    return pools


def lineage_path(name: str, parents: dict[str, str | None]) -> list[str]:
    path = [name]
    seen = {name}
    current = name
    while parents.get(current):
        parent = parents[current]
        if parent in seen:
            path.append(str(parent))
            break
        path.append(str(parent))
        seen.add(str(parent))
        current = str(parent)
    return list(reversed(path))


def recommended_state(
    pool_status: str,
    promotion: dict[str, Any] | None,
    scorecard: dict[str, Any] | None,
    failure: dict[str, Any] | None,
) -> str:
    if promotion and promotion.get("ready_for_manual_dryrun_review"):
        return "manual_dryrun_review"
    if pool_status == "candidate":
        return "research_candidate"
    if pool_status == "watchlist":
        return "watchlist"
    if pool_status == "rejected":
        return "archive_or_redesign"
    tier = (scorecard or {}).get("tier")
    if tier == "promotable_research_candidate":
        return "promotion_recheck"
    top_mode = (failure or {}).get("top_mode")
    if top_mode in {"insufficient_sample", "loss_exit_quality", "regime_fragility"}:
        return "redesign"
    if top_mode:
        return "blocked_research"
    return "unclassified_research"


def concise_metrics(pool: dict[str, Any] | None, scorecard: dict[str, Any] | None) -> dict[str, Any]:
    source = pool or scorecard or {}
    return {
        "trades": source.get("trades"),
        "total_profit_pct": source.get("total_profit_pct", source.get("base_return_pct")),
        "adjusted_return_pct": source.get("adjusted_return_pct"),
        "market_change_pct": source.get("market_change_pct"),
        "profit_factor": source.get("profit_factor"),
        "max_drawdown_pct": source.get("max_drawdown_pct"),
        "classification": source.get("classification"),
        "reasons": source.get("reasons", []),
    }


def build_payload() -> dict[str, Any]:
    registry_nodes = load_registry_nodes()
    pool_entries = load_pool_entries()
    assessment = load_json(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json")
    promotion_report = load_json(AGENT_ROOT / "promotion_reports/latest_promotion_report.json")
    behavior_report = load_json(AGENT_ROOT / "trade_behavior/latest_trade_behavior.json")
    failure_report = load_json(AGENT_ROOT / "failure_attribution/latest_failure_attribution.json")

    scorecards = index_by(assessment.get("scorecards", []))
    promotions = index_by(promotion_report.get("verdicts", []))
    behaviors = index_by(behavior_report.get("summaries", []))
    failures = index_by(failure_report.get("attributions", []))

    names = sorted(set(registry_nodes) | set(pool_entries) | set(scorecards) | set(promotions) | set(behaviors) | set(failures))
    parents = {name: parent_of(registry_nodes.get(name, {})) for name in names}
    children: dict[str, list[str]] = defaultdict(list)
    for name, parent in parents.items():
        if parent:
            children[parent].append(name)

    nodes = []
    by_state: dict[str, int] = defaultdict(int)
    by_generation: dict[str, int] = defaultdict(int)
    roots: dict[str, int] = defaultdict(int)
    for name in names:
        registry = registry_nodes.get(name, {})
        pool = pool_entries.get(name)
        scorecard = scorecards.get(name)
        promotion = promotions.get(name)
        behavior = behaviors.get(name)
        failure = failures.get(name)
        path = lineage_path(name, parents)
        root = path[0]
        pool_status = (pool or {}).get("_pool_status", "none")
        generation = registry.get("_generation", "unknown_pool")
        state = recommended_state(pool_status, promotion, scorecard, failure)
        top_failure = (failure or {}).get("failure_modes", [{}])[0] if (failure or {}).get("failure_modes") else {}
        node = {
            "name": name,
            "generation": generation,
            "family": registry.get("family"),
            "source": registry.get("source"),
            "parent": parents.get(name),
            "root": root,
            "children": sorted(children.get(name, [])),
            "lineage_path": path,
            "hypothesis": registry.get("hypothesis"),
            "risk_notes": registry.get("risk_notes"),
            "experiment_id": registry.get("experiment_id"),
            "source_review_id": registry.get("source_review_id"),
            "leverage_cap": registry.get("leverage_cap"),
            "regime": registry.get("regime"),
            "direction": registry.get("direction"),
            "change_set": registry.get("change_set"),
            "success_gate": registry.get("success_gate"),
            "pool_status": pool_status,
            "metrics": concise_metrics(pool, scorecard),
            "scorecard": {
                "tier": (scorecard or {}).get("tier"),
                "score": (scorecard or {}).get("score"),
                "primary_failures": (scorecard or {}).get("primary_failures", []),
            },
            "promotion": {
                "verdict": (promotion or {}).get("verdict"),
                "ready_for_manual_dryrun_review": (promotion or {}).get("ready_for_manual_dryrun_review", False),
                "blocks": (promotion or {}).get("blocks", []),
                "next_actions": (promotion or {}).get("next_actions", []),
            },
            "behavior": {
                "win_rate_pct": (behavior or {}).get("win_rate_pct"),
                "long_trades": (behavior or {}).get("long_trades"),
                "short_trades": (behavior or {}).get("short_trades"),
                "max_consecutive_losses": (behavior or {}).get("max_consecutive_losses"),
                "diagnostics": (behavior or {}).get("diagnostics", []),
            },
            "failure_attribution": {
                "top_mode": (failure or {}).get("top_mode"),
                "top_severity": top_failure.get("severity"),
                "linked_experiments": top_failure.get("linked_experiments", []),
                "recommendation": top_failure.get("recommendation"),
            },
            "recommended_state": state,
            "evidence_paths": sorted(
                item
                for item in [
                    registry.get("_registry_path"),
                    (pool or {}).get("_pool_path"),
                    rel(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json") if scorecard else None,
                    rel(AGENT_ROOT / "promotion_reports/latest_promotion_report.json") if promotion else None,
                    rel(AGENT_ROOT / "trade_behavior/latest_trade_behavior.json") if behavior else None,
                    rel(AGENT_ROOT / "failure_attribution/latest_failure_attribution.json") if failure else None,
                ]
                if item
            ),
        }
        nodes.append(node)
        by_state[state] += 1
        by_generation[generation] += 1
        roots[root] += 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "strategy_count": len(nodes),
        "root_count": len(roots),
        "summary": {
            "by_recommended_state": [{"state": key, "count": by_state[key]} for key in sorted(by_state)],
            "by_generation": [{"generation": key, "count": by_generation[key]} for key in sorted(by_generation)],
            "largest_roots": [
                {"root": key, "count": count}
                for key, count in sorted(roots.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        },
        "nodes": sorted(nodes, key=lambda item: (item["root"], len(item["lineage_path"]), item["name"])),
        "source_artifacts": {
            "registries": [rel(path) for _, path in REGISTRIES if path.exists()],
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
        f"- Strategies: `{payload['strategy_count']}`",
        f"- Roots: `{payload['root_count']}`",
        "",
        "## Recommended States",
        "",
        "| State | Count |",
        "|---|---:|",
    ]
    for item in payload["summary"]["by_recommended_state"]:
        lines.append(f"| {item['state']} | {item['count']} |")
    lines.extend(["", "## Largest Roots", "", "| Root | Count |", "|---|---:|"])
    for item in payload["summary"]["largest_roots"]:
        lines.append(f"| {item['root']} | {item['count']} |")
    lines.extend(
        [
            "",
            "## Strategy Nodes",
            "",
            "| Strategy | Gen | Root | Parent | Children | Pool | State | Score | Top Failure | Next Action |",
            "|---|---|---|---|---:|---|---|---:|---|---|",
        ]
    )
    for item in payload["nodes"]:
        next_actions = item.get("promotion", {}).get("next_actions", [])
        top_failure = item.get("failure_attribution", {}).get("top_mode") or ""
        score = item.get("scorecard", {}).get("score")
        lines.append(
            "| {name} | {generation} | {root} | {parent} | {children} | {pool} | {state} | {score} | {top_failure} | {next_action} |".format(
                name=item["name"],
                generation=item["generation"],
                root=item["root"],
                parent=item.get("parent") or "",
                children=len(item.get("children", [])),
                pool=item.get("pool_status"),
                state=item.get("recommended_state"),
                score="" if score is None else score,
                top_failure=top_failure,
                next_action=(next_actions[0] if next_actions else item.get("failure_attribution", {}).get("recommendation") or ""),
            )
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Lineage state is a research routing aid, not a live-trading permission.",
            "- Parent/root relationships come from local registries; missing parents are kept visible instead of inferred.",
            "- Promotion remains blocked unless the promotion gate explicitly says ready for manual dry-run review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = OUTPUT_DIR / f"strategy_lineage_{timestamp}.json"
    md_path = OUTPUT_DIR / f"strategy_lineage_{timestamp}.md"
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
