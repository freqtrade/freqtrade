#!/usr/bin/env python3
"""Build a diversified strategy-family experiment from generated local strategies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_DIR = AGENT_ROOT / "family_diversity"
PLAN_JSON = OUTPUT_DIR / "latest_family_diversity_plan.json"
PLAN_MD = OUTPUT_DIR / "latest_family_diversity_plan.md"
EXPERIMENT_PATH = AGENT_ROOT / "experiments/family_diversity_experiment.json"

REGISTRY_PATHS = [
    AGENT_ROOT / "experiments/autonomous_strategy_registry.json",
    AGENT_ROOT / "experiments/iterative_strategy_registry.json",
    AGENT_ROOT / "experiments/behavior_experiment_strategy_registry.json",
    AGENT_ROOT / "experiments/memory_guided_strategy_registry.json",
    AGENT_ROOT / "experiments/source_translated_registry.json",
]

BUCKETS = {
    "cost_pressure": ("cost", "defensive", "stop_loss", "cooldown", "walk_forward"),
    "mean_reversion": ("mean", "range", "fade"),
    "short_momentum": ("momentum", "breakout", "squeeze", "failed-bounce", "short"),
    "trend": ("trend", "pullback", "continuation"),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_strategies() -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen = set()
    for registry_path in REGISTRY_PATHS:
        payload = load_json_if_exists(registry_path)
        for item in payload.get("strategies", []):
            name = item.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            row = dict(item)
            row["registry"] = rel(registry_path)
            row["bucket"] = classify_bucket(row)
            collected.append(row)
    return collected


def classify_bucket(item: dict[str, Any]) -> str:
    core_text = " ".join(str(item.get(key, "")) for key in ("name", "family", "source", "blocker")).lower()
    if any(needle in core_text for needle in BUCKETS["cost_pressure"]):
        return "cost_pressure"
    if any(needle in core_text for needle in BUCKETS["mean_reversion"]):
        return "mean_reversion"
    if any(needle in core_text for needle in BUCKETS["trend"]):
        return "trend"
    if any(needle in core_text for needle in BUCKETS["short_momentum"]):
        return "short_momentum"
    text = f"{core_text} {item.get('hypothesis', '')}".lower()
    for bucket, needles in BUCKETS.items():
        if any(needle in text for needle in needles):
            return bucket
    return "other"


def select_diverse(strategies: list[dict[str, Any]], per_bucket: int, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_names = set()
    for bucket in BUCKETS:
        bucket_rows = [item for item in strategies if item["bucket"] == bucket]
        for item in bucket_rows[:per_bucket]:
            selected.append(item)
            selected_names.add(item["name"])
    for item in strategies:
        if len(selected) >= limit:
            break
        if item["name"] not in selected_names:
            selected.append(item)
            selected_names.add(item["name"])
    return selected[:limit]


def build_payload(per_bucket: int, limit: int, timerange: str, smoke_timerange: str) -> dict[str, Any]:
    strategies = collect_strategies()
    selected = select_diverse(strategies, per_bucket, limit)
    grouped = {bucket: [item["name"] for item in selected if item["bucket"] == bucket] for bucket in [*BUCKETS, "other"]}
    experiment = {
        "id": "family_diversity_strategy_lab",
        "title": "Diversified strategy-family research iteration",
        "profile_ref": "strategy_registry.json",
        "strategy_path": "user_data/strategies/research_generated",
        "timeframes": ["1m"],
        "timeranges": [timerange],
        "matrix": {
            "timeranges": [
                {"name": "smoke", "label": "Recent smoke", "timerange": smoke_timerange},
                {"name": "full", "label": "Full local sample", "timerange": timerange},
            ]
        },
        "fee": 0.0005,
        "strategies": [item["name"] for item in selected],
        "strategy_groups": {key: value for key, value in grouped.items() if value},
        "checks": {"backtesting": True, "recursive_analysis": False, "lookahead_analysis": False},
        "notes": [
            "Generated after no-candidate-yield diagnosis.",
            "Keeps each iteration broad across trend, mean reversion, short momentum, and cost-pressure hypotheses.",
            "Uses only already generated local research strategies; no external code is executed.",
        ],
    }
    return {
        "generated_at_utc": utc_stamp(),
        "trigger": "no_candidate_yield",
        "strategy_count_seen": len(strategies),
        "strategy_count_selected": len(selected),
        "bucket_counts": {bucket: len([item for item in selected if item["bucket"] == bucket]) for bucket in [*BUCKETS, "other"]},
        "selected_strategies": selected,
        "experiment": experiment,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Family Diversity Experiment Plan",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Trigger: `{payload['trigger']}`",
        f"- Seen strategies: `{payload['strategy_count_seen']}`",
        f"- Selected strategies: `{payload['strategy_count_selected']}`",
        f"- Experiment: `{rel(EXPERIMENT_PATH)}`",
        "",
        "## Bucket Counts",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for bucket, count in payload["bucket_counts"].items():
        lines.append(f"| {bucket} | {count} |")
    lines.extend(["", "## Selected Strategies", "", "| Strategy | Bucket | Family | Source | Registry |", "|---|---|---|---|---|"])
    for item in payload["selected_strategies"]:
        lines.append(
            "| {name} | {bucket} | {family} | {source} | {registry} |".format(
                name=item.get("name", ""),
                bucket=item.get("bucket", ""),
                family=item.get("family", ""),
                source=item.get("source", ""),
                registry=item.get("registry", ""),
            )
        )
    PLAN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload(per_bucket=2, limit=12, timerange="20240101-20260622", smoke_timerange="20260101-20260622")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    EXPERIMENT_PATH.write_text(json.dumps(payload["experiment"], indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload)
    print(f"Wrote {rel(PLAN_JSON)}")
    print(f"Wrote {rel(PLAN_MD)}")
    print(f"Wrote {rel(EXPERIMENT_PATH)}")


if __name__ == "__main__":
    main()
