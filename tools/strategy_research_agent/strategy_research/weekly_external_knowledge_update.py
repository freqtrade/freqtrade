#!/usr/bin/env python3
"""Run the weekly external knowledge update layer.

This coordinator refreshes external/source knowledge and rebuilds the Agent
brain artifacts. It is research-only and deliberately does not download videos,
read exchange keys, modify trading configs, or promote strategies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
UPDATE_DIR = AGENT_ROOT / "knowledge_updates"
LATEST_JSON = UPDATE_DIR / "latest_weekly_knowledge_update.json"
LATEST_MD = UPDATE_DIR / "latest_weekly_knowledge_update.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-bilibili",
        action="store_true",
        help="Also try authenticated Bilibili subtitle refresh. Never downloads video.",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip final report/dashboard refresh.",
    )
    return parser.parse_args()


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


def run_step(name: str, command: list[str], optional: bool = False) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    status = "ok" if completed.returncode == 0 else ("warning" if optional else "failed")
    return {
        "name": name,
        "status": status,
        "optional": optional,
        "command": command,
        "returncode": completed.returncode,
        "output_tail": completed.stdout[-3000:],
    }


def count_state_before_after() -> dict[str, Any]:
    graph = load_json(AGENT_ROOT / "knowledge/graph/strategy_agent_graph_context.json")
    source_discovery = load_json(AGENT_ROOT / "source_discovery/latest_source_discovery.json")
    memory = load_json(AGENT_ROOT / "research_memory/latest_research_memory.json")
    consolidation = load_json(AGENT_ROOT / "consolidation/latest_research_consolidation.json")
    return {
        "active_knowledge_cards": graph.get("active_card_count", 0),
        "source_candidates": source_discovery.get("candidate_count", 0),
        "memory_version": memory.get("memory_version"),
        "solidified_rules": consolidation.get("observed_counts", {}).get("solidified_rules", 0),
    }


def build_payload(args: argparse.Namespace, started_at: str, before: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    after = count_state_before_after()
    failed_required = [step for step in steps if step["status"] == "failed" and not step["optional"]]
    warning_steps = [step for step in steps if step["status"] == "warning"]
    graph = load_json(AGENT_ROOT / "knowledge/graph/strategy_agent_graph_context.json")
    source_discovery = load_json(AGENT_ROOT / "source_discovery/latest_source_discovery.json")
    knowledge_plan = load_json(AGENT_ROOT / "experiments/knowledge_guided_hypothesis_plan.json")
    memory_plan = load_json(AGENT_ROOT / "experiments/memory_guided_hypothesis_plan.json")
    return {
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "status": "failed" if failed_required else ("warning" if warning_steps else "ok"),
        "with_bilibili": args.with_bilibili,
        "research_only": True,
        "before": before,
        "after": after,
        "delta": {
            key: after.get(key, 0) - before.get(key, 0)
            for key in ["active_knowledge_cards", "source_candidates", "solidified_rules"]
        },
        "steps": steps,
        "knowledge_summary": {
            "active_card_count": graph.get("active_card_count", 0),
            "knowledge_hypotheses": knowledge_plan.get("hypothesis_count", 0),
            "memory_hypotheses": memory_plan.get("hypothesis_count", 0),
            "source_action_summary": source_discovery.get("action_summary", []),
        },
        "safety_policy": {
            "downloads_video": False,
            "reads_exchange_keys": False,
            "modifies_trading_config": False,
            "promotes_strategies": False,
            "external_sources_are_quarantined": True,
        },
        "source_artifacts": {
            "source_discovery": rel(AGENT_ROOT / "source_discovery/latest_source_discovery.md"),
            "knowledge_report": rel(AGENT_ROOT / "knowledge/latest_price_action_knowledge_layer_report.md"),
            "knowledge_graph": rel(AGENT_ROOT / "knowledge/graph/knowledge_graph.md"),
            "graph_context": rel(AGENT_ROOT / "knowledge/graph/strategy_agent_graph_context.json"),
            "research_memory": rel(AGENT_ROOT / "research_memory/latest_research_memory.md"),
            "consolidation": rel(AGENT_ROOT / "consolidation/latest_research_consolidation.md"),
            "operating_rules": rel(AGENT_ROOT / "consolidation/agent_operating_rules.json"),
            "dashboard": rel(AGENT_ROOT / "dashboard/index.html"),
        },
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Weekly External Knowledge Update",
        "",
        f"- Started UTC: `{payload['started_at_utc']}`",
        f"- Finished UTC: `{payload['finished_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Bilibili refresh attempted: `{payload['with_bilibili']}`",
        "",
        "## Summary",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("active_knowledge_cards", "Active knowledge cards"),
        ("source_candidates", "External source candidates"),
        ("solidified_rules", "Solidified rules"),
    ]:
        lines.append(
            f"| {label} | {payload['before'].get(key, 0)} | {payload['after'].get(key, 0)} | {payload['delta'].get(key, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Steps",
            "",
            "| Step | Status | Optional | Return Code |",
            "|---|---|---|---:|",
        ]
    )
    for step in payload["steps"]:
        lines.append(f"| {step['name']} | {step['status']} | {step['optional']} | {step['returncode']} |")
    lines.extend(
        [
            "",
            "## Knowledge Output",
            "",
            f"- Active cards: `{payload['knowledge_summary']['active_card_count']}`",
            f"- Knowledge hypotheses: `{payload['knowledge_summary']['knowledge_hypotheses']}`",
            f"- Memory hypotheses: `{payload['knowledge_summary']['memory_hypotheses']}`",
            "",
            "## Safety Boundary",
            "",
            "- No live trading.",
            "- No exchange API keys.",
            "- No dry-run/live config changes.",
            "- No video downloads.",
            "- New external knowledge can only create research hypotheses until evidence gates pass.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, path in payload["source_artifacts"].items():
        lines.append(f"- {key}: `{path}`")
    LATEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    UPDATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = payload["finished_at_utc"]
    json_path = UPDATE_DIR / f"weekly_knowledge_update_{stamp}.json"
    md_path = UPDATE_DIR / f"weekly_knowledge_update_{stamp}.md"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(text, encoding="utf-8")
    LATEST_JSON.write_text(text, encoding="utf-8")
    write_markdown(payload)
    md_path.write_text(LATEST_MD.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")


def main() -> None:
    args = parse_args()
    python = str(REPO_ROOT / ".venv/bin/python")
    started_at = now_utc()
    before = count_state_before_after()
    steps: list[dict[str, Any]] = []
    steps.append(run_step("external_source_scout", [python, "user_data/strategy_research/scout_external_sources.py"]))
    steps.append(run_step("external_source_review", [python, "user_data/strategy_research/review_sources.py"]))
    if args.with_bilibili:
        steps.append(
            run_step(
                "bilibili_transcript_refresh",
                [python, "user_data/strategy_research/fetch_bilibili_transcripts.py"],
                optional=True,
            )
        )
    steps.extend(
        [
            run_step("price_action_knowledge_base", [python, "user_data/strategy_research/build_price_action_knowledge_base.py"]),
            run_step("price_action_knowledge_layer", [python, "user_data/strategy_research/build_price_action_knowledge_layer.py"]),
            run_step("price_action_knowledge_graph", [python, "user_data/strategy_research/build_price_action_knowledge_graph.py"]),
            run_step("strategy_lineage", [python, "user_data/strategy_research/build_strategy_lineage.py"]),
            run_step("research_memory", [python, "user_data/strategy_research/build_research_memory.py"]),
            run_step("knowledge_guided_hypotheses", [python, "user_data/strategy_research/plan_knowledge_guided_hypotheses.py"]),
            run_step("memory_guided_hypotheses", [python, "user_data/strategy_research/plan_memory_guided_hypotheses.py"]),
            run_step("research_consolidation", [python, "user_data/strategy_research/build_research_consolidation.py"]),
        ]
    )
    if not args.skip_dashboard:
        steps.append(run_step("dashboard_refresh", [python, "user_data/strategy_research/run_research_agent.py", "--skip-backtests"]))
    payload = build_payload(args, started_at, before, steps)
    write_outputs(payload)
    if payload["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
