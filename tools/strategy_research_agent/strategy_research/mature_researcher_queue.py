#!/usr/bin/env python3
"""Turn mature researcher decisions into a safe executable response queue."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
DECISION_PATH = AGENT_ROOT / "mature_researcher/latest_researcher_decision.json"
REPORT_DIR = AGENT_ROOT / "mature_researcher"
LATEST_QUEUE_JSON = REPORT_DIR / "latest_response_queue.json"
LATEST_QUEUE_MD = REPORT_DIR / "latest_response_queue.md"
LATEST_RECEIPT_JSON = REPORT_DIR / "latest_response_execution.json"
LATEST_RECEIPT_MD = REPORT_DIR / "latest_response_execution.md"


@dataclass
class QueueItem:
    priority: int
    strategy: str
    experiment: str
    objective: str
    command: list[str]
    success_gate: str
    promotion_block: str
    safe_to_execute: bool
    expected_runtime: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-next", action="store_true", help="Execute the highest priority safe queue item.")
    parser.add_argument("--dry-run", action="store_true", help="Write queue and receipt without executing.")
    return parser.parse_args()


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


def command_for(strategy: str, experiment: str) -> tuple[list[str], bool, str]:
    if experiment == "fee_sensitivity_grid":
        return (
            [
                "./.venv/bin/python",
                "user_data/strategy_research/run_research_agent.py",
                "--experiment",
                "user_data/strategy_research/experiments/short_cycle_scalping_experiment.json",
                "--strategy",
                strategy,
                "--timerange",
                "20260101-20260301",
                "--fee",
                "0.0015",
            ],
            True,
            "medium",
        )
    if experiment == "entry_delay_confirmation":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--behavior-experiments"],
            True,
            "short",
        )
    if experiment == "time_stop_exit_grid":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--behavior-variants"],
            True,
            "short",
        )
    if experiment == "low_leverage_edge_grid":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--memory-guided-hypotheses"],
            True,
            "short",
        )
    if experiment == "inverse_signal_retest":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--memory-guided-hypotheses"],
            True,
            "short",
        )
    if experiment == "single_condition_relaxation":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--memory-guided-hypotheses"],
            True,
            "short",
        )
    if experiment == "walk_forward_validation":
        return (["user_data/strategy_research/start_manual_research.sh", "--walk-forward"], True, "long")
    if experiment in {"recursive_analysis", "lookahead_analysis", "regime_matrix", "stress_cost_validation"}:
        return (["user_data/strategy_research/start_manual_research.sh", "--promotion-gate"], True, "short")
    return (
        ["user_data/strategy_research/start_manual_research.sh", "--mature-researcher"],
        False,
        "short",
    )


def objective_for(diagnosis: str, experiment: str) -> str:
    if experiment == "fee_sensitivity_grid":
        return "验证该策略在更高手续费压力下是否还有净 edge。"
    if experiment == "inverse_signal_retest":
        return "测试原信号是否更像反向指标，而不是继续调高杠杆。"
    if experiment == "low_leverage_edge_grid":
        return "先在低杠杆下证明信号本身有正期望。"
    if experiment == "entry_delay_confirmation":
        return "测试延迟和确认是否能改善入场后的不利波动。"
    if experiment == "time_stop_exit_grid":
        return "测试短持仓失效退出是否能减少亏损拖延。"
    if experiment == "single_condition_relaxation":
        return "样本不足时每次只放宽一个条件，避免复杂过拟合。"
    return diagnosis


def build_queue(decisions: dict[str, Any]) -> list[QueueItem]:
    items: list[QueueItem] = []
    for decision in decisions.get("top_decisions", []):
        experiments = decision.get("next_experiments", [])
        for rank, experiment in enumerate(experiments[:3]):
            command, safe, runtime = command_for(decision.get("strategy", ""), experiment)
            items.append(
                QueueItem(
                    priority=int(decision.get("priority", 0)) * 10 - rank,
                    strategy=decision.get("strategy", ""),
                    experiment=experiment,
                    objective=objective_for(decision.get("diagnosis", ""), experiment),
                    command=command,
                    success_gate=decision.get("success_gate", ""),
                    promotion_block=decision.get("promotion_block", ""),
                    safe_to_execute=safe,
                    expected_runtime=runtime,
                )
            )
    return sorted(items, key=lambda item: (-item.priority, item.strategy, item.experiment))


def build_payload() -> dict[str, Any]:
    decisions = load_json(DECISION_PATH)
    queue = build_queue(decisions)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_decision": rel(DECISION_PATH),
        "queue_count": len(queue),
        "safe_count": sum(1 for item in queue if item.safe_to_execute),
        "queue": [asdict(item) for item in queue],
        "policy": [
            "Execute one queue item at a time.",
            "Do not execute live trading or private-key operations.",
            "Promotion gates remain closed until the success gate is met by fresh evidence.",
        ],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Mature Researcher Response Queue",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Source decision: `{payload['source_decision']}`",
        f"- Queue items: `{payload['queue_count']}`",
        f"- Safe items: `{payload['safe_count']}`",
        "",
        "## Queue",
        "",
        "| Priority | Strategy | Experiment | Objective | Safe | Runtime | Command | Success Gate |",
        "|---:|---|---|---|---:|---|---|---|",
    ]
    for item in payload["queue"]:
        row = dict(item)
        row["command"] = " ".join(item["command"])
        lines.append(
            "| {priority} | {strategy} | {experiment} | {objective} | {safe_to_execute} | {expected_runtime} | `{command}` | {success_gate} |".format(
                **row,
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_queue(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"response_queue_{timestamp}.json"
    md_path = REPORT_DIR / f"response_queue_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_QUEUE_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_QUEUE_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_QUEUE_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_QUEUE_MD.relative_to(REPO_ROOT)}")
    print(f"Queue items: {payload['queue_count']}")


def write_receipt(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"response_execution_{timestamp}.json"
    md_path = REPORT_DIR / f"response_execution_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_RECEIPT_JSON.write_text(json_text, encoding="utf-8")
    lines = [
        "# Mature Researcher Response Execution",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Strategy: `{payload.get('item', {}).get('strategy', '')}`",
        f"- Experiment: `{payload.get('item', {}).get('experiment', '')}`",
        f"- Command: `{' '.join(payload.get('command') or [])}`",
        f"- Return code: `{payload.get('returncode')}`",
        "",
        "## Output Tail",
        "",
        "```text",
        payload.get("output_tail") or "",
        "```",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    LATEST_RECEIPT_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")


def execute_next(queue: list[dict[str, Any]], dry_run: bool) -> None:
    selected = next((item for item in queue if item.get("safe_to_execute")), None)
    if not selected:
        write_receipt(
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "status": "blocked",
                "mode": "dry_run" if dry_run else "execute",
                "item": None,
                "command": None,
                "returncode": None,
                "output_tail": "No safe queue item found.",
            }
        )
        return
    command = selected["command"]
    if dry_run:
        write_receipt(
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "status": "dry_run",
                "mode": "dry_run",
                "item": selected,
                "command": command,
                "returncode": None,
                "output_tail": "Command was not executed.",
            }
        )
        return
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    write_receipt(
        {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "status": "ok" if completed.returncode == 0 else "failed",
            "mode": "execute",
            "item": selected,
            "command": command,
            "returncode": completed.returncode,
            "output_tail": completed.stdout[-4000:],
        }
    )


def main() -> None:
    args = parse_args()
    payload = build_payload()
    write_queue(payload)
    if args.execute_next or args.dry_run:
        execute_next(payload["queue"], args.dry_run)


if __name__ == "__main__":
    main()
