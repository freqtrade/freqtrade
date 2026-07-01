#!/usr/bin/env python3
"""Turn mature researcher decisions into a safe executable response queue."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
DECISION_PATH = AGENT_ROOT / "mature_researcher/latest_researcher_decision.json"
STRATEGY_REGISTRY_PATH = AGENT_ROOT / "strategy_registry.json"
REPORT_DIR = AGENT_ROOT / "mature_researcher"
LATEST_QUEUE_JSON = REPORT_DIR / "latest_response_queue.json"
LATEST_QUEUE_MD = REPORT_DIR / "latest_response_queue.md"
LATEST_RECEIPT_JSON = REPORT_DIR / "latest_response_execution.json"
LATEST_RECEIPT_MD = REPORT_DIR / "latest_response_execution.md"
EXECUTION_HISTORY_JSONL = REPORT_DIR / "response_execution_history.jsonl"
EXPERIMENT_DIR = AGENT_ROOT / "experiments"


@dataclass
class QueueItem:
    key: str
    priority: int
    strategy: str
    experiment: str
    objective: str
    command: list[str]
    success_gate: str
    promotion_block: str
    safe_to_execute: bool
    expected_runtime: str
    attempts_24h: int
    last_executed_utc: str | None
    cooldown_until_utc: str | None
    skip_reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-next", action="store_true", help="Execute the highest priority safe queue item.")
    parser.add_argument("--dry-run", action="store_true", help="Write queue and receipt without executing.")
    parser.add_argument("--cooldown-hours", type=float, default=6.0, help="Skip queue items executed in the last N hours.")
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(value: datetime | None = None) -> str:
    return (value or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def parse_stamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def queue_key(strategy: str, experiment: str, command: list[str]) -> str:
    return "::".join([strategy, experiment, " ".join(command)])


def load_execution_history() -> list[dict[str, Any]]:
    if not EXECUTION_HISTORY_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in EXECUTION_HISTORY_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def recent_history_by_key(history: list[dict[str, Any]], now: datetime, hours: float = 24.0) -> dict[str, list[dict[str, Any]]]:
    cutoff = now - timedelta(hours=hours)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in history:
        executed_at = parse_stamp(item.get("generated_at_utc"))
        if not executed_at or executed_at < cutoff:
            continue
        key = item.get("key")
        if not key:
            continue
        grouped.setdefault(key, []).append(item)
    return grouped


def append_history(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with EXECUTION_HISTORY_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_current_registry_strategies() -> set[str]:
    payload = load_json(STRATEGY_REGISTRY_PATH)
    strategies = payload.get("strategies", [])
    current: set[str] = set()
    if not isinstance(strategies, list):
        return current
    for item in strategies:
        if not isinstance(item, dict):
            continue
        name = item.get("strategy") or item.get("strategy_name") or item.get("name")
        if name:
            current.add(str(name))
    return current


def experiment_contains_strategy(path: Path, strategy: str) -> bool:
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return False
    strategies = payload.get("strategies", [])
    if not isinstance(strategies, list):
        return False
    return strategy in {str(item) for item in strategies}


def find_strategy_experiment(strategy: str) -> Path | None:
    if not strategy or not EXPERIMENT_DIR.exists():
        return None
    for path in sorted(EXPERIMENT_DIR.glob("*_experiment.json")):
        if experiment_contains_strategy(path, strategy):
            return path
    return None


def validate_command(command: list[str], safe: bool, skip_reason: str | None) -> tuple[bool, str | None]:
    if not safe:
        return safe, skip_reason
    for index, token in enumerate(command):
        if token != "--experiment" or index + 1 >= len(command):
            continue
        experiment_path = REPO_ROOT / command[index + 1]
        if not experiment_path.exists():
            return False, f"missing experiment file: {rel(experiment_path)}"
    return safe, skip_reason


def command_for(strategy: str, experiment: str) -> tuple[list[str], bool, str, str | None]:
    if experiment == "fee_sensitivity_grid":
        experiment_path = find_strategy_experiment(strategy)
        if not experiment_path:
            return (
                [],
                False,
                "medium",
                f"no existing experiment contains strategy: {strategy}",
            )
        return (
            [
                "./.venv/bin/python",
                "user_data/strategy_research/run_research_agent.py",
                "--experiment",
                rel(experiment_path),
                "--strategy",
                strategy,
                "--timerange",
                "20260101-20260301",
                "--fee",
                "0.0015",
            ],
            True,
            "medium",
            None,
        )
    if experiment == "entry_delay_confirmation":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--memory-guided-hypotheses"],
            True,
            "short",
            None,
        )
    if experiment == "time_stop_exit_grid":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--memory-guided-hypotheses"],
            True,
            "short",
            None,
        )
    if experiment == "fixed_50x_signal_edge_check":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--memory-guided-hypotheses"],
            True,
            "short",
            None,
        )
    if experiment == "inverse_signal_retest":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--memory-guided-hypotheses"],
            True,
            "short",
            None,
        )
    if experiment == "single_condition_relaxation":
        return (
            ["user_data/strategy_research/start_manual_research.sh", "--memory-guided-hypotheses"],
            True,
            "short",
            None,
        )
    if experiment == "walk_forward_validation":
        return (["user_data/strategy_research/start_manual_research.sh", "--walk-forward"], True, "long", None)
    if experiment in {"recursive_analysis", "lookahead_analysis", "regime_matrix", "stress_cost_validation"}:
        return (["user_data/strategy_research/start_manual_research.sh", "--promotion-gate"], True, "short", None)
    return (
        ["user_data/strategy_research/start_manual_research.sh", "--mature-researcher"],
        False,
        "short",
        f"no command mapping for experiment: {experiment}",
    )


def objective_for(diagnosis: str, experiment: str) -> str:
    if experiment == "fee_sensitivity_grid":
        return "验证该策略在更高手续费压力下是否还有净 edge。"
    if experiment == "inverse_signal_retest":
        return "测试原信号是否更像反向指标，而不是继续调高杠杆。"
    if experiment == "fixed_50x_signal_edge_check":
        return "在固定 50x 合约口径下证明信号本身有正期望。"
    if experiment == "entry_delay_confirmation":
        return "测试延迟和确认是否能改善入场后的不利波动。"
    if experiment == "time_stop_exit_grid":
        return "测试短持仓失效退出是否能减少亏损拖延。"
    if experiment == "single_condition_relaxation":
        return "样本不足时每次只放宽一个条件，避免复杂过拟合。"
    return diagnosis


def build_queue(decisions: dict[str, Any], cooldown_hours: float = 6.0) -> list[QueueItem]:
    items: list[QueueItem] = []
    now = utc_now()
    history = load_execution_history()
    recent_24h = recent_history_by_key(history, now, 24.0)
    cooldown_recent = recent_history_by_key(history, now, cooldown_hours)
    current_strategies = load_current_registry_strategies()
    seen_keys: set[str] = set()
    for decision in decisions.get("top_decisions", []):
        strategy = str(decision.get("strategy", ""))
        if current_strategies and strategy not in current_strategies:
            continue
        experiments = decision.get("next_experiments", [])
        for rank, experiment in enumerate(experiments[:3]):
            command, safe, runtime, skip_reason = command_for(strategy, experiment)
            safe, skip_reason = validate_command(command, safe, skip_reason)
            key = queue_key(strategy, experiment, command)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cooldown_items = cooldown_recent.get(key, [])
            last = max(
                (parse_stamp(item.get("generated_at_utc")) for item in recent_24h.get(key, [])),
                default=None,
            )
            cooldown_until = last + timedelta(hours=cooldown_hours) if last else None
            safe_now = safe
            if safe and cooldown_items:
                safe_now = False
                skip_reason = f"cooldown: executed within {cooldown_hours:g}h"
            items.append(
                QueueItem(
                    key=key,
                    priority=int(decision.get("priority", 0)) * 10 - rank,
                    strategy=strategy,
                    experiment=experiment,
                    objective=objective_for(decision.get("diagnosis", ""), experiment),
                    command=command,
                    success_gate=decision.get("success_gate", ""),
                    promotion_block=decision.get("promotion_block", ""),
                    safe_to_execute=safe_now,
                    expected_runtime=runtime,
                    attempts_24h=len(recent_24h.get(key, [])),
                    last_executed_utc=utc_stamp(last) if last else None,
                    cooldown_until_utc=utc_stamp(cooldown_until) if cooldown_until else None,
                    skip_reason=skip_reason,
                )
            )
    return sorted(items, key=lambda item: (-item.priority, item.strategy, item.experiment))


def build_payload(cooldown_hours: float = 6.0) -> dict[str, Any]:
    decisions = load_json(DECISION_PATH)
    queue = build_queue(decisions, cooldown_hours=cooldown_hours)
    return {
        "generated_at_utc": utc_stamp(),
        "source_decision": rel(DECISION_PATH),
        "queue_count": len(queue),
        "safe_count": sum(1 for item in queue if item.safe_to_execute),
        "cooldown_hours": cooldown_hours,
        "queue": [asdict(item) for item in queue],
        "policy": [
            "Execute one queue item at a time.",
            "Do not repeat the same strategy/experiment command inside the cooldown window.",
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
        f"- Cooldown hours: `{payload['cooldown_hours']}`",
        "",
        "## Queue",
        "",
        "| Priority | Strategy | Experiment | Safe | Attempts 24h | Last | Cooldown Until | Skip Reason | Runtime | Command |",
        "|---:|---|---|---:|---:|---|---|---|---|---|",
    ]
    for item in payload["queue"]:
        row = dict(item)
        row["command"] = " ".join(item["command"])
        lines.append(
            "| {priority} | {strategy} | {experiment} | {safe_to_execute} | {attempts_24h} | {last_executed_utc} | {cooldown_until_utc} | {skip_reason} | {expected_runtime} | `{command}` |".format(
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
    item = payload.get("item") or {}
    lines = [
        "# Mature Researcher Response Execution",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Strategy: `{item.get('strategy', '')}`",
        f"- Experiment: `{item.get('experiment', '')}`",
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
    if payload.get("history_record"):
        append_history(payload["history_record"])
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")


def execute_next(queue: list[dict[str, Any]], dry_run: bool) -> None:
    selected = next((item for item in queue if item.get("safe_to_execute")), None)
    if not selected:
        generated_at = utc_stamp()
        write_receipt(
            {
                "generated_at_utc": generated_at,
                "status": "blocked",
                "mode": "dry_run" if dry_run else "execute",
                "item": None,
                "command": None,
                "returncode": None,
                "output_tail": "No safe queue item found.",
                "history_record": None,
            }
        )
        return
    command = selected["command"]
    if dry_run:
        generated_at = utc_stamp()
        write_receipt(
            {
                "generated_at_utc": generated_at,
                "status": "dry_run",
                "mode": "dry_run",
                "item": selected,
                "command": command,
                "returncode": None,
                "output_tail": "Command was not executed.",
                "history_record": None,
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
    generated_at = utc_stamp()
    status = "ok" if completed.returncode == 0 else "failed"
    write_receipt(
        {
            "generated_at_utc": generated_at,
            "status": status,
            "mode": "execute",
            "item": selected,
            "command": command,
            "returncode": completed.returncode,
            "output_tail": completed.stdout[-4000:],
            "history_record": {
                "generated_at_utc": generated_at,
                "key": selected.get("key"),
                "strategy": selected.get("strategy"),
                "experiment": selected.get("experiment"),
                "command": command,
                "status": status,
                "returncode": completed.returncode,
            },
        }
    )


def main() -> None:
    args = parse_args()
    payload = build_payload(cooldown_hours=args.cooldown_hours)
    write_queue(payload)
    if args.execute_next or args.dry_run:
        execute_next(payload["queue"], args.dry_run)


if __name__ == "__main__":
    main()
