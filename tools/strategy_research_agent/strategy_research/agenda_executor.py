#!/usr/bin/env python3
"""Select and optionally run the next safe research agenda item."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
AGENDA_PATH = AGENT_ROOT / "research_agendas/latest_research_agenda.json"
RUN_DIR = AGENT_ROOT / "agenda_runs"
LATEST_RUN_JSON = RUN_DIR / "latest_agenda_run.json"
LATEST_RUN_MD = RUN_DIR / "latest_agenda_run.md"


SAFE_COMMANDS: dict[str, dict[str, Any]] = {
    "./.venv/bin/python user_data/strategy_research/run_research_agent.py --run-recursive --run-lookahead": {
        "argv": [
            "./.venv/bin/python",
            "user_data/strategy_research/run_research_agent.py",
            "--run-recursive",
            "--run-lookahead",
        ],
        "long": False,
    },
    "./.venv/bin/python user_data/strategy_research/estimate_trade_costs.py": {
        "argv": ["./.venv/bin/python", "user_data/strategy_research/estimate_trade_costs.py"],
        "long": False,
    },
    "./.venv/bin/python user_data/strategy_research/analyze_strategy_research.py": {
        "argv": ["./.venv/bin/python", "user_data/strategy_research/analyze_strategy_research.py"],
        "long": False,
    },
    "./.venv/bin/python user_data/strategy_research/strategy_iteration_engine.py": {
        "argv": ["./.venv/bin/python", "user_data/strategy_research/strategy_iteration_engine.py"],
        "long": False,
    },
    "user_data/strategy_research/start_manual_research.sh --walk-forward": {
        "argv": ["user_data/strategy_research/start_manual_research.sh", "--walk-forward"],
        "long": True,
    },
    "user_data/strategy_research/run_full_research_cycle.sh --skip-aux-fetch": {
        "argv": ["user_data/strategy_research/run_full_research_cycle.sh", "--skip-aux-fetch"],
        "long": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually run the selected safe command.")
    parser.add_argument("--allow-long", action="store_true", help="Allow long walk-forward/full-cycle tasks.")
    parser.add_argument("--strategy", help="Prefer a specific strategy from the agenda.")
    parser.add_argument("--blocker", help="Prefer a specific blocker from the agenda.")
    parser.add_argument("--index", type=int, default=0, help="Select the Nth matching agenda item.")
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


def select_item(agenda: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    items = agenda.get("items", [])
    if args.strategy:
        items = [item for item in items if item.get("strategy") == args.strategy]
    if args.blocker:
        items = [item for item in items if item.get("blocker") == args.blocker]
    if not items or args.index >= len(items):
        return None
    return items[args.index]


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    agenda = load_json(AGENDA_PATH)
    selected = select_item(agenda, args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt: dict[str, Any] = {
        "generated_at_utc": timestamp,
        "source_agenda": rel(AGENDA_PATH),
        "mode": "execute" if args.execute else "dry_run",
        "status": "no_matching_agenda_item",
        "selected_item": selected,
        "command": None,
        "returncode": None,
        "stdout_tail": None,
    }
    if not selected:
        return receipt

    command_text = selected.get("next_command", "")
    safe = SAFE_COMMANDS.get(command_text)
    receipt["command"] = command_text
    if not safe:
        receipt["status"] = "blocked_not_in_allowlist"
        return receipt
    if safe["long"] and not args.allow_long:
        receipt["status"] = "blocked_requires_allow_long"
        return receipt
    if not args.execute:
        receipt["status"] = "dry_run_selected"
        return receipt

    completed = subprocess.run(
        safe["argv"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    receipt["returncode"] = completed.returncode
    receipt["stdout_tail"] = completed.stdout[-4000:]
    receipt["status"] = "executed_ok" if completed.returncode == 0 else "executed_failed"
    return receipt


def write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    item = receipt.get("selected_item") or {}
    lines = [
        "# Agenda Run Receipt",
        "",
        f"- Generated UTC: `{receipt['generated_at_utc']}`",
        f"- Mode: `{receipt['mode']}`",
        f"- Status: `{receipt['status']}`",
        f"- Source agenda: `{receipt['source_agenda']}`",
        f"- Strategy: `{item.get('strategy', '')}`",
        f"- Blocker: `{item.get('blocker', '')}`",
        f"- Objective: `{item.get('objective', '')}`",
        f"- Command: `{receipt.get('command') or ''}`",
        f"- Return code: `{receipt.get('returncode')}`",
        "",
        "## Success Gate",
        "",
        item.get("success_gate", ""),
    ]
    if receipt.get("stdout_tail"):
        lines.extend(["", "## Output Tail", "", "```text", receipt["stdout_tail"], "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(receipt: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = receipt["generated_at_utc"]
    json_path = RUN_DIR / f"agenda_run_{timestamp}.json"
    md_path = RUN_DIR / f"agenda_run_{timestamp}.md"
    json_text = json.dumps(receipt, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_RUN_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, receipt)
    LATEST_RUN_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_RUN_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_RUN_MD.relative_to(REPO_ROOT)}")
    print(f"Status: {receipt['status']}")


def main() -> int:
    args = parse_args()
    receipt = build_receipt(args)
    write_outputs(receipt)
    return 1 if receipt["status"] == "executed_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
