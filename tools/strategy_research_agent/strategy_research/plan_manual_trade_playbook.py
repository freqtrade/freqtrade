#!/usr/bin/env python3
"""Write a research playbook that converts discretionary crypto trading into testable layers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_DIR = AGENT_ROOT / "manual_playbook"
PLAYBOOK_JSON = OUTPUT_DIR / "latest_manual_trade_playbook.json"
PLAYBOOK_MD = OUTPUT_DIR / "latest_manual_trade_playbook.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def payload() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "purpose": "Turn discretionary direction + location + abstention logic into auditable Freqtrade experiments.",
        "layers": [
            {
                "layer": "direction",
                "question": "Should the bot prefer long, short, or flat?",
                "simple_signals": [
                    "BTC breaks or reclaims a recent 4h/1h structure level.",
                    "BTC 24h and 72h returns agree with the side.",
                    "ETH follows BTC after a delay instead of diverging strongly.",
                ],
                "failure_mode": "If direction has no edge, no stop/ROI setting can rescue the strategy.",
            },
            {
                "layer": "entry_location",
                "question": "Where is the entry allowed after direction is known?",
                "simple_signals": [
                    "Pullback toward EMA after directional break.",
                    "Short only after bounce failure; long only after reclaim holds.",
                    "No entry immediately on first signal candle.",
                ],
                "failure_mode": "Entering on every signal creates fee drag and bad MAE.",
            },
            {
                "layer": "exit_quality",
                "question": "Does the exit match the trade type?",
                "simple_signals": [
                    "Scalp exits quickly when momentum stalls.",
                    "Trend exits only when structure invalidates.",
                    "Loss is cut by invalidation, not by hoping for ROI.",
                ],
                "failure_mode": "High win rate can still lose if losses are much larger than wins.",
            },
            {
                "layer": "abstention",
                "question": "When should the bot skip trading?",
                "simple_signals": [
                    "BTC/ETH disagreement is large.",
                    "Noise is high but directional range is small.",
                    "Recent generated family shows negative edge or low sample quality.",
                ],
                "failure_mode": "A weak strategy often gets worse when forced to trade every condition.",
            },
        ],
        "experiment_backlog": [
            {
                "id": "manual_direction_structure_break",
                "goal": "Test direction first with BTC/ETH structure and return agreement.",
                "max_conditions": 3,
                "promotion_gate": "Positive PF with enough trades before adding entry filters.",
            },
            {
                "id": "manual_entry_delayed_confirmation",
                "goal": "Enter only after a small delay and renewed movement in favor.",
                "max_conditions": 3,
                "promotion_gate": "Improves PF or MAE without collapsing trade count.",
            },
            {
                "id": "manual_abstention_noise_filter",
                "goal": "Skip high-noise disagreement windows before signal generation.",
                "max_conditions": 3,
                "promotion_gate": "Improves PF under same fee without only deleting all trades.",
            },
        ],
    }


def write_markdown(data: dict[str, Any]) -> None:
    lines = [
        "# Manual Trade Playbook",
        "",
        f"- Generated UTC: `{data['generated_at_utc']}`",
        f"- Purpose: {data['purpose']}",
        "",
        "## Layers",
        "",
        "| Layer | Question | Simple Signals | Failure Mode |",
        "|---|---|---|---|",
    ]
    for layer in data["layers"]:
        lines.append(
            "| {layer} | {question} | {signals} | {failure_mode} |".format(
                layer=layer["layer"],
                question=layer["question"],
                signals="<br>".join(layer["simple_signals"]),
                failure_mode=layer["failure_mode"],
            )
        )
    lines.extend(["", "## Experiment Backlog", "", "| ID | Goal | Max Conditions | Promotion Gate |", "|---|---|---:|---|"])
    for item in data["experiment_backlog"]:
        lines.append("| {id} | {goal} | {max_conditions} | {promotion_gate} |".format(**item))
    PLAYBOOK_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = payload()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAYBOOK_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(data)
    print(f"Wrote {rel(PLAYBOOK_JSON)}")
    print(f"Wrote {rel(PLAYBOOK_MD)}")


if __name__ == "__main__":
    main()
