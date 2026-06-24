#!/usr/bin/env python3
"""Review manual-style direction/entry/abstention experiments and recommend next research action."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REPORT_INDEX = AGENT_ROOT / "reports/agent_report_index.json"
OUTPUT_DIR = AGENT_ROOT / "manual_playbook"
LATEST_JSON = OUTPUT_DIR / "latest_manual_research_review.json"
LATEST_MD = OUTPUT_DIR / "latest_manual_research_review.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_report() -> tuple[Path, dict[str, Any]]:
    index = load_json(REPORT_INDEX)
    item = index.get("latest_report") or {}
    path = item.get("path")
    if not path:
        raise SystemExit("No latest report found.")
    report_path = REPO_ROOT / path
    return report_path, load_json(report_path)


def classify_manual_layer(strategy: str) -> str:
    if "Strong" in strategy:
        return "strong_confirmation"
    if "Abstention" in strategy:
        return "abstention"
    if "Entry" in strategy:
        return "entry_confirmation"
    if "Direction" in strategy:
        return "direction"
    return "other"


def build_review(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for result in report.get("results", []):
        strategy = result.get("strategy", "")
        layer = classify_manual_layer(strategy)
        if layer == "other":
            continue
        rows.append(
            {
                "strategy": strategy,
                "layer": layer,
                "return_pct": result.get("total_profit_pct"),
                "profit_factor": result.get("profit_factor"),
                "trades": result.get("trades"),
                "long_trades": result.get("long_trades"),
                "short_trades": result.get("short_trades"),
                "reasons": result.get("reasons") or [],
            }
        )
    negative = [row for row in rows if (row.get("profit_factor") or 0) < 1.0]
    high_trade_negative = [row for row in rows if (row.get("profit_factor") or 0) < 1.0 and (row.get("trades") or 0) >= 100]
    if high_trade_negative:
        recommendation = "pause_manual_family_change_signal_source"
        diagnosis = "Manual direction/entry/abstention variants have enough smoke trades but PF remains below 1, so the current signal source lacks edge."
    elif negative:
        recommendation = "retest_manual_family_on_longer_windows"
        diagnosis = "Manual-style variants are negative but sample may still be inconclusive."
    else:
        recommendation = "validate_manual_family"
        diagnosis = "Manual-style variants are not clearly broken; run matrix and validation gates."
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_report": rel(report_path),
        "manual_results": rows,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "next_experiments": [
            {
                "id": "change_signal_source",
                "reason": "Current manual-style rules still derive direction from simple returns/EMA/structure and do not show edge.",
                "examples": ["orderbook imbalance", "funding/mark context", "BTC ETH relative strength", "higher timeframe regime labels"],
            },
            {
                "id": "direction_model_before_entry",
                "reason": "Train or score direction separately before testing entries; do not tune ROI/stop first.",
                "examples": ["future 30m sign hit-rate", "long/short lane precision", "regime-conditioned direction score"],
            },
        ],
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Manual Research Review",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Source report: `{payload['source_report']}`",
        f"- Diagnosis: {payload['diagnosis']}",
        f"- Recommendation: `{payload['recommendation']}`",
        "",
        "## Manual Results",
        "",
        "| Strategy | Layer | Return % | PF | Trades | L/S | Reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["manual_results"]:
        lines.append(
            "| {strategy} | {layer} | {return_pct} | {profit_factor} | {trades} | {long_trades}/{short_trades} | {reasons} |".format(
                strategy=row["strategy"],
                layer=row["layer"],
                return_pct=row["return_pct"],
                profit_factor=row["profit_factor"],
                trades=row["trades"],
                long_trades=row["long_trades"],
                short_trades=row["short_trades"],
                reasons=", ".join(row["reasons"]),
            )
        )
    lines.extend(["", "## Next Experiments", "", "| ID | Reason | Examples |", "|---|---|---|"])
    for item in payload["next_experiments"]:
        lines.append("| {id} | {reason} | {examples} |".format(id=item["id"], reason=item["reason"], examples=", ".join(item["examples"])))
    LATEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report_path, report = latest_report()
    payload = build_review(report_path, report)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload)
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")


if __name__ == "__main__":
    main()
