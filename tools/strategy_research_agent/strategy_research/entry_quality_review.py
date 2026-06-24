#!/usr/bin/env python3
"""Review entry quality from the latest strategy research backtest report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REPORT_INDEX = AGENT_ROOT / "reports/agent_report_index.json"
OUTPUT_DIR = AGENT_ROOT / "entry_quality"
LATEST_JSON = OUTPUT_DIR / "latest_entry_quality_review.json"
LATEST_MD = OUTPUT_DIR / "latest_entry_quality_review.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_result_report() -> Path:
    index = load_json(REPORT_INDEX)
    latest = index.get("latest_report") or {}
    path = latest.get("path")
    if not path:
        raise SystemExit("No latest result report found. Run a backtest iteration first.")
    return REPO_ROOT / path


def decision(row: dict[str, Any]) -> tuple[str, str]:
    trades = row.get("trades") or 0
    pf = row.get("profit_factor")
    profit = row.get("total_profit_pct")
    long_trades = row.get("long_trades") or 0
    short_trades = row.get("short_trades") or 0
    reasons = set(row.get("reasons") or [])
    if trades < 30 and (pf or 0) >= 1.0:
        return "careful_expand", "PF is not broken, but sample is too small; test a narrow expansion before judging edge."
    if trades < 30:
        return "discard_or_reframe", "Sample is tiny and quality is weak; avoid more threshold tweaks unless hypothesis changes."
    if trades >= 50 and (pf or 0) < 1.0:
        return "negative_edge", "Trade count is enough for a smoke read and PF is below 1; tightening alone is unlikely to rescue it."
    if profit is not None and profit < 0 and "negative total return" in reasons:
        return "tighten_or_split", "Return is negative; split by direction or regime before widening further."
    if long_trades == 0 or short_trades == 0:
        return "direction_split", "Signals are one-sided; evaluate the active side separately before adding the missing side."
    return "hold_for_validation", "Entry quality is not obviously broken; validate on longer windows and cost stress."


def build_review(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    rows = []
    for row in report.get("results", []):
        action, diagnosis = decision(row)
        rows.append(
            {
                "strategy": row.get("strategy"),
                "family": row.get("family"),
                "classification": row.get("classification"),
                "trades": row.get("trades"),
                "daily_trades": row.get("daily_trades"),
                "profit_factor": row.get("profit_factor"),
                "total_profit_pct": row.get("total_profit_pct"),
                "max_drawdown_pct": row.get("max_drawdown_pct"),
                "long_trades": row.get("long_trades"),
                "short_trades": row.get("short_trades"),
                "reasons": row.get("reasons") or [],
                "action": action,
                "diagnosis": diagnosis,
            }
        )
    action_counts: dict[str, int] = {}
    for row in rows:
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_report": rel(report_path),
        "strategy_count": len(rows),
        "action_counts": action_counts,
        "reviews": rows,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Entry Quality Review",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Source report: `{payload['source_report']}`",
        f"- Strategies reviewed: `{payload['strategy_count']}`",
        "",
        "## Action Counts",
        "",
        "| Action | Count |",
        "|---|---:|",
    ]
    for action, count in sorted(payload["action_counts"].items()):
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## Strategy Reviews",
            "",
            "| Strategy | Action | Return % | PF | Trades | L/S | Diagnosis |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["reviews"]:
        lines.append(
            "| {strategy} | {action} | {total_profit_pct} | {profit_factor} | {trades} | {long_trades}/{short_trades} | {diagnosis} |".format(
                **row
            )
        )
    LATEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report_path = latest_result_report()
    report = load_json(report_path)
    payload = build_review(report, report_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload)
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")


if __name__ == "__main__":
    main()
