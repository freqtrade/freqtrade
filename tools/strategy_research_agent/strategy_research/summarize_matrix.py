#!/usr/bin/env python3
"""Summarize regime/cost matrix reports into a strategy robustness view."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REPORT_DIR = AGENT_ROOT / "reports"
SUMMARY_DIR = AGENT_ROOT / "matrix_summaries"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric(value: Any) -> float | None:
    return None if value is None else float(value)


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for report in reports:
        cost_model = report.get("experiment", {}).get("cost_model", {})
        experiment_id = report.get("experiment", {}).get("id")
        for item in report.get("results", []):
            row = {
                "experiment": experiment_id,
                "strategy": item.get("strategy"),
                "regime": item.get("regime"),
                "timerange": item.get("timerange"),
                "fee": item.get("fee", cost_model.get("fee")),
                "classification": item.get("classification"),
                "trades": item.get("trades"),
                "return_pct": item.get("total_profit_pct"),
                "drawdown_pct": item.get("max_drawdown_pct"),
                "profit_factor": item.get("profit_factor"),
                "reasons": item.get("reasons", []),
            }
            rows.append(row)
            grouped[row["strategy"]].append(row)

    strategy_summary = []
    for strategy, items in grouped.items():
        returns = [numeric(item["return_pct"]) for item in items if item.get("return_pct") is not None]
        pfs = [numeric(item["profit_factor"]) for item in items if item.get("profit_factor") is not None]
        trades = [int(item["trades"]) for item in items if item.get("trades") is not None]
        positives = sum(1 for item in items if numeric(item.get("return_pct")) is not None and numeric(item.get("return_pct")) > 0)
        rejected = sum(1 for item in items if item.get("classification") == "rejected")
        too_few = sum(1 for item in items if "too few trades" in item.get("reasons", []))
        stress_negative = sum(
            1
            for item in items
            if item.get("fee") == 0.001
            and numeric(item.get("return_pct")) is not None
            and numeric(item.get("return_pct")) < 0
        )
        verdict = "fragile"
        if positives == len(items) and rejected == 0:
            verdict = "robust_candidate"
        elif positives >= len(items) / 2 and stress_negative == 0:
            verdict = "watchlist"
        strategy_summary.append(
            {
                "strategy": strategy,
                "runs": len(items),
                "positive_runs": positives,
                "rejected_runs": rejected,
                "too_few_trade_runs": too_few,
                "stress_negative_runs": stress_negative,
                "min_return_pct": round(min(returns), 2) if returns else None,
                "max_return_pct": round(max(returns), 2) if returns else None,
                "min_profit_factor": round(min(pfs), 2) if pfs else None,
                "total_trades": sum(trades) if trades else 0,
                "verdict": verdict,
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_reports": [report.get("generated_at_utc") for report in reports],
        "rows": rows,
        "strategy_summary": sorted(strategy_summary, key=lambda item: (item["verdict"], item["strategy"])),
        "interpretation": [
            "This summary compares candidate strategies across BTC-derived market regimes and fee scenarios.",
            "Funding, mark price, and true slippage are still not included unless a report explicitly says otherwise.",
            "A high too_few_trade_runs count means the current strategy is too sparse to prove regime robustness.",
        ],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Matrix Robustness Summary",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Verdict | Runs | Positive | Rejected | Too Few Trades | Stress Negative | Min Return | Max Return | Min PF | Total Trades |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["strategy_summary"]:
        lines.append(
            "| {strategy} | {verdict} | {runs} | {positive_runs} | {rejected_runs} | {too_few_trade_runs} | {stress_negative_runs} | {min_return_pct}% | {max_return_pct}% | {min_profit_factor} | {total_trades} |".format(
                **item
            )
        )
    lines.extend(["", "## Rows", "", "| Experiment | Strategy | Regime | Fee | Trades | Return | DD | PF | Class |", "|---|---|---|---:|---:|---:|---:|---:|---|"])
    for item in payload["rows"]:
        lines.append(
            "| {experiment} | {strategy} | {regime} | {fee} | {trades} | {return_pct}% | {drawdown_pct}% | {profit_factor} | {classification} |".format(
                **item
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports = [load_json(path) for path in args.report]
    payload = summarize(reports)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = SUMMARY_DIR / f"matrix_summary_{timestamp}.json"
    md_path = SUMMARY_DIR / f"matrix_summary_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    latest_json = SUMMARY_DIR / "latest_matrix_summary.json"
    latest_md = SUMMARY_DIR / "latest_matrix_summary.md"
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {latest_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {latest_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
