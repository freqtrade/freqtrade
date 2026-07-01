#!/usr/bin/env python3
"""Build and summarize walk-forward validation experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
EXPERIMENT_PATH = AGENT_ROOT / "experiments/walk_forward_validation_experiment.json"
SUMMARY_DIR = AGENT_ROOT / "walk_forward_summaries"
LATEST_SUMMARY_JSON = SUMMARY_DIR / "latest_walk_forward_summary.json"
LATEST_SUMMARY_MD = SUMMARY_DIR / "latest_walk_forward_summary.md"
ITERATIVE_REGISTRY = AGENT_ROOT / "experiments/iterative_strategy_registry.json"
MEMORY_GUIDED_REGISTRY = AGENT_ROOT / "experiments/memory_guided_strategy_registry.json"
BASE_REGISTRY = AGENT_ROOT / "strategy_registry.json"


WINDOWS = [
    {"name": "wf_2024_h1", "label": "2024 H1", "timerange": "20240101-20240701"},
    {"name": "wf_2024_h2", "label": "2024 H2", "timerange": "20240701-20250101"},
    {"name": "wf_2025_h1", "label": "2025 H1", "timerange": "20250101-20250701"},
    {"name": "wf_2025_h2", "label": "2025 H2", "timerange": "20250701-20260101"},
    {"name": "wf_2026_h1", "label": "2026 H1", "timerange": "20260101-20260622"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build a walk-forward experiment JSON.")
    build.add_argument(
        "--source",
        choices=["memory_guided", "iterative", "base", "all"],
        default="base",
        help="Strategy universe for the walk-forward experiment.",
    )
    build.add_argument("--limit", type=int, default=6)
    build.add_argument("--dry-run", action="store_true")

    summarize = sub.add_parser("summarize", help="Summarize a walk-forward agent report.")
    summarize.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def registry_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = load_json(path)
    return [item["name"] for item in payload.get("strategies", []) if item.get("name")]


def strategy_universe(source: str, limit: int) -> list[str]:
    names: list[str] = []
    if source in {"memory_guided", "all"}:
        names.extend(registry_names(MEMORY_GUIDED_REGISTRY))
    if source in {"iterative", "all"}:
        names.extend(registry_names(ITERATIVE_REGISTRY))
    if source in {"base", "all"}:
        names.extend(registry_names(BASE_REGISTRY))
    deduped = list(dict.fromkeys(names))
    return deduped[:limit]


def build_experiment(source: str, limit: int) -> dict[str, Any]:
    strategies = strategy_universe(source, limit)
    if not strategies:
        raise SystemExit(f"No strategies found for source={source}.")
    return {
        "id": "walk_forward_validation",
        "title": "Walk-forward validation across fixed half-year windows",
        "profile_ref": "strategy_registry.json",
        "strategy_path": "user_data/strategies/research_generated",
        "timeframes": ["15m"],
        "timeranges": ["20240101-20260622"],
        "matrix": {"timeranges": WINDOWS},
        "fee": 0.0005,
        "strategies": strategies,
        "checks": {
            "backtesting": True,
            "recursive_analysis": False,
            "lookahead_analysis": False,
        },
        "validation_policy": {
            "min_positive_windows": 3,
            "max_negative_windows": 1,
            "min_total_trades": 100,
            "min_median_profit_factor": 1.05,
            "max_worst_drawdown_pct": 10.0,
        },
        "notes": [
            "Walk-forward windows are fixed calendar windows, not optimized per strategy.",
            "This validation is designed to reject one-window luck before dry-run promotion.",
        ],
    }


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def verdict_for(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    returns = [numeric(row.get("total_profit_pct")) for row in rows]
    returns = [value for value in returns if value is not None]
    pfs = [numeric(row.get("profit_factor")) for row in rows]
    pfs = [value for value in pfs if value is not None]
    drawdowns = [numeric(row.get("max_drawdown_pct")) for row in rows]
    drawdowns = [value for value in drawdowns if value is not None]
    trades = [int(row.get("trades") or 0) for row in rows]

    positive_windows = sum(1 for value in returns if value > 0)
    negative_windows = sum(1 for value in returns if value < 0)
    total_trades = sum(trades)
    median_pf = median(pfs)
    worst_dd = max(drawdowns) if drawdowns else None

    if positive_windows < 3:
        reasons.append("too few positive windows")
    if negative_windows > 1:
        reasons.append("too many negative windows")
    if total_trades < 100:
        reasons.append("too few total trades")
    if median_pf is None or median_pf < 1.05:
        reasons.append("median PF below threshold")
    if worst_dd is None or worst_dd > 10.0:
        reasons.append("worst drawdown above threshold")

    if not reasons:
        return "walk_forward_candidate", ["passes fixed-window stability gates"]
    if positive_windows >= 2 and total_trades >= 50 and (median_pf or 0) >= 1.0:
        return "walk_forward_watchlist", reasons
    return "walk_forward_rejected", reasons


def summarize_report(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report.get("results", []):
        grouped[row["strategy"]].append(row)

    summaries = []
    for strategy, rows in grouped.items():
        returns = [numeric(row.get("total_profit_pct")) for row in rows if numeric(row.get("total_profit_pct")) is not None]
        pfs = [numeric(row.get("profit_factor")) for row in rows if numeric(row.get("profit_factor")) is not None]
        drawdowns = [numeric(row.get("max_drawdown_pct")) for row in rows if numeric(row.get("max_drawdown_pct")) is not None]
        verdict, reasons = verdict_for(rows)
        summaries.append(
            {
                "strategy": strategy,
                "verdict": verdict,
                "windows": len(rows),
                "positive_windows": sum(1 for value in returns if value > 0),
                "negative_windows": sum(1 for value in returns if value < 0),
                "total_trades": sum(int(row.get("trades") or 0) for row in rows),
                "median_return_pct": round(median(returns), 4) if returns else None,
                "median_profit_factor": round(median(pfs), 4) if pfs else None,
                "worst_drawdown_pct": round(max(drawdowns), 4) if drawdowns else None,
                "reasons": reasons,
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_report": str(report_path.relative_to(REPO_ROOT)),
        "experiment": report.get("experiment", {}).get("id"),
        "policy": report.get("experiment", {}).get("validation_policy", {}),
        "strategy_summary": sorted(summaries, key=lambda item: (item["verdict"], item["strategy"])),
        "rows": report.get("results", []),
        "interpretation": [
            "A walk_forward_candidate must work across fixed calendar windows, not only one favorable sample.",
            "This does not replace recursive-analysis, lookahead-analysis, or dry-run monitoring.",
        ],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Walk-Forward Validation Summary",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Source report: `{payload['source_report']}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Verdict | Windows | Positive | Negative | Trades | Median Return | Median PF | Worst DD | Reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in payload["strategy_summary"]:
        row = dict(item)
        row["reasons"] = ", ".join(item.get("reasons", []))
        lines.append(
            "| {strategy} | {verdict} | {windows} | {positive_windows} | {negative_windows} | {total_trades} | {median_return_pct}% | {median_profit_factor} | {worst_drawdown_pct}% | {reasons} |".format(
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Window Rows",
            "",
            "| Strategy | Window | Timerange | Class | Trades | Return | DD | PF |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| {strategy} | {regime} | {timerange} | {classification} | {trades} | {total_profit_pct}% | {max_drawdown_pct}% | {profit_factor} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.command == "build":
        experiment = build_experiment(args.source, args.limit)
        if args.dry_run:
            print(json.dumps(experiment, indent=2, ensure_ascii=False))
            return
        EXPERIMENT_PATH.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {EXPERIMENT_PATH.relative_to(REPO_ROOT)}")
        print(f"Strategies: {len(experiment['strategies'])}")
        print(f"Windows: {len(WINDOWS)}")
        return

    report_path = args.report
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report = load_json(report_path)
    payload = summarize_report(report, report_path)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = SUMMARY_DIR / f"walk_forward_summary_{timestamp}.json"
    md_path = SUMMARY_DIR / f"walk_forward_summary_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_SUMMARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LATEST_SUMMARY_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_SUMMARY_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_SUMMARY_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
