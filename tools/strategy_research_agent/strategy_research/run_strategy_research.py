#!/usr/bin/env python3
"""Run a local Freqtrade strategy research loop and grade candidate strategies."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = Path("user_data/strategy_research/reports")


@dataclass
class BacktestMetrics:
    strategy: str
    status: str
    backtesting_from: str | None = None
    backtesting_to: str | None = None
    trades: int | None = None
    daily_trades: float | None = None
    total_profit_pct: float | None = None
    profit_factor: float | None = None
    market_change_pct: float | None = None
    max_drawdown_pct: float | None = None
    long_trades: int | None = None
    short_trades: int | None = None
    long_profit_pct: float | None = None
    short_profit_pct: float | None = None
    command: list[str] | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("user_data/strategy_research/strategy_registry.json"),
    )
    parser.add_argument("--strategy", action="append", help="Run only the named strategy. Repeatable.")
    parser.add_argument("--timerange", help="Override registry timerange.")
    parser.add_argument("--timeframe", help="Override registry timeframe.")
    parser.add_argument("--fee", type=float, help="Override registry fee.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running backtests.")
    return parser.parse_args()


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct_value(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else None


def int_value(value: str) -> int | None:
    match = re.search(r"\d+", value.replace(",", ""))
    return int(match.group(0)) if match else None


def table_value(output: str, label: str) -> str | None:
    pattern = re.compile(rf"│\s*{re.escape(label)}\s*│\s*([^│]+?)\s*│")
    match = pattern.search(output)
    return match.group(1).strip() if match else None


def parse_long_short(value: str | None) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    match = re.search(r"(\d+)\s*/\s*(\d+)", value)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_long_short_profit(value: str | None) -> tuple[float | None, float | None]:
    if not value:
        return None, None
    match = re.search(r"([-+]?\d+(?:\.\d+)?)%\s*/\s*([-+]?\d+(?:\.\d+)?)%", value)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def parse_trade_count(value: str | None) -> tuple[int | None, float | None]:
    if not value:
        return None, None
    match = re.search(r"(\d+)\s*/\s*([-+]?\d+(?:\.\d+)?)", value)
    if not match:
        return None, None
    return int(match.group(1)), float(match.group(2))


def parse_metrics(strategy: str, output: str, command: list[str]) -> BacktestMetrics:
    trades, daily_trades = parse_trade_count(table_value(output, "Total/Daily Avg Trades"))
    long_trades, short_trades = parse_long_short(table_value(output, "Long / Short trades"))
    long_profit_pct, short_profit_pct = parse_long_short_profit(table_value(output, "Long / Short profit %"))
    if trades is None and "No trades made." in output:
        trades = 0
        daily_trades = 0.0
        long_trades = 0
        short_trades = 0
        long_profit_pct = 0.0
        short_profit_pct = 0.0
    return BacktestMetrics(
        strategy=strategy,
        status="ok",
        backtesting_from=table_value(output, "Backtesting from"),
        backtesting_to=table_value(output, "Backtesting to"),
        trades=trades,
        daily_trades=daily_trades,
        total_profit_pct=0.0 if trades == 0 else pct_value(table_value(output, "Total profit %") or ""),
        profit_factor=0.0 if trades == 0 else pct_value(table_value(output, "Profit factor") or ""),
        market_change_pct=pct_value(table_value(output, "Market change") or ""),
        max_drawdown_pct=0.0 if trades == 0 else pct_value(table_value(output, "Max % of account underwater") or ""),
        long_trades=long_trades,
        short_trades=short_trades,
        long_profit_pct=long_profit_pct,
        short_profit_pct=short_profit_pct,
        command=command,
    )


def run_backtest(
    strategy: str,
    profile: dict[str, Any],
    args: argparse.Namespace,
    dry_run: bool,
) -> BacktestMetrics:
    timeframe = args.timeframe or profile["timeframe"]
    timerange = args.timerange or profile["timerange"]
    fee = args.fee if args.fee is not None else profile["fee"]
    config = profile["config"]

    command = [
        str(REPO_ROOT / ".venv/bin/freqtrade"),
        "backtesting",
        "-c",
        config,
        "--strategy",
        strategy,
        "--timeframe",
        timeframe,
        "--timerange",
        timerange,
        "--fee",
        str(fee),
        "--cache",
        "none",
        "--export",
        "none",
    ]

    if dry_run:
        return BacktestMetrics(strategy=strategy, status="dry_run", command=command)

    env = os.environ.copy()
    offline_path = str(REPO_ROOT / "user_data/offline_exchange")
    env["PYTHONPATH"] = (
        offline_path
        if not env.get("PYTHONPATH")
        else f"{offline_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lowered = completed.stdout.lower()
    if completed.returncode != 0 or "configuration error" in lowered or " - error -" in lowered:
        return BacktestMetrics(
            strategy=strategy,
            status="failed",
            command=command,
            error=completed.stdout[-4000:],
        )
    return parse_metrics(strategy, completed.stdout, command)


def classify(metrics: BacktestMetrics, thresholds: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if metrics.status == "dry_run":
        return "dry_run", ["command preview only"]

    if metrics.status != "ok":
        return "failed", [metrics.error or metrics.status]

    if metrics.total_profit_pct is None or metrics.profit_factor is None or metrics.max_drawdown_pct is None:
        return "needs_review", ["missing parsed metrics"]

    if metrics.total_profit_pct < thresholds["reject_if_total_profit_pct_below"]:
        reasons.append("negative total return")
    if metrics.profit_factor < thresholds["reject_if_profit_factor_below"]:
        reasons.append("profit factor below threshold")
    if metrics.max_drawdown_pct > thresholds["reject_if_max_drawdown_pct_above"]:
        reasons.append("drawdown above threshold")
    if metrics.trades is not None and metrics.trades < thresholds["reject_if_total_trades_below"]:
        reasons.append("too few trades")

    if reasons:
        return "rejected", reasons

    if (
        metrics.profit_factor >= thresholds["candidate_if_profit_factor_at_least"]
        and metrics.max_drawdown_pct <= thresholds["candidate_if_max_drawdown_pct_at_most"]
    ):
        if (
            metrics.market_change_pct is not None
            and metrics.total_profit_pct < metrics.market_change_pct
        ):
            return "research_candidate", ["positive and controlled, but below market change"]
        return "dryrun_candidate", ["passes first-pass return, PF, and drawdown gates"]

    return "needs_more_data", ["positive but weak edge"]


def metrics_to_dict(metrics: BacktestMetrics, classification: str, reasons: list[str]) -> dict[str, Any]:
    data = metrics.__dict__.copy()
    data["classification"] = classification
    data["reasons"] = reasons
    return data


def write_reports(
    report_dir: Path,
    registry: dict[str, Any],
    results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = REPO_ROOT / report_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"strategy_research_{timestamp}.json"
    md_path = output_dir / f"strategy_research_{timestamp}.md"

    payload = {
        "generated_at_utc": timestamp,
        "profile": registry["profile"],
        "thresholds": registry["thresholds"],
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Strategy Research Report",
        "",
        f"- Generated UTC: `{timestamp}`",
        f"- Market: `{registry['profile']['market']}`",
        f"- Instrument: `{registry['profile']['instrument']}`",
        f"- Timeframe: `{registry['profile']['timeframe']}`",
        f"- Timerange: `{registry['profile']['timerange']}`",
        f"- Fee: `{registry['profile']['fee']}`",
        "",
        "| Strategy | Class | Trades | Return | DD | PF | Market | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            "| {strategy} | {classification} | {trades} | {ret} | {dd} | {pf} | {market} | {reasons} |".format(
                strategy=item["strategy"],
                classification=item["classification"],
                trades=item.get("trades"),
                ret=f"{item.get('total_profit_pct')}%",
                dd=f"{item.get('max_drawdown_pct')}%",
                pf=item.get("profit_factor"),
                market=f"{item.get('market_change_pct')}%",
                reasons="; ".join(item.get("reasons", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Safety Notes",
            "",
            "- This report is research output only.",
            "- Do not connect live keys or promote a strategy without manual review.",
            "- Current futures tests do not include funding-rate and mark-price history.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    registry = load_registry(REPO_ROOT / args.registry)
    selected = set(args.strategy or [])
    strategies = [
        item for item in registry["strategies"] if not selected or item["name"] in selected
    ]
    if selected:
        found = {item["name"] for item in strategies}
        missing = selected - found
        if missing:
            raise SystemExit(f"Strategy not found in registry: {', '.join(sorted(missing))}")

    results: list[dict[str, Any]] = []
    for item in strategies:
        metrics = run_backtest(item["name"], registry["profile"], args, args.dry_run)
        classification, reasons = classify(metrics, registry["thresholds"])
        result = metrics_to_dict(metrics, classification, reasons)
        result.update(
            {
                "family": item.get("family"),
                "source": item.get("source"),
                "hypothesis": item.get("hypothesis"),
                "risk_notes": item.get("risk_notes"),
            }
        )
        results.append(result)
        print(f"{item['name']}: {classification} ({'; '.join(reasons)})")

    json_path, md_path = write_reports(args.report_dir, registry, results)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
