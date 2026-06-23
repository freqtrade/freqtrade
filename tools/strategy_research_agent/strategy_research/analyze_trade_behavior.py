#!/usr/bin/env python3
"""Analyze exported Freqtrade trades for behavior-level strategy diagnostics."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
COST_ESTIMATE = AGENT_ROOT / "cost_adjustments/latest_trade_cost_estimate.json"
REPORT_DIR = AGENT_ROOT / "trade_behavior"
LATEST_REPORT_JSON = REPORT_DIR / "latest_trade_behavior.json"
LATEST_REPORT_MD = REPORT_DIR / "latest_trade_behavior.md"


@dataclass
class BehaviorSummary:
    strategy: str
    source_zip: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    total_profit_abs: float
    avg_profit_abs: float
    avg_win_abs: float
    avg_loss_abs: float
    payoff_ratio: float | None
    profit_factor: float | None
    expectancy_abs: float
    median_duration_min: float | None
    avg_duration_min: float | None
    long_trades: int
    short_trades: int
    long_profit_abs: float
    short_profit_abs: float
    stop_loss_trades: int
    stop_loss_profit_abs: float
    max_consecutive_losses: int
    avg_mfe_pct: float | None
    avg_mae_pct: float | None
    pair_breakdown: list[dict[str, Any]]
    exit_reason_breakdown: list[dict[str, Any]]
    enter_tag_breakdown: list[dict[str, Any]]
    diagnostics: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-zip", action="append", type=Path)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_zips() -> list[Path]:
    payload = load_json(COST_ESTIMATE)
    paths = []
    seen = set()
    for item in payload.get("estimates", []):
        path = REPO_ROOT / item.get("source_zip", "")
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def read_backtest_json(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path.resolve()) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json") and not name.endswith("_config.json")]
        if len(names) != 1:
            raise ValueError(f"Expected one result json in {path}, found {names}")
        with archive.open(names[0]) as handle:
            return json.loads(handle.read().decode("utf-8"))


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100.0, 4)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def profit_factor(wins: list[float], losses: list[float]) -> float | None:
    loss_abs = abs(sum(losses))
    if loss_abs == 0:
        return None
    return sum(wins) / loss_abs


def max_consecutive_losses(trades: list[dict[str, Any]]) -> int:
    current = 0
    maximum = 0
    for trade in sorted(trades, key=lambda item: item.get("close_timestamp") or 0):
        if float(trade.get("profit_abs") or 0) < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def trade_excursions(trade: dict[str, Any]) -> tuple[float | None, float | None]:
    open_rate = float(trade.get("open_rate") or 0)
    if open_rate <= 0:
        return None, None
    min_rate = float(trade.get("min_rate") or open_rate)
    max_rate = float(trade.get("max_rate") or open_rate)
    if trade.get("is_short"):
        mfe = (open_rate - min_rate) / open_rate
        mae = (max_rate - open_rate) / open_rate
    else:
        mfe = (max_rate - open_rate) / open_rate
        mae = (open_rate - min_rate) / open_rate
    return pct(mfe), pct(mae)


def grouped_breakdown(trades: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.get(key) or "")].append(trade)
    rows = []
    for value, items in grouped.items():
        profits = [float(item.get("profit_abs") or 0) for item in items]
        rows.append(
            {
                key: value,
                "trades": len(items),
                "wins": sum(1 for item in profits if item > 0),
                "profit_abs": round(sum(profits), 6),
                "avg_profit_abs": round(sum(profits) / len(profits), 6) if profits else 0.0,
            }
        )
    return sorted(rows, key=lambda item: (-abs(item["profit_abs"]), item[key]))


def diagnostics(summary: dict[str, Any]) -> list[str]:
    notes = []
    if summary["win_rate_pct"] >= 55 and summary["total_profit_abs"] < 0:
        notes.append("High win rate but losing overall: average loss is too large relative to average win.")
    if summary["stop_loss_trades"] and summary["stop_loss_profit_abs"] < 0:
        notes.append("Stop-loss exits dominate realized losses; review entry confirmation and stop distance.")
    if summary["short_trades"] > summary["long_trades"] * 3 and summary["short_profit_abs"] < 0:
        notes.append("Short exposure is heavily dominant and unprofitable; split short-only logic from long/short regimes.")
    if summary["avg_mae_pct"] is not None and summary["avg_mfe_pct"] is not None and summary["avg_mae_pct"] > summary["avg_mfe_pct"]:
        notes.append("Average adverse excursion exceeds favorable excursion; entries are likely early or poorly confirmed.")
    if summary["max_consecutive_losses"] >= 5:
        notes.append("Long losing streak detected; add circuit breakers or regime cooldowns.")
    if not notes:
        notes.append("No single behavior failure dominates; inspect pair/tag breakdowns for localized weaknesses.")
    return notes


def summarize_strategy(strategy: str, source_zip: Path, trades: list[dict[str, Any]]) -> BehaviorSummary:
    profits = [float(trade.get("profit_abs") or 0) for trade in trades]
    wins = [item for item in profits if item > 0]
    losses = [item for item in profits if item < 0]
    durations = [float(trade.get("trade_duration") or 0) for trade in trades if trade.get("trade_duration") is not None]
    long_trades = [trade for trade in trades if not trade.get("is_short")]
    short_trades = [trade for trade in trades if trade.get("is_short")]
    stop_loss_trades = [trade for trade in trades if str(trade.get("exit_reason", "")).lower() == "stop_loss"]
    excursions = [trade_excursions(trade) for trade in trades]
    mfes = [item[0] for item in excursions if item[0] is not None]
    maes = [item[1] for item in excursions if item[1] is not None]
    summary = {
        "strategy": strategy,
        "source_zip": rel(source_zip),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 2) if trades else 0.0,
        "total_profit_abs": round(sum(profits), 6),
        "avg_profit_abs": round(mean(profits) or 0.0, 6),
        "avg_win_abs": round(mean(wins) or 0.0, 6),
        "avg_loss_abs": round(mean(losses) or 0.0, 6),
        "payoff_ratio": round((mean(wins) or 0.0) / abs(mean(losses) or 1.0), 4) if losses and wins else None,
        "profit_factor": round(profit_factor(wins, losses), 4) if profit_factor(wins, losses) is not None else None,
        "expectancy_abs": round(mean(profits) or 0.0, 6),
        "median_duration_min": round(median(durations), 2) if durations else None,
        "avg_duration_min": round(mean(durations), 2) if durations else None,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_profit_abs": round(sum(float(trade.get("profit_abs") or 0) for trade in long_trades), 6),
        "short_profit_abs": round(sum(float(trade.get("profit_abs") or 0) for trade in short_trades), 6),
        "stop_loss_trades": len(stop_loss_trades),
        "stop_loss_profit_abs": round(sum(float(trade.get("profit_abs") or 0) for trade in stop_loss_trades), 6),
        "max_consecutive_losses": max_consecutive_losses(trades),
        "avg_mfe_pct": round(mean(mfes), 4) if mfes else None,
        "avg_mae_pct": round(mean(maes), 4) if maes else None,
        "pair_breakdown": grouped_breakdown(trades, "pair"),
        "exit_reason_breakdown": grouped_breakdown(trades, "exit_reason"),
        "enter_tag_breakdown": grouped_breakdown(trades, "enter_tag"),
    }
    summary["diagnostics"] = diagnostics(summary)
    return BehaviorSummary(**summary)


def build_payload(paths: list[Path]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        data = read_backtest_json(path)
        for strategy, payload in data.get("strategy", {}).items():
            summaries.append(asdict(summarize_strategy(strategy, path, payload.get("trades", []))))
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_zips": [rel(path) for path in paths],
        "strategy_count": len(summaries),
        "summaries": sorted(summaries, key=lambda item: item["strategy"]),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Trade Behavior Analysis",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Strategy count: `{payload['strategy_count']}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Trades | Win % | Profit Abs | PF | Payoff | Avg Dur Min | Long/Short | Stop Losses | MFE % | MAE % | Max Loss Streak |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in payload["summaries"]:
        lines.append(
            "| {strategy} | {trades} | {win_rate_pct} | {total_profit_abs} | {profit_factor} | {payoff_ratio} | {avg_duration_min} | {long_trades}/{short_trades} | {stop_loss_trades} | {avg_mfe_pct} | {avg_mae_pct} | {max_consecutive_losses} |".format(
                **item
            )
        )
    lines.extend(["", "## Diagnostics", ""])
    for item in payload["summaries"]:
        lines.append(f"### {item['strategy']}")
        for note in item["diagnostics"]:
            lines.append(f"- {note}")
        lines.append("")
        lines.append("| Pair | Trades | Wins | Profit Abs | Avg Profit |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in item["pair_breakdown"]:
            lines.append("| {pair} | {trades} | {wins} | {profit_abs} | {avg_profit_abs} |".format(**row))
        lines.append("")
        lines.append("| Exit Reason | Trades | Wins | Profit Abs | Avg Profit |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in item["exit_reason_breakdown"]:
            lines.append(
                "| {exit_reason} | {trades} | {wins} | {profit_abs} | {avg_profit_abs} |".format(**row)
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"trade_behavior_{timestamp}.json"
    md_path = REPORT_DIR / f"trade_behavior_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_REPORT_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_REPORT_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"Strategies analyzed: {payload['strategy_count']}")


def main() -> None:
    args = parse_args()
    paths = args.backtest_zip or default_zips()
    if not paths:
        raise SystemExit("No backtest zips provided and no cost-adjustment source zips found.")
    payload = build_payload(paths)
    write_outputs(payload)


if __name__ == "__main__":
    main()
