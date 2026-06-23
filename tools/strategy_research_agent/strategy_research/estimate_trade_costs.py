#!/usr/bin/env python3
"""Estimate strategy-level funding and slippage impact from exported backtest trades."""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
AUX_ROOT = REPO_ROOT / "user_data/data/binance/futures_aux"
REPORT_DIR = REPO_ROOT / "user_data/strategy_research/cost_adjustments"


@dataclass
class StrategyCostEstimate:
    strategy: str
    trades: int
    base_profit_abs: float
    base_profit_pct: float
    funding_abs: float
    funding_pct_of_start_balance: float
    slippage_abs: float
    slippage_pct_of_start_balance: float
    adjusted_profit_abs: float
    adjusted_profit_pct: float
    funding_events: int
    missing_funding_trades: int
    funding_coverage_missing_trades: int
    slippage_bps_round_trip: float
    source_zip: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-zip", action="append", type=Path, required=True)
    parser.add_argument("--slippage-bps", type=float, default=4.0)
    return parser.parse_args()


def pair_key(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def load_funding(pair: str) -> pd.DataFrame:
    path = AUX_ROOT / "funding_rate" / f"{pair_key(pair)}.feather"
    if not path.exists():
        return pd.DataFrame(columns=["date", "funding_rate"])
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame[["date", "funding_rate"]].sort_values("date")


def read_backtest_json(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path.resolve()) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith(".json") and not name.endswith("_config.json")
        ]
        if len(names) != 1:
            raise ValueError(f"Expected one result json in {path}, found {names}")
        with archive.open(names[0]) as handle:
            return json.loads(handle.read().decode("utf-8"))


def safe_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def estimate_funding_for_trade(trade: dict[str, Any], funding_cache: dict[str, pd.DataFrame]) -> tuple[float, int, bool, bool]:
    pair = trade["pair"]
    funding = funding_cache.setdefault(pair, load_funding(pair))
    if funding.empty:
        return 0.0, 0, True, True
    open_time = pd.to_datetime(trade["open_date"], utc=True)
    close_time = pd.to_datetime(trade["close_date"], utc=True)
    coverage_missing = bool(open_time < funding["date"].min() or close_time > funding["date"].max())
    window = funding[(funding["date"] > open_time) & (funding["date"] <= close_time)]
    if window.empty:
        return 0.0, 0, False, coverage_missing
    notional = float(trade["amount"]) * float(trade["open_rate"])
    sign = 1.0 if trade.get("is_short") else -1.0
    funding_abs = float((window["funding_rate"] * notional * sign).sum())
    return funding_abs, int(len(window)), False, coverage_missing


def estimate_strategy(path: Path, slippage_bps: float) -> list[StrategyCostEstimate]:
    data = read_backtest_json(path)
    estimates: list[StrategyCostEstimate] = []
    funding_cache: dict[str, pd.DataFrame] = {}
    for strategy, payload in data["strategy"].items():
        trades = payload["trades"]
        starting_balance = float(payload["starting_balance"])
        base_profit_abs = float(payload["profit_total_abs"])
        funding_abs = 0.0
        funding_events = 0
        missing_funding_trades = 0
        funding_coverage_missing_trades = 0
        slippage_abs = 0.0
        for trade in trades:
            trade_funding, events, missing, coverage_missing = estimate_funding_for_trade(trade, funding_cache)
            funding_abs += trade_funding
            funding_events += events
            missing_funding_trades += int(missing)
            funding_coverage_missing_trades += int(coverage_missing)
            notional = float(trade["amount"]) * float(trade["open_rate"])
            slippage_abs -= notional * (slippage_bps / 10000.0)
        adjusted = base_profit_abs + funding_abs + slippage_abs
        estimates.append(
            StrategyCostEstimate(
                strategy=strategy,
                trades=int(len(trades)),
                base_profit_abs=round(base_profit_abs, 6),
                base_profit_pct=round(base_profit_abs / starting_balance * 100, 4),
                funding_abs=round(funding_abs, 6),
                funding_pct_of_start_balance=round(funding_abs / starting_balance * 100, 4),
                slippage_abs=round(slippage_abs, 6),
                slippage_pct_of_start_balance=round(slippage_abs / starting_balance * 100, 4),
                adjusted_profit_abs=round(adjusted, 6),
                adjusted_profit_pct=round(adjusted / starting_balance * 100, 4),
                funding_events=funding_events,
                missing_funding_trades=missing_funding_trades,
                funding_coverage_missing_trades=funding_coverage_missing_trades,
                slippage_bps_round_trip=slippage_bps,
                source_zip=safe_repo_path(path),
            )
        )
    return estimates


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Trade Cost Adjustment Estimate",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Round-trip slippage bps: `{payload['slippage_bps_round_trip']}`",
        "",
        "| Strategy | Trades | Base % | Funding % | Slippage % | Adjusted % | Funding Events | Missing Funding Files | Coverage Missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["estimates"]:
        lines.append(
            "| {strategy} | {trades} | {base_profit_pct} | {funding_pct_of_start_balance} | {slippage_pct_of_start_balance} | {adjusted_profit_pct} | {funding_events} | {missing_funding_trades} | {funding_coverage_missing_trades} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Funding estimate aligns exported trade open/close windows to downloaded Binance funding rates.",
            "- Positive funding rates are treated as beneficial to shorts and costly to longs.",
            "- Slippage is a simple round-trip notional haircut, not a reconstruction of order-book fills.",
            "- Coverage missing means the trade window extends outside the downloaded funding-rate date range.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    estimates = []
    for path in args.backtest_zip:
        estimates.extend(asdict(item) for item in estimate_strategy(path, args.slippage_bps))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "slippage_bps_round_trip": args.slippage_bps,
        "estimates": estimates,
    }
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"trade_cost_estimate_{timestamp}.json"
    md_path = REPORT_DIR / f"trade_cost_estimate_{timestamp}.md"
    latest_json = REPORT_DIR / "latest_trade_cost_estimate.json"
    latest_md = REPORT_DIR / "latest_trade_cost_estimate.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {latest_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {latest_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
