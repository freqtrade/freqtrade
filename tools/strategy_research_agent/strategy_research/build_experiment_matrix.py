#!/usr/bin/env python3
"""Build market-regime and cost-scenario experiments from local OHLCV data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REGIME_DIR = AGENT_ROOT / "market_regimes"
EXPERIMENT_DIR = AGENT_ROOT / "experiments"
BTC_1M = REPO_ROOT / "user_data/data/binance/futures/BTC_USDT_USDT-1m-futures.feather"


@dataclass
class RegimeSlice:
    name: str
    label: str
    timerange: str
    start_utc: str
    end_utc: str
    days: int
    btc_return_pct: float
    realized_vol_pct: float
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", default=[], help="Strategy to include. Repeatable.")
    parser.add_argument("--max-days", type=int, default=90, help="Maximum days per generated slice.")
    return parser.parse_args()


def load_btc_daily() -> pd.DataFrame:
    frame = pd.read_feather(BTC_1M)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.set_index("date").sort_index()
    daily = frame["close"].resample("1D").last().dropna().to_frame("close")
    daily["return_1d"] = daily["close"].pct_change()
    daily["return_30d"] = daily["close"] / daily["close"].shift(30) - 1.0
    daily["return_90d"] = daily["close"] / daily["close"].shift(90) - 1.0
    daily["vol_30d"] = daily["return_1d"].rolling(30).std() * (365**0.5)
    return daily.dropna()


def choose_window(daily: pd.DataFrame, score: pd.Series, max_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    center = score.dropna().idxmax()
    start = max(daily.index[0], center - pd.Timedelta(days=max_days // 2))
    end = min(daily.index[-1], start + pd.Timedelta(days=max_days - 1))
    if (end - start).days + 1 < max_days:
        start = max(daily.index[0], end - pd.Timedelta(days=max_days - 1))
    return start, end


def build_slice(daily: pd.DataFrame, name: str, label: str, start: pd.Timestamp, end: pd.Timestamp, notes: str) -> RegimeSlice:
    window = daily.loc[start:end]
    btc_return = window["close"].iloc[-1] / window["close"].iloc[0] - 1.0
    realized_vol = window["return_1d"].std() * (365**0.5)
    timerange = f"{start.strftime('%Y%m%d')}-{(end + pd.Timedelta(days=1)).strftime('%Y%m%d')}"
    return RegimeSlice(
        name=name,
        label=label,
        timerange=timerange,
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        days=int((end - start).days + 1),
        btc_return_pct=round(float(btc_return * 100), 2),
        realized_vol_pct=round(float(realized_vol * 100), 2),
        notes=notes,
    )


def detect_regimes(max_days: int) -> list[RegimeSlice]:
    daily = load_btc_daily()
    bull_start, bull_end = choose_window(daily, daily["return_90d"], max_days)
    bear_start, bear_end = choose_window(daily, -daily["return_90d"], max_days)
    high_vol_start, high_vol_end = choose_window(daily, daily["vol_30d"], max_days)
    range_score = -(daily["return_90d"].abs()) - daily["vol_30d"].rank(pct=True) * 0.05
    range_start, range_end = choose_window(daily, range_score, max_days)
    return [
        build_slice(daily, "bull", "BTC 90d strongest uptrend", bull_start, bull_end, "Highest BTC trailing 90d return window."),
        build_slice(daily, "bear", "BTC 90d strongest downtrend", bear_start, bear_end, "Lowest BTC trailing 90d return window."),
        build_slice(daily, "range", "BTC flattest lower-volatility range", range_start, range_end, "Low absolute BTC 90d return with lower volatility preference."),
        build_slice(daily, "high_vol", "BTC highest realized volatility", high_vol_start, high_vol_end, "Highest BTC trailing 30d annualized realized volatility window."),
    ]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_experiment(
    slices: list[RegimeSlice],
    strategies: list[str],
    fee: float,
    scenario_id: str,
    title: str,
    cost_notes: list[str],
) -> dict[str, Any]:
    return {
        "id": scenario_id,
        "title": title,
        "profile_ref": "strategy_registry.json",
        "timeframes": ["1m"],
        "fee": fee,
        "strategies": strategies,
        "matrix": {
            "timeranges": [
                {
                    "name": item.name,
                    "label": item.label,
                    "timerange": item.timerange,
                    "btc_return_pct": item.btc_return_pct,
                    "realized_vol_pct": item.realized_vol_pct,
                }
                for item in slices
            ]
        },
        "checks": {
            "backtesting": True,
            "recursive_analysis": False,
            "lookahead_analysis": False,
        },
        "cost_model": {
            "fee": fee,
            "slippage_included": False,
            "funding_included": False,
            "mark_price_included": False,
            "notes": cost_notes,
        },
    }


def main() -> None:
    args = parse_args()
    registry = load_json(AGENT_ROOT / "strategy_registry.json")
    default_strategies = [
        item["name"]
        for item in registry["strategies"]
        if item["name"] in {
            "BtcEthFuturesRegime10xPullbackShortOnlyStrategy",
            "BtcEthFuturesEthSelfPullbackShortOnlyStrategy",
        }
    ]
    strategies = args.strategy or default_strategies
    slices = detect_regimes(args.max_days)
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    regime_payload = {
        "generated_at_utc": generated_at,
        "source_data": str(BTC_1M.relative_to(REPO_ROOT)),
        "method": "BTC daily close derived from local 1m futures OHLCV; windows selected by trailing 90d return and 30d realized volatility.",
        "slices": [asdict(item) for item in slices],
    }
    write_json(REGIME_DIR / "btc_market_regime_slices.json", regime_payload)
    base = build_experiment(
        slices,
        strategies,
        0.0005,
        "candidate_regime_matrix_base_cost",
        "Candidate strategies across BTC market regimes, base fee",
        ["Uses explicit Freqtrade fee 0.05%. Funding, mark price, and slippage are not included yet."],
    )
    stress = build_experiment(
        slices,
        strategies,
        0.0010,
        "candidate_regime_matrix_stress_cost",
        "Candidate strategies across BTC market regimes, stress fee",
        ["Uses doubled explicit Freqtrade fee 0.10% as a simple stress proxy for fee/slippage drag.", "Funding and mark price are still not included."],
    )
    write_json(EXPERIMENT_DIR / "candidate_regime_matrix_base_cost.json", base)
    write_json(EXPERIMENT_DIR / "candidate_regime_matrix_stress_cost.json", stress)
    print(f"Wrote {REGIME_DIR.relative_to(REPO_ROOT)}/btc_market_regime_slices.json")
    print("Wrote user_data/strategy_research/experiments/candidate_regime_matrix_base_cost.json")
    print("Wrote user_data/strategy_research/experiments/candidate_regime_matrix_stress_cost.json")


if __name__ == "__main__":
    main()
