#!/usr/bin/env python3
"""Mine short-cycle OHLCV factors before strategy generation.

The factor layer is part of the same research-only Strategy Agent. It reads
local Binance USDT-M futures candles, scores simple 3m/5m/15m factors against
forward returns/MFE/MAE, and writes evidence that later strategy synthesis must
consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_DIR = AGENT_ROOT / "factors"
LATEST_JSON = OUTPUT_DIR / "latest_factor_research.json"
LATEST_MD = OUTPUT_DIR / "latest_factor_research.md"
PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
TIMEFRAMES = ["3m", "5m", "15m"]
FEE_ROUND_TRIP = 0.001
MIN_SAMPLE = 80


@dataclass(frozen=True)
class FactorSpec:
    name: str
    description: str
    column: str
    direction: str
    source: str


FACTOR_SPECS = [
    FactorSpec("ret_3", "3-bar momentum", "ret_3", "high_long_low_short", "price_momentum"),
    FactorSpec("ret_12", "12-bar trend pressure", "ret_12", "high_long_low_short", "price_momentum"),
    FactorSpec("ema_gap", "Fast/slow EMA gap", "ema_gap", "high_long_low_short", "trend_structure"),
    FactorSpec("atr_pct", "Realized volatility floor", "atr_pct", "high_abs", "volatility"),
    FactorSpec("volume_ratio", "Relative volume expansion", "volume_ratio", "high_abs", "volume"),
    FactorSpec("breakout_pos", "Position inside recent 36-bar range", "breakout_pos", "high_long_low_short", "structure"),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def pair_data_path(pair: str, timeframe: str) -> Path:
    stem = pair.replace("/", "_").replace(":", "_")
    return REPO_ROOT / f"user_data/data/binance/futures/{stem}-{timeframe}-futures.feather"


def load_frame(pair: str, timeframe: str) -> pd.DataFrame:
    path = pair_data_path(pair, timeframe)
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.sort_values("date").reset_index(drop=True)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"] / out["close"].shift(3) - 1.0
    out["ret_12"] = out["close"] / out["close"].shift(12) - 1.0
    out["ema_fast"] = out["close"].ewm(span=8, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=55, adjust=False).mean()
    out["ema_gap"] = out["ema_fast"] / out["ema_slow"] - 1.0
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_pct"] = tr.rolling(14).mean() / out["close"]
    out["volume_ratio"] = out["volume"] / out["volume"].rolling(36).mean()
    recent_high = out["high"].rolling(36).max()
    recent_low = out["low"].rolling(36).min()
    denom = (recent_high - recent_low).replace(0, pd.NA)
    out["breakout_pos"] = (out["close"] - recent_low) / denom
    return out


def add_forward_labels(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = frame.copy()
    out["forward_return"] = out["close"].shift(-horizon) / out["close"] - 1.0
    forward_high = out["high"].shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    forward_low = out["low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    out["long_mfe"] = forward_high / out["close"] - 1.0
    out["long_mae"] = forward_low / out["close"] - 1.0
    out["short_mfe"] = out["close"] / forward_low - 1.0
    out["short_mae"] = out["close"] / forward_high - 1.0
    return out


def side_score(sample: pd.DataFrame, side: str) -> dict[str, Any]:
    if side == "long":
        returns = sample["forward_return"]
        mfe = sample["long_mfe"]
        mae = sample["long_mae"]
    else:
        returns = -sample["forward_return"]
        mfe = sample["short_mfe"]
        mae = sample["short_mae"]
    returns_after_fee = returns - FEE_ROUND_TRIP
    count = int(returns_after_fee.count())
    if count == 0:
        return {
            "sample": 0,
            "mean_forward_return_pct": 0.0,
            "mean_after_fee_pct": 0.0,
            "win_rate": 0.0,
            "mean_mfe_pct": 0.0,
            "mean_mae_pct": 0.0,
            "verdict": "insufficient_sample",
        }
    mean_after_fee = float(returns_after_fee.mean())
    win_rate = float((returns_after_fee > 0).mean())
    verdict = "edge_candidate" if count >= MIN_SAMPLE and mean_after_fee > 0 and win_rate >= 0.52 else "reject"
    return {
        "sample": count,
        "mean_forward_return_pct": round(float(returns.mean()) * 100, 4),
        "mean_after_fee_pct": round(mean_after_fee * 100, 4),
        "win_rate": round(win_rate, 4),
        "mean_mfe_pct": round(float(mfe.mean()) * 100, 4),
        "mean_mae_pct": round(float(mae.mean()) * 100, 4),
        "verdict": verdict,
    }


def quantile_sample(frame: pd.DataFrame, column: str, direction: str, side: str) -> pd.DataFrame:
    data = frame.dropna(subset=[column, "forward_return", "long_mfe", "long_mae", "short_mfe", "short_mae"])
    if data.empty:
        return data
    if direction == "high_abs":
        threshold = data[column].quantile(0.80)
        return data[data[column] >= threshold]
    if side == "long":
        threshold = data[column].quantile(0.80)
        return data[data[column] >= threshold]
    threshold = data[column].quantile(0.20)
    return data[data[column] <= threshold]


def evaluate_pair_timeframe(pair: str, timeframe: str) -> list[dict[str, Any]]:
    frame = add_features(load_frame(pair, timeframe))
    horizon = {"3m": 12, "5m": 12, "15m": 8}[timeframe]
    frame = add_forward_labels(frame, horizon)
    rows: list[dict[str, Any]] = []
    for spec in FACTOR_SPECS:
        for side in ["long", "short"]:
            sample = quantile_sample(frame, spec.column, spec.direction, side)
            score = side_score(sample, side)
            rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "horizon_bars": horizon,
                    "factor": spec.name,
                    "description": spec.description,
                    "side": side,
                    "source": spec.source,
                    **score,
                }
            )
    return rows


def build_payload() -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for pair in PAIRS:
        for timeframe in TIMEFRAMES:
            path = pair_data_path(pair, timeframe)
            audit = {"pair": pair, "timeframe": timeframe, "path": rel(path), "exists": path.exists()}
            if path.exists():
                frame = load_frame(pair, timeframe)
                audit.update(
                    {
                        "rows": int(len(frame)),
                        "first_utc": frame["date"].iloc[0].isoformat() if len(frame) else None,
                        "last_utc": frame["date"].iloc[-1].isoformat() if len(frame) else None,
                    }
                )
                evaluations.extend(evaluate_pair_timeframe(pair, timeframe))
            audits.append(audit)
    candidates = [
        item
        for item in evaluations
        if item["verdict"] == "edge_candidate"
    ]
    return {
        "generated_at_utc": now_utc(),
        "research_only": True,
        "market": "Binance USDT-M futures",
        "timeframes": TIMEFRAMES,
        "pairs": PAIRS,
        "fee_round_trip": FEE_ROUND_TRIP,
        "min_sample": MIN_SAMPLE,
        "data_audit": audits,
        "evaluations": evaluations,
        "edge_candidates": candidates,
        "summary": {
            "evaluations": len(evaluations),
            "edge_candidates": len(candidates),
            "verdict": "has_edge_candidates" if candidates else "no_edge_candidates",
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Factor Research Report",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Market: `{payload['market']}`",
        f"- Timeframes: `{', '.join(payload['timeframes'])}`",
        f"- Round-trip fee assumption: `{payload['fee_round_trip']}`",
        f"- Edge candidates: `{len(payload['edge_candidates'])}`",
        "",
        "## Edge Candidates",
        "",
        "| Pair | TF | Factor | Side | Sample | Mean After Fee % | Win Rate | MFE % | MAE % |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in payload["edge_candidates"]:
        lines.append(
            "| {pair} | {timeframe} | {factor} | {side} | {sample} | {mean_after_fee_pct} | {win_rate} | {mean_mfe_pct} | {mean_mae_pct} |".format(
                **item
            )
        )
    if not payload["edge_candidates"]:
        lines.append("| none | - | - | - | 0 | 0 | 0 | 0 | 0 |")
    lines.extend(
        [
            "",
            "## Top Evaluations By Mean After Fee",
            "",
            "| Pair | TF | Factor | Side | Sample | Mean After Fee % | Win Rate | Verdict |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    top = sorted(payload["evaluations"], key=lambda item: item["mean_after_fee_pct"], reverse=True)[:20]
    for item in top:
        lines.append(
            "| {pair} | {timeframe} | {factor} | {side} | {sample} | {mean_after_fee_pct} | {win_rate} | {verdict} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- This report is evidence for the same Strategy Agent, not a separate Agent.",
            "- Strategy synthesis may only consume rows with `verdict=edge_candidate`, unless the next run is explicitly a negative-control or redesign experiment.",
            "- This factor layer does not modify Freqtrade config, dry-run config, live config, or exchange credentials.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    timestamp = payload["generated_at_utc"]
    json_path = OUTPUT_DIR / f"factor_research_{timestamp}.json"
    md_path = OUTPUT_DIR / f"factor_research_{timestamp}.md"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(text, encoding="utf-8")
    LATEST_JSON.write_text(text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")


if __name__ == "__main__":
    main()
