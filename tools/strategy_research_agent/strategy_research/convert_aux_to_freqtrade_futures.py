#!/usr/bin/env python3
"""Convert downloaded Binance futures aux data to Freqtrade futures candle files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
AUX_ROOT = REPO_ROOT / "user_data/data/binance/futures_aux"
FUTURES_DATA_ROOT = REPO_ROOT / "user_data/data/binance/futures"
REPORT_DIR = REPO_ROOT / "user_data/strategy_research/cost_audits"
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT"]
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass
class ConvertedFile:
    pair: str
    candle_type: str
    timeframe: str
    rows: int
    first_utc: str | None
    last_utc: str | None
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=PAIRS)
    parser.add_argument("--timeframe", default="1h")
    return parser.parse_args()


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def summarize(pair: str, candle_type: str, timeframe: str, frame: pd.DataFrame, path: Path) -> ConvertedFile:
    return ConvertedFile(
        pair=pair,
        candle_type=candle_type,
        timeframe=timeframe,
        rows=int(len(frame)),
        first_utc=frame["date"].iloc[0].isoformat() if not frame.empty else None,
        last_utc=frame["date"].iloc[-1].isoformat() if not frame.empty else None,
        output_path=rel_path(path),
    )


def convert_funding(pair: str, timeframe: str) -> ConvertedFile:
    source = AUX_ROOT / "funding_rate" / f"{pair}.feather"
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_feather(source)
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.floor("s")
    rate = frame["funding_rate"].astype(float)
    output = pd.DataFrame(
        {
            "date": frame["date"],
            "open": rate,
            "high": rate,
            "low": rate,
            "close": rate,
            "volume": 0.0,
        }
    )[OHLCV_COLUMNS].sort_values("date")
    output_path = FUTURES_DATA_ROOT / f"{pair}-{timeframe}-funding_rate.feather"
    output.reset_index(drop=True).to_feather(output_path, compression_level=9, compression="lz4")
    return summarize(pair, "funding_rate", timeframe, output, output_path)


def convert_mark(pair: str, timeframe: str) -> ConvertedFile:
    source = AUX_ROOT / "mark_price" / f"{pair}-1m.feather"
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_feather(source)
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.floor("min")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame = frame.set_index("date").sort_index()
    output = (
        frame.resample(timeframe)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    output = output[OHLCV_COLUMNS].sort_values("date")
    output_path = FUTURES_DATA_ROOT / f"{pair}-{timeframe}-mark.feather"
    output.reset_index(drop=True).to_feather(output_path, compression_level=9, compression="lz4")
    return summarize(pair, "mark", timeframe, output, output_path)


def write_report(items: list[ConvertedFile]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "converted_files": [asdict(item) for item in items],
    }
    json_path = REPORT_DIR / "latest_freqtrade_futures_aux_conversion.json"
    lines = [
        "# Freqtrade Futures Aux Conversion",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        "",
        "| Pair | Candle Type | Timeframe | Rows | First | Last | Output |",
        "|---|---|---|---:|---|---|---|",
    ]
    for item in payload["converted_files"]:
        lines.append(
            "| {pair} | {candle_type} | {timeframe} | {rows} | {first_utc} | {last_utc} | {output_path} |".format(
                **item
            )
        )
    md_path = REPORT_DIR / "latest_freqtrade_futures_aux_conversion.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {rel_path(json_path)}")
    print(f"Wrote {rel_path(md_path)}")


def main() -> None:
    args = parse_args()
    FUTURES_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    converted: list[ConvertedFile] = []
    for pair in args.pairs:
        converted.append(convert_funding(pair, args.timeframe))
        converted.append(convert_mark(pair, args.timeframe))
    write_report(converted)


if __name__ == "__main__":
    main()
