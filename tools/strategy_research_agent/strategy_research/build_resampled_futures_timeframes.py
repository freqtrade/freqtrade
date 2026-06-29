#!/usr/bin/env python3
"""Build local 3m/5m futures OHLCV files from existing 1m data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
DATA_ROOT = REPO_ROOT / "user_data/data/binance/futures"
REPORT_DIR = AGENT_ROOT / "data_updates"
REPORT_JSON = REPORT_DIR / "latest_resampled_timeframes.json"
REPORT_MD = REPORT_DIR / "latest_resampled_timeframes.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=["BTC_USDT_USDT", "ETH_USDT_USDT"])
    parser.add_argument("--timeframes", nargs="+", default=["3m", "5m"])
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def pandas_rule(timeframe: str) -> str:
    if not timeframe.endswith("m"):
        raise ValueError(f"Only minute resampling is supported: {timeframe}")
    return f"{int(timeframe[:-1])}min"


def resample_pair(pair: str, timeframe: str) -> dict[str, Any]:
    source = DATA_ROOT / f"{pair}-1m-futures.feather"
    target = DATA_ROOT / f"{pair}-{timeframe}-futures.feather"
    if not source.exists():
        return {"pair": pair, "timeframe": timeframe, "status": "missing_source", "source": rel(source)}
    frame = pd.read_feather(source)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.set_index("date").sort_index()
    output = (
        frame.resample(pandas_rule(timeframe), label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    output.to_feather(target)
    return {
        "pair": pair,
        "timeframe": timeframe,
        "status": "ok",
        "source": rel(source),
        "target": rel(target),
        "rows": int(len(output)),
        "first_utc": output["date"].iloc[0].isoformat() if len(output) else None,
        "last_utc": output["date"].iloc[-1].isoformat() if len(output) else None,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Resampled Futures Timeframes",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        "",
        "| Pair | Timeframe | Status | Rows | First | Last | Target |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in payload["outputs"]:
        lines.append(
            "| {pair} | {timeframe} | {status} | {rows} | {first_utc} | {last_utc} | {target} |".format(
                pair=row.get("pair", ""),
                timeframe=row.get("timeframe", ""),
                status=row.get("status", ""),
                rows=row.get("rows", 0),
                first_utc=row.get("first_utc", ""),
                last_utc=row.get("last_utc", ""),
                target=row.get("target", row.get("source", "")),
            )
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [resample_pair(pair, timeframe) for pair in args.pairs for timeframe in args.timeframes]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "outputs": outputs,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload)
    print(f"Wrote {rel(REPORT_JSON)}")
    print(f"Wrote {rel(REPORT_MD)}")


if __name__ == "__main__":
    main()
