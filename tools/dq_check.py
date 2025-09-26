#!/usr/bin/env python3

"""
Data Quality report generator for orderbook parquet data.
- Computes missing rate, inter-second gaps, negative/zero spreads, outliers by quantile.
- Writes YYYY-MM-DD_dq.json next to parquet partitions.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


@dataclass
class DQMetrics:
    date: str
    rows: int
    missing_rate: float
    negative_spread_pct: float
    median_spread_bps: float | None
    p99_spread_bps: float | None
    max_gap_seconds: int


def run_dq(root: str, date: str) -> None:
    year, month, day = [int(x) for x in date.split("-")]
    # Support numeric directory partitioning (year/month/day)
    try:
        schema = pa.schema(
            [
                pa.field("year", pa.int32()),
                pa.field("month", pa.int32()),
                pa.field("day", pa.int32()),
            ]
        )
        partitioning = ds.partitioning(schema=schema, flavor="directory")
        dataset = ds.dataset(root, format="parquet", partitioning=partitioning)
        filt = (ds.field("year") == year) & (ds.field("month") == month) & (ds.field("day") == day)
        tbl = dataset.to_table(filter=filt)
    except Exception:
        dataset = ds.dataset(root, format="parquet")
        tbl = dataset.to_table()
    df = tbl.to_pandas()
    if not df.empty and "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
        # If we could not filter by partition, filter by date here
        df = df[df.index.date == pd.to_datetime(date).date()]
    if df.empty:
        print("No data for", date)
        return

    df = df.sort_index()

    rows = len(df)
    missing_rate = float(df.isna().mean().mean())
    negative_spread_pct = float((df.get("spread", 0) < 0).mean())
    if "spread" in df.columns and "mid" in df.columns:
        spread_bps = 1e4 * df["spread"] / df["mid"].replace(0, pd.NA)
        median_spread_bps = float(spread_bps.median()) if spread_bps.notna().any() else None
        p99_spread_bps = float(spread_bps.quantile(0.99)) if spread_bps.notna().any() else None
    else:
        median_spread_bps = None
        p99_spread_bps = None

    gaps = df.index.to_series().diff().dt.total_seconds().fillna(1).astype(int)
    max_gap_seconds = int(gaps.max())

    metrics = DQMetrics(
        date=date,
        rows=rows,
        missing_rate=missing_rate,
        negative_spread_pct=negative_spread_pct,
        median_spread_bps=median_spread_bps,
        p99_spread_bps=p99_spread_bps,
        max_gap_seconds=max_gap_seconds,
    )

    out_path = Path(root) / f"{date}_dq.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print("DQ written:", str(out_path))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "root", help="root directory of parquet (e.g., user_data/featurestore/bybit/BTCUSDT/1s)"
    )
    p.add_argument("date", help="date YYYY-MM-DD")
    args = p.parse_args()
    run_dq(args.root, args.date)
