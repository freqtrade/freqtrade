#!/usr/bin/env python3
"""Feature store inspection helper.

Usage (example):
  PYTHONPATH=. ./tools/featurestore_inspect.py \
    --exchange bybit --symbol BTCUSDT \
    --root user_data/featurestore/bybit/BTCUSDT/1s \
    --lookback-min 30 --depth 200 --embargo 1

What it does:
  1. Reads raw 1s parquet dataset under --root, prints basic stats.
  2. Computes gap statistics between consecutive seconds.
  3. Loads engineered minute features via freqtrade_ext.feature_store.load_orderbook_features.
  4. Prints feature DataFrame shape, column sample, head/tail times, NA ratios (top 15).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd


try:
    import pyarrow.dataset as ds
except Exception as e:  # pragma: no cover
    print(f"[inspect] pyarrow unavailable: {e}")
    ds = None

from typing import Any, Protocol


class LoaderFn(Protocol):
    def __call__(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        timerange: tuple[Any, Any] | None = None,
        embargo_secs: int = 1,
        depth: int = 200,
        root_dir: str | None = None,
    ) -> Any: ...


load_orderbook_features: Any
try:
    from freqtrade_ext.feature_store import load_orderbook_features as _real_loader

    load_orderbook_features = _real_loader
except Exception as e:  # pragma: no cover
    print(f"[inspect] feature_store import failed: {e}")
    load_orderbook_features = None


def read_raw(root: pathlib.Path) -> pd.DataFrame:
    if ds is None:
        raise RuntimeError("pyarrow.dataset not available")
    dataset = ds.dataset(root, format="parquet")
    table = dataset.to_table()
    if table.num_rows == 0:
        return pd.DataFrame()
    df = table.to_pandas()
    # Ensure ts column exists
    if "ts" not in df.columns:
        raise ValueError("Column 'ts' not found in raw dataset")
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    return df


def stats_raw(df: pd.DataFrame) -> None:
    if df.empty:
        print("[raw] empty dataset")
        return
    uniq = df["ts"].nunique()
    dup = len(df) - uniq
    print(f"[raw] rows={len(df)} cols={len(df.columns)} unique_ts={uniq} dup_rows={dup}")
    print(f"[raw] ts_range {df['ts'].min()} -> {df['ts'].max()}")
    ts = pd.to_datetime(df["ts"], utc=True).sort_values()
    if len(ts) > 1:
        gaps = ts.diff().dropna().dt.total_seconds()
        print(
            f"[raw] gap_max={gaps.max()} gap_p95={gaps.quantile(0.95)} gap_mean={gaps.mean():.3f}"
        )
    na = df.isna().mean().sort_values()
    print("[raw] na_ratio head10:")
    print(na.head(10).to_string())


def stats_features(exchange: str, symbol: str, lookback_min: int, depth: int, embargo: int) -> None:
    loader = load_orderbook_features
    if loader is None:
        print("[features] loader unavailable")
        return
    end = datetime.now(UTC).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=lookback_min)
    # mypy: load_orderbook_features is Optional; guarded above
    feats = loader(
        exchange,
        symbol,
        "1m",
        (pd.Timestamp(start), pd.Timestamp(end)),
        embargo_secs=embargo,
        depth=depth,
    )
    if feats.empty:
        print("[features] empty dataframe")
        return
    print(f"[features] shape={feats.shape} cols={len(feats.columns)}")
    print(f"[features] index range {feats.index.min()} -> {feats.index.max()}")
    print("[features] sample cols:", list(feats.columns[:12]))
    na = feats.isna().mean().sort_values(ascending=False)
    print("[features] top15 NA ratios:")
    print(na.head(15).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument(
        "--root", required=True, help="Path to 1s featurestore root (year/... partitions inside)"
    )
    ap.add_argument("--lookback-min", type=int, default=30)
    ap.add_argument("--depth", type=int, default=200)
    ap.add_argument("--embargo", type=int, default=1)
    args = ap.parse_args()

    root_path = pathlib.Path(args.root)
    if not root_path.exists():
        print(f"[inspect] root not found: {root_path}")
        sys.exit(1)

    try:
        raw_df = read_raw(root_path)
        stats_raw(raw_df)
    except Exception as e:
        print(f"[inspect] raw stats error: {e}")

    try:
        stats_features(args.exchange, args.symbol, args.lookback_min, args.depth, args.embargo)
    except Exception as e:
        print(f"[inspect] feature stats error: {e}")


if __name__ == "__main__":
    main()
