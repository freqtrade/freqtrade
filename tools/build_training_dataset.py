#!/usr/bin/env python3
"""
Build a flat training dataset by left-joining OHLCV with orderbook features.

Usage example:
  python tools/build_training_dataset.py \
    --exchange binanceusdm \
    --datadir user_data/data/binanceusdm \
    --pairs BTC/USDT:USDT ETH/USDT:USDT \
    --timeframe 5m \
    --timerange 20250710-20250805 \
    --out user_data/datasets/ob_dataset_5m.feather \
    --label-period 12

Notes:
- Joins orderbook features using a left join on UTC DatetimeIndex('date'),
  applying embargo and shift(1) inside load_orderbook_features to avoid leakage.
- Adds simple forward-return label column '&-target' if --label-period is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.data.history.history_utils import load_pair_history
from freqtrade.enums import CandleType


try:
    from freqtrade_ext.feature_store import load_orderbook_features
except Exception:  # pragma: no cover
    raise SystemExit(
        "freqtrade_ext.feature_store not available. Install extras from requirements-ext.txt"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build flat dataset with OB features (left join)")
    p.add_argument(
        "--exchange",
        required=True,
        help="Exchange id used for featurestore path (e.g., binanceusdm)",
    )
    p.add_argument(
        "--datadir",
        required=True,
        help=("Path to OHLCV data directory (e.g., user_data/data/binanceusdm)"),
    )
    p.add_argument(
        "--pairs", nargs="+", required=True, help="Pairs, e.g. BTC/USDT:USDT ETH/USDT:USDT"
    )
    p.add_argument("--timeframe", required=True, help="Timeframe, e.g. 5m")
    p.add_argument("--timerange", required=True, help="Timerange, e.g. 20250710-20250805")
    p.add_argument("--out", required=True, help="Output file (.feather/.parquet/.csv)")
    p.add_argument(
        "--embargo-secs", type=int, default=1, help="Embargo seconds before minute close"
    )
    p.add_argument("--depth", type=int, default=200, help="Orderbook aggregation depth selector")
    p.add_argument(
        "--label-period",
        type=int,
        default=0,
        help="Forward label period in candles (if > 0, adds '&-target')",
    )
    p.add_argument(
        "--featurestore-root",
        default=None,
        help=("Override featurestore root (default: user_data/featurestore/<exchange>/<base>/1s)"),
    )
    return p.parse_args()


def _join_features(ohlcv: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    # Ensure timezone-aware datetime alignment
    had_date_col = "date" in df.columns
    if had_date_col:
        idx = pd.to_datetime(df["date"], utc=True, errors="coerce")
        # Drop original 'date' to avoid duplicate after reset_index
        df = df.drop(columns=["date"])  # avoid duplicate column names
    else:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df["__tmp_idx__"] = idx
    df = df.set_index("__tmp_idx__")
    out = df.join(feats, how="left")  # left join on DatetimeIndex
    out = out.reset_index().rename(columns={"index": "date", "__tmp_idx__": "date"})
    # Basic NA/infs cleanup
    out = out.replace([float("inf"), float("-inf")], pd.NA).fillna(method="ffill").fillna(0)
    # Prefix representative OB columns for clarity
    for c in [
        "spread_bps",
        "microprice",
        "ob_imbalance",
        "ob_depth_delta",
        "ofi_top",
        "book_slope",
    ]:
        if c in out.columns:
            out[f"feat__{c}"] = out[c]
    return out


def _add_forward_label(df: pd.DataFrame, label_period: int) -> pd.DataFrame:
    if label_period and label_period > 0 and "close" in df.columns:
        df["&-target"] = df["close"].shift(-int(label_period)) / df["close"] - 1.0
    return df


def main() -> None:
    args = _parse_args()
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    tr = TimeRange.parse_timerange(args.timerange)
    datadir = Path(args.datadir)

    frames: list[pd.DataFrame] = []
    for pair in args.pairs:
        # Load candles (respect trading mode: futures for ':USDT' suffix pairs)
        candle_type = CandleType.FUTURES if ":" in pair else CandleType.SPOT
        ohlcv = load_pair_history(
            pair=pair,
            timeframe=args.timeframe,
            datadir=datadir,
            timerange=tr,
            candle_type=candle_type,
        )
        if ohlcv.empty:
            continue
        # OB features for the same timerange
        start = pd.to_datetime(ohlcv["date"].min(), utc=True)
        end = pd.to_datetime(ohlcv["date"].max(), utc=True)
        try:
            feats = load_orderbook_features(
                exchange=args.exchange,
                pair=pair,
                timeframe=args.timeframe,
                timerange=(start, end),
                embargo_secs=int(args.embargo_secs),
                depth=int(args.depth),
                root_dir=args.featurestore_root,
            )
        except Exception:
            feats = pd.DataFrame()
        # Left join
        merged = _join_features(ohlcv, feats)
        merged["pair"] = pair
        # Optional forward label
        merged = _add_forward_label(merged, int(args.label_period))
        frames.append(merged)

    if not frames:
        raise SystemExit("No data merged. Check pairs/timerange/datadir.")

    out = pd.concat(frames, ignore_index=True)
    # Save in requested format
    suf = outpath.suffix.lower()
    if suf == ".csv":
        out.to_csv(outpath, index=False)
    elif suf in (".feather", ".ft"):  # feather
        out.to_feather(outpath)
    elif suf == ".parquet":
        out.to_parquet(outpath, index=False)
    else:
        # default to feather
        out.to_feather(outpath)
    print(f"Saved: {outpath} rows={len(out)} cols={len(out.columns)}")


if __name__ == "__main__":
    main()
