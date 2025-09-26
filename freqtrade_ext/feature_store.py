from __future__ import annotations

import pathlib

import pandas as pd


try:  # Lazy import for test environments without pyarrow
    import pyarrow.dataset as ds
except Exception:  # pragma: no cover
    ds = None


# Public API ---------------------------------------------------------------


def load_orderbook_features(
    exchange: str,
    pair: str,
    timeframe: str,
    timerange: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    embargo_secs: int = 1,
    depth: int = 200,
    root_dir: str | None = None,
) -> pd.DataFrame:
    """
    Load 1-second parquet orderbook summaries and aggregate to minute bars using
    right-open window [t-60s, t), then apply an embargo (drop last n seconds) and
    finally shift(1) to prevent look-ahead leakage.

    Returns a DataFrame indexed by DatetimeIndex(name='date').
    """
    # Normalize pair like "BTC/USDT:USDT" -> "BTCUSDT"
    base_token = pair.split(":")[0]
    base = base_token.replace("/", "")
    if root_dir is None:
        root = f"user_data/featurestore/{exchange}/{base}/1s"
    else:
        root = str(pathlib.Path(root_dir) / exchange / base / "1s")

    if timerange is not None:
        start, end = timerange
    else:
        end = pd.Timestamp.utcnow().tz_localize("UTC")
        start = end - pd.Timedelta(days=7)

    dataset, part_filter = _scanner_for_timerange(root, start, end)
    try:
        tbl = (
            dataset.to_table(filter=part_filter) if part_filter is not None else dataset.to_table()
        )
    except Exception:
        return _empty_index()

    if tbl.num_rows == 0:
        return _empty_index()

    df = tbl.to_pandas(types_mapper=None)
    # ts が列ではなく index として復元されたケースへ対応
    if "ts" not in df.columns:
        if df.index.name in ("ts", "date"):
            df = df.reset_index().rename(columns={df.index.name: "ts"})
    if "ts" not in df.columns:
        return _empty_index()

    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df = df.set_index(pd.to_datetime(df["ts"], utc=True)).drop(columns=["ts"])

    # Aggregate to minute using right-open window and embargo
    agg = _minute_last_right_open(df, embargo_secs=embargo_secs)

    # Derive additional features
    agg = _derive_features(agg, depth=depth)

    # Shift to prevent leakage
    agg = _safe_shift(agg, list(agg.columns), n=1)

    return agg


# Internal helpers ---------------------------------------------------------


def _empty_index() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))


def _safe_shift(df: pd.DataFrame, cols: list[str], n: int = 1) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].shift(n)
    return out


def _scanner_for_timerange(root: str, start: pd.Timestamp, end: pd.Timestamp):
    """
    Build a dataset scanner for the featurestore path, filtering out non-parquet files.
    Tries directory partitioning first; on failure, falls back to explicit parquet file listing.
    """
    if ds is None:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to read the feature store. Install requirements-ext.txt"
        )

    dataset = None
    part_filter = None
    try:
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("year", pa.int32()),
                pa.field("month", pa.int32()),
                pa.field("day", pa.int32()),
            ]
        )
        partitioning = ds.partitioning(schema=schema, flavor="directory")
        dataset = ds.dataset(root, format="parquet", partitioning=partitioning)
        part_filter = (ds.field("year") >= int(start.year)) & (ds.field("year") <= int(end.year))
        return dataset, part_filter
    except Exception:
        # Fallback: enumerate parquet files explicitly to avoid non-parquet (e.g., *.json) in root
        try:
            p = pathlib.Path(root)
            files = [str(x) for x in p.rglob("*.parquet")]
            if not files:
                raise FileNotFoundError("No parquet files found under featurestore path")
            dataset = ds.dataset(files, format="parquet")
            part_filter = None
            return dataset, part_filter
        except Exception:
            # Re-raise the original error for visibility
            dataset = ds.dataset(root, format="parquet")
            part_filter = None
            return dataset, part_filter


def _minute_last_right_open(df_sec: pd.DataFrame, embargo_secs: int = 1) -> pd.DataFrame:
    s = df_sec.copy()
    # Drop last n seconds within each minute (right-open window [t-60, t))
    s = s[s.index.second <= (59 - max(0, int(embargo_secs)))]
    out = s.groupby(s.index.floor("min")).last()
    out.index.name = "date"
    return out


def _derive_features(df: pd.DataFrame, depth: int) -> pd.DataFrame:
    out = df.copy()

    # Relative spread and bps
    if {"spread", "mid"}.issubset(out.columns):
        out["rel_spread"] = out["spread"] / out["mid"].clip(lower=1e-9)
        out["spread_bps"] = 1e4 * out["rel_spread"]

    # Normalize imbalance names
    if f"imb_{depth}" in out.columns:
        out["ob_imbalance"] = out[f"imb_{depth}"]
    if f"depth_delta_{depth}" in out.columns:
        out["ob_depth_delta"] = out[f"depth_delta_{depth}"]

    # Microprice (top-of-book weighted)
    for col in ["top_bid_qty", "top_ask_qty", "best_bid", "best_ask"]:
        if col not in out.columns:
            out[col] = pd.NA
    denom = out.get("top_bid_qty", 0).fillna(0) + out.get("top_ask_qty", 0).fillna(0)
    out["microprice"] = (
        (out["best_ask"] * out.get("top_bid_qty", 0).fillna(0))
        + (out["best_bid"] * out.get("top_ask_qty", 0).fillna(0))
    ) / denom.replace(0, pd.NA)

    # OFI (top-of-book proxy)
    out["d_best_bid"] = out.get("best_bid").diff()
    out["d_best_ask"] = out.get("best_ask").diff()
    out["d_top_bid_q"] = out.get("top_bid_qty").diff()
    out["d_top_ask_q"] = out.get("top_ask_qty").diff()
    out["ofi_top"] = (out["d_top_bid_q"].clip(lower=0) - out["d_top_ask_q"].clip(lower=0)).fillna(0)

    # Book slope proxy
    if "ob_depth_delta" in out.columns:
        out["book_slope"] = out["ob_depth_delta"].diff()

    # Cleanup
    out = out.replace([float("inf"), float("-inf")], pd.NA)

    return out
