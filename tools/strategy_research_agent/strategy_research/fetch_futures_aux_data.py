#!/usr/bin/env python3
"""Download Binance USDT-M futures funding-rate and mark-price static data."""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "user_data/data/binance/futures_aux"
BASE_URL = "https://data.binance.vision/data/futures/um/monthly"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["funding_rate", "mark_price"],
        default=[],
        help="Dataset to download. Defaults to both.",
    )
    parser.add_argument("--timeframe", default="1m")
    return parser.parse_args()


def month_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    months = []
    year, month = start_dt.year, start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def fetch_zip(url: str) -> bytes | None:
    request = Request(url, headers={"User-Agent": "local-strategy-researcher/0.1"})
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - public static dataset.
            return response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except URLError:
        raise


def read_single_csv_from_zip(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one csv in archive, found {names}")
        with archive.open(names[0]) as handle:
            return pd.read_csv(handle)


def funding_url(pair: str, month: str) -> str:
    return f"{BASE_URL}/fundingRate/{pair}/{pair}-fundingRate-{month}.zip"


def mark_url(pair: str, timeframe: str, month: str) -> str:
    return f"{BASE_URL}/markPriceKlines/{pair}/{timeframe}/{pair}-{timeframe}-{month}.zip"


def normalize_funding(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(frame["calc_time"], unit="ms", utc=True),
            "pair": pair,
            "funding_interval_hours": frame["funding_interval_hours"].astype(float),
            "funding_rate": frame["last_funding_rate"].astype(float),
        }
    )


def normalize_mark(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(frame["open_time"], unit="ms", utc=True),
            "pair": pair,
            "open": frame["open"].astype(float),
            "high": frame["high"].astype(float),
            "low": frame["low"].astype(float),
            "close": frame["close"].astype(float),
        }
    )


def write_dataset(pair: str, dataset: str, frames: list[pd.DataFrame], suffix: str = "") -> Path:
    if not frames:
        raise ValueError(f"No frames downloaded for {pair} {dataset}")
    output_dir = OUTPUT_ROOT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
    pair_name = pair.replace("USDT", "_USDT_USDT")
    path = output_dir / f"{pair_name}{suffix}.feather"
    combined.reset_index(drop=True).to_feather(path)
    return path


def main() -> None:
    args = parse_args()
    datasets = args.dataset or ["funding_rate", "mark_price"]
    months = month_range(args.start_month, args.end_month)
    for pair in args.pair:
        for dataset in datasets:
            frames: list[pd.DataFrame] = []
            for month in months:
                if dataset == "funding_rate":
                    url = funding_url(pair, month)
                else:
                    url = mark_url(pair, args.timeframe, month)
                raw = fetch_zip(url)
                if raw is None:
                    print(f"missing {dataset} {pair} {month}")
                    continue
                frame = read_single_csv_from_zip(raw)
                if dataset == "funding_rate":
                    frames.append(normalize_funding(frame, pair))
                else:
                    frames.append(normalize_mark(frame, pair))
                print(f"downloaded {dataset} {pair} {month}")
            suffix = f"-{args.timeframe}" if dataset == "mark_price" else ""
            path = write_dataset(pair, dataset, frames, suffix)
            print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
