#!/usr/bin/env python3
"""Download Binance USDT-M 1m klines and convert them to Freqtrade feather data."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://data.binance.vision/data/futures/um"
COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
REPORT_DIR = Path("user_data/strategy_research/data_updates")


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def default_end_day() -> date:
    return datetime.now(timezone.utc).date()


def month_start(day: date) -> date:
    return day.replace(day=1)


def add_month(day: date) -> date:
    year = day.year + (day.month // 12)
    month = (day.month % 12) + 1
    return date(year, month, 1)


def month_end_exclusive(day: date) -> date:
    return add_month(month_start(day))


def iter_urls(symbol: str, start: date, end: date):
    """Yield monthly archives for whole months and daily archives for partial months."""
    current = start
    while current < end:
        if current.day == 1 and month_end_exclusive(current) <= end:
            yield (
                "monthly",
                current,
                f"{BASE_URL}/monthly/klines/{symbol}/1m/{symbol}-1m-{current:%Y-%m}.zip",
            )
            current = add_month(current)
            continue

        yield (
            "daily",
            current,
            f"{BASE_URL}/daily/klines/{symbol}/1m/{symbol}-1m-{current:%Y-%m-%d}.zip",
        )
        current += timedelta(days=1)


def fetch_bytes(url: str, cache_file: Path) -> bytes | None:
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file.read_bytes()

    request = Request(url, headers={"User-Agent": "freqtrade-local-research/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            print(f"skip missing: {url}")
            return None
        raise
    except URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(payload)
    return payload


def frame_from_zip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"expected one csv in archive, found {csv_names}")
        with archive.open(csv_names[0]) as handle:
            frame = pd.read_csv(handle, header=None, names=COLUMNS)

    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
    frame = frame.dropna(subset=["open_time"])
    frame["date"] = pd.to_datetime(frame["open_time"].astype("int64"), unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["date", "open", "high", "low", "close", "volume"]].dropna()


def output_name(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    return f"{base}_USDT_USDT-1m-futures.feather"


def existing_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame[["date", "open", "high", "low", "close", "volume"]]


def incremental_start(existing: pd.DataFrame, fallback_start: date) -> date:
    if existing.empty:
        return fallback_start
    last_ts = existing["date"].max()
    return (last_ts + pd.Timedelta(minutes=1)).date()


def count_gaps(data: pd.DataFrame) -> int:
    gaps = int(data["date"].diff().ne(pd.Timedelta(minutes=1)).sum() - 1)
    return max(gaps, 0)


def download_symbol(
    symbol: str,
    start: date,
    end: date,
    data_dir: Path,
    cache_dir: Path,
    incremental: bool,
) -> dict[str, object]:
    out_file = data_dir / "futures" / output_name(symbol)
    prior = existing_data(out_file) if incremental else pd.DataFrame()
    effective_start = incremental_start(prior, start) if incremental else start
    if effective_start >= end:
        print(f"{symbol}: up to date through {prior['date'].max() if not prior.empty else 'none'}")
        return {
            "symbol": symbol,
            "status": "up_to_date",
            "rows": int(len(prior)),
            "first_utc": prior["date"].min().isoformat() if not prior.empty else None,
            "last_utc": prior["date"].max().isoformat() if not prior.empty else None,
            "gaps": count_gaps(prior) if not prior.empty else None,
            "archives": 0,
            "output": str(out_file),
        }

    frames: list[pd.DataFrame] = []
    downloaded = 0
    for archive_type, archive_date, url in iter_urls(symbol, effective_start, end):
        suffix = archive_date.strftime("%Y-%m" if archive_type == "monthly" else "%Y-%m-%d")
        cache_file = cache_dir / archive_type / symbol / "1m" / f"{symbol}-1m-{suffix}.zip"
        payload = fetch_bytes(url, cache_file)
        if payload is None:
            continue
        frames.append(frame_from_zip(payload))
        downloaded += 1
        print(f"{symbol}: loaded {archive_type} {suffix}")

    if not frames:
        if incremental and not prior.empty:
            print(f"{symbol}: no new archives available after {prior['date'].max()}")
            return {
                "symbol": symbol,
                "status": "no_new_archives",
                "rows": int(len(prior)),
                "first_utc": prior["date"].min().isoformat(),
                "last_utc": prior["date"].max().isoformat(),
                "gaps": count_gaps(prior),
                "archives": 0,
                "output": str(out_file),
            }
        raise RuntimeError(f"no data downloaded for {symbol}")

    new_data = pd.concat(frames, ignore_index=True)
    start_ts = pd.Timestamp(effective_start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    new_data = new_data[(new_data["date"] >= start_ts) & (new_data["date"] < end_ts)]
    data = pd.concat([prior, new_data], ignore_index=True) if incremental and not prior.empty else new_data
    data = data.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    expected = int((data["date"].max() - data["date"].min()).total_seconds() // 60) + 1
    actual = len(data)
    gaps = count_gaps(data)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_feather(out_file, compression="lz4", compression_level=9)

    print(
        f"{symbol}: wrote {out_file} | rows={actual:,}/{expected:,} "
        f"| range={data['date'].min()} -> {data['date'].max()} | gaps={gaps} | archives={downloaded}"
    )
    return {
        "symbol": symbol,
        "status": "updated",
        "rows": actual,
        "expected_rows": expected,
        "first_utc": data["date"].min().isoformat(),
        "last_utc": data["date"].max().isoformat(),
        "gaps": gaps,
        "archives": downloaded,
        "incremental_start": effective_start.isoformat(),
        "end_exclusive": end.isoformat(),
        "output": str(out_file),
    }


def write_report(results: list[dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "results": results,
    }
    json_path = REPORT_DIR / "latest_ohlcv_1m_update.json"
    md_path = REPORT_DIR / "latest_ohlcv_1m_update.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Binance USDT-M 1m OHLCV Update",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        "",
        "| Symbol | Status | Rows | First | Last | Gaps | Archives |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {symbol} | {status} | {rows} | {first_utc} | {last_utc} | {gaps} | {archives} |".format(
                **item
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start", type=parse_day, default=parse_day("2024-01-01"))
    parser.add_argument("--end", type=parse_day, default=default_end_day())
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument("--cache-dir", type=Path, default=Path("user_data/data/binance/public_data_cache"))
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    if args.end <= args.start:
        raise SystemExit("--end must be after --start")

    results = []
    for symbol in args.symbols:
        results.append(
            download_symbol(
                symbol,
                args.start,
                args.end,
                args.data_dir,
                args.cache_dir,
                args.incremental,
            )
        )
    write_report(results)


if __name__ == "__main__":
    main()
