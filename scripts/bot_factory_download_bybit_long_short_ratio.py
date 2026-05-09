#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.bybit_long_short_ratio import (
    BybitLongShortRatioDownloadInputs,
    default_bybit_long_short_ratio_path,
    download_bybit_long_short_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download public Bybit V5 long/short account ratio market data "
            "for Bot Factory local research screens. This command uses no API "
            "keys and does not call exchange order endpoints."
        )
    )
    parser.add_argument("--symbol", required=True, help="Bybit symbol, e.g. BTCUSDT.")
    parser.add_argument(
        "--category",
        default="linear",
        choices=["linear", "inverse"],
        help="Bybit market category.",
    )
    parser.add_argument(
        "--period",
        default="1h",
        choices=["5min", "15min", "30min", "1h", "4h", "1d"],
        help="Bybit account-ratio period.",
    )
    parser.add_argument(
        "--timerange",
        default=None,
        help="UTC timerange in YYYYMMDD-YYYYMMDD form. End is exclusive.",
    )
    parser.add_argument("--start", default=None, help="UTC ISO start time.")
    parser.add_argument("--end", default=None, help="UTC ISO end time.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet/CSV path. Defaults to user_data/data/bybit/futures/.",
    )
    parser.add_argument("--base-url", default="https://api.bybit.com")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time, end_time = _resolve_time_window(args)
    output = (
        Path(args.output)
        if args.output
        else default_bybit_long_short_ratio_path(
            args.symbol,
            category=args.category,
            period=args.period,
        )
    )
    artifact = download_bybit_long_short_ratio(
        BybitLongShortRatioDownloadInputs(
            root_dir=ROOT_DIR,
            symbol=args.symbol,
            category=args.category,
            period=args.period,
            start_time=start_time,
            end_time=end_time,
            output_path=output,
            base_url=args.base_url,
            limit=args.limit,
            max_pages=args.max_pages,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0 if artifact["status"] == "completed" else 1


def _resolve_time_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.timerange:
        return _parse_timerange(args.timerange)
    if not args.start or not args.end:
        raise SystemExit("Provide --timerange or both --start and --end.")
    return _parse_datetime(args.start), _parse_datetime(args.end)


def _parse_timerange(value: str) -> tuple[datetime, datetime]:
    if "-" not in value:
        raise SystemExit("--timerange must use YYYYMMDD-YYYYMMDD.")
    start_raw, end_raw = value.split("-", 1)
    return _parse_date(start_raw), _parse_date(end_raw)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
