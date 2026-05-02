#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.data_quality import (
    check_ohlcv_parquet,
    default_quality_output_path,
    write_quality_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Bot Factory OHLCV parquet quality.")
    parser.add_argument("files", nargs="+", help="OHLCV parquet files to check.")
    parser.add_argument("--timeframe", default=None, help="Expected timeframe, for example 5m or 1h.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path. Defaults to registry/strategies/checks/.",
    )
    parser.add_argument("--no-fail", action="store_true", help="Always exit with status 0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [check_ohlcv_parquet(Path(file_path), args.timeframe) for file_path in args.files]
    output = Path(args.output) if args.output else default_quality_output_path()
    write_quality_reports(reports, output)

    for report in reports:
        print(report.to_json())
    print(f"OHLCV quality report written: {output}")

    ok = all(report.ok for report in reports)
    if not ok and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
