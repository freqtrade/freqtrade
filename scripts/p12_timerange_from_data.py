import argparse
import logging
import sys
from pathlib import Path
from datetime import timezone

# Add current directory to path to allow importing from workspace freqtrade
import os

sys.path.append(os.getcwd())

# Avoid rapidjson dependency if possible by mocking it before other imports if they use it
try:
    import rapidjson
except ImportError:
    import json

    sys.modules["rapidjson"] = json

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("p12_timerange_from_data")


def get_timerange(pair: str, timeframe: str, datadir: str, data_format: str):
    try:
        data = load_pair_history(
            datadir=Path(datadir),
            timeframe=timeframe,
            pair=pair,
            data_format=data_format,
            candle_type=CandleType.SPOT,
        )

        if data.empty:
            print(
                f"ERROR: No data found for {pair} {timeframe} in {datadir} (format: {data_format})",
                file=sys.stderr,
            )
            sys.exit(1)

        start_date = data.iloc[0]["date"].replace(tzinfo=timezone.utc)
        end_date = data.iloc[-1]["date"].replace(tzinfo=timezone.utc)

        # Format: YYYYMMDD-YYYYMMDD
        print(f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}")

    except Exception as e:
        print(f"ERROR: Failed to load data: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Compute timerange from stored OHLCV data.")
    parser.add_argument("--pair", required=True, help="Pair (e.g., RELIANCE/INR)")
    parser.add_argument("--tf", required=True, help="Timeframe (e.g., 5m)")
    parser.add_argument("--datadir", required=True, help="Data directory")
    parser.add_argument("--format", default="feather", help="Data format (json or feather)")

    args = parser.parse_args()
    get_timerange(args.pair, args.tf, args.datadir, args.format)


if __name__ == "__main__":
    main()
