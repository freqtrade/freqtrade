import argparse
import json
import sys
from pathlib import Path


def generate_metrics(trades_file: str, out_file: str, pair: str, timeframe: str, timerange: str):
    try:
        trades_path = Path(trades_file)
        if not trades_path.exists():
            print(f"ERROR: Trades file not found: {trades_file}", file=sys.stderr)
            sys.exit(1)

        with trades_path.open("r") as f:
            data = json.load(f)

        if isinstance(data, list):
            trades = data
        elif isinstance(data, dict) and "strategy" in data:
            # Extract from nested Freqtrade results structure
            # {"strategy": {"StrategyName": {"trades": [...]}}}
            strat_name = next(iter(data["strategy"].keys()))
            trades = data["strategy"][strat_name].get("trades", [])
        else:
            trades = []

        trades_count = len(trades)
        total_profit = sum(t.get("profit_abs", 0) for t in trades)

        metrics = {
            "pair": pair,
            "timeframe": timeframe,
            "timerange": timerange,
            "trades_count": trades_count,
            "total_profit_abs": total_profit,
        }

        with Path(out_file).open("w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics written to {out_file}")

    except Exception as e:
        print(f"ERROR: Failed to generate metrics: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Extract metrics from backtest trades.")
    parser.add_argument("--trades", required=True, help="Path to backtest-trades.json")
    parser.add_argument("--out", required=True, help="Path to output metrics_summary.json")
    parser.add_argument("--pair", required=True, help="Pair")
    parser.add_argument("--tf", required=True, help="Timeframe")
    parser.add_argument("--timerange", required=True, help="Timerange used")

    args = parser.parse_args()
    generate_metrics(args.trades, args.out, args.pair, args.tf, args.timerange)


if __name__ == "__main__":
    main()
