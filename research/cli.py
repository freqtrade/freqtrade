# research/cli.py
"""Command-line entry point for the research package: `python -m research.cli gate
...` runs the promotion gate for a strategy and prints its verdict."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from freqtrade.configuration import Configuration
from research.gate import run_promotion_gate


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and dispatch to the requested subcommand. Currently only
    `gate` is implemented: it loads the freqtrade config, runs
    `run_promotion_gate`, prints the verdict and supporting statistics, and
    returns exit code 0 on pass or 1 on fail (for use in CI/scripts)."""
    parser = argparse.ArgumentParser(prog="research", description="Freqtrade research gate")
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="Run the promotion gate for a strategy")
    gate.add_argument("--strategy", required=True)
    gate.add_argument("--config", required=True, help="Path to a freqtrade config.json")
    gate.add_argument(
        "--pairs", required=True, help="Comma-separated pairs, e.g. BTC/USDT,ETH/USDT"
    )
    gate.add_argument("--timeframe", required=True)
    gate.add_argument("--start", required=True, help="YYYY-MM-DD")
    gate.add_argument("--end", required=True, help="YYYY-MM-DD")
    gate.add_argument("--train-days", type=int, required=True)
    gate.add_argument("--test-days", type=int, required=True)
    gate.add_argument("--param-grid", required=True, help="JSON list of param dicts")
    gate.add_argument("--db-path", default="user_data/research.sqlite")

    args = parser.parse_args(argv)

    if args.command == "gate":
        ft_config = Configuration.from_files([args.config])
        ft_config["strategy"] = args.strategy
        result = run_promotion_gate(
            config=ft_config,
            strategy_id=args.strategy,
            pairs=args.pairs.split(","),
            timeframe=args.timeframe,
            datadir=Path(ft_config["datadir"]),
            # freqtrade's Backtesting.backtest() compares against tz-aware (UTC) pandas
            # Timestamps internally -- a naive datetime here crashes deep inside it.
            start=datetime.fromisoformat(args.start).replace(tzinfo=UTC),
            end=datetime.fromisoformat(args.end).replace(tzinfo=UTC),
            train_days=args.train_days,
            test_days=args.test_days,
            param_grid=json.loads(args.param_grid),
            db_path=args.db_path,
        )
        verdict = "PASS" if result.passed else "FAIL"
        print(f"{result.strategy_id}: {verdict}")
        print(f"  deflated_sharpe   {result.deflated_sharpe:.3f}")
        print(f"  permutation p     {result.permutation_p:.3f}")
        print(f"  PBO               {result.pbo:.3f}")
        print(f"  mean OOS sharpe   {result.mean_test_sharpe:.3f}")
        print(f"  trials (ledger)   {result.n_trials}")
        for reason in result.reasons:
            print(f"  - {reason}")
        return 0 if result.passed else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
