# research/cli.py
"""Command-line entry point for the research package: `python -m research.cli gate
...` runs the promotion gate for a strategy and prints its verdict."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from freqtrade.configuration import Configuration
from research.cost_stress import DEFAULT_FEE_MULTIPLIERS
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
    gate.add_argument(
        "--fee-sensitivity",
        action="store_true",
        help="Also run a fee-sensitivity stress test if the gate passes (informational)",
    )
    gate.add_argument(
        "--regime-breakdown",
        action="store_true",
        help=(
            "Also compute a regime (Trend x Volatility) breakdown of walk-forward "
            "results, regardless of pass/fail (informational)"
        ),
    )

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
            fee_sensitivity_multipliers=DEFAULT_FEE_MULTIPLIERS if args.fee_sensitivity else None,
            include_regime_breakdown=args.regime_breakdown,
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
        if result.fee_sensitivity:
            print("  fee sensitivity (informational, not part of PASS/FAIL):")
            for mult, stats in result.fee_sensitivity.items():
                label = f"{mult:.2f}x fee" + (" (baseline)" if mult == 1.0 else "")
                print(
                    f"    {label:<22} mean OOS sharpe {stats['mean_test_sharpe']:>6.2f}"
                    f"   deflated_sharpe (n_trials=1) {stats['deflated_sharpe']:.3f}"
                )
        if result.regime_breakdown:
            print(
                f"  regime breakdown ({args.pairs.split(',')[0]}, informational, "
                "not part of PASS/FAIL):"
            )
            for label, stats in result.regime_breakdown.items():
                print(
                    f"    {label:<15} {stats['n_windows']:>2} windows"
                    f"   {stats['n_trades']:>3} trades"
                    f"   mean sharpe {stats['mean_test_sharpe']:>6.2f}"
                    f"   total return {stats['total_return']:>8.4f}"
                )
        return 0 if result.passed else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
