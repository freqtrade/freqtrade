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
from research.db import get_engine, get_session
from research.gate import run_promotion_gate
from research.models import ReconstructedTrade
from research.scoring import strategy_report
from research.trader_mining.engine import reconstruct_and_persist_trades
from research.trader_mining.ingestion import ingest_hyperliquid_fills, ingest_hyperliquid_ledger
from research.trader_mining.metrics import compute_metrics, format_report
from research.trader_mining.split_report import compute_split_report, format_split_report
from research.trader_mining.splitting import PeriodBoundaries


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
    gate.add_argument(
        "--parameter-stability",
        action="store_true",
        help=(
            "Also compute the fraction of grid variants profitable in-sample across "
            "the walk-forward run, regardless of pass/fail (informational)"
        ),
    )

    trader_import = sub.add_parser(
        "trader-import", help="Import one wallet's fill history from Hyperliquid"
    )
    trader_import.add_argument("--trader", required=True, help="Wallet address")
    trader_import.add_argument("--since", help="YYYY-MM-DD, earliest fill to fetch")
    trader_import.add_argument("--db-path", default="user_data/research.sqlite")

    trader_analyze = sub.add_parser(
        "trader-analyze", help="Reconstruct trades from a wallet's ingested fills"
    )
    trader_analyze.add_argument("--trader", required=True, help="Wallet address")
    trader_analyze.add_argument("--symbol", help="Limit to one symbol (default: all)")
    trader_analyze.add_argument("--db-path", default="user_data/research.sqlite")

    trader_report = sub.add_parser(
        "trader-report", help="Print a performance report for a wallet's reconstructed trades"
    )
    trader_report.add_argument("--trader", required=True, help="Wallet address")
    trader_report.add_argument("--symbol", help="Limit to one symbol (default: all)")
    trader_report.add_argument("--db-path", default="user_data/research.sqlite")
    trader_report.add_argument(
        "--train-end",
        help=(
            "YYYY-MM-DD, TRAIN/VALIDATION boundary (exclusive of VALIDATION). Requires "
            "--validation-end and --test-end."
        ),
    )
    trader_report.add_argument(
        "--validation-end",
        help=(
            "YYYY-MM-DD, VALIDATION/TEST boundary (exclusive of TEST). Requires "
            "--train-end and --test-end."
        ),
    )
    trader_report.add_argument(
        "--test-end",
        help=(
            "YYYY-MM-DD, TEST/FORWARD boundary (exclusive of FORWARD; FORWARD is "
            "open-ended). Requires --train-end and --validation-end."
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
            include_parameter_stability=args.parameter_stability,
        )
        print(strategy_report(result, pair=args.pairs.split(",")[0]))
        return 0 if result.passed else 1

    elif args.command == "trader-import":
        engine = get_engine(args.db_path)
        session = get_session(engine)
        since = datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
        ingest_result = ingest_hyperliquid_fills(session, trader=args.trader, since=since)
        print(f"n_fetched: {ingest_result.n_fetched}")
        print(f"n_new: {ingest_result.n_new}")
        print(f"history_completeness: {ingest_result.history_completeness}")
        if ingest_result.history_completeness == "truncated_by_provider_limit":
            print(
                "WARNING: history_completeness=truncated_by_provider_limit -- "
                "Hyperliquid's 10,000-fill ceiling was reached; earlier fills may exist "
                "but are not retrievable via this endpoint."
            )
        ledger_result = ingest_hyperliquid_ledger(session, trader=args.trader)
        print(f"n_ledger_events_fetched: {ledger_result.n_fetched}")
        print(f"n_ledger_events_new: {ledger_result.n_new}")
        return 0

    elif args.command == "trader-analyze":
        engine = get_engine(args.db_path)
        session = get_session(engine)
        analyze_result = reconstruct_and_persist_trades(
            session, trader=args.trader, symbol=args.symbol
        )
        print(f"n_trades: {analyze_result.n_trades}")
        print(f"symbols: {', '.join(analyze_result.symbols)}")
        if analyze_result.reconciled_gaps:
            print(f"reconciled_gaps ({len(analyze_result.reconciled_gaps)}):")
            for gap in analyze_result.reconciled_gaps:
                print(f"  {gap}")
        return 0

    elif args.command == "trader-report":
        split_flags = (args.train_end, args.validation_end, args.test_end)
        if any(split_flags) and not all(split_flags):
            trader_report.error(
                "--train-end, --validation-end, and --test-end must be given together "
                "(all three or none)"
            )

        engine = get_engine(args.db_path)
        session = get_session(engine)
        query = session.query(ReconstructedTrade).filter(ReconstructedTrade.trader == args.trader)
        if args.symbol:
            query = query.filter(ReconstructedTrade.symbol == args.symbol)
        trades = query.all()

        if all(split_flags):
            boundaries = PeriodBoundaries(
                train_end=datetime.fromisoformat(args.train_end).replace(tzinfo=UTC),
                validation_end=datetime.fromisoformat(args.validation_end).replace(tzinfo=UTC),
                test_end=datetime.fromisoformat(args.test_end).replace(tzinfo=UTC),
            )
            split_report = compute_split_report(trades, boundaries)
            print(format_split_report(split_report, args.trader))
        else:
            metrics = compute_metrics(trades)
            print(format_report(metrics, args.trader))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
