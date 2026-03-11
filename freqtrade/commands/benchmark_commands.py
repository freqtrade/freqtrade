"""
PortfolioBench benchmark subcommand.

Integrates the benchmarking suite (formerly benchmark.py / benchmark_all.py)
into the portbench CLI:

    portbench benchmark                     # full suite
    portbench benchmark --quick             # smoke test
    portbench benchmark --trading-only      # trading strategies only
    portbench benchmark --portfolio-only    # portfolio strategies only
    portbench benchmark --export report.json
"""

import logging
import sys
from typing import Any


logger = logging.getLogger(__name__)


def start_benchmark(args: dict[str, Any]) -> None:
    """Entry point for ``portbench benchmark``."""
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Use benchmark.py's run_benchmark (the richer, single-file version)
    from benchmark import run_benchmark

    include_trading = not args.get("portfolio_only", False)
    include_portfolio = not args.get("trading_only", False)
    quick = args.get("quick", False)
    export_path = args.get("benchmark_export", None)

    results = run_benchmark(
        include_trading=include_trading,
        include_portfolio=include_portfolio,
        quick=quick,
        export_path=export_path,
    )

    sys.exit(1 if results["summary"]["failed"] > 0 else 0)


def start_benchmark_all(args: dict[str, Any]) -> None:
    """Entry point for ``portbench benchmark-all``."""
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from benchmark_all import main as benchmark_all_main

    # Reconstruct sys.argv from args so benchmark_all's own argparse works.
    argv = []
    if args.get("quick"):
        argv.append("--quick")
    if args.get("trading_only"):
        argv.append("--trading-only")
    if args.get("portfolio_only"):
        argv.append("--portfolio-only")
    if args.get("skip_backtests"):
        argv.append("--skip-backtests")
    if args.get("strategies"):
        argv.extend(["--strategies"] + args["strategies"])
    if args.get("benchmark_timeframes"):
        argv.extend(["--timeframes"] + args["benchmark_timeframes"])
    if args.get("benchmark_categories"):
        argv.extend(["--categories"] + args["benchmark_categories"])
    if args.get("json_output"):
        argv.extend(["--json-output", args["json_output"]])

    # Patch sys.argv so benchmark_all's argparse picks them up.
    old_argv = sys.argv
    sys.argv = ["portbench benchmark-all"] + argv
    try:
        benchmark_all_main()
    finally:
        sys.argv = old_argv
