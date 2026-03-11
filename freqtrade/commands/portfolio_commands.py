"""
PortfolioBench portfolio and data-generation subcommands.

    portbench portfolio                  # run the standalone portfolio pipeline
    portbench generate-data              # generate synthetic test data
"""

import logging
import sys
from typing import Any


logger = logging.getLogger(__name__)


def start_portfolio(args: dict[str, Any]) -> None:
    """Entry point for ``portbench portfolio``."""
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from portfolio.PortfolioManagement import run_portfolio

    data_dir = args.get("datadir_portfolio", None)
    pairs = args.get("pairs", None)
    timeframe = args.get("timeframe", "1d")
    initial_capital = args.get("initial_capital", 10_000.0)

    run_portfolio(
        data_dir=data_dir,
        pairs=pairs,
        timeframe=timeframe,
        initial_capital=initial_capital,
    )


def start_generate_data(args: dict[str, Any]) -> None:
    """Entry point for ``portbench generate-data``."""
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from utils.generate_test_data import main as generate_main

    generate_main()
