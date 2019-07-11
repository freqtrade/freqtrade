#!/usr/bin/env python3
"""
Script to display profits

Use `python plot_profit.py --help` to display the command line arguments
"""
import logging
import sys
from typing import Any, Dict, List

from freqtrade.configuration import Arguments, ARGS_PLOT_PROFIT
from freqtrade.optimize import setup_configuration
from freqtrade.plot.plotting import init_plotscript, generate_profit_graph, store_plot_file
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def plot_profit(config: Dict[str, Any]) -> None:
    """
    Plots the total profit for all pairs.
    Note, the profit calculation isn't realistic.
    But should be somewhat proportional, and therefor useful
    in helping out to find a good algorithm.
    """
    plot_elements = init_plotscript(config)
    trades = plot_elements['trades']
    # Filter trades to relevant pairs
    trades = trades[trades['pair'].isin(plot_elements["pairs"])]

    # Create an average close price of all the pairs that were involved.
    # this could be useful to gauge the overall market trend
    fig = generate_profit_graph(plot_elements["pairs"], plot_elements["tickers"], trades)
    store_plot_file(fig, filename='freqtrade-profit-plot.html', auto_open=True)


def plot_parse_args(args: List[str]) -> Dict[str, Any]:
    """
    Parse args passed to the script
    :param args: Cli arguments
    :return: args: Array with all arguments
    """
    arguments = Arguments(args, 'Graph profits')
    arguments.build_args(optionlist=ARGS_PLOT_PROFIT)

    parsed_args = arguments.parse_args()

    # Load the configuration
    config = setup_configuration(parsed_args, RunMode.OTHER)
    return config


def main(sysargv: List[str]) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """
    logger.info('Starting Plot Dataframe')
    plot_profit(
        plot_parse_args(sysargv)
    )


if __name__ == '__main__':
    main(sys.argv[1:])
