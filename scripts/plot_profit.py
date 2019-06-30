#!/usr/bin/env python3
"""
Script to display profits

Use `python plot_profit.py --help` to display the command line arguments
"""
import logging
import sys
from typing import Any, Dict, List

import plotly.graph_objs as go
from plotly import tools

from freqtrade.arguments import ARGS_PLOT_PROFIT, Arguments
from freqtrade.data.btanalysis import create_cum_profit, combine_tickers_with_mean
from freqtrade.optimize import setup_configuration
from freqtrade.plot.plotting import FTPlots, store_plot_file, add_profit
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def plot_profit(config: Dict[str, Any]) -> None:
    """
    Plots the total profit for all pairs.
    Note, the profit calculation isn't realistic.
    But should be somewhat proportional, and therefor useful
    in helping out to find a good algorithm.
    """
    plot = FTPlots(config)

    trades = plot.trades[plot.trades['pair'].isin(plot.pairs)]

    # Create an average close price of all the pairs that were involved.
    # this could be useful to gauge the overall market trend

    # Combine close-values for all pairs, rename columns to "pair"
    df_comb = combine_tickers_with_mean(plot.tickers, "close")

    # Add combined cumulative profit
    df_comb = create_cum_profit(df_comb, trades, 'cum_profit')

    # Plot the pairs average close prices, and total profit growth
    avgclose = go.Scattergl(
        x=df_comb.index,
        y=df_comb['mean'],
        name='Avg close price',
    )

    fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=True, row_width=[1, 1, 1])

    fig.append_trace(avgclose, 1, 1)
    fig = add_profit(fig, 2, df_comb, 'cum_profit', 'Profit')

    for pair in plot.pairs:
        profit_col = f'cum_profit_{pair}'
        df_comb = create_cum_profit(df_comb, trades[trades['pair'] == pair], profit_col)

        fig = add_profit(fig, 3, df_comb, profit_col, f"Profit {pair}")

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
