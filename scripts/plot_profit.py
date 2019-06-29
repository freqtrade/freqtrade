#!/usr/bin/env python3
"""
Script to display profits

Use `python plot_profit.py --help` to display the command line arguments
"""
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import plot

from freqtrade.arguments import ARGS_PLOT_PROFIT, Arguments
from freqtrade.configuration import Configuration
from freqtrade.data import history
from freqtrade.data.btanalysis import create_cum_profit, load_trades
from freqtrade.plot.plotting import generate_plot_file
from freqtrade.resolvers import StrategyResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def plot_profit(args: Namespace) -> None:
    """
    Plots the total profit for all pairs.
    Note, the profit calculation isn't realistic.
    But should be somewhat proportional, and therefor useful
    in helping out to find a good algorithm.
    """

    # We need to use the same pairs and the same ticker_interval
    # as used in backtesting / trading
    # to match the tickerdata against the results
    timerange = Arguments.parse_timerange(args.timerange)

    config = Configuration(args, RunMode.OTHER).get_config()

    # Init strategy
    strategy = StrategyResolver(config).strategy

    # Take pairs from the cli otherwise switch to the pair in the config file
    if args.pairs:
        filter_pairs = args.pairs
        filter_pairs = filter_pairs.split(',')
    else:
        filter_pairs = config['exchange']['pair_whitelist']

    # Load the profits results
    trades = load_trades(config)

    trades = trades[trades['pair'].isin(filter_pairs)]

    ticker_interval = strategy.ticker_interval
    pairs = config['exchange']['pair_whitelist']

    if filter_pairs:
        pairs = list(set(pairs) & set(filter_pairs))
        logger.info('Filter, keep pairs %s' % pairs)

    tickers = history.load_data(
        datadir=Path(str(config.get('datadir'))),
        pairs=pairs,
        ticker_interval=ticker_interval,
        refresh_pairs=False,
        timerange=timerange
    )

    # Create an average close price of all the pairs that were involved.
    # this could be useful to gauge the overall market trend

    # Combine close-values for all pairs, rename columns to "pair"
    df_comb = pd.concat([tickers[pair].set_index('date').rename(
        {'close': pair}, axis=1)[pair] for pair in tickers], axis=1)
    df_comb['mean'] = df_comb.mean(axis=1)

    # Add combined cumulative profit
    df_comb = create_cum_profit(df_comb, trades, 'cum_profit')

    # Plot the pairs average close prices, and total profit growth
    avgclose = go.Scattergl(
        x=df_comb.index,
        y=df_comb['mean'],
        name='Avg close price',
    )

    profit = go.Scattergl(
        x=df_comb.index,
        y=df_comb['cum_profit'],
        name='Profit',
    )

    fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=True, row_width=[1, 1, 1])

    fig.append_trace(avgclose, 1, 1)
    fig.append_trace(profit, 2, 1)

    for pair in pairs:
        profit_col = f'cum_profit_{pair}'
        df_comb = create_cum_profit(df_comb, trades[trades['pair'] == pair], profit_col)

        pair_profit = go.Scattergl(
            x=df_comb.index,
            y=df_comb[profit_col],
            name=f"Profit {pair}",
        )
        fig.append_trace(pair_profit, 3, 1)

    generate_plot_file(fig,
                       filename='freqtrade-profit-plot.html',
                       auto_open=True)


def plot_parse_args(args: List[str]) -> Namespace:
    """
    Parse args passed to the script
    :param args: Cli arguments
    :return: args: Array with all arguments
    """
    arguments = Arguments(args, 'Graph profits')
    arguments.build_args(optionlist=ARGS_PLOT_PROFIT)

    return arguments.parse_args()


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
