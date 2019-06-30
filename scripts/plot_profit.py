#!/usr/bin/env python3
"""
Script to display profits

Use `python plot_profit.py --help` to display the command line arguments
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objs as go
from plotly import tools

from freqtrade.arguments import ARGS_PLOT_PROFIT, Arguments
from freqtrade.data import history
from freqtrade.data.btanalysis import create_cum_profit, load_trades
from freqtrade.optimize import setup_configuration
from freqtrade.plot.plotting import store_plot_file
from freqtrade.resolvers import ExchangeResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def plot_profit(config: Dict[str, Any]) -> None:
    """
    Plots the total profit for all pairs.
    Note, the profit calculation isn't realistic.
    But should be somewhat proportional, and therefor useful
    in helping out to find a good algorithm.
    """

    exchange = ExchangeResolver(config.get('exchange', {}).get('name'), config).exchange

    # Take pairs from the cli otherwise switch to the pair in the config file
    if "pairs" in config:
        pairs = config["pairs"].split(',')
    else:
        pairs = config["exchange"]["pair_whitelist"]

    # We need to use the same pairs and the same ticker_interval
    # as used in backtesting / trading
    # to match the tickerdata against the results
    timerange = Arguments.parse_timerange(config["timerange"])

    tickers = history.load_data(
        datadir=Path(str(config.get("datadir"))),
        pairs=pairs,
        ticker_interval=config['ticker_interval'],
        refresh_pairs=config.get('refresh_pairs', False),
        timerange=timerange,
        exchange=exchange,
        live=config.get("live", False),
    )

    # Load the profits results
    trades = load_trades(config)

    trades = trades[trades['pair'].isin(pairs)]

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

    store_plot_file(fig,
                       filename='freqtrade-profit-plot.html',
                       auto_open=True)


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
