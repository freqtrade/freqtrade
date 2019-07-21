#!/usr/bin/env python3
"""
Script to display when the bot will buy on specific pair(s)

Use `python plot_dataframe.py --help` to display the command line arguments

Indicators recommended
Row 1: sma, ema3, ema5, ema10, ema50
Row 3: macd, rsi, fisher_rsi, mfi, slowd, slowk, fastd, fastk

Example of usage:
> python3 scripts/plot_dataframe.py --pairs BTC/EUR,XRP/BTC -d user_data/data/
  --indicators1 sma,ema3 --indicators2 fastk,fastd
"""
import logging
import sys
from typing import Any, Dict, List

import pandas as pd

from freqtrade.configuration import Arguments
from freqtrade.configuration.arguments import ARGS_PLOT_DATAFRAME
from freqtrade.data.btanalysis import extract_trades_of_period
from freqtrade.optimize import setup_configuration
from freqtrade.plot.plotting import (init_plotscript, generate_candlestick_graph,
                                     store_plot_file,
                                     generate_plot_filename)
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def generate_dataframe(strategy, tickers, pair) -> pd.DataFrame:
    """
    Get tickers then Populate strategy indicators and signals, then return the full dataframe
    :return: the DataFrame of a pair
    """

    dataframes = strategy.tickerdata_to_dataframe(tickers)
    dataframe = dataframes[pair]
    dataframe = strategy.advise_buy(dataframe, {'pair': pair})
    dataframe = strategy.advise_sell(dataframe, {'pair': pair})

    return dataframe


def analyse_and_plot_pairs(config: Dict[str, Any]):
    """
    From arguments provided in cli:
    -Initialise backtest env
    -Get tickers data
    -Generate Dafaframes populated with indicators and signals
    -Load trades excecuted on same periods
    -Generate Plotly plot objects
    -Generate plot files
    :return: None
    """
    plot_elements = init_plotscript(config)
    trades = plot_elements['trades']

    pair_counter = 0
    for pair, data in plot_elements["tickers"].items():
        pair_counter += 1
        logger.info("analyse pair %s", pair)
        tickers = {}
        tickers[pair] = data
        dataframe = generate_dataframe(plot_elements["strategy"], tickers, pair)

        trades_pair = trades.loc[trades['pair'] == pair]
        trades_pair = extract_trades_of_period(dataframe, trades_pair)

        fig = generate_candlestick_graph(
            pair=pair,
            data=dataframe,
            trades=trades_pair,
            indicators1=config["indicators1"].split(","),
            indicators2=config["indicators2"].split(",")
        )

        store_plot_file(fig, filename=generate_plot_filename(pair, config['ticker_interval']),
                        folder=config['user_data_dir'] / "plot")

    logger.info('End of ploting process %s plots generated', pair_counter)


def plot_parse_args(args: List[str]) -> Dict[str, Any]:
    """
    Parse args passed to the script
    :param args: Cli arguments
    :return: args: Array with all arguments
    """
    arguments = Arguments(args, 'Graph dataframe')
    arguments._build_args(optionlist=ARGS_PLOT_DATAFRAME)
    parsed_args = arguments._parse_args()

    # Load the configuration
    config = setup_configuration(parsed_args, RunMode.OTHER)
    return config


def main(sysargv: List[str]) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """
    logger.info('Starting Plot Dataframe')
    analyse_and_plot_pairs(
        plot_parse_args(sysargv)
    )
    exit()


if __name__ == '__main__':
    main(sys.argv[1:])
