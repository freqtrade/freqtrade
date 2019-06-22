#!/usr/bin/env python3
"""
Script to display when the bot will buy on specific pair(s)

Mandatory Cli parameters:
-p / --pairs: pair(s) to examine

Option but recommended
-s / --strategy: strategy to use


Optional Cli parameters
-d / --datadir: path to pair(s) backtest data
--timerange: specify what timerange of data to use.
-l / --live: Live, to download the latest ticker for the pair(s)
-db / --db-url: Show trades stored in database


Indicators recommended
Row 1: sma, ema3, ema5, ema10, ema50
Row 3: macd, rsi, fisher_rsi, mfi, slowd, slowk, fastd, fastk

Example of usage:
> python3 scripts/plot_dataframe.py --pairs BTC/EUR,XRP/BTC -d user_data/data/
  --indicators1 sma,ema3 --indicators2 fastk,fastd
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from freqtrade.arguments import Arguments, TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import (extract_trades_of_period,
                                       load_backtest_data, load_trades_from_db)
from freqtrade.optimize import setup_configuration
from freqtrade.plot.plotting import generate_graph, generate_plot_file
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def get_tickers_data(strategy, exchange, pairs: List[str], timerange: TimeRange,
                     datadir: Path, refresh_pairs: bool, live: bool):
    """
    Get tickers data for each pairs on live or local, option defined in args
    :return: dictionary of tickers. output format: {'pair': tickersdata}
    """

    ticker_interval = strategy.ticker_interval

    tickers = history.load_data(
        datadir=datadir,
        pairs=pairs,
        ticker_interval=ticker_interval,
        refresh_pairs=refresh_pairs,
        timerange=timerange,
        exchange=exchange,
        live=live,
    )

    # No ticker found, impossible to download, len mismatch
    for pair, data in tickers.copy().items():
        logger.debug("checking tickers data of pair: %s", pair)
        logger.debug("data.empty: %s", data.empty)
        logger.debug("len(data): %s", len(data))
        if data.empty:
            del tickers[pair]
            logger.info(
                'An issue occured while retreiving data of %s pair, please retry '
                'using -l option for live or --refresh-pairs-cached', pair)
    return tickers


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
    exchange_name = config.get('exchange', {}).get('name').title()
    exchange = ExchangeResolver(exchange_name, config).exchange

    strategy = StrategyResolver(config).strategy
    if "pairs" in config:
        pairs = config["pairs"].split(',')
    else:
        pairs = config["exchange"]["pair_whitelist"]

    # Set timerange to use
    timerange = Arguments.parse_timerange(config["timerange"])
    ticker_interval = strategy.ticker_interval

    tickers = get_tickers_data(strategy, exchange, pairs, timerange,
                               datadir=Path(str(config.get("datadir"))),
                               refresh_pairs=config.get('refresh_pairs', False),
                               live=config.get("live", False))
    pair_counter = 0
    for pair, data in tickers.items():
        pair_counter += 1
        logger.info("analyse pair %s", pair)
        tickers = {}
        tickers[pair] = data
        dataframe = generate_dataframe(strategy, tickers, pair)
        trades = None
        if config["trade_source"] == "DB":
            trades = load_trades_from_db(config["db_url"])
        elif config["trade_source"] == "file":
            trades = load_backtest_data(Path(config["exportfilename"]))

        trades = trades.loc[trades['pair'] == pair]
        trades = extract_trades_of_period(dataframe, trades)

        fig = generate_graph(
            pair=pair,
            data=dataframe,
            trades=trades,
            indicators1=config["indicators1"].split(","),
            indicators2=config["indicators2"].split(",")
        )

        generate_plot_file(fig, pair, ticker_interval)

    logger.info('End of ploting process %s plots generated', pair_counter)


def plot_parse_args(args: List[str]) -> Dict[str, Any]:
    """
    Parse args passed to the script
    :param args: Cli arguments
    :return: args: Array with all arguments
    """
    arguments = Arguments(args, 'Graph dataframe')
    arguments.common_options()
    arguments.main_options()
    arguments.common_optimize_options()
    arguments.backtesting_options()
    arguments.common_scripts_options()
    arguments.plot_dataframe_options()
    parsed_args = arguments.parse_args()

    # Load the configuration
    config = setup_configuration(parsed_args, RunMode.BACKTEST)
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
