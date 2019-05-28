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
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytz
from plotly.offline import plot

from freqtrade import persistence
from freqtrade.arguments import Arguments, TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, load_backtest_data
from freqtrade.plot.plotting import generate_graph
from freqtrade.exchange import Exchange
from freqtrade.optimize import setup_configuration
from freqtrade.persistence import Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)
_CONF: Dict[str, Any] = {}

timeZone = pytz.UTC


def load_trades(args: Namespace, pair: str) -> pd.DataFrame:
    trades: pd.DataFrame = pd.DataFrame()
    if args.db_url:
        persistence.init(args.db_url, clean_open_orders=False)

        columns = ["pair", "profit", "open_time", "close_time",
                   "open_rate", "close_rate", "duration"]

        for x in Trade.query.all():
            print("date: {}".format(x.open_date))

        trades = pd.DataFrame([(t.pair, t.calc_profit(),
                                t.open_date.replace(tzinfo=timeZone),
                                t.close_date.replace(tzinfo=timeZone) if t.close_date else None,
                                t.open_rate, t.close_rate,
                                t.close_date.timestamp() - t.open_date.timestamp()
                                if t.close_date else None)
                               for t in Trade.query.filter(Trade.pair.is_(pair)).all()],
                              columns=columns)

    elif args.exportfilename:

        file = Path(args.exportfilename)
        if file.exists():
            trades = load_backtest_data(file)

        else:
            trades = pd.DataFrame([], columns=BT_DATA_COLUMNS)

    return trades


def generate_plot_file(fig, pair, ticker_interval, is_last) -> None:
    """
    Generate a plot html file from pre populated fig plotly object
    :return: None
    """
    logger.info('Generate plot file for %s', pair)

    pair_name = pair.replace("/", "_")
    file_name = 'freqtrade-plot-' + pair_name + '-' + ticker_interval + '.html'

    Path("user_data/plots").mkdir(parents=True, exist_ok=True)

    plot(fig, filename=str(Path('user_data/plots').joinpath(file_name)),
         auto_open=False)
    if is_last:
        plot(fig, filename=str(Path('user_data').joinpath('freqtrade-plot.html')), auto_open=False)


def get_trading_env(args: Namespace):
    """
    Initalize freqtrade Exchange and Strategy, split pairs recieved in parameter
    :return: Strategy
    """
    global _CONF

    # Load the configuration
    _CONF.update(setup_configuration(args, RunMode.BACKTEST))
    print(_CONF)

    pairs = args.pairs.split(',')
    if pairs is None:
        logger.critical('Parameter --pairs mandatory;. E.g --pairs ETH/BTC,XRP/BTC')
        exit()

    # Load the strategy
    try:
        strategy = StrategyResolver(_CONF).strategy
        exchange = Exchange(_CONF)
    except AttributeError:
        logger.critical(
            'Impossible to load the strategy. Please check the file "user_data/strategies/%s.py"',
            args.strategy
        )
        exit()

    return [strategy, exchange, pairs]


def get_tickers_data(strategy, exchange, pairs: List[str], timerange: TimeRange, live: bool):
    """
    Get tickers data for each pairs on live or local, option defined in args
    :return: dictionary of tickers. output format: {'pair': tickersdata}
    """

    ticker_interval = strategy.ticker_interval

    tickers = history.load_data(
        datadir=Path(str(_CONF.get("datadir"))),
        pairs=pairs,
        ticker_interval=ticker_interval,
        refresh_pairs=_CONF.get('refresh_pairs', False),
        timerange=timerange,
        exchange=Exchange(_CONF),
        live=args.live,
    )

    # No ticker found, impossible to download, len mismatch
    for pair, data in tickers.copy().items():
        logger.debug("checking tickers data of pair: %s", pair)
        logger.debug("data.empty: %s", data.empty)
        logger.debug("len(data): %s", len(data))
        if data.empty:
            del tickers[pair]
            logger.info(
                'An issue occured while retreiving datas of %s pair, please retry '
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


def extract_trades_of_period(dataframe, trades) -> pd.DataFrame:
    """
    Compare trades and backtested pair DataFrames to get trades performed on backtested period
    :return: the DataFrame of a trades of period
    """
    trades = trades.loc[(trades['open_time'] >= dataframe.iloc[0]['date']) &
                        (trades['close_time'] <= dataframe.iloc[-1]['date'])]
    return trades


def analyse_and_plot_pairs(args: Namespace):
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
    strategy, exchange, pairs = get_trading_env(args)
    # Set timerange to use
    timerange = Arguments.parse_timerange(args.timerange)
    ticker_interval = strategy.ticker_interval

    tickers = get_tickers_data(strategy, exchange, pairs, timerange, args.live)
    pair_counter = 0
    for pair, data in tickers.items():
        pair_counter += 1
        logger.info("analyse pair %s", pair)
        tickers = {}
        tickers[pair] = data
        dataframe = generate_dataframe(strategy, tickers, pair)

        trades = load_trades(args, pair)
        trades = extract_trades_of_period(dataframe, trades)

        fig = generate_graph(
            pair=pair,
            data=dataframe,
            trades=trades,
            indicators1=args.indicators1.split(","),
            indicators2=args.indicators2.split(",")
        )

        is_last = (False, True)[pair_counter == len(tickers)]
        generate_plot_file(fig, pair, ticker_interval, is_last)

    logger.info('End of ploting process %s plots generated', pair_counter)


def plot_parse_args(args: List[str]) -> Namespace:
    """
    Parse args passed to the script
    :param args: Cli arguments
    :return: args: Array with all arguments
    """
    arguments = Arguments(args, 'Graph dataframe')
    arguments.scripts_options()
    arguments.parser.add_argument(
        '--indicators1',
        help='Set indicators from your strategy you want in the first row of the graph. Separate '
             'them with a coma. E.g: ema3,ema5 (default: %(default)s)',
        type=str,
        default='sma,ema3,ema5',
        dest='indicators1',
    )

    arguments.parser.add_argument(
        '--indicators2',
        help='Set indicators from your strategy you want in the third row of the graph. Separate '
             'them with a coma. E.g: fastd,fastk (default: %(default)s)',
        type=str,
        default='macd,macdsignal',
        dest='indicators2',
    )
    arguments.parser.add_argument(
        '--plot-limit',
        help='Specify tick limit for plotting - too high values cause huge files - '
             'Default: %(default)s',
        dest='plot_limit',
        default=750,
        type=int,
    )
    arguments.common_args_parser()
    arguments.optimizer_shared_options(arguments.parser)
    arguments.backtesting_options(arguments.parser)
    return arguments.parse_args()

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
