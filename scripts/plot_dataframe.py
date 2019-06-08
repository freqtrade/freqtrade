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
import plotly.graph_objs as go
import pytz
from plotly import tools
from plotly.offline import plot

from freqtrade import persistence
from freqtrade.arguments import Arguments, TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, load_backtest_data
from freqtrade.exchange import Exchange
from freqtrade.optimize import setup_configuration
from freqtrade.persistence import Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)
_CONF: Dict[str, Any] = {}

timeZone = pytz.UTC


def load_trades(args: Namespace, pair: str, timerange: TimeRange) -> pd.DataFrame:
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

    plot(fig, filename=str(Path('user_data/plots').joinpath(file_name)), auto_open=False)
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


def get_tickers_data(strategy, exchange, pairs: List[str], args):
    """
    Get tickers data for each pairs on live or local, option defined in args
    :return: dictinnary of tickers. output format: {'pair': tickersdata}
    """

    ticker_interval = strategy.ticker_interval
    timerange = Arguments.parse_timerange(args.timerange)

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
    trades = trades.loc[trades['open_time'] >= dataframe.iloc[0]['date']]
    return trades


def generate_graph(
                    pair: str,
                    trades: pd.DataFrame,
                    data: pd.DataFrame,
                    indicators1: str,
                    indicators2: str
                ) -> tools.make_subplots:
    """
    Generate the graph from the data generated by Backtesting or from DB
    :param pair: Pair to Display on the graph
    :param trades: All trades created
    :param data: Dataframe
    :indicators1: String Main plot indicators
    :indicators2: String Sub plot indicators
    :return: None
    """

    # Define the graph
    fig = tools.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_width=[1, 1, 4],
        vertical_spacing=0.0001,
    )
    fig['layout'].update(title=pair)
    fig['layout']['yaxis1'].update(title='Price')
    fig['layout']['yaxis2'].update(title='Volume')
    fig['layout']['yaxis3'].update(title='Other')
    fig['layout']['xaxis']['rangeslider'].update(visible=False)

    # Common information
    candles = go.Candlestick(
        x=data.date,
        open=data.open,
        high=data.high,
        low=data.low,
        close=data.close,
        name='Price'
    )

    df_buy = data[data['buy'] == 1]
    buys = go.Scattergl(
        x=df_buy.date,
        y=df_buy.close,
        mode='markers',
        name='buy',
        marker=dict(
            symbol='triangle-up-dot',
            size=9,
            line=dict(width=1),
            color='green',
        )
    )
    df_sell = data[data['sell'] == 1]
    sells = go.Scattergl(
        x=df_sell.date,
        y=df_sell.close,
        mode='markers',
        name='sell',
        marker=dict(
            symbol='triangle-down-dot',
            size=9,
            line=dict(width=1),
            color='red',
        )
    )

    trade_buys = go.Scattergl(
        x=trades["open_time"],
        y=trades["open_rate"],
        mode='markers',
        name='trade_buy',
        marker=dict(
            symbol='square-open',
            size=11,
            line=dict(width=2),
            color='green'
        )
    )
    trade_sells = go.Scattergl(
        x=trades["close_time"],
        y=trades["close_rate"],
        mode='markers',
        name='trade_sell',
        marker=dict(
            symbol='square-open',
            size=11,
            line=dict(width=2),
            color='red'
        )
    )

    # Row 1
    fig.append_trace(candles, 1, 1)

    if 'bb_lowerband' in data and 'bb_upperband' in data:
        bb_lower = go.Scatter(
            x=data.date,
            y=data.bb_lowerband,
            name='BB lower',
            line={'color': 'rgba(255,255,255,0)'},
        )
        bb_upper = go.Scatter(
            x=data.date,
            y=data.bb_upperband,
            name='BB upper',
            fill="tonexty",
            fillcolor="rgba(0,176,246,0.2)",
            line={'color': 'rgba(255,255,255,0)'},
        )
        fig.append_trace(bb_lower, 1, 1)
        fig.append_trace(bb_upper, 1, 1)

    fig = generate_row(fig=fig, row=1, raw_indicators=indicators1, data=data)
    fig.append_trace(buys, 1, 1)
    fig.append_trace(sells, 1, 1)
    fig.append_trace(trade_buys, 1, 1)
    fig.append_trace(trade_sells, 1, 1)

    # Row 2
    volume = go.Bar(
        x=data['date'],
        y=data['volume'],
        name='Volume'
    )
    fig.append_trace(volume, 2, 1)

    # Row 3
    fig = generate_row(fig=fig, row=3, raw_indicators=indicators2, data=data)

    return fig


def generate_row(fig, row, raw_indicators, data) -> tools.make_subplots:
    """
    Generator all the indicator selected by the user for a specific row
    """
    for indicator in raw_indicators.split(','):
        if indicator in data:
            scattergl = go.Scattergl(
                x=data['date'],
                y=data[indicator],
                name=indicator
            )
            fig.append_trace(scattergl, row, 1)
        else:
            logger.info(
                'Indicator "%s" ignored. Reason: This indicator is not found '
                'in your strategy.',
                indicator
            )

    return fig


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

    tickers = get_tickers_data(strategy, exchange, pairs, args)
    pair_counter = 0
    for pair, data in tickers.items():
        pair_counter += 1
        logger.info("analyse pair %s", pair)
        tickers = {}
        tickers[pair] = data
        dataframe = generate_dataframe(strategy, tickers, pair)

        trades = load_trades(args, pair, timerange)
        trades = extract_trades_of_period(dataframe, trades)

        fig = generate_graph(
            pair=pair,
            trades=trades,
            data=dataframe,
            indicators1=args.indicators1,
            indicators2=args.indicators2
        )

        is_last = (False, True)[pair_counter == len(tickers)]
        generate_plot_file(fig, pair, ticker_interval, is_last)

    logger.info('End of ploting process %s plots generated', pair_counter)


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
