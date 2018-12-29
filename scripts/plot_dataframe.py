#!/usr/bin/env python3
"""
Script to display when the bot will buy a specific pair

Mandatory Cli parameters:
-p / --pair: pair to examine

Option but recommended
-s / --strategy: strategy to use


Optional Cli parameters
-d / --datadir: path to pair backtest data
--timerange: specify what timerange of data to use.
-l / --live: Live, to download the latest ticker for the pair
-db / --db-url: Show trades stored in database


Indicators recommended
Row 1: sma, ema3, ema5, ema10, ema50
Row 3: macd, rsi, fisher_rsi, mfi, slowd, slowk, fastd, fastk

Example of usage:
> python3 scripts/plot_dataframe.py --pair BTC/EUR -d user_data/data/ --indicators1 sma,ema3
--indicators2 fastk,fastd
"""
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import plotly.graph_objs as go
import pytz

from plotly import tools
from plotly.offline import plot

from freqtrade import persistence
from freqtrade.arguments import Arguments, TimeRange
from freqtrade.data import history
from freqtrade.exchange import Exchange
from freqtrade.optimize.backtesting import setup_configuration
from freqtrade.persistence import Trade
from freqtrade.resolvers import StrategyResolver

logger = logging.getLogger(__name__)
_CONF: Dict[str, Any] = {}

timeZone = pytz.UTC


def load_trades(args: Namespace, pair: str, timerange: TimeRange) -> pd.DataFrame:
    trades: pd.DataFrame = pd.DataFrame()
    if args.db_url:
        persistence.init(_CONF)
        columns = ["pair", "profit", "opents", "closets", "open_rate", "close_rate", "duration"]

        for x in Trade.query.all():
            print("date: {}".format(x.open_date))

        trades = pd.DataFrame([(t.pair, t.calc_profit(),
                                t.open_date.replace(tzinfo=timeZone),
                                t.close_date.replace(tzinfo=timeZone) if t.close_date else None,
                                t.open_rate, t.close_rate,
                                t.close_date.timestamp() - t.open_date.timestamp() if t.close_date else None)
                               for t in Trade.query.filter(Trade.pair.is_(pair)).all()],
                              columns=columns)

    elif args.exportfilename:
        file = Path(args.exportfilename)
        # must align with columns in backtest.py
        columns = ["pair", "profit", "opents", "closets", "index", "duration",
                   "open_rate", "close_rate", "open_at_end", "sell_reason"]
        with file.open() as f:
            data = json.load(f)
            trades = pd.DataFrame(data, columns=columns)
        trades = trades.loc[trades["pair"] == pair]
        if timerange:
            if timerange.starttype == 'date':
                trades = trades.loc[trades["opents"] >= timerange.startts]
            if timerange.stoptype == 'date':
                trades = trades.loc[trades["opents"] <= timerange.stopts]

        trades['opents'] = pd.to_datetime(trades['opents'],
                                          unit='s',
                                          utc=True,
                                          infer_datetime_format=True)
        trades['closets'] = pd.to_datetime(trades['closets'],
                                           unit='s',
                                           utc=True,
                                           infer_datetime_format=True)
    return trades


def plot_analyzed_dataframe(args: Namespace) -> None:
    """
    Calls analyze() and plots the returned dataframe
    :return: None
    """
    global _CONF

    # Load the configuration
    _CONF.update(setup_configuration(args))

    print(_CONF)
    # Set the pair to audit
    pair = args.pair

    if pair is None:
        logger.critical('Parameter --pair mandatory;. E.g --pair ETH/BTC')
        exit()

    if '/' not in pair:
        logger.critical('--pair format must be XXX/YYY')
        exit()

    # Set timerange to use
    timerange = Arguments.parse_timerange(args.timerange)

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

    # Set the ticker to use
    tick_interval = strategy.ticker_interval

    # Load pair tickers
    tickers = {}
    if args.live:
        logger.info('Downloading pair.')
        exchange.refresh_latest_ohlcv([pair], tick_interval)
        tickers[pair] = exchange.klines(pair)
    else:
        tickers = history.load_data(
            datadir=Path(_CONF.get("datadir")),
            pairs=[pair],
            ticker_interval=tick_interval,
            refresh_pairs=_CONF.get('refresh_pairs', False),
            timerange=timerange,
            exchange=Exchange(_CONF)
        )

        # No ticker found, or impossible to download
        if tickers == {}:
            exit()

    # Get trades already made from the DB
    trades = load_trades(args, pair, timerange)

    dataframes = strategy.tickerdata_to_dataframe(tickers)

    dataframe = dataframes[pair]
    dataframe = strategy.advise_buy(dataframe, {'pair': pair})
    dataframe = strategy.advise_sell(dataframe, {'pair': pair})

    if len(dataframe.index) > args.plot_limit:
        logger.warning('Ticker contained more than %s candles as defined '
                       'with --plot-limit, clipping.', args.plot_limit)
    dataframe = dataframe.tail(args.plot_limit)

    trades = trades.loc[trades['opents'] >= dataframe.iloc[0]['date']]
    fig = generate_graph(
        pair=pair,
        trades=trades,
        data=dataframe,
        args=args
    )

    plot(fig, filename=str(Path('user_data').joinpath('freqtrade-plot.html')))


def generate_graph(pair, trades: pd.DataFrame, data: pd.DataFrame, args) -> tools.make_subplots:
    """
    Generate the graph from the data generated by Backtesting or from DB
    :param pair: Pair to Display on the graph
    :param trades: All trades created
    :param data: Dataframe
    :param args: sys.argv that contrains the two params indicators1, and indicators2
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
        x=trades["opents"],
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
        x=trades["closets"],
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

    fig = generate_row(fig=fig, row=1, raw_indicators=args.indicators1, data=data)
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
    fig = generate_row(fig=fig, row=3, raw_indicators=args.indicators2, data=data)

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
        default='macd',
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
    plot_analyzed_dataframe(
        plot_parse_args(sysargv)
    )


if __name__ == '__main__':
    main(sys.argv[1:])
