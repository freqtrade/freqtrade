#!/usr/bin/env python3
"""
Script to display when the bot will buy a specific pair

Mandatory Cli parameters:
-p / --pair: pair to examine

Optional Cli parameters
-s / --strategy: strategy to use
-d / --datadir: path to pair backtest data
--timerange: specify what timerange of data to use.
-l / --live: Live, to download the latest ticker for the pair
"""

import sys
from argparse import Namespace

from typing import List

from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go

from freqtrade.arguments import Arguments
from freqtrade.analyze import Analyze
from freqtrade import exchange
from freqtrade.logger import Logger
import freqtrade.optimize as optimize


logger = Logger(name="Graph dataframe").get_logger()


def plot_analyzed_dataframe(args: Namespace) -> None:
    """
    Calls analyze() and plots the returned dataframe
    :return: None
    """
    pair = args.pair.replace('-', '_')
    timerange = Arguments.parse_timerange(args.timerange)

    # Init strategy
    try:
        analyze = Analyze({'strategy': args.strategy})
    except AttributeError:
        logger.critical(
            'Impossible to load the strategy. Please check the file "user_data/strategies/%s.py"',
            args.strategy
        )
        exit()

    tick_interval = analyze.strategy.ticker_interval

    tickers = {}
    if args.live:
        logger.info('Downloading pair.')
        # Init Bittrex to use public API
        exchange._API = exchange.Bittrex({'key': '', 'secret': ''})
        tickers[pair] = exchange.get_ticker_history(pair, tick_interval)
    else:
        tickers = optimize.load_data(
            datadir=args.datadir,
            pairs=[pair],
            ticker_interval=tick_interval,
            refresh_pairs=False,
            timerange=timerange
        )
    dataframes = analyze.tickerdata_to_dataframe(tickers)
    dataframe = dataframes[pair]
    dataframe = analyze.populate_buy_trend(dataframe)
    dataframe = analyze.populate_sell_trend(dataframe)

    if len(dataframe.index) > 750:
        logger.warning('Ticker contained more than 750 candles, clipping.')
    data = dataframe.tail(750)

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

    bb_lower = go.Scatter(
        x=data.date,
        y=data.bb_lowerband,
        name='BB lower',
        line={'color': "transparent"},
    )
    bb_upper = go.Scatter(
        x=data.date,
        y=data.bb_upperband,
        name='BB upper',
        fill="tonexty",
        fillcolor="rgba(0,176,246,0.2)",
        line={'color': "transparent"},
    )
    macd = go.Scattergl(x=data['date'], y=data['macd'], name='MACD')
    macdsignal = go.Scattergl(x=data['date'], y=data['macdsignal'], name='MACD signal')
    volume = go.Bar(x=data['date'], y=data['volume'], name='Volume')

    fig = tools.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_width=[1, 1, 4],
        vertical_spacing=0.0001,
    )

    fig.append_trace(candles, 1, 1)
    fig.append_trace(bb_lower, 1, 1)
    fig.append_trace(bb_upper, 1, 1)
    fig.append_trace(buys, 1, 1)
    fig.append_trace(sells, 1, 1)
    fig.append_trace(volume, 2, 1)
    fig.append_trace(macd, 3, 1)
    fig.append_trace(macdsignal, 3, 1)

    fig['layout'].update(title=args.pair)
    fig['layout']['yaxis1'].update(title='Price')
    fig['layout']['yaxis2'].update(title='Volume')
    fig['layout']['yaxis3'].update(title='MACD')

    plot(fig, filename='freqtrade-plot.html')


def plot_parse_args(args: List[str]) -> Namespace:
    """
    Parse args passed to the script
    :param args: Cli arguments
    :return: args: Array with all arguments
    """
    arguments = Arguments(args, 'Graph dataframe')
    arguments.scripts_options()
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
