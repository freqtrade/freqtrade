#!/usr/bin/env python3

import sys
import logging
import argparse
import os

from pandas import DataFrame
import talib.abstract as ta

import plotly
from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade import exchange, analyze
from freqtrade.misc import common_args_parser
from freqtrade.strategy.strategy import Strategy
import freqtrade.misc as misc
import freqtrade.optimize as optimize
import freqtrade.analyze as analyze


logger = logging.getLogger(__name__)


def plot_parse_args(args):
    parser = misc.common_args_parser('Graph dataframe')
    misc.backtesting_options(parser)
    misc.scripts_options(parser)
    return parser.parse_args(args)


def plot_analyzed_dataframe(args) -> None:
    """
    Calls analyze() and plots the returned dataframe
    :param pair: pair as str
    :return: None
    """
    pair = args.pair.replace('-', '_')
    timerange = misc.parse_timerange(args.timerange)

    # Init strategy
    strategy = Strategy()
    strategy.init({'strategy': args.strategy})
    tick_interval = strategy.ticker_interval

    tickers = {}
    if args.live:
        logger.info('Downloading pair.')
        # Init Bittrex to use public API
        exchange._API = exchange.Bittrex({'key': '', 'secret': ''})
        tickers[pair] = exchange.get_ticker_history(pair, tick_interval)
    else:
        tickers = optimize.load_data(args.datadir, pairs=[pair],
                                     ticker_interval=tick_interval,
                                     refresh_pairs=False,
                                     timerange=timerange)
    dataframes = optimize.tickerdata_to_dataframe(tickers)
    dataframe = dataframes[pair]
    dataframe = analyze.populate_buy_trend(dataframe)
    dataframe = analyze.populate_sell_trend(dataframe)
    dates = misc.datesarray_to_datetimearray(dataframe['date'])

    if (len(dataframe.index) > 750):
        logger.warn('Ticker contained more than 750 candles, clipping.')
    df = dataframe.tail(750)

    candles = go.Candlestick(x=df.date,
                        open=df.open,
                        high=df.high,
                        low=df.low,
                        close=df.close,
                        name='Price')

    df_buy = df[df['buy'] == 1]
    buys = go.Scattergl(
        x=df_buy.date,
        y=df_buy.close,
        mode='markers',
        name='buy',
        marker=dict(
            symbol='triangle-up-dot',
            size=9,
            line=dict(
                width=1,
            ),
            color='green',
        )
    )
    df_sell = df[df['sell'] == 1]
    sells = go.Scattergl(
        x=df_sell.date,
        y=df_sell.close,
        mode='markers',
        name='sell',
        marker=dict(
            symbol='triangle-down-dot',
            size=9,
            line=dict(
                width=1,
            ),
            color='red',
        )
    )

    bb_lower = go.Scatter(
        x=df.date,
        y=df.bb_lowerband,
        name='BB lower',
        line={'color': "transparent"},
    )
    bb_upper = go.Scatter(
        x=df.date,
        y=df.bb_upperband,
        name='BB upper',
        fill="tonexty",
        fillcolor="rgba(0,176,246,0.2)", 
        line={'color': "transparent"},
    )

    macd = go.Scattergl(
        x=df['date'],
        y=df['macd'],
        name='MACD'
    )
    macdsignal = go.Scattergl(
        x=df['date'],
        y=df['macdsignal'],
        name='MACD signal'
    )

    volume = go.Bar(
        x=df['date'],
        y=df['volume'],
        name='Volume'
    )

    fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=True, row_width=[1, 1, 4],vertical_spacing=0.0001)

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


if __name__ == '__main__':
    args = plot_parse_args(sys.argv[1:])
    plot_analyzed_dataframe(args)
