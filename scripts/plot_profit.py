#!/usr/bin/env python3

import sys
import json
import numpy as np

from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go

import freqtrade.optimize as optimize
import freqtrade.misc as misc
from freqtrade.strategy.strategy import Strategy


def plot_parse_args(args):
    parser = misc.common_args_parser('Graph profits')
    # FIX: perhaps delete those backtesting options that are not feasible (shows up in -h)
    misc.backtesting_options(parser)
    misc.scripts_options(parser)
    return parser.parse_args(args)


# data:: [ pair,      profit-%,  enter,         exit,        time, duration]
# data:: ['BTC_XMR', 0.00537847, '1511176800', '1511178000', 5057, 1]
# FIX: make use of the enter/exit dates to insert the
# profit more precisely into the pg array
def make_profit_array(data, px, filter_pairs=[]):
    pg = np.zeros(px)
    # Go through the trades
    # and make an total profit
    # array
    for trade in data:
        pair = trade[0]
        if filter_pairs and pair not in filter_pairs:
            continue
        profit = trade[1]
        tim = trade[4]
        dur = trade[5]
        ix = tim + dur - 1
        if ix < px:
            pg[ix] += profit

    # rewrite the pg array to go from
    # total profits at each timeframe
    # to accumulated profits
    pa = 0
    for x in range(0, len(pg)):
        p = pg[x]  # Get current total percent
        pa += p  # Add to the accumulated percent
        pg[x] = pa  # write back to save memory

    return pg


def plot_profit(args) -> None:
    """
    Plots the total profit for all pairs.
    Note, the profit calculation isn't realistic.
    But should be somewhat proportional, and therefor useful
    in helping out to find a good algorithm.
    """

    # We need to use the same pairs, same tick_interval
    # and same timeperiod as used in backtesting
    # to match the tickerdata against the profits-results

    filter_pairs = args.pair

    config = misc.load_config(args.config)
    config.update({'strategy': args.strategy})

    # Init strategy
    strategy = Strategy()
    strategy.init(config)

    pairs = config['exchange']['pair_whitelist']

    if filter_pairs:
        filter_pairs = filter_pairs.split(',')
        pairs = list(set(pairs) & set(filter_pairs))
        print('Filter, keep pairs %s' % pairs)

    timerange = misc.parse_timerange(args.timerange)
    tickers = optimize.load_data(args.datadir, pairs=pairs,
                                 ticker_interval=strategy.ticker_interval,
                                 refresh_pairs=False,
                                 timerange=timerange)
    dataframes = optimize.preprocess(tickers)

    # NOTE: the dataframes are of unequal length,
    # 'dates' is an merged date array of them all.

    dates = misc.common_datearray(dataframes)
    max_x = dates.size

    # Make an average close price of all the pairs that was involved.
    # this could be useful to gauge the overall market trend
    # We are essentially saying:
    #  array <- sum dataframes[*]['close'] / num_items dataframes
    #  FIX: there should be some onliner numpy/panda for this
    avgclose = np.zeros(max_x)
    num = 0
    for pair, pair_data in dataframes.items():
        close = pair_data['close']
        maxprice = max(close)  # Normalize price to [0,1]
        print('Pair %s has length %s' % (pair, len(close)))
        for x in range(0, len(close)):
            avgclose[x] += close[x] / maxprice
        # avgclose += close
        num += 1
    avgclose /= num

    # Load the profits results
    # And make an profits-growth array

    filename = 'backtest-result.json'
    with open(filename) as file:
        data = json.load(file)
    pg = make_profit_array(data, max_x, filter_pairs)

    #
    # Plot the pairs average close prices, and total profit growth
    #

    avgclose = go.Scattergl(
        x=dates,
        y=avgclose,
        name='Avg close price',
    )
    profit = go.Scattergl(
        x=dates,
        y=pg,
        name='Profit',
    )

    fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=True, row_width=[1, 1, 1])

    fig.append_trace(avgclose, 1, 1)
    fig.append_trace(profit, 2, 1)

    for pair in pairs:
        pg = make_profit_array(data, max_x, pair)
        pair_profit = go.Scattergl(
            x=dates,
            y=pg,
            name=pair,
        )
        fig.append_trace(pair_profit, 3, 1)

    plot(fig, filename='freqtrade-profit-plot.html')


if __name__ == '__main__':
    args = plot_parse_args(sys.argv[1:])
    plot_profit(args)
