#!/usr/bin/env python3

import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

import freqtrade.optimize as optimize
import freqtrade.misc as misc
import freqtrade.exchange as exchange
import freqtrade.analyze  as analyze


def plot_parse_args(args ):
    parser = misc.common_args_parser('Graph utility')
    # FIX: perhaps delete those backtesting options that are not feasible (shows up in -h)
    misc.backtesting_options(parser)
    parser.add_argument(
        '-p', '--pair',
        help = 'Show profits for only this pairs. Pairs are comma-separated.',
        dest = 'pair',
        default = None
    )
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
        pg[tim+dur-1] += profit

    # rewrite the pg array to go from
    # total profits at each timeframe
    # to accumulated profits
    pa = 0
    for x in range(0,len(pg)):
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
    pairs = config['exchange']['pair_whitelist']
    if filter_pairs:
        filter_pairs = filter_pairs.split(',')
        pairs = list(set(pairs) & set(filter_pairs))
        print('Filter, keep pairs %s' % pairs)

    tickers = optimize.load_data(args.datadir, pairs=pairs,
                                 ticker_interval=args.ticker_interval,
                                 refresh_pairs=False)
    dataframes = optimize.preprocess(tickers)

    # Make an average close price of all the pairs that was involved.
    # this could be useful to gauge the overall market trend

    # FIX: since the dataframes are of unequal length,
    # andor has different dates, we need to merge them
    # But we dont have the date information in the
    # backtesting results, this is needed to match the dates
    # For now, assume the dataframes are aligned.
    max_x = 0
    for pair, pair_data in dataframes.items():
        n = len(pair_data['close'])
        max_x = max(max_x, n)
    #    if max_x != n:
    #        raise Exception('Please rerun script. Input data has different lengths %s'
    #                         %('Different pair length: %s <=> %s' %(max_x, n)))
    print('max_x: %s' %(max_x))

    # We are essentially saying:
    #  array <- sum dataframes[*]['close'] / num_items dataframes
    #  FIX: there should be some onliner numpy/panda for this
    avgclose = np.zeros(max_x)
    num = 0
    for pair, pair_data in dataframes.items():
        close = pair_data['close']
        maxprice = max(close)  # Normalize price to [0,1]
        print('Pair %s has length %s' %(pair, len(close)))
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

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('total profit')
    ax1.plot(avgclose, label='avgclose')
    ax2.plot(pg, label='profit')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    # FIX if we have one line pair in paris
    #     then skip the plotting of the third graph,
    #     or change what we plot
    # In third graph, we plot each profit separately
    for pair in pairs:
        pg = make_profit_array(data, max_x, pair)
        ax3.plot(pg, label=pair)
    ax3.legend(loc='upper left')
    # black background to easier see multiple colors
    ax3.set_facecolor('black')

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()


if __name__ == '__main__':
    args = plot_parse_args(sys.argv[1:])
    plot_profit(args)
