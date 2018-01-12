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
    # FIX: perhaps delete those backtesting options that are not feasible
    misc.backtesting_options(parser)
    # TODO: Make the pair argument take a comma separated list
    parser.add_argument(
        '-p', '--pair',
        help = 'Show profits for only this pair',
        dest = 'pair',
        default = None
    )

    return parser.parse_args(args)


def make_profit_array(data, filter_pair):
    xmin = 0
    xmax = 0

    #  pair       profit-%    time  duration
    # ['BTC_XMR', 0.00537847, 5057, 1]
    for trade in data:
        pair = trade[0]
        profit = trade[1]
        x = trade[2]
        dur = trade[3]
        xmax = max(xmax, x + dur)

    pg = np.zeros(xmax)

    # Go through the trades
    # and make an total profit
    # array
    for trade in data:
        pair = trade[0]
        if filter_pair and pair != filter_pair:
            continue
        profit = trade[1]
        tim = trade[2]
        dur = trade[3]
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

    filter_pair = args.pair

    config = misc.load_config(args.config)
    pairs = config['exchange']['pair_whitelist']
    if filter_pair:
        print('Filtering out pair %s' % filter_pair)
        pairs = list(filter(lambda pair: pair == filter_pair, pairs))

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

    # We are essentially saying:
    #  array <- sum dataframes[*]['close'] / num_items dataframes
    #  FIX: there should be some onliner numpy/panda for this

    first = True
    avgclose = None
    num = 0
    for pair, pair_data in dataframes.items():
      close = pair_data['close']
      print('Pair %s has length %s' %(pair, len(close)))
      num += 1
      if first:
        first = False
        avgclose = np.copy(close)
      else:
        avgclose += close
    avgclose /= num

    # Load the profits results
    # And make an profits-growth array

    filename = 'backtest-result.json'
    with open(filename) as file:
      data = json.load(file)
    pg = make_profit_array(data, filter_pair)

    #
    # Plot the pairs average close prices, and total profit growth
    #

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('total profit')
    ax1.plot(avgclose, label='avgclose')
    ax2.plot(pg, label='profit')
    ax1.legend()
    ax2.legend()

    # FIX if we have one line pair in paris
    #     then skip the plotting of the third graph,
    #     or change what we plot
    # In third graph, we plot each profit separately
    for pair in pairs:
        pg = make_profit_array(data, pair)
        ax3.plot(pg, label=pair)
    ax3.legend()

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()


if __name__ == '__main__':
    args = plot_parse_args(sys.argv[1:])
    plot_profit(args)
