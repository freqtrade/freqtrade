# pragma pylint: disable=missing-docstring,W0212


import logging
from typing import Tuple, Dict

import arrow
from pandas import DataFrame
from tabulate import tabulate

from freqtrade import exchange
from freqtrade.analyze import populate_buy_trend, populate_sell_trend
from freqtrade.exchange import Bittrex
from freqtrade.main import min_roi_reached
from freqtrade.misc import load_config
from freqtrade.optimize import load_data, preprocess
from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)


def get_timeframe(data: Dict[str, Dict]) -> Tuple[arrow.Arrow, arrow.Arrow]:
    """
    Get the maximum timeframe for the given backtest data
    :param data: dictionary with backtesting data
    :return: tuple containing min_date, max_date
    """
    min_date, max_date = None, None
    for values in data.values():
        sorted_values = sorted(values, key=lambda d: arrow.get(d['T']))
        if not min_date or sorted_values[0]['T'] < min_date:
            min_date = sorted_values[0]['T']
        if not max_date or sorted_values[-1]['T'] > max_date:
            max_date = sorted_values[-1]['T']
    return arrow.get(min_date), arrow.get(max_date)


def generate_text_table(data: Dict[str, Dict], results: DataFrame, stake_currency) -> str:
    """
    Generates and returns a text table for the given backtest data and the results dataframe
    :return: pretty printed table with tabulate as str
    """
    tabular_data = []
    headers = ['pair', 'buy count', 'avg profit', 'total profit', 'avg duration']
    for pair in data:
        result = results[results.currency == pair]
        tabular_data.append([
            pair,
            len(result.index),
            '{:.2f}%'.format(result.profit.mean() * 100.0),
            '{:.08f} {}'.format(result.profit.sum(), stake_currency),
            '{:.2f}'.format(result.duration.mean() * 5),
        ])

    # Append Total
    tabular_data.append([
        'TOTAL',
        len(results.index),
        '{:.2f}%'.format(results.profit.mean() * 100.0),
        '{:.08f} {}'.format(results.profit.sum(), stake_currency),
        '{:.2f}'.format(results.duration.mean() * 5),
    ])
    return tabulate(tabular_data, headers=headers)


def backtest(config: Dict, processed: Dict[str, DataFrame],
             max_open_trades: int = 0, realistic: bool = True) -> DataFrame:
    """
    Implements backtesting functionality
    :param config: config to use
    :param processed: a processed dictionary with format {pair, data}
    :param max_open_trades: maximum number of concurrent trades (default: 0, disabled)
    :param realistic: do we try to simulate realistic trades? (default: True)
    :return: DataFrame
    """
    trades = []
    trade_count_lock = {}
    exchange._API = Bittrex({'key': '', 'secret': ''})
    for pair, pair_data in processed.items():
        pair_data['buy'], pair_data['sell'] = 0, 0
        ticker = populate_sell_trend(populate_buy_trend(pair_data))
        # for each buy point
        lock_pair_until = None
        for row in ticker[ticker.buy == 1].itertuples(index=True):
            if realistic:
                if lock_pair_until is not None and row.Index <= lock_pair_until:
                    continue
            if max_open_trades > 0:
                # Check if max_open_trades has already been reached for the given date
                if not trade_count_lock.get(row.date, 0) < max_open_trades:
                    continue

            if max_open_trades > 0:
                # Increase lock
                trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

            trade = Trade(
                open_rate=row.close,
                open_date=row.date,
                amount=config['stake_amount'],
                fee=exchange.get_fee() * 2
            )

            # calculate win/lose forwards from buy point
            for row2 in ticker[row.Index + 1:].itertuples(index=True):
                if max_open_trades > 0:
                    # Increase trade_count_lock for every iteration
                    trade_count_lock[row2.date] = trade_count_lock.get(row2.date, 0) + 1

                if min_roi_reached(trade, row2.close, row2.date) or row2.sell == 1:
                    current_profit = trade.calc_profit(row2.close)
                    lock_pair_until = row2.Index

                    trades.append((pair, current_profit, row2.Index - row.Index))
                    break
    labels = ['currency', 'profit', 'duration']
    return DataFrame.from_records(trades, columns=labels)


def start(args):
    print('')
    exchange._API = Bittrex({'key': '', 'secret': ''})

    print('Using config: {} ...'.format(args.config))
    config = load_config(args.config)

    print('Using ticker_interval: {} ...'.format(args.ticker_interval))

    data = {}
    if args.live:
        print('Downloading data for all pairs in whitelist ...')
        for pair in config['exchange']['pair_whitelist']:
            data[pair] = exchange.get_ticker_history(pair, args.ticker_interval)
    else:
        print('Using local backtesting data (ignoring whitelist in given config)...')
        data = load_data(args.ticker_interval)

    print('Using stake_currency: {} ...\nUsing stake_amount: {} ...'.format(
        config['stake_currency'], config['stake_amount']
    ))

    # Print timeframe
    min_date, max_date = get_timeframe(data)
    print('Measuring data from {} up to {} ...'.format(
        min_date.isoformat(), max_date.isoformat()
    ))

    max_open_trades = 0
    if args.realistic_simulation:
        print('Using max_open_trades: {} ...'.format(config['max_open_trades']))
        max_open_trades = config['max_open_trades']

    from freqtrade import main
    main._CONF = config

    # Execute backtest and print results
    results = backtest(
        config, preprocess(data), max_open_trades, args.realistic_simulation
    )
    print('====================== BACKTESTING REPORT ======================================\n\n')
    print(generate_text_table(data, results, config['stake_currency']))
