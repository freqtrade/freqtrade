# pragma pylint: disable=missing-docstring


import logging
import os
from typing import Tuple, Dict

import arrow
import pytest
from pandas import DataFrame
from tabulate import tabulate

from freqtrade import exchange
from freqtrade.analyze import parse_ticker_dataframe, populate_indicators, \
    populate_buy_trend, populate_sell_trend
from freqtrade.exchange import Bittrex
from freqtrade.main import min_roi_reached
from freqtrade.misc import load_config
from freqtrade.persistence import Trade
from freqtrade.tests import load_backtesting_data

logger = logging.getLogger(__name__)


def format_results(results: DataFrame):
    return 'Made {} buys. Average profit {:.2f}%. ' \
           'Total profit was {:.3f}. Average duration {:.1f} mins.'.format(
               len(results.index),
               results.profit.mean() * 100.0,
               results.profit.sum(),
               results.duration.mean() * 5,
           )


def preprocess(backdata) -> Dict[str, DataFrame]:
    processed = {}
    for pair, pair_data in backdata.items():
        processed[pair] = populate_indicators(parse_ticker_dataframe(pair_data))
    return processed


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


def backtest(backtest_conf, processed, mocker):
    trades = []
    exchange._API = Bittrex({'key': '', 'secret': ''})
    mocker.patch.dict('freqtrade.main._CONF', backtest_conf)
    for pair, pair_data in processed.items():
        pair_data['buy'] = 0
        pair_data['sell'] = 0
        ticker = populate_sell_trend(populate_buy_trend(pair_data))
        # for each buy point
        for row in ticker[ticker.buy == 1].itertuples(index=True):
            trade = Trade(
                open_rate=row.close,
                open_date=row.date,
                amount=backtest_conf['stake_amount'],
                fee=exchange.get_fee() * 2
            )
            # calculate win/lose forwards from buy point
            for row2 in ticker[row.Index:].itertuples(index=True):
                if min_roi_reached(trade, row2.close, row2.date) or row2.sell == 1:
                    current_profit = trade.calc_profit(row2.close)

                    trades.append((pair, current_profit, row2.Index - row.Index))
                    break
    labels = ['currency', 'profit', 'duration']
    return DataFrame.from_records(trades, columns=labels)


@pytest.mark.skipif(not os.environ.get('BACKTEST'), reason="BACKTEST not set")
def test_backtest(backtest_conf, mocker):
    print('')
    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Load configuration file based on env variable
    conf_path = os.environ.get('BACKTEST_CONFIG')
    if conf_path:
        print('Using config: {} ...'.format(conf_path))
        config = load_config(conf_path)
    else:
        config = backtest_conf

    # Parse ticker interval
    ticker_interval = int(os.environ.get('BACKTEST_TICKER_INTERVAL') or 5)
    print('Using ticker_interval: {} ...'.format(ticker_interval))

    data = {}
    if os.environ.get('BACKTEST_LIVE'):
        print('Downloading data for all pairs in whitelist ...')
        for pair in config['exchange']['pair_whitelist']:
            data[pair] = exchange.get_ticker_history(pair, ticker_interval)
    else:
        print('Using local backtesting data (ignoring whitelist in given config)...')
        data = load_backtesting_data(ticker_interval)

    print('Using stake_currency: {} ...\nUsing stake_amount: {} ...'.format(
        config['stake_currency'], config['stake_amount']
    ))

    # Print timeframe
    min_date, max_date = get_timeframe(data)
    print('Measuring data from {} up to {} ...'.format(
        min_date.isoformat(), max_date.isoformat()
    ))

    # Execute backtest and print results
    results = backtest(config, preprocess(data), mocker)
    print('====================== BACKTESTING REPORT ======================================\n\n'
          'NOTE: This Report doesn\'t respect the limits of max_open_trades, \n'
          '      so the projected values should be taken with a grain of salt.\n')
    print(generate_text_table(data, results, config['stake_currency']))
