# pragma pylint: disable=missing-docstring

import json
import logging
import os
from typing import Tuple, Dict

import arrow
import pytest
from arrow import Arrow
from pandas import DataFrame

from freqtrade import exchange
from freqtrade.analyze import parse_ticker_dataframe, populate_indicators, \
    populate_buy_trend, populate_sell_trend
from freqtrade.exchange import Bittrex
from freqtrade.main import min_roi_reached
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


def format_results(results: DataFrame):
    return 'Made {} buys. Average profit {:.2f}%. ' \
           'Total profit was {:.3f}. Average duration {:.1f} mins.'.format(
               len(results.index),
               results.profit.mean() * 100.0,
               results.profit.sum(),
               results.duration.mean() * 5,
           )


def print_pair_results(pair: str, results: DataFrame):
    print('For currency {}:'.format(pair))
    print(format_results(results[results.currency == pair]))


def preprocess(backdata) -> Dict[str, DataFrame]:
    processed = {}
    for pair, pair_data in backdata.items():
        processed[pair] = populate_indicators(parse_ticker_dataframe(pair_data))
    return processed


def get_timeframe(backdata: Dict[str, Dict]) -> Tuple[Arrow, Arrow]:
    min_date, max_date = None, None
    for values in backdata.values():
        values = sorted(values, key=lambda d: arrow.get(d['T']))
        if not min_date or values[0]['T'] < min_date:
            min_date = values[0]['T']
        if not max_date or values[-1]['T'] > max_date:
            max_date = values[-1]['T']
    return arrow.get(min_date), arrow.get(max_date)


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
                amount=1,
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
def test_backtest(backtest_conf, backdata, mocker):
    print('')

    config = None
    conf_path = os.environ.get('BACKTEST_CONFIG')
    if conf_path:
        print('Using config: {} ...'.format(conf_path))
        with open(conf_path, 'r') as fp:
            config = json.load(fp)

    ticker_interval = int(os.environ.get('BACKTEST_TICKER_INTERVAL') or 5)
    print('Using ticker_interval: {}'.format(ticker_interval))

    livedata = {}
    if os.environ.get('BACKTEST_LIVE'):
        print('Downloading data for all pairs in whitelist ...'.format(conf_path))
        exchange._API = Bittrex({'key': '', 'secret': ''})
        for pair in config['exchange']['pair_whitelist']:
            livedata[pair] = exchange.get_ticker_history(pair, ticker_interval)

    config = config or backtest_conf
    data = livedata or backdata

    min_date, max_date = get_timeframe(data)
    print('Measuring data from {} up to {} ...'.format(
        min_date.isoformat(), max_date.isoformat()
    ))

    results = backtest(config, preprocess(data), mocker)
    print('====================== BACKTESTING REPORT ================================')
    for pair in data:
        print_pair_results(pair, results)
    print('TOTAL OVER ALL TRADES:')
    print(format_results(results))
