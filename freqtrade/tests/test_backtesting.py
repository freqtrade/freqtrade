# pragma pylint: disable=missing-docstring
from typing import Dict
import logging
import os

import pytest
import arrow
from pandas import DataFrame

from freqtrade import exchange
from freqtrade.analyze import parse_ticker_dataframe, populate_indicators, \
    populate_buy_trend, populate_sell_trend
from freqtrade.exchange import Bittrex
from freqtrade.main import min_roi_reached
from freqtrade.persistence import Trade

logging.disable(logging.DEBUG)  # disable debug logs that slow backtesting a lot


def format_results(results):
    return 'Made {} buys. Average profit {:.2f}%. Total profit was {:.3f}. Average duration {:.1f} mins.'.format(
        len(results.index), results.profit.mean() * 100.0, results.profit.sum(), results.duration.mean() * 5)


def print_pair_results(pair, results):
    print('For currency {}:'.format(pair))
    print(format_results(results[results.currency == pair]))


def preprocess(backdata) -> Dict[str, DataFrame]:
    processed = {}
    for pair, pair_data in backdata.items():
        processed[pair] = populate_indicators(parse_ticker_dataframe(pair_data))
    return processed


def backtest(backtest_conf, processed, mocker):
    trades = []
    exchange._API = Bittrex({'key': '', 'secret': ''})
    mocker.patch.dict('freqtrade.main._CONF', backtest_conf)
    mocker.patch('arrow.utcnow', return_value=arrow.get('2017-08-20T14:50:00'))
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


@pytest.mark.skipif(not os.environ.get('BACKTEST', False), reason="BACKTEST not set")
def test_backtest(backtest_conf, backdata, mocker, report=True):
    results = backtest(backtest_conf, preprocess(backdata), mocker)

    print('====================== BACKTESTING REPORT ================================')
    for pair in backdata:
        print_pair_results(pair, results)
    print('TOTAL OVER ALL TRADES:')
    print(format_results(results))
