# pragma pylint: disable=missing-docstring
import json
import logging
import os

import pytest
import arrow
from pandas import DataFrame

from freqtrade.analyze import analyze_ticker
from freqtrade.main import should_sell
from freqtrade.persistence import Trade

logging.disable(logging.DEBUG) # disable debug logs that slow backtesting a lot

def format_results(results):
    return 'Made {} buys. Average profit {:.2f}%. Total profit was {:.3f}. Average duration {:.1f} mins.'.format(
        len(results.index),
        results.profit.mean() * 100.0,
        results.profit.sum(),
        results.duration.mean() * 5
    )

def print_pair_results(pair, results):
    print('For currency {}:'.format(pair))
    print(format_results(results[results.currency == pair]))

@pytest.fixture
def pairs():
    return ['btc-neo', 'btc-eth', 'btc-omg', 'btc-edg', 'btc-pay',
            'btc-pivx', 'btc-qtum', 'btc-mtl', 'btc-etc', 'btc-ltc']

@pytest.fixture
def conf():
    return {
        "minimal_roi": {
            "50":  0.0,
            "40":  0.01,
            "30":  0.02,
            "0":  0.045
        },
        "stoploss": -0.40
    }

def backtest(conf, pairs, mocker):
    trades = []
    mocker.patch.dict('freqtrade.main._CONF', conf)
    for pair in pairs:
        with open('freqtrade/tests/testdata/'+pair+'.json') as data_file:
            data = json.load(data_file)

            mocker.patch('freqtrade.analyze.get_ticker_history', return_value=data)
            mocker.patch('arrow.utcnow', return_value=arrow.get('2017-08-20T14:50:00'))
            ticker = analyze_ticker(pair)[['close', 'date', 'buy']].copy()
            # for each buy point
            for index, row in ticker[ticker.buy == 1].iterrows():
                trade = Trade(
                    open_rate=row['close'],
                    open_date=row['date'],
                    amount=1,
                )
                # calculate win/lose forwards from buy point
                for index2, row2 in ticker[index:].iterrows():
                    if should_sell(trade, row2['close'], row2['date']):
                        current_profit = (row2['close'] - trade.open_rate) / trade.open_rate

                        trades.append((pair, current_profit, index2 - index))
                        break
    labels = ['currency', 'profit', 'duration']
    results = DataFrame.from_records(trades, columns=labels)
    return results

@pytest.mark.skipif(not os.environ.get('BACKTEST', False), reason="BACKTEST not set")
def test_backtest(conf, pairs, mocker, report=True):
    results = backtest(conf, pairs, mocker)

    print('====================== BACKTESTING REPORT ================================')
    [print_pair_results(pair, results) for pair in pairs]
    print('TOTAL OVER ALL TRADES:')
    print(format_results(results))
