# pragma pylint: disable=missing-docstring
import unittest
from unittest.mock import patch
import os
import json
import logging
import arrow
from pandas import DataFrame
from analyze import analyze_ticker
from persistence import Trade
from main import should_sell

def print_results(results):
    print('Made {} buys. Average profit {:.2f}%. Total profit was {:.3f}. Average duration {:.1f} mins.'.format(
        len(results.index),
        results.profit.mean() * 100.0,
        results.profit.sum(),
        results.duration.mean()*5
    ))

class TestMain(unittest.TestCase):
    pairs = ['btc-neo', 'btc-eth', 'btc-omg', 'btc-edg', 'btc-pay', 'btc-pivx', 'btc-qtum', 'btc-mtl', 'btc-etc', 'btc-ltc']
    conf = {
        "minimal_roi": {
            "2880": 0.005,
            "720":  0.01,
            "0":  0.02
        },
        "stoploss": -0.10
    }

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.DEBUG) # disable debug logs that slow backtesting a lot

    @unittest.skipIf(not os.environ.get('BACKTEST', False), "slow, should be run manually")
    def test_backtest(self):
        trades = []
        with patch.dict('main._CONF', self.conf):
            for pair in self.pairs:
                with open('test/testdata/'+pair+'.json') as data_file:
                    data = json.load(data_file)

                    with patch('analyze.get_ticker', return_value=data):
                        with patch('arrow.utcnow', return_value=arrow.get('2017-08-20T14:50:00')):
                            ticker = analyze_ticker(pair)
                            # for each buy point
                            for index, row in ticker[ticker.buy == 1].iterrows():
                                trade = Trade(
                                    open_rate=row['close'],
                                    open_date=arrow.get(row['date']).datetime,
                                    amount=1,
                                )
                                # calculate win/lose forwards from buy point
                                for index2, row2 in ticker[index:].iterrows():
                                    if should_sell(trade, row2['close'], arrow.get(row2['date']).datetime):
                                        current_profit = (row2['close'] - trade.open_rate) / trade.open_rate

                                        trades.append((pair, current_profit, index2 - index))
                                        break
            
        labels = ['currency', 'profit', 'duration']
        results = DataFrame.from_records(trades, columns=labels)

        print('====================== BACKTESTING REPORT ================================')

        for pair in self.pairs:
            print('For currency {}:'.format(pair))
            print_results(results[results.currency == pair])
        print('TOTAL OVER ALL TRADES:')
        print_results(results)
