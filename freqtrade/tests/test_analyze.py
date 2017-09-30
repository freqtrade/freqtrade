# pragma pylint: disable=missing-docstring
import unittest
from unittest.mock import patch

import arrow
from pandas import DataFrame

from freqtrade.analyze import parse_ticker_dataframe, populate_buy_trend, populate_indicators, \
    get_buy_signal

RESULT_BITTREX = {
    'success': True,
    'message': '',
    'result': [
        {'O': 0.00065311, 'H': 0.00065311, 'L': 0.00065311, 'C': 0.00065311, 'V': 22.17210568, 'T': '2017-08-30T10:40:00', 'BV': 0.01448082},
        {'O': 0.00066194, 'H': 0.00066195, 'L': 0.00066194, 'C': 0.00066195, 'V': 33.4727437, 'T': '2017-08-30T10:34:00', 'BV': 0.02215696},
        {'O': 0.00065311, 'H': 0.00065311, 'L': 0.00065311, 'C': 0.00065311, 'V': 53.85127609, 'T': '2017-08-30T10:37:00', 'BV': 0.0351708},
        {'O': 0.00066194, 'H': 0.00066194, 'L': 0.00065311, 'C': 0.00065311, 'V': 46.29210665, 'T': '2017-08-30T10:42:00', 'BV': 0.03063118},
    ]
}

class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.result = parse_ticker_dataframe(RESULT_BITTREX['result'], arrow.get('2017-08-30T10:00:00'))

    def test_1_dataframe_has_correct_columns(self):
        self.assertEqual(self.result.columns.tolist(),
                         ['close', 'high', 'low', 'open', 'date', 'volume'])

    def test_2_orders_by_date(self):
        self.assertEqual(self.result['date'].tolist(),
                         ['2017-08-30T10:34:00',
                          '2017-08-30T10:37:00',
                          '2017-08-30T10:40:00',
                          '2017-08-30T10:42:00'])

    def test_3_populates_buy_trend(self):
        dataframe = populate_buy_trend(populate_indicators(self.result))
        self.assertTrue('buy' in dataframe.columns)
        self.assertTrue('buy_price' in dataframe.columns)

    def test_4_returns_latest_buy_signal(self):
        buydf = DataFrame([{'buy': 1, 'date': arrow.utcnow()}])
        with patch('freqtrade.analyze.analyze_ticker', return_value=buydf):
            self.assertEqual(get_buy_signal('BTC-ETH'), True)
        buydf = DataFrame([{'buy': 0, 'date': arrow.utcnow()}])
        with patch('freqtrade.analyze.analyze_ticker', return_value=buydf):
            self.assertEqual(get_buy_signal('BTC-ETH'), False)


if __name__ == '__main__':
    unittest.main()
