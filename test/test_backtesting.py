# pragma pylint: disable=missing-docstring
import unittest
from unittest.mock import patch
import json
import arrow
from analyze import analyze_ticker

class TestMain(unittest.TestCase):

    def test_1_create_trade(self):
        with open('test/testdata/btc-neo.json') as data_file:
            data = json.load(data_file)

            with patch('analyze.get_ticker', return_value=data):
                with patch('arrow.utcnow', return_value=arrow.get('2017-08-20T14:50:00')):
                    print(analyze_ticker('BTC-NEO'))
