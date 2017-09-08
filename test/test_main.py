import unittest
from unittest.mock import patch, MagicMock

import os
from jsonschema import validate

import exchange
from main import create_trade, handle_trade, close_trade_if_fulfilled, init
from misc import conf_schema
from persistence import Trade


class TestMain(unittest.TestCase):
    conf = {
        "max_open_trades": 3,
        "stake_currency": "BTC",
        "stake_amount": 0.05,
        "dry_run": True,
        "minimal_roi": {
            "2880": 0.005,
            "720": 0.01,
            "0": 0.02
        },
        "poloniex": {
            "enabled": False,
            "key": "key",
            "secret": "secret",
            "pair_whitelist": []
        },
        "bittrex": {
            "enabled": True,
            "key": "key",
            "secret": "secret",
            "pair_whitelist": [
                "BTC_ETH"
            ]
        },
        "telegram": {
            "enabled": True,
            "token": "token",
            "chat_id": "chat_id"
        }
    }

    def test_1_create_trade(self):
        with patch.dict('main._conf', self.conf):
            with patch('main.get_buy_signal', side_effect=lambda _: True) as buy_signal:
                with patch.multiple('main.telegram', init=MagicMock(), send_msg=MagicMock()):
                    with patch.multiple('main.exchange',
                                        get_ticker=MagicMock(return_value={
                                            'bid': 0.07256061,
                                            'ask': 0.072661,
                                            'last': 0.07256061
                                        }),
                                        buy=MagicMock(return_value='mocked_order_id')):
                        init(self.conf, 'sqlite://')
                        trade = create_trade(15.0, exchange.Exchange.BITTREX)
                        Trade.session.add(trade)
                        Trade.session.flush()
                        self.assertIsNotNone(trade)
                        self.assertEqual(trade.open_rate, 0.072661)
                        self.assertEqual(trade.pair, 'BTC_ETH')
                        self.assertEqual(trade.exchange, exchange.Exchange.BITTREX)
                        self.assertEqual(trade.amount, 206.43811673387373)
                        self.assertEqual(trade.btc_amount, 15.0)
                        self.assertEqual(trade.is_open, True)
                        self.assertIsNotNone(trade.open_date)
                        buy_signal.assert_called_once_with('BTC_ETH')

    def test_2_handle_trade(self):
        with patch.dict('main._conf', self.conf):
            with patch.multiple('main.telegram', init=MagicMock(), send_msg=MagicMock()):
                with patch.multiple('main.exchange',
                                    get_ticker=MagicMock(return_value={
                                        'bid': 0.17256061,
                                        'ask': 0.172661,
                                        'last': 0.17256061
                                    }),
                                    buy=MagicMock(return_value='mocked_order_id')):
                    trade = Trade.query.filter(Trade.is_open.is_(True)).first()
                    self.assertTrue(trade)
                    handle_trade(trade)
                    self.assertEqual(trade.close_rate, 0.17256061)
                    self.assertEqual(trade.close_profit, 137.4872490056564)
                    self.assertIsNotNone(trade.close_date)
                    self.assertEqual(trade.open_order_id, 'dry_run')

    def test_3_close_trade(self):
        with patch.dict('main._conf', self.conf):
            trade = Trade.query.filter(Trade.is_open.is_(True)).first()
            self.assertTrue(trade)

            # Simulate that there is no open order
            trade.open_order_id = None

            closed = close_trade_if_fulfilled(trade)
            self.assertTrue(closed)
            self.assertEqual(trade.is_open, False)

    @classmethod
    def setUpClass(cls):
        validate(cls.conf, conf_schema)


if __name__ == '__main__':
    unittest.main()
