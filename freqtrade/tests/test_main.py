import unittest
from unittest.mock import patch, MagicMock, call

import copy
from jsonschema import validate

from freqtrade import exchange
from freqtrade.main import create_trade, handle_trade, close_trade_if_fulfilled, init, \
    get_target_bid
from freqtrade.misc import CONF_SCHEMA
from freqtrade.persistence import Trade


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
        "bid_strategy": {
            "ask_last_balance": 0.0
        },
        "bittrex": {
            "enabled": True,
            "key": "key",
            "secret": "secret",
            "pair_whitelist": [
                "BTC_ETH",
                "BTC_TKN",
                "BTC_TRST",
                "BTC_SWT",
            ]
        },
        "telegram": {
            "enabled": True,
            "token": "token",
            "chat_id": "chat_id"
        }
    }

    def test_1_create_trade(self):
        with patch.dict('freqtrade.main._CONF', self.conf):
            with patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True) as buy_signal:
                with patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock()):
                    with patch.multiple('freqtrade.main.exchange',
                                        get_ticker=MagicMock(return_value={
                                            'bid': 0.07256061,
                                            'ask': 0.072661,
                                            'last': 0.07256061
                                        }),
                                        buy=MagicMock(return_value='mocked_order_id')):
                        # Save state of current whitelist
                        whitelist = copy.deepcopy(self.conf['bittrex']['pair_whitelist'])

                        init(self.conf, 'sqlite://')
                        for pair in ['BTC_ETH', 'BTC_TKN', 'BTC_TRST', 'BTC_SWT']:
                            trade = create_trade(15.0, exchange.Exchange.BITTREX)
                            Trade.session.add(trade)
                            Trade.session.flush()
                            self.assertIsNotNone(trade)
                            self.assertEqual(trade.open_rate, 0.072661)
                            self.assertEqual(trade.pair, pair)
                            self.assertEqual(trade.exchange, exchange.Exchange.BITTREX)
                            self.assertEqual(trade.amount, 206.43811673387373)
                            self.assertEqual(trade.stake_amount, 15.0)
                            self.assertEqual(trade.is_open, True)
                            self.assertIsNotNone(trade.open_date)
                            self.assertEqual(whitelist, self.conf['bittrex']['pair_whitelist'])

                        buy_signal.assert_has_calls(
                            [call('BTC_ETH'), call('BTC_TKN'), call('BTC_TRST'), call('BTC_SWT')]
                        )

    def test_2_handle_trade(self):
        with patch.dict('freqtrade.main._CONF', self.conf):
            with patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock()):
                with patch.multiple('freqtrade.main.exchange',
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
        with patch.dict('freqtrade.main._CONF', self.conf):
            trade = Trade.query.filter(Trade.is_open.is_(True)).first()
            self.assertTrue(trade)

            # Simulate that there is no open order
            trade.open_order_id = None

            closed = close_trade_if_fulfilled(trade)
            self.assertTrue(closed)
            self.assertEqual(trade.is_open, False)

    def test_balance_fully_ask_side(self):
        with patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 0.0}}):
            self.assertEqual(get_target_bid({'ask': 20, 'last': 10}), 20)

    def test_balance_fully_last_side(self):
        with patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}}):
            self.assertEqual(get_target_bid({'ask': 20, 'last': 10}), 10)

    def test_balance_when_last_bigger_than_ask(self):
        with patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}}):
            self.assertEqual(get_target_bid({'ask': 5, 'last': 10}), 5)

    @classmethod
    def setUpClass(cls):
        validate(cls.conf, CONF_SCHEMA)


if __name__ == '__main__':
    unittest.main()
