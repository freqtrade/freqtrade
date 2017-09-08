import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from jsonschema import validate
from telegram import Bot, Update, Message, Chat

import exchange
from main import init, create_trade, update_state, State, get_state
from misc import CONF_SCHEMA
from persistence import Trade
from rpc.telegram import _status, _profit, _forcesell, _performance, _start, _stop


class MagicBot(MagicMock, Bot):
    pass


class TestTelegram(unittest.TestCase):

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
            "chat_id": "0"
        }
    }

    def test_1_status_handle(self):
        with patch.dict('main._CONF', self.conf):
            with patch('main.get_buy_signal', side_effect=lambda _: True):
                msg_mock = MagicMock()
                with patch.multiple('main.telegram', _CONF=self.conf, init=MagicMock(), send_msg=msg_mock):
                    with patch.multiple('main.exchange',
                                        get_ticker=MagicMock(return_value={
                                            'bid': 0.07256061,
                                            'ask': 0.072661,
                                            'last': 0.07256061
                                        }),
                                        buy=MagicMock(return_value='mocked_order_id')):
                        init(self.conf, 'sqlite://')

                        # Create some test data
                        trade = create_trade(15.0, exchange.Exchange.BITTREX)
                        self.assertTrue(trade)
                        Trade.session.add(trade)
                        Trade.session.flush()

                        _status(bot=MagicBot(), update=self.update)
                        self.assertEqual(msg_mock.call_count, 2)
                        self.assertIn('[BTC_ETH]', msg_mock.call_args_list[-1][0][0])

    def test_2_profit_handle(self):
        with patch.dict('main._CONF', self.conf):
            with patch('main.get_buy_signal', side_effect=lambda _: True):
                msg_mock = MagicMock()
                with patch.multiple('main.telegram', _CONF=self.conf, init=MagicMock(), send_msg=msg_mock):
                    with patch.multiple('main.exchange',
                                        get_ticker=MagicMock(return_value={
                                            'bid': 0.07256061,
                                            'ask': 0.072661,
                                            'last': 0.07256061
                                        }),
                                        buy=MagicMock(return_value='mocked_order_id')):
                        init(self.conf, 'sqlite://')

                        # Create some test data
                        trade = create_trade(15.0, exchange.Exchange.BITTREX)
                        self.assertTrue(trade)
                        trade.close_rate = 0.07256061
                        trade.close_profit = 100.00
                        trade.close_date = datetime.utcnow()
                        trade.open_order_id = None
                        trade.is_open = False
                        Trade.session.add(trade)
                        Trade.session.flush()

                        _profit(bot=MagicBot(), update=self.update)
                        self.assertEqual(msg_mock.call_count, 2)
                        self.assertIn('(100.00%)', msg_mock.call_args_list[-1][0][0])

    def test_3_forcesell_handle(self):
        with patch.dict('main._CONF', self.conf):
            with patch('main.get_buy_signal', side_effect=lambda _: True):
                msg_mock = MagicMock()
                with patch.multiple('main.telegram', _CONF=self.conf, init=MagicMock(), send_msg=msg_mock):
                    with patch.multiple('main.exchange',
                                        get_ticker=MagicMock(return_value={
                                            'bid': 0.07256061,
                                            'ask': 0.072661,
                                            'last': 0.07256061
                                        }),
                                        buy=MagicMock(return_value='mocked_order_id')):
                        init(self.conf, 'sqlite://')

                        # Create some test data
                        trade = create_trade(15.0, exchange.Exchange.BITTREX)
                        self.assertTrue(trade)
                        Trade.session.add(trade)
                        Trade.session.flush()

                        self.update.message.text = '/forcesell 1'
                        _forcesell(bot=MagicBot(), update=self.update)

                        self.assertEqual(msg_mock.call_count, 2)
                        self.assertIn('Selling [BTC/ETH]', msg_mock.call_args_list[-1][0][0])
                        self.assertIn('0.072561', msg_mock.call_args_list[-1][0][0])

    def test_4_performance_handle(self):
        with patch.dict('main._CONF', self.conf):
            with patch('main.get_buy_signal', side_effect=lambda _: True):
                msg_mock = MagicMock()
                with patch.multiple('main.telegram', _CONF=self.conf, init=MagicMock(), send_msg=msg_mock):
                    with patch.multiple('main.exchange',
                                        get_ticker=MagicMock(return_value={
                                            'bid': 0.07256061,
                                            'ask': 0.072661,
                                            'last': 0.07256061
                                        }),
                                        buy=MagicMock(return_value='mocked_order_id')):
                        init(self.conf, 'sqlite://')

                        # Create some test data
                        trade = create_trade(15.0, exchange.Exchange.BITTREX)
                        self.assertTrue(trade)
                        trade.close_rate = 0.07256061
                        trade.close_profit = 100.00
                        trade.close_date = datetime.utcnow()
                        trade.open_order_id = None
                        trade.is_open = False
                        Trade.session.add(trade)
                        Trade.session.flush()

                        _performance(bot=MagicBot(), update=self.update)
                        self.assertEqual(msg_mock.call_count, 2)
                        self.assertIn('Performance', msg_mock.call_args_list[-1][0][0])
                        self.assertIn('BTC_ETH	100.00%', msg_mock.call_args_list[-1][0][0])

    def test_5_start_handle(self):
        with patch.dict('main._CONF', self.conf):
            msg_mock = MagicMock()
            with patch.multiple('main.telegram', _CONF=self.conf, init=MagicMock(), send_msg=msg_mock):
                init(self.conf, 'sqlite://')

                update_state(State.PAUSED)
                self.assertEqual(get_state(), State.PAUSED)
                _start(bot=MagicBot(), update=self.update)
                self.assertEqual(get_state(), State.RUNNING)
                self.assertEqual(msg_mock.call_count, 0)

    def test_6_stop_handle(self):
        with patch.dict('main._CONF', self.conf):
            msg_mock = MagicMock()
            with patch.multiple('main.telegram', _CONF=self.conf, init=MagicMock(), send_msg=msg_mock):
                init(self.conf, 'sqlite://')

                update_state(State.RUNNING)
                self.assertEqual(get_state(), State.RUNNING)
                _stop(bot=MagicBot(), update=self.update)
                self.assertEqual(get_state(), State.PAUSED)
                self.assertEqual(msg_mock.call_count, 1)
                self.assertIn('Stopping trader', msg_mock.call_args_list[0][0][0])

    def setUp(self):
        self.update = Update(0)
        self.update.message = Message(0, 0, datetime.utcnow(), Chat(0, 0))

    @classmethod
    def setUpClass(cls):
        validate(cls.conf, CONF_SCHEMA)


if __name__ == '__main__':
    unittest.main()
