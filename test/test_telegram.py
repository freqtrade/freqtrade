import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

import os
from jsonschema import validate
from telegram import Bot, Update, Message, Chat

import exchange
from main import init, create_trade
from misc import conf_schema
from persistence import Trade
from rpc.telegram import _status, _profit


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
        with patch.dict('main._conf', self.conf):
            with patch('main.get_buy_signal', side_effect=lambda _: True):
                msg_mock = MagicMock()
                with patch.multiple('main.telegram', _conf=self.conf, init=MagicMock(), send_msg=msg_mock):
                    with patch.multiple('main.exchange',
                                        get_ticker=MagicMock(return_value={
                                            'bid': 0.07256061,
                                            'ask': 0.072661,
                                            'last': 0.07256061
                                        }),
                                        buy=MagicMock(return_value='mocked_order_id')):
                        init(self.conf)

                        # Create some test data
                        trade = create_trade(15.0, exchange.Exchange.BITTREX)
                        Trade.session.add(trade)
                        Trade.session.flush()

                        _status(bot=MagicBot(), update=self.update)
                        self.assertEqual(msg_mock.call_count, 2)
                        self.assertIn('[BTC_ETH]', msg_mock.call_args_list[-1][0][0])

    def test_2_profit_handle(self):
        with patch.dict('main._conf', self.conf):
            with patch('main.get_buy_signal', side_effect=lambda _: True):
                msg_mock = MagicMock()
                with patch.multiple('main.telegram', _conf=self.conf, init=MagicMock(), send_msg=msg_mock):
                    with patch.multiple('main.exchange',
                                        get_ticker=MagicMock(return_value={
                                            'bid': 0.07256061,
                                            'ask': 0.072661,
                                            'last': 0.07256061
                                        }),
                                        buy=MagicMock(return_value='mocked_order_id')):
                        init(self.conf)

                        # Create some test data
                        trade = create_trade(15.0, exchange.Exchange.BITTREX)
                        trade.close_rate = 0.07256061
                        trade.close_profit = 137.4872490056564
                        trade.close_date = datetime.utcnow()
                        trade.open_order_id = None
                        trade.is_open = False
                        Trade.session.add(trade)
                        Trade.session.flush()

                        _profit(bot=MagicBot(), update=self.update)
                        self.assertEqual(msg_mock.call_count, 2)
                        self.assertIn('(137.49%)', msg_mock.call_args_list[-1][0][0])

    def setUp(self):
        try:
            os.remove('./tradesv2.dry_run.sqlite')
        except FileNotFoundError:
            pass
        self.update = Update(0)
        self.update.message = Message(0, 0, MagicMock(), Chat(0, 0))

    @classmethod
    def setUpClass(cls):
        validate(cls.conf, conf_schema)


if __name__ == '__main__':
    unittest.main()
