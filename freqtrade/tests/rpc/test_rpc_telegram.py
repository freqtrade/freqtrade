# pragma pylint: disable=protected-access, unused-argument, invalid-name
# pragma pylint: disable=too-many-lines, too-many-arguments

"""
Unit test file for rpc/telegram.py
"""

import re
from copy import deepcopy
from datetime import datetime
from random import randint
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from telegram import Update, Message, Chat
from telegram.error import NetworkError

from freqtrade import __version__
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.rpc.telegram import Telegram
from freqtrade.rpc.telegram import authorized_only
from freqtrade.state import State
from freqtrade.tests.conftest import get_patched_freqtradebot, log_has
from freqtrade.tests.test_freqtradebot import patch_get_signal, patch_coinmarketcap


class DummyCls(Telegram):
    """
    Dummy class for testing the Telegram @authorized_only decorator
    """
    def __init__(self, freqtrade) -> None:
        super().__init__(freqtrade)
        self.state = {'called': False}

    @authorized_only
    def dummy_handler(self, *args, **kwargs) -> None:
        """
        Fake method that only change the state of the object
        """
        self.state['called'] = True

    @authorized_only
    def dummy_exception(self, *args, **kwargs) -> None:
        """
        Fake method that throw an exception
        """
        raise Exception('test')


def test__init__(default_conf, mocker) -> None:
    """
    Test __init__() method
    """
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())

    telegram = Telegram(get_patched_freqtradebot(mocker, default_conf))
    assert telegram._updater is None
    assert telegram._config == default_conf


def test_init(default_conf, mocker, caplog) -> None:
    """
    Test _init() method
    """
    start_polling = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock(return_value=start_polling))

    Telegram(get_patched_freqtradebot(mocker, default_conf))
    assert start_polling.call_count == 0

    # number of handles registered
    assert start_polling.dispatcher.add_handler.call_count == 11
    assert start_polling.start_polling.call_count == 1

    message_str = "rpc.telegram is listening for following commands: [['status'], ['profit'], " \
                  "['balance'], ['start'], ['stop'], ['forcesell'], ['performance'], ['daily'], " \
                  "['count'], ['help'], ['version']]"

    assert log_has(message_str, caplog.record_tuples)


def test_init_disabled(default_conf, mocker, caplog) -> None:
    """
    Test _init() method when Telegram is disabled
    """
    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    Telegram(get_patched_freqtradebot(mocker, conf))

    message_str = "rpc.telegram is listening for following commands: [['status'], ['profit'], " \
                  "['balance'], ['start'], ['stop'], ['forcesell'], ['performance'], ['daily'], " \
                  "['count'], ['help'], ['version']]"

    assert not log_has(message_str, caplog.record_tuples)


def test_cleanup(default_conf, mocker) -> None:
    """
    Test cleanup() method
    """
    updater_mock = MagicMock()
    updater_mock.stop = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Updater', updater_mock)

    # not enabled
    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    telegram = Telegram(get_patched_freqtradebot(mocker, conf))
    telegram.cleanup()
    assert telegram._updater is None
    assert updater_mock.call_count == 0
    assert not hasattr(telegram._updater, 'stop')
    assert updater_mock.stop.call_count == 0

    # enabled
    conf['telegram']['enabled'] = True
    telegram = Telegram(get_patched_freqtradebot(mocker, conf))
    telegram.cleanup()
    assert telegram._updater.stop.call_count == 1


def test_is_enabled(default_conf, mocker) -> None:
    """
    Test is_enabled() method
    """
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())

    telegram = Telegram(get_patched_freqtradebot(mocker, default_conf))
    assert telegram.is_enabled()


def test_is_not_enabled(default_conf, mocker) -> None:
    """
    Test is_enabled() method
    """
    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    telegram = Telegram(get_patched_freqtradebot(mocker, conf))

    assert not telegram.is_enabled()


def test_authorized_only(default_conf, mocker, caplog) -> None:
    """
    Test authorized_only() method when we are authorized
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())

    chat = Chat(0, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), chat)

    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    dummy = DummyCls(FreqtradeBot(conf, create_engine('sqlite://')))
    dummy.dummy_handler(bot=MagicMock(), update=update)
    assert dummy.state['called'] is True
    assert log_has(
        'Executing handler: dummy_handler for chat_id: 0',
        caplog.record_tuples
    )
    assert not log_has(
        'Rejected unauthorized message from: 0',
        caplog.record_tuples
    )
    assert not log_has(
        'Exception occurred within Telegram module',
        caplog.record_tuples
    )


def test_authorized_only_unauthorized(default_conf, mocker, caplog) -> None:
    """
    Test authorized_only() method when we are unauthorized
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())

    chat = Chat(0xdeadbeef, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), chat)

    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    dummy = DummyCls(FreqtradeBot(conf, create_engine('sqlite://')))
    dummy.dummy_handler(bot=MagicMock(), update=update)
    assert dummy.state['called'] is False
    assert not log_has(
        'Executing handler: dummy_handler for chat_id: 3735928559',
        caplog.record_tuples
    )
    assert log_has(
        'Rejected unauthorized message from: 3735928559',
        caplog.record_tuples
    )
    assert not log_has(
        'Exception occurred within Telegram module',
        caplog.record_tuples
    )


def test_authorized_only_exception(default_conf, mocker, caplog) -> None:
    """
    Test authorized_only() method when an exception is thrown
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())

    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), Chat(0, 0))

    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    dummy = DummyCls(FreqtradeBot(conf, create_engine('sqlite://')))
    dummy.dummy_exception(bot=MagicMock(), update=update)
    assert dummy.state['called'] is False
    assert not log_has(
        'Executing handler: dummy_handler for chat_id: 0',
        caplog.record_tuples
    )
    assert not log_has(
        'Rejected unauthorized message from: 0',
        caplog.record_tuples
    )
    assert log_has(
        'Exception occurred within Telegram module',
        caplog.record_tuples
    )


def test_status(default_conf, update, mocker, ticker) -> None:
    """
    Test _status() method
    """
    update.message.chat.id = 123
    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    conf['telegram']['chat_id'] = 123

    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_pair_detail_url=MagicMock()
    )
    msg_mock = MagicMock()
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        rpc_trade_status=MagicMock(return_value=(False, [1, 2, 3])),
        _status_table=status_table,
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Create some test data
    for _ in range(3):
        freqtradebot.create_trade()

    telegram._status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 3

    update.message.text = MagicMock()
    update.message.text.replace = MagicMock(return_value='table 2 3')
    telegram._status(bot=MagicMock(), update=update)
    assert status_table.call_count == 1


def test_status_handle(default_conf, update, ticker, mocker) -> None:
    """
    Test _status() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker
    )
    msg_mock = MagicMock()
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _status_table=status_table,
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    freqtradebot.update_state(State.STOPPED)
    telegram._status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'trader is not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.update_state(State.RUNNING)
    telegram._status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.create_trade()
    # Trigger status while we have a fulfilled order for the open trade
    telegram._status(bot=MagicMock(), update=update)

    assert msg_mock.call_count == 1
    assert '[ETH/BTC]' in msg_mock.call_args_list[0][0][0]


def test_status_table_handle(default_conf, update, ticker, mocker) -> None:
    """
    Test _status_table() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value='mocked_order_id')
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    conf = deepcopy(default_conf)
    conf['stake_amount'] = 15.0
    freqtradebot = FreqtradeBot(conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    freqtradebot.update_state(State.STOPPED)
    telegram._status_table(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'trader is not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.update_state(State.RUNNING)
    telegram._status_table(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no active order' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.create_trade()

    telegram._status_table(bot=MagicMock(), update=update)

    text = re.sub('</?pre>', '', msg_mock.call_args_list[-1][0][0])
    line = text.split("\n")
    fields = re.sub('[ ]+', ' ', line[2].strip()).split(' ')

    assert int(fields[0]) == 1
    assert fields[1] == 'ETH/BTC'
    assert msg_mock.call_count == 1


def test_daily_handle(default_conf, update, ticker, limit_buy_order,
                      limit_sell_order, mocker) -> None:
    """
    Test _daily() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch(
        'freqtrade.fiat_convert.CryptoToFiatConverter._find_price',
        return_value=15000.0
    )
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_pair_detail_url=MagicMock()
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()
    trade = Trade.query.first()
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    # Try valid data
    update.message.text = '/daily 2'
    telegram._daily(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'Daily' in msg_mock.call_args_list[0][0][0]
    assert str(datetime.utcnow().date()) in msg_mock.call_args_list[0][0][0]
    assert str('  0.00006217 BTC') in msg_mock.call_args_list[0][0][0]
    assert str('  0.933 USD') in msg_mock.call_args_list[0][0][0]
    assert str('  1 trade') in msg_mock.call_args_list[0][0][0]
    assert str('  0 trade') in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()
    # Add two other trades
    freqtradebot.create_trade()
    freqtradebot.create_trade()

    trades = Trade.query.all()
    for trade in trades:
        trade.update(limit_buy_order)
        trade.update(limit_sell_order)
        trade.close_date = datetime.utcnow()
        trade.is_open = False

    update.message.text = '/daily 1'

    telegram._daily(bot=MagicMock(), update=update)
    assert str('  0.00018651 BTC') in msg_mock.call_args_list[0][0][0]
    assert str('  2.798 USD') in msg_mock.call_args_list[0][0][0]
    assert str('  3 trades') in msg_mock.call_args_list[0][0][0]


def test_daily_wrong_input(default_conf, update, ticker, mocker) -> None:
    """
    Test _daily() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.update_state(State.RUNNING)
    update.message.text = '/daily -2'
    telegram._daily(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.update_state(State.RUNNING)
    update.message.text = '/daily today'
    telegram._daily(bot=MagicMock(), update=update)
    assert str('Daily Profit over the last 7 days') in msg_mock.call_args_list[0][0][0]


def test_profit_handle(default_conf, update, ticker, ticker_sell_up,
                       limit_buy_order, limit_sell_order, mocker) -> None:
    """
    Test _profit() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    telegram._profit(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no closed trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.create_trade()
    trade = Trade.query.first()

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    telegram._profit(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no closed trade' in msg_mock.call_args_list[-1][0][0]
    msg_mock.reset_mock()

    # Update the ticker with a market going up
    mocker.patch('freqtrade.freqtradebot.exchange.get_ticker', ticker_sell_up)
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    telegram._profit(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert '*ROI:* Close trades' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.00006217 BTC (6.20%)`' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.933 USD`' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.00006217 BTC (6.20%)`' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.933 USD`' in msg_mock.call_args_list[-1][0][0]

    assert '*Best Performing:* `ETH/BTC: 6.20%`' in msg_mock.call_args_list[-1][0][0]


def test_telegram_balance_handle(default_conf, update, mocker) -> None:
    """
    Test _balance() method
    """
    mock_balance = [
        {
            'Currency': 'BTC',
            'Balance': 10.0,
            'Available': 12.0,
            'Pending': 0.0,
            'CryptoAddress': 'XXXX',
        },
        {
            'Currency': 'ETH',
            'Balance': 0.0,
            'Available': 0.0,
            'Pending': 0.0,
            'CryptoAddress': 'XXXX',
        },
        {
            'Currency': 'USDT',
            'Balance': 10000.0,
            'Available': 0.0,
            'Pending': 0.0,
            'CryptoAddress': 'XXXX',
        },
        {
            'Currency': 'LTC',
            'Balance': 10.0,
            'Available': 10.0,
            'Pending': 0.0,
            'CryptoAddress': 'XXXX',
        }
    ]

    def mock_ticker(symbol, refresh):
        """
        Mock Bittrex.get_ticker() response
        """
        if symbol == 'USDT_BTC':
            return {
                'bid': 10000.00,
                'ask': 10000.00,
                'last': 10000.00,
            }

        return {
            'bid': 0.1,
            'ask': 0.1,
            'last': 0.1,
        }

    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.exchange.get_balances', return_value=mock_balance)
    mocker.patch('freqtrade.freqtradebot.exchange.get_ticker', side_effect=mock_ticker)

    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    telegram._balance(bot=MagicMock(), update=update)
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert '*Currency*: BTC' in result
    assert '*Currency*: ETH' not in result
    assert '*Currency*: USDT' in result
    assert 'Balance' in result
    assert 'Est. BTC' in result
    assert '*BTC*:  12.00000000' in result


def test_zero_balance_handle(default_conf, update, mocker) -> None:
    """
    Test _balance() method when the Exchange platform returns nothing
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.exchange.get_balances', return_value=[])

    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    telegram._balance(bot=MagicMock(), update=update)
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert '`All balances are zero.`' in result


def test_start_handle(default_conf, update, mocker) -> None:
    """
    Test _start() method
    """
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    freqtradebot.update_state(State.STOPPED)
    assert freqtradebot.get_state() == State.STOPPED
    telegram._start(bot=MagicMock(), update=update)
    assert freqtradebot.get_state() == State.RUNNING
    assert msg_mock.call_count == 0


def test_start_handle_already_running(default_conf, update, mocker) -> None:
    """
    Test _start() method
    """
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    freqtradebot.update_state(State.RUNNING)
    assert freqtradebot.get_state() == State.RUNNING
    telegram._start(bot=MagicMock(), update=update)
    assert freqtradebot.get_state() == State.RUNNING
    assert msg_mock.call_count == 1
    assert 'already running' in msg_mock.call_args_list[0][0][0]


def test_stop_handle(default_conf, update, mocker) -> None:
    """
    Test _stop() method
    """
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    freqtradebot.update_state(State.RUNNING)
    assert freqtradebot.get_state() == State.RUNNING
    telegram._stop(bot=MagicMock(), update=update)
    assert freqtradebot.get_state() == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'Stopping trader' in msg_mock.call_args_list[0][0][0]


def test_stop_handle_already_stopped(default_conf, update, mocker) -> None:
    """
    Test _stop() method
    """
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    freqtradebot.update_state(State.STOPPED)
    assert freqtradebot.get_state() == State.STOPPED
    telegram._stop(bot=MagicMock(), update=update)
    assert freqtradebot.get_state() == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'already stopped' in msg_mock.call_args_list[0][0][0]


def test_forcesell_handle(default_conf, update, ticker, ticker_sell_up, mocker) -> None:
    """
    Test _forcesell() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch('freqtrade.freqtradebot.exchange.get_ticker', ticker_sell_up)

    update.message.text = '/forcesell 1'
    telegram._forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 2
    assert 'Selling' in rpc_mock.call_args_list[-1][0][0]
    assert '[ETH/BTC]' in rpc_mock.call_args_list[-1][0][0]
    assert 'Amount' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001172' in rpc_mock.call_args_list[-1][0][0]
    assert 'profit: 6.11%, 0.00006126' in rpc_mock.call_args_list[-1][0][0]
    assert '0.919 USD' in rpc_mock.call_args_list[-1][0][0]


def test_forcesell_down_handle(default_conf, update, ticker, ticker_sell_down, mocker) -> None:
    """
    Test _forcesell() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_down
    )

    trade = Trade.query.first()
    assert trade

    update.message.text = '/forcesell 1'
    telegram._forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 2
    assert 'Selling' in rpc_mock.call_args_list[-1][0][0]
    assert '[ETH/BTC]' in rpc_mock.call_args_list[-1][0][0]
    assert 'Amount' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001044' in rpc_mock.call_args_list[-1][0][0]
    assert 'loss: -5.48%, -0.00005492' in rpc_mock.call_args_list[-1][0][0]
    assert '-0.824 USD' in rpc_mock.call_args_list[-1][0][0]


def test_forcesell_all_handle(default_conf, update, ticker, mocker) -> None:
    """
    Test _forcesell() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    mocker.patch('freqtrade.exchange.get_pair_detail_url', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Create some test data
    for _ in range(4):
        freqtradebot.create_trade()
    rpc_mock.reset_mock()

    update.message.text = '/forcesell all'
    telegram._forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 4
    for args in rpc_mock.call_args_list:
        assert '0.00001098' in args[0][0]
        assert 'loss: -0.59%, -0.00000591 BTC' in args[0][0]
        assert '-0.089 USD' in args[0][0]


def test_forcesell_handle_invalid(default_conf, update, mocker) -> None:
    """
    Test _forcesell() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.exchange.validate_pairs', MagicMock())

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Trader is not running
    freqtradebot.update_state(State.STOPPED)
    update.message.text = '/forcesell 1'
    telegram._forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]

    # No argument
    msg_mock.reset_mock()
    freqtradebot.update_state(State.RUNNING)
    update.message.text = '/forcesell'
    telegram._forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'Invalid argument' in msg_mock.call_args_list[0][0][0]

    # Invalid argument
    msg_mock.reset_mock()
    freqtradebot.update_state(State.RUNNING)
    update.message.text = '/forcesell 123456'
    telegram._forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'Invalid argument.' in msg_mock.call_args_list[0][0][0]


def test_performance_handle(default_conf, update, ticker, limit_buy_order,
                            limit_sell_order, mocker) -> None:
    """
    Test _performance() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()
    trade = Trade.query.first()
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False
    telegram._performance(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'Performance' in msg_mock.call_args_list[0][0][0]
    assert '<code>ETH/BTC\t6.20% (1)</code>' in msg_mock.call_args_list[0][0][0]


def test_performance_handle_invalid(default_conf, update, mocker) -> None:
    """
    Test _performance() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.exchange.validate_pairs', MagicMock())
    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    # Trader is not running
    freqtradebot.update_state(State.STOPPED)
    telegram._performance(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]


def test_count_handle(default_conf, update, ticker, mocker) -> None:
    """
    Test _count() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value='mocked_order_id')
    )
    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    freqtradebot.update_state(State.STOPPED)
    telegram._count(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.update_state(State.RUNNING)

    # Create some test data
    freqtradebot.create_trade()
    msg_mock.reset_mock()
    telegram._count(bot=MagicMock(), update=update)

    msg = '<pre>  current    max\n---------  -----\n        1      {}</pre>'.format(
        default_conf['max_open_trades']
    )
    assert msg in msg_mock.call_args_list[0][0][0]


def test_help_handle(default_conf, update, mocker) -> None:
    """
    Test _help() method
    """
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    telegram._help(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert '*/help:* `This help message`' in msg_mock.call_args_list[0][0][0]


def test_version_handle(default_conf, update, mocker) -> None:
    """
    Test _version() method
    """
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        send_msg=msg_mock
    )
    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    telegram._version(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert '*Version:* `{}`'.format(__version__) in msg_mock.call_args_list[0][0][0]


def test_send_msg(default_conf, mocker) -> None:
    """
    Test send_msg() method
    """
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    conf = deepcopy(default_conf)
    bot = MagicMock()
    freqtradebot = FreqtradeBot(conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    telegram._config['telegram']['enabled'] = False
    telegram.send_msg('test', bot)
    assert not bot.method_calls
    bot.reset_mock()

    telegram._config['telegram']['enabled'] = True
    telegram.send_msg('test', bot)
    assert len(bot.method_calls) == 1


def test_send_msg_network_error(default_conf, mocker, caplog) -> None:
    """
    Test send_msg() method
    """
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    conf = deepcopy(default_conf)
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError('Oh snap'))
    freqtradebot = FreqtradeBot(conf, create_engine('sqlite://'))
    telegram = Telegram(freqtradebot)

    telegram._config['telegram']['enabled'] = True
    telegram.send_msg('test', bot)

    # Bot should've tried to send it twice
    assert len(bot.method_calls) == 2
    assert log_has(
        'Telegram NetworkError: Oh snap! Trying one more time.',
        caplog.record_tuples
    )
