# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, unused-argument, invalid-name
# pragma pylint: disable=too-many-lines, too-many-arguments

import re
from datetime import datetime
from random import randint
from unittest.mock import MagicMock, PropertyMock

import arrow
import pytest
from telegram import Chat, Message, Update
from telegram.error import NetworkError

from freqtrade import __version__
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.rpc import RPCMessageType
from freqtrade.rpc.telegram import Telegram, authorized_only
from freqtrade.strategy.interface import SellType
from freqtrade.state import State
from freqtrade.tests.conftest import (get_patched_freqtradebot, log_has,
                                      patch_exchange)
from freqtrade.tests.test_freqtradebot import patch_get_signal
from freqtrade.tests.conftest import patch_coinmarketcap


class DummyCls(Telegram):
    """
    Dummy class for testing the Telegram @authorized_only decorator
    """
    def __init__(self, freqtrade) -> None:
        super().__init__(freqtrade)
        self.state = {'called': False}

    def _init(self):
        pass

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
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())

    telegram = Telegram(get_patched_freqtradebot(mocker, default_conf))
    assert telegram._updater is None
    assert telegram._config == default_conf


def test_init(default_conf, mocker, caplog) -> None:
    start_polling = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock(return_value=start_polling))

    Telegram(get_patched_freqtradebot(mocker, default_conf))
    assert start_polling.call_count == 0

    # number of handles registered
    assert start_polling.dispatcher.add_handler.call_count > 0
    assert start_polling.start_polling.call_count == 1

    message_str = "rpc.telegram is listening for following commands: [['status'], ['profit'], " \
                  "['balance'], ['start'], ['stop'], ['forcesell'], ['forcebuy'], " \
                  "['performance'], ['daily'], ['count'], ['reload_conf'], " \
                  "['stopbuy'], ['whitelist'], ['blacklist'], ['help'], ['version']]"

    assert log_has(message_str, caplog.record_tuples)


def test_cleanup(default_conf, mocker) -> None:
    updater_mock = MagicMock()
    updater_mock.stop = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Updater', updater_mock)

    telegram = Telegram(get_patched_freqtradebot(mocker, default_conf))
    telegram.cleanup()
    assert telegram._updater.stop.call_count == 1


def test_authorized_only(default_conf, mocker, caplog) -> None:
    patch_coinmarketcap(mocker)
    patch_exchange(mocker, None)

    chat = Chat(0, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), chat)

    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    patch_get_signal(bot, (True, False))
    dummy = DummyCls(bot)
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
    patch_coinmarketcap(mocker)
    patch_exchange(mocker, None)
    chat = Chat(0xdeadbeef, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), chat)

    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    patch_get_signal(bot, (True, False))
    dummy = DummyCls(bot)
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
    patch_coinmarketcap(mocker)
    patch_exchange(mocker)

    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), Chat(0, 0))

    default_conf['telegram']['enabled'] = False

    bot = FreqtradeBot(default_conf)
    patch_get_signal(bot, (True, False))
    dummy = DummyCls(bot)

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


def test_status(default_conf, update, mocker, fee, ticker, markets) -> None:
    update.message.chat.id = 123
    default_conf['telegram']['enabled'] = False
    default_conf['telegram']['chat_id'] = 123

    patch_coinmarketcap(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(markets)
    )
    msg_mock = MagicMock()
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _rpc_trade_status=MagicMock(return_value=[{
            'trade_id': 1,
            'pair': 'ETH/BTC',
            'date': arrow.utcnow(),
            'open_rate': 1.099e-05,
            'close_rate': None,
            'current_rate': 1.098e-05,
            'amount': 90.99181074,
            'close_profit': None,
            'current_profit': -0.59,
            'stop_loss': 1.099e-05,
            'stop_loss_percentage': '-2%',
            'open_order': '(limit buy rem=0.00000000)'
        }]),
        _status_table=status_table,
        _send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    for _ in range(3):
        freqtradebot.create_trade()

    telegram._status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1

    update.message.text = MagicMock()
    update.message.text.replace = MagicMock(return_value='table 2 3')
    telegram._status(bot=MagicMock(), update=update)
    assert status_table.call_count == 1


def test_status_handle(default_conf, update, ticker, fee, markets, mocker) -> None:
    patch_coinmarketcap(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(markets)
    )
    msg_mock = MagicMock()
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _status_table=status_table,
        _send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.STOPPED
    # Status is also enabled when stopped
    telegram._status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    telegram._status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.create_trade()
    # Trigger status while we have a fulfilled order for the open trade
    telegram._status(bot=MagicMock(), update=update)

    # close_rate should not be included in the message as the trade is not closed
    # and no line should be empty
    lines = msg_mock.call_args_list[0][0][0].split('\n')
    assert '' not in lines
    assert 'Close Rate' not in ''.join(lines)
    assert 'Close Profit' not in ''.join(lines)

    assert msg_mock.call_count == 1
    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]


def test_status_table_handle(default_conf, update, ticker, fee, markets, mocker) -> None:
    patch_coinmarketcap(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': 'mocked_order_id'}),
        get_fee=fee,
        markets=PropertyMock(markets)
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    default_conf['stake_amount'] = 15.0
    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.STOPPED
    # Status table is also enabled when stopped
    telegram._status_table(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no active order' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
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


def test_daily_handle(default_conf, update, ticker, limit_buy_order, fee,
                      limit_sell_order, markets, mocker) -> None:
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.rpc.rpc.CryptoToFiatConverter._find_price',
        return_value=15000.0
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(markets)
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
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
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    update.message.text = '/daily -2'
    telegram._daily(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    update.message.text = '/daily today'
    telegram._daily(bot=MagicMock(), update=update)
    assert str('Daily Profit over the last 7 days') in msg_mock.call_args_list[0][0][0]


def test_profit_handle(default_conf, update, ticker, ticker_sell_up, fee,
                       limit_buy_order, limit_sell_order, markets, mocker) -> None:
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(markets)
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
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
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', ticker_sell_up)
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
    mock_balance = {
        'BTC': {
            'total': 12.0,
            'free': 12.0,
            'used': 0.0
        },
        'ETH': {
            'total': 0.0,
            'free': 0.0,
            'used': 0.0
        },
        'USDT': {
            'total': 10000.0,
            'free': 10000.0,
            'used': 0.0
        },
        'LTC': {
            'total': 10.0,
            'free': 10.0,
            'used': 0.0
        },
        'XRP': {
            'total': 1.0,
            'free': 1.0,
            'used': 0.0
            }
    }

    def mock_ticker(symbol, refresh):
        if symbol == 'BTC/USDT':
            return {
                'bid': 10000.00,
                'ask': 10000.00,
                'last': 10000.00,
            }
        elif symbol == 'XRP/BTC':
            return {
                'bid': 0.00001,
                'ask': 0.00001,
                'last': 0.00001,
            }
        return {
            'bid': 0.1,
            'ask': 0.1,
            'last': 0.1,
        }

    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.exchange.Exchange.get_balances', return_value=mock_balance)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', side_effect=mock_ticker)

    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram = Telegram(freqtradebot)

    telegram._balance(bot=MagicMock(), update=update)
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert '*BTC:*' in result
    assert '*ETH:*' not in result
    assert '*USDT:*' in result
    assert 'Balance:' in result
    assert 'Est. BTC:' in result
    assert 'BTC:  12.00000000' in result
    assert '*XRP:* not showing <1$ amount' in result


def test_balance_handle_empty_response(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_balances', return_value={})

    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram = Telegram(freqtradebot)

    telegram._balance(bot=MagicMock(), update=update)
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert 'all balances are zero' in result


def test_start_handle(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    telegram._start(bot=MagicMock(), update=update)
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1


def test_start_handle_already_running(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    telegram._start(bot=MagicMock(), update=update)
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1
    assert 'already running' in msg_mock.call_args_list[0][0][0]


def test_stop_handle(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    telegram._stop(bot=MagicMock(), update=update)
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'stopping trader' in msg_mock.call_args_list[0][0][0]


def test_stop_handle_already_stopped(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    telegram._stop(bot=MagicMock(), update=update)
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'already stopped' in msg_mock.call_args_list[0][0][0]


def test_stopbuy_handle(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    assert freqtradebot.config['max_open_trades'] != 0
    telegram._stopbuy(bot=MagicMock(), update=update)
    assert freqtradebot.config['max_open_trades'] == 0
    assert msg_mock.call_count == 1
    assert 'No more buy will occur from now. Run /reload_conf to reset.' \
        in msg_mock.call_args_list[0][0][0]


def test_reload_conf_handle(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    telegram._reload_conf(bot=MagicMock(), update=update)
    assert freqtradebot.state == State.RELOAD_CONF
    assert msg_mock.call_count == 1
    assert 'reloading config' in msg_mock.call_args_list[0][0][0]


def test_forcesell_handle(default_conf, update, ticker, fee,
                          ticker_sell_up, markets, mocker) -> None:
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        validate_pairs=MagicMock(return_value={})
    )

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', ticker_sell_up)

    update.message.text = '/forcesell 1'
    telegram._forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'limit': 1.172e-05,
        'amount': 90.99181073703367,
        'open_rate': 1.099e-05,
        'current_rate': 1.172e-05,
        'profit_amount': 6.126e-05,
        'profit_percent': 0.0611052,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value
    } == last_msg


def test_forcesell_down_handle(default_conf, update, ticker, fee,
                               ticker_sell_down, markets, mocker) -> None:
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        validate_pairs=MagicMock(return_value={})
    )

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_down
    )

    trade = Trade.query.first()
    assert trade

    update.message.text = '/forcesell 1'
    telegram._forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 2

    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.044e-05,
        'amount': 90.99181073703367,
        'open_rate': 1.099e-05,
        'current_rate': 1.044e-05,
        'profit_amount': -5.492e-05,
        'profit_percent': -0.05478342,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value
    } == last_msg


def test_forcesell_all_handle(default_conf, update, ticker, fee, markets, mocker) -> None:
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        validate_pairs=MagicMock(return_value={})
    )

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    for _ in range(4):
        freqtradebot.create_trade()
    rpc_mock.reset_mock()

    update.message.text = '/forcesell all'
    telegram._forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 4
    msg = rpc_mock.call_args_list[0][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.098e-05,
        'amount': 90.99181073703367,
        'open_rate': 1.099e-05,
        'current_rate': 1.098e-05,
        'profit_amount': -5.91e-06,
        'profit_percent': -0.00589291,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value
    } == msg


def test_forcesell_handle_invalid(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    patch_exchange(mocker)

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Trader is not running
    freqtradebot.state = State.STOPPED
    update.message.text = '/forcesell 1'
    telegram._forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]

    # No argument
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    update.message.text = '/forcesell'
    telegram._forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'invalid argument' in msg_mock.call_args_list[0][0][0]

    # Invalid argument
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    update.message.text = '/forcesell 123456'
    telegram._forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'invalid argument' in msg_mock.call_args_list[0][0][0]


def test_forcebuy_handle(default_conf, update, markets, mocker) -> None:
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram._send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        markets=PropertyMock(markets),
        validate_pairs=MagicMock(return_value={})
        )
    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.RPC._rpc_forcebuy', fbuy_mock)

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    update.message.text = '/forcebuy ETH/BTC'
    telegram._forcebuy(bot=MagicMock(), update=update)

    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == 'ETH/BTC'
    assert fbuy_mock.call_args_list[0][0][1] is None

    # Reset and retry with specified price
    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.RPC._rpc_forcebuy', fbuy_mock)
    update.message.text = '/forcebuy ETH/BTC 0.055'
    telegram._forcebuy(bot=MagicMock(), update=update)

    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == 'ETH/BTC'
    assert isinstance(fbuy_mock.call_args_list[0][0][1], float)
    assert fbuy_mock.call_args_list[0][0][1] == 0.055


def test_forcebuy_handle_exception(default_conf, update, markets, mocker) -> None:
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram._send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        markets=PropertyMock(markets),
        validate_pairs=MagicMock(return_value={})
    )
    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    update.message.text = '/forcebuy ETH/Nonepair'
    telegram._forcebuy(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 1
    assert rpc_mock.call_args_list[0][0][0] == 'Forcebuy not enabled.'


def test_performance_handle(default_conf, update, ticker, fee,
                            limit_buy_order, limit_sell_order, markets, mocker) -> None:
    patch_coinmarketcap(mocker)
    patch_exchange(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(markets),
        validate_pairs=MagicMock(return_value={})
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
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


def test_count_handle(default_conf, update, ticker, fee, markets, mocker) -> None:
    patch_coinmarketcap(mocker)
    patch_exchange(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': 'mocked_order_id'}),
        markets=PropertyMock(markets)
    )
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.STOPPED
    telegram._count(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING

    # Create some test data
    freqtradebot.create_trade()
    msg_mock.reset_mock()
    telegram._count(bot=MagicMock(), update=update)

    msg = '<pre>  current    max    total stake\n---------  -----  -------------\n' \
          '        1      {}          {}</pre>'\
        .format(
            default_conf['max_open_trades'],
            default_conf['stake_amount']
        )
    assert msg in msg_mock.call_args_list[0][0][0]


def test_whitelist_static(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    telegram = Telegram(freqtradebot)

    telegram._whitelist(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert ('Using whitelist `StaticPairList` with 4 pairs\n`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`'
            in msg_mock.call_args_list[0][0][0])


def test_whitelist_dynamic(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    default_conf['pairlist'] = {'method': 'VolumePairList',
                                'config': {'number_assets': 4}
                                }
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    telegram = Telegram(freqtradebot)

    telegram._whitelist(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert ('Using whitelist `VolumePairList` with 4 pairs\n`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`'
            in msg_mock.call_args_list[0][0][0])


def test_blacklist_static(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    telegram = Telegram(freqtradebot)

    telegram._blacklist(bot=MagicMock(), update=update, args=[])
    assert msg_mock.call_count == 1
    assert ("Blacklist contains 2 pairs\n`DOGE/BTC, HOT/BTC`"
            in msg_mock.call_args_list[0][0][0])

    msg_mock.reset_mock()
    telegram._blacklist(bot=MagicMock(), update=update, args=["ETH/BTC"])
    assert msg_mock.call_count == 1
    assert ("Blacklist contains 3 pairs\n`DOGE/BTC, HOT/BTC, ETH/BTC`"
            in msg_mock.call_args_list[0][0][0])
    assert freqtradebot.pairlists.blacklist == ["DOGE/BTC", "HOT/BTC", "ETH/BTC"]


def test_help_handle(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    telegram = Telegram(freqtradebot)

    telegram._help(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert '*/help:* `This help message`' in msg_mock.call_args_list[0][0][0]


def test_version_handle(default_conf, update, mocker) -> None:
    patch_coinmarketcap(mocker)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    telegram._version(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert '*Version:* `{}`'.format(__version__) in msg_mock.call_args_list[0][0][0]


def test_send_msg_buy_notification(default_conf, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    telegram.send_msg({
        'type': RPCMessageType.BUY_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'limit': 1.099e-05,
        'stake_amount': 0.001,
        'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD'
    })
    assert msg_mock.call_args[0][0] \
        == '*Bittrex:* Buying ETH/BTC\n' \
           'with limit `0.00001099\n' \
           '(0.001000 BTC,0.000 USD)`'


def test_send_msg_sell_notification(default_conf, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    old_convamount = telegram._fiat_converter.convert_amount
    telegram._fiat_converter.convert_amount = lambda a, b, c: -24.812
    telegram.send_msg({
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'gain': 'loss',
        'limit': 3.201e-05,
        'amount': 1333.3333333333335,
        'open_rate': 7.5e-05,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_percent': -0.57405275,
        'stake_currency': 'ETH',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value
    })
    assert msg_mock.call_args[0][0] \
        == ('*Binance:* Selling KEY/ETH\n'
            '*Limit:* `0.00003201`\n'
            '*Amount:* `1333.33333333`\n'
            '*Open Rate:* `0.00007500`\n'
            '*Current Rate:* `0.00003201`\n'
            '*Sell Reason:* `stop_loss`\n'
            '*Profit:* `-57.41%`` (loss: -0.05746268 ETH`` / -24.812 USD)`')

    msg_mock.reset_mock()
    telegram.send_msg({
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'gain': 'loss',
        'limit': 3.201e-05,
        'amount': 1333.3333333333335,
        'open_rate': 7.5e-05,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_percent': -0.57405275,
        'stake_currency': 'ETH',
        'sell_reason': SellType.STOP_LOSS.value
    })
    assert msg_mock.call_args[0][0] \
        == ('*Binance:* Selling KEY/ETH\n'
            '*Limit:* `0.00003201`\n'
            '*Amount:* `1333.33333333`\n'
            '*Open Rate:* `0.00007500`\n'
            '*Current Rate:* `0.00003201`\n'
            '*Sell Reason:* `stop_loss`\n'
            '*Profit:* `-57.41%`')
    # Reset singleton function to avoid random breaks
    telegram._fiat_converter.convert_amount = old_convamount


def test_send_msg_status_notification(default_conf, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    telegram.send_msg({
        'type': RPCMessageType.STATUS_NOTIFICATION,
        'status': 'running'
    })
    assert msg_mock.call_args[0][0] == '*Status:* `running`'


def test_warning_notification(default_conf, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    telegram.send_msg({
        'type': RPCMessageType.WARNING_NOTIFICATION,
        'status': 'message'
    })
    assert msg_mock.call_args[0][0] == '*Warning:* `message`'


def test_custom_notification(default_conf, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    telegram.send_msg({
        'type': RPCMessageType.CUSTOM_NOTIFICATION,
        'status': '*Custom:* `Hello World`'
    })
    assert msg_mock.call_args[0][0] == '*Custom:* `Hello World`'


def test_send_msg_unknown_type(default_conf, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    with pytest.raises(NotImplementedError, match=r'Unknown message type: None'):
        telegram.send_msg({
            'type': None,
        })


def test_send_msg_buy_notification_no_fiat(default_conf, mocker) -> None:
    del default_conf['fiat_display_currency']
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    telegram.send_msg({
        'type': RPCMessageType.BUY_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'limit': 1.099e-05,
        'stake_amount': 0.001,
        'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC',
        'fiat_currency': None
    })
    assert msg_mock.call_args[0][0] \
        == '*Bittrex:* Buying ETH/BTC\n' \
           'with limit `0.00001099\n' \
           '(0.001000 BTC)`'


def test_send_msg_sell_notification_no_fiat(default_conf, mocker) -> None:
    del default_conf['fiat_display_currency']
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    telegram.send_msg({
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'gain': 'loss',
        'limit': 3.201e-05,
        'amount': 1333.3333333333335,
        'open_rate': 7.5e-05,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_percent': -0.57405275,
        'stake_currency': 'ETH',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value
    })
    assert msg_mock.call_args[0][0] \
        == '*Binance:* Selling KEY/ETH\n' \
           '*Limit:* `0.00003201`\n' \
           '*Amount:* `1333.33333333`\n' \
           '*Open Rate:* `0.00007500`\n' \
           '*Current Rate:* `0.00003201`\n' \
           '*Sell Reason:* `stop_loss`\n' \
           '*Profit:* `-57.41%`'


def test__send_msg(default_conf, mocker) -> None:
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    telegram._config['telegram']['enabled'] = True
    telegram._send_msg('test', bot)
    assert len(bot.method_calls) == 1


def test__send_msg_network_error(default_conf, mocker, caplog) -> None:
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError('Oh snap'))
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    telegram._config['telegram']['enabled'] = True
    telegram._send_msg('test', bot)

    # Bot should've tried to send it twice
    assert len(bot.method_calls) == 2
    assert log_has(
        'Telegram NetworkError: Oh snap! Trying one more time.',
        caplog.record_tuples
    )
