# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, unused-argument, invalid-name
# pragma pylint: disable=too-many-lines, too-many-arguments

import re
from datetime import datetime
from random import choice, randint
from string import ascii_uppercase
from unittest.mock import MagicMock, PropertyMock

import arrow
import pytest
from telegram import Chat, Message, Update
from telegram.error import NetworkError

from freqtrade import __version__
from freqtrade.edge import PairInfo
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.rpc import RPCMessageType
from freqtrade.rpc.telegram import Telegram, authorized_only
from freqtrade.state import State
from freqtrade.strategy.interface import SellType
from tests.conftest import (get_patched_freqtradebot, log_has, patch_exchange,
                            patch_get_signal, patch_whitelist)


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
                  "['performance'], ['daily'], ['count'], ['reload_conf'], ['show_config'], " \
                  "['stopbuy'], ['whitelist'], ['blacklist'], ['edge'], ['help'], ['version']]"

    assert log_has(message_str, caplog)


def test_cleanup(default_conf, mocker) -> None:
    updater_mock = MagicMock()
    updater_mock.stop = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Updater', updater_mock)

    telegram = Telegram(get_patched_freqtradebot(mocker, default_conf))
    telegram.cleanup()
    assert telegram._updater.stop.call_count == 1


def test_authorized_only(default_conf, mocker, caplog) -> None:
    patch_exchange(mocker)

    chat = Chat(0, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), chat)

    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    patch_get_signal(bot, (True, False))
    dummy = DummyCls(bot)
    dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state['called'] is True
    assert log_has('Executing handler: dummy_handler for chat_id: 0', caplog)
    assert not log_has('Rejected unauthorized message from: 0', caplog)
    assert not log_has('Exception occurred within Telegram module', caplog)


def test_authorized_only_unauthorized(default_conf, mocker, caplog) -> None:
    patch_exchange(mocker)
    chat = Chat(0xdeadbeef, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), chat)

    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    patch_get_signal(bot, (True, False))
    dummy = DummyCls(bot)
    dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 3735928559', caplog)
    assert log_has('Rejected unauthorized message from: 3735928559', caplog)
    assert not log_has('Exception occurred within Telegram module', caplog)


def test_authorized_only_exception(default_conf, mocker, caplog) -> None:
    patch_exchange(mocker)

    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), Chat(0, 0))

    default_conf['telegram']['enabled'] = False

    bot = FreqtradeBot(default_conf)
    patch_get_signal(bot, (True, False))
    dummy = DummyCls(bot)

    dummy.dummy_exception(update=update, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 0', caplog)
    assert not log_has('Rejected unauthorized message from: 0', caplog)
    assert log_has('Exception occurred within Telegram module', caplog)


def test_status(default_conf, update, mocker, fee, ticker,) -> None:
    update.message.chat.id = 123
    default_conf['telegram']['enabled'] = False
    default_conf['telegram']['chat_id'] = 123

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
    )
    msg_mock = MagicMock()
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _rpc_trade_status=MagicMock(return_value=[{
            'trade_id': 1,
            'pair': 'ETH/BTC',
            'base_currency': 'BTC',
            'open_date': arrow.utcnow(),
            'open_date_hum': arrow.utcnow().humanize,
            'close_date': None,
            'close_date_hum': None,
            'open_rate': 1.099e-05,
            'close_rate': None,
            'current_rate': 1.098e-05,
            'amount': 90.99181074,
            'stake_amount': 90.99181074,
            'close_profit': None,
            'current_profit': -0.59,
            'initial_stop_loss': 1.098e-05,
            'stop_loss': 1.099e-05,
            'initial_stop_loss_pct': -0.05,
            'stop_loss_pct': -0.01,
            'open_order': '(limit buy rem=0.00000000)'
        }]),
        _status_table=status_table,
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    for _ in range(3):
        freqtradebot.create_trades()

    telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1

    context = MagicMock()
    # /status table 2 3
    context.args = ["table", "2", "3"]
    telegram._status(update=update, context=context)
    assert status_table.call_count == 1


def test_status_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
    )
    msg_mock = MagicMock()
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _status_table=status_table,
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    patch_get_signal(freqtradebot, (True, False))

    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.STOPPED
    # Status is also enabled when stopped
    telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.create_trades()
    # Trigger status while we have a fulfilled order for the open trade
    telegram._status(update=update, context=MagicMock())

    # close_rate should not be included in the message as the trade is not closed
    # and no line should be empty
    lines = msg_mock.call_args_list[0][0][0].split('\n')
    assert '' not in lines
    assert 'Close Rate' not in ''.join(lines)
    assert 'Close Profit' not in ''.join(lines)

    assert msg_mock.call_count == 1
    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]


def test_status_table_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': 'mocked_order_id'}),
        get_fee=fee,
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    default_conf['stake_amount'] = 15.0
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.STOPPED
    # Status table is also enabled when stopped
    telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active order' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active order' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.create_trades()

    telegram._status_table(update=update, context=MagicMock())

    text = re.sub('</?pre>', '', msg_mock.call_args_list[-1][0][0])
    line = text.split("\n")
    fields = re.sub('[ ]+', ' ', line[2].strip()).split(' ')

    assert int(fields[0]) == 1
    assert fields[1] == 'ETH/BTC'
    assert msg_mock.call_count == 1


def test_daily_handle(default_conf, update, ticker, limit_buy_order, fee,
                      limit_sell_order, mocker) -> None:
    default_conf['max_open_trades'] = 1
    mocker.patch(
        'freqtrade.rpc.rpc.CryptoToFiatConverter._find_price',
        return_value=15000.0
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trades()
    trade = Trade.query.first()
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    # Try valid data
    # /daily 2
    context = MagicMock()
    context.args = ["2"]
    telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Daily' in msg_mock.call_args_list[0][0][0]
    assert str(datetime.utcnow().date()) in msg_mock.call_args_list[0][0][0]
    assert str('  0.00006217 BTC') in msg_mock.call_args_list[0][0][0]
    assert str('  0.933 USD') in msg_mock.call_args_list[0][0][0]
    assert str('  1 trade') in msg_mock.call_args_list[0][0][0]
    assert str('  0 trade') in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()
    freqtradebot.config['max_open_trades'] = 2
    # Add two other trades
    freqtradebot.create_trades()

    trades = Trade.query.all()
    for trade in trades:
        trade.update(limit_buy_order)
        trade.update(limit_sell_order)
        trade.close_date = datetime.utcnow()
        trade.is_open = False

    # /daily 1
    context = MagicMock()
    context.args = ["1"]
    telegram._daily(update=update, context=context)
    assert str('  0.00018651 BTC') in msg_mock.call_args_list[0][0][0]
    assert str('  2.798 USD') in msg_mock.call_args_list[0][0][0]
    assert str('  3 trades') in msg_mock.call_args_list[0][0][0]


def test_daily_wrong_input(default_conf, update, ticker, mocker) -> None:
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

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /daily -2
    context = MagicMock()
    context.args = ["-2"]
    telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /daily today
    context = MagicMock()
    context.args = ["today"]
    telegram._daily(update=update, context=context)
    assert str('Daily Profit over the last 7 days') in msg_mock.call_args_list[0][0][0]


def test_profit_handle(default_conf, update, ticker, ticker_sell_up, fee,
                       limit_buy_order, limit_sell_order, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
    )
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no closed trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.create_trades()
    trade = Trade.query.first()

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no closed trade' in msg_mock.call_args_list[-1][0][0]
    msg_mock.reset_mock()

    # Update the ticker with a market going up
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', ticker_sell_up)
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*ROI:* Close trades' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.00006217 BTC (6.20%)`' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.933 USD`' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.00006217 BTC (6.20%)`' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.933 USD`' in msg_mock.call_args_list[-1][0][0]

    assert '*Best Performing:* `ETH/BTC: 6.20%`' in msg_mock.call_args_list[-1][0][0]


def test_telegram_balance_handle(default_conf, update, mocker, rpc_balance) -> None:

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

    mocker.patch('freqtrade.exchange.Exchange.get_balances', return_value=rpc_balance)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', side_effect=mock_ticker)
    mocker.patch('freqtrade.exchange.Exchange.get_valid_pair_combination',
                 side_effect=lambda a, b: f"{a}/{b}")

    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram = Telegram(freqtradebot)

    telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert '*BTC:*' in result
    assert '*ETH:*' not in result
    assert '*USDT:*' in result
    assert '*EUR:*' in result
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

    freqtradebot.config['dry_run'] = False
    telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert 'All balances are zero.' in result


def test_balance_handle_empty_response_dry(default_conf, update, mocker) -> None:
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

    telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert "Running in Dry Run, balances are not available." in result


def test_balance_handle_too_large_response(default_conf, update, mocker) -> None:
    balances = []
    for i in range(100):
        curr = choice(ascii_uppercase) + choice(ascii_uppercase) + choice(ascii_uppercase)
        balances.append({
            'currency': curr,
            'free': 1.0,
            'used': 0.5,
            'balance': i,
            'est_btc': 1
        })
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_balance', return_value={
        'currencies': balances,
        'total': 100.0,
        'symbol': 100.0,
        'value': 1000.0,
    })

    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram = Telegram(freqtradebot)

    telegram._balance(update=update, context=MagicMock())
    assert msg_mock.call_count > 1
    # Test if wrap happens around 4000 -
    # and each single currency-output is around 120 characters long so we need
    # an offset to avoid random test failures
    assert len(msg_mock.call_args_list[0][0][0]) < 4096
    assert len(msg_mock.call_args_list[0][0][0]) > (4096 - 120)


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
    telegram._start(update=update, context=MagicMock())
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
    telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1
    assert 'already running' in msg_mock.call_args_list[0][0][0]


def test_stop_handle(default_conf, update, mocker) -> None:
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
    telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'stopping trader' in msg_mock.call_args_list[0][0][0]


def test_stop_handle_already_stopped(default_conf, update, mocker) -> None:
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
    telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'already stopped' in msg_mock.call_args_list[0][0][0]


def test_stopbuy_handle(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    assert freqtradebot.config['max_open_trades'] != 0
    telegram._stopbuy(update=update, context=MagicMock())
    assert freqtradebot.config['max_open_trades'] == 0
    assert msg_mock.call_count == 1
    assert 'No more buy will occur from now. Run /reload_conf to reset.' \
        in msg_mock.call_args_list[0][0][0]


def test_reload_conf_handle(default_conf, update, mocker) -> None:
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
    telegram._reload_conf(update=update, context=MagicMock())
    assert freqtradebot.state == State.RELOAD_CONF
    assert msg_mock.call_count == 1
    assert 'reloading config' in msg_mock.call_args_list[0][0][0]


def test_forcesell_handle(default_conf, update, ticker, fee,
                          ticker_sell_up, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trades()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', ticker_sell_up)

    # /forcesell 1
    context = MagicMock()
    context.args = ["1"]
    telegram._forcesell(update=update, context=context)

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'limit': 1.172e-05,
        'amount': 90.99181073703367,
        'order_type': 'limit',
        'open_rate': 1.099e-05,
        'current_rate': 1.172e-05,
        'profit_amount': 6.126e-05,
        'profit_percent': 0.0611052,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value
    } == last_msg


def test_forcesell_down_handle(default_conf, update, ticker, fee,
                               ticker_sell_down, mocker) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trades()

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_down
    )

    trade = Trade.query.first()
    assert trade

    # /forcesell 1
    context = MagicMock()
    context.args = ["1"]
    telegram._forcesell(update=update, context=context)

    assert rpc_mock.call_count == 2

    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.044e-05,
        'amount': 90.99181073703367,
        'order_type': 'limit',
        'open_rate': 1.099e-05,
        'current_rate': 1.044e-05,
        'profit_amount': -5.492e-05,
        'profit_percent': -0.05478342,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value
    } == last_msg


def test_forcesell_all_handle(default_conf, update, ticker, fee, mocker) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
    )
    default_conf['max_open_trades'] = 4
    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trades()
    rpc_mock.reset_mock()

    # /forcesell all
    context = MagicMock()
    context.args = ["all"]
    telegram._forcesell(update=update, context=context)

    assert rpc_mock.call_count == 4
    msg = rpc_mock.call_args_list[0][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.098e-05,
        'amount': 90.99181073703367,
        'order_type': 'limit',
        'open_rate': 1.099e-05,
        'current_rate': 1.098e-05,
        'profit_amount': -5.91e-06,
        'profit_percent': -0.00589291,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value
    } == msg


def test_forcesell_handle_invalid(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Trader is not running
    freqtradebot.state = State.STOPPED
    # /forcesell 1
    context = MagicMock()
    context.args = ["1"]
    telegram._forcesell(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]

    # No argument
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    context = MagicMock()
    context.args = []
    telegram._forcesell(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'invalid argument' in msg_mock.call_args_list[0][0][0]

    # Invalid argument
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /forcesell 123456
    context = MagicMock()
    context.args = ["123456"]
    telegram._forcesell(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'invalid argument' in msg_mock.call_args_list[0][0][0]


def test_forcebuy_handle(default_conf, update, markets, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram._send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(markets),
        )
    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.RPC._rpc_forcebuy', fbuy_mock)

    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # /forcebuy ETH/BTC
    context = MagicMock()
    context.args = ["ETH/BTC"]
    telegram._forcebuy(update=update, context=context)

    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == 'ETH/BTC'
    assert fbuy_mock.call_args_list[0][0][1] is None

    # Reset and retry with specified price
    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.RPC._rpc_forcebuy', fbuy_mock)
    # /forcebuy ETH/BTC 0.055
    context = MagicMock()
    context.args = ["ETH/BTC", "0.055"]
    telegram._forcebuy(update=update, context=context)

    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == 'ETH/BTC'
    assert isinstance(fbuy_mock.call_args_list[0][0][1], float)
    assert fbuy_mock.call_args_list[0][0][1] == 0.055


def test_forcebuy_handle_exception(default_conf, update, markets, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    rpc_mock = mocker.patch('freqtrade.rpc.telegram.Telegram._send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(markets),
    )
    freqtradebot = FreqtradeBot(default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    update.message.text = '/forcebuy ETH/Nonepair'
    telegram._forcebuy(update=update, context=MagicMock())

    assert rpc_mock.call_count == 1
    assert rpc_mock.call_args_list[0][0][0] == 'Forcebuy not enabled.'


def test_performance_handle(default_conf, update, ticker, fee,
                            limit_buy_order, limit_sell_order, mocker) -> None:
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
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    # Create some test data
    freqtradebot.create_trades()
    trade = Trade.query.first()
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False
    telegram._performance(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Performance' in msg_mock.call_args_list[0][0][0]
    assert '<code>ETH/BTC\t6.20% (1)</code>' in msg_mock.call_args_list[0][0][0]


def test_count_handle(default_conf, update, ticker, fee, mocker) -> None:
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
        get_fee=fee,
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    telegram = Telegram(freqtradebot)

    freqtradebot.state = State.STOPPED
    telegram._count(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING

    # Create some test data
    freqtradebot.create_trades()
    msg_mock.reset_mock()
    telegram._count(update=update, context=MagicMock())

    msg = '<pre>  current    max    total stake\n---------  -----  -------------\n' \
          '        1      {}          {}</pre>'\
        .format(
            default_conf['max_open_trades'],
            default_conf['stake_amount']
        )
    assert msg in msg_mock.call_args_list[0][0][0]


def test_whitelist_static(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    telegram = Telegram(freqtradebot)

    telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert ('Using whitelist `StaticPairList` with 4 pairs\n`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`'
            in msg_mock.call_args_list[0][0][0])


def test_whitelist_dynamic(default_conf, update, mocker) -> None:
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

    telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert ('Using whitelist `VolumePairList` with 4 pairs\n`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`'
            in msg_mock.call_args_list[0][0][0])


def test_blacklist_static(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    telegram = Telegram(freqtradebot)

    telegram._blacklist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert ("Blacklist contains 2 pairs\n`DOGE/BTC, HOT/BTC`"
            in msg_mock.call_args_list[0][0][0])

    msg_mock.reset_mock()

    # /blacklist ETH/BTC
    context = MagicMock()
    context.args = ["ETH/BTC"]
    telegram._blacklist(update=update, context=context)
    assert msg_mock.call_count == 1
    assert ("Blacklist contains 3 pairs\n`DOGE/BTC, HOT/BTC, ETH/BTC`"
            in msg_mock.call_args_list[0][0][0])
    assert freqtradebot.pairlists.blacklist == ["DOGE/BTC", "HOT/BTC", "ETH/BTC"]


def test_edge_disabled(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    telegram = Telegram(freqtradebot)

    telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "Edge is not enabled." in msg_mock.call_args_list[0][0][0]


def test_edge_enabled(edge_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'E/F': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
        }
    ))
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )

    freqtradebot = get_patched_freqtradebot(mocker, edge_conf)

    telegram = Telegram(freqtradebot)

    telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '<b>Edge only validated following pairs:</b>\n<pre>' in msg_mock.call_args_list[0][0][0]
    assert 'Pair      Winrate    Expectancy    Stoploss' in msg_mock.call_args_list[0][0][0]


def test_help_handle(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    telegram = Telegram(freqtradebot)

    telegram._help(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*/help:* `This help message`' in msg_mock.call_args_list[0][0][0]


def test_version_handle(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Version:* `{}`'.format(__version__) in msg_mock.call_args_list[0][0][0]


def test_show_config_handle(default_conf, update, mocker) -> None:
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
        _send_msg=msg_mock
    )
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)

    telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Mode:* `{}`'.format('Dry-run') in msg_mock.call_args_list[0][0][0]
    assert '*Exchange:* `bittrex`' in msg_mock.call_args_list[0][0][0]
    assert '*Strategy:* `DefaultStrategy`' in msg_mock.call_args_list[0][0][0]


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
        'order_type': 'limit',
        'stake_amount': 0.001,
        'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD'
    })
    assert msg_mock.call_args[0][0] \
        == '*Bittrex:* Buying ETH/BTC\n' \
           'at rate `0.00001099\n' \
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
        'order_type': 'market',
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
            '*Rate:* `0.00003201`\n'
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
        'order_type': 'market',
        'open_rate': 7.5e-05,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_percent': -0.57405275,
        'stake_currency': 'ETH',
        'sell_reason': SellType.STOP_LOSS.value
    })
    assert msg_mock.call_args[0][0] \
        == ('*Binance:* Selling KEY/ETH\n'
            '*Rate:* `0.00003201`\n'
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
        'order_type': 'limit',
        'stake_amount': 0.001,
        'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC',
        'fiat_currency': None
    })
    assert msg_mock.call_args[0][0] \
        == '*Bittrex:* Buying ETH/BTC\n' \
           'at rate `0.00001099\n' \
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
        'order_type': 'limit',
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
           '*Rate:* `0.00003201`\n' \
           '*Amount:* `1333.33333333`\n' \
           '*Open Rate:* `0.00007500`\n' \
           '*Current Rate:* `0.00003201`\n' \
           '*Sell Reason:* `stop_loss`\n' \
           '*Profit:* `-57.41%`'


def test__send_msg(default_conf, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    telegram._updater = MagicMock()
    telegram._updater.bot = bot

    telegram._config['telegram']['enabled'] = True
    telegram._send_msg('test')
    assert len(bot.method_calls) == 1


def test__send_msg_network_error(default_conf, mocker, caplog) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError('Oh snap'))
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    telegram = Telegram(freqtradebot)
    telegram._updater = MagicMock()
    telegram._updater.bot = bot

    telegram._config['telegram']['enabled'] = True
    telegram._send_msg('test')

    # Bot should've tried to send it twice
    assert len(bot.method_calls) == 2
    assert log_has('Telegram NetworkError: Oh snap! Trying one more time.', caplog)
