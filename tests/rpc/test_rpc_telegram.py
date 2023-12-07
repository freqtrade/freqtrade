# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, unused-argument, invalid-name
# pragma pylint: disable=too-many-lines, too-many-arguments

import asyncio
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from functools import reduce
from random import choice, randint
from string import ascii_uppercase
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
import time_machine
from pandas import DataFrame
from sqlalchemy import select
from telegram import Chat, Message, ReplyKeyboardMarkup, Update
from telegram.error import BadRequest, NetworkError, TelegramError

from freqtrade import __version__
from freqtrade.constants import CANCEL_REASON
from freqtrade.edge import PairInfo
from freqtrade.enums import (ExitType, MarketDirection, RPCMessageType, RunMode, SignalDirection,
                             State)
from freqtrade.exceptions import OperationalException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.loggers import setup_logging
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc import RPC
from freqtrade.rpc.rpc import RPCException
from freqtrade.rpc.telegram import Telegram, authorized_only
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import (CURRENT_TEST_STRATEGY, EXMS, create_mock_trades,
                            create_mock_trades_usdt, get_patched_freqtradebot, log_has, log_has_re,
                            patch_exchange, patch_get_signal, patch_whitelist)


@pytest.fixture(autouse=True)
def mock_exchange_loop(mocker):
    mocker.patch('freqtrade.exchange.exchange.Exchange._init_async_loop')


@pytest.fixture
def default_conf(default_conf) -> dict:
    # Telegram is enabled by default
    default_conf['telegram']['enabled'] = True
    return default_conf


@pytest.fixture
def update():
    message = Message(0, datetime.now(timezone.utc), Chat(0, 0))
    _update = Update(0, message=message)

    return _update


def patch_eventloop_threading(telegrambot):
    is_init = False

    def thread_fuck():
        nonlocal is_init
        telegrambot._loop = asyncio.new_event_loop()
        is_init = True
        telegrambot._loop.run_forever()
    x = threading.Thread(target=thread_fuck, daemon=True)
    x.start()
    while not is_init:
        pass


class DummyCls(Telegram):
    """
    Dummy class for testing the Telegram @authorized_only decorator
    """

    def __init__(self, rpc: RPC, config) -> None:
        super().__init__(rpc, config)
        self.state = {'called': False}

    def _init(self):
        pass

    @authorized_only
    async def dummy_handler(self, *args, **kwargs) -> None:
        """
        Fake method that only change the state of the object
        """
        self.state['called'] = True

    @authorized_only
    async def dummy_exception(self, *args, **kwargs) -> None:
        """
        Fake method that throw an exception
        """
        raise Exception('test')


def get_telegram_testobject(mocker, default_conf, mock=True, ftbot=None):
    msg_mock = AsyncMock()
    if mock:
        mocker.patch.multiple(
            'freqtrade.rpc.telegram.Telegram',
            _init=MagicMock(),
            _send_msg=msg_mock,
            _start_thread=MagicMock(),
        )
    if not ftbot:
        ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    telegram = Telegram(rpc, default_conf)
    telegram._loop = MagicMock()
    patch_eventloop_threading(telegram)

    return telegram, ftbot, msg_mock


def test_telegram__init__(default_conf, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    assert telegram._config == default_conf


def test_telegram_init(default_conf, mocker, caplog) -> None:
    app_mock = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Telegram._start_thread', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init_telegram_app', return_value=app_mock)
    mocker.patch('freqtrade.rpc.telegram.Telegram._startup_telegram', AsyncMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._init()
    assert app_mock.call_count == 0

    # number of handles registered
    assert app_mock.add_handler.call_count > 0
    # assert start_polling.start_polling.call_count == 1

    message_str = ("rpc.telegram is listening for following commands: [['status'], ['profit'], "
                   "['balance'], ['start'], ['stop'], "
                   "['forceexit', 'forcesell', 'fx'], ['forcebuy', 'forcelong'], ['forceshort'], "
                   "['reload_trade'], ['trades'], ['delete'], ['cancel_open_order', 'coo'], "
                   "['performance'], ['buys', 'entries'], ['exits', 'sells'], ['mix_tags'], "
                   "['stats'], ['daily'], ['weekly'], ['monthly'], "
                   "['count'], ['locks'], ['delete_locks', 'unlock'], "
                   "['reload_conf', 'reload_config'], ['show_conf', 'show_config'], "
                   "['stopbuy', 'stopentry'], ['whitelist'], ['blacklist'], "
                   "['bl_delete', 'blacklist_delete'], "
                   "['logs'], ['edge'], ['health'], ['help'], ['version'], ['marketdir'], "
                   "['order']]")

    assert log_has(message_str, caplog)


async def test_telegram_startup(default_conf, mocker) -> None:
    app_mock = MagicMock()
    app_mock.initialize = AsyncMock()
    app_mock.start = AsyncMock()
    app_mock.updater.start_polling = AsyncMock()
    app_mock.updater.running = False
    sleep_mock = mocker.patch('freqtrade.rpc.telegram.asyncio.sleep', AsyncMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    await telegram._startup_telegram()
    assert app_mock.initialize.call_count == 1
    assert app_mock.start.call_count == 1
    assert app_mock.updater.start_polling.call_count == 1
    assert sleep_mock.call_count == 1


async def test_telegram_cleanup(default_conf, mocker, ) -> None:
    app_mock = MagicMock()
    app_mock.stop = AsyncMock()
    app_mock.initialize = AsyncMock()

    updater_mock = MagicMock()
    updater_mock.stop = AsyncMock()
    app_mock.updater = updater_mock
    # mocker.patch('freqtrade.rpc.telegram.Application', app_mock)

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    telegram._loop = asyncio.get_running_loop()
    telegram._thread = MagicMock()
    telegram.cleanup()
    await asyncio.sleep(0.1)
    assert app_mock.stop.call_count == 1
    assert telegram._thread.join.call_count == 1


async def test_authorized_only(default_conf, mocker, caplog, update) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    patch_get_signal(bot)
    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state['called'] is True
    assert log_has('Executing handler: dummy_handler for chat_id: 0', caplog)
    assert not log_has('Rejected unauthorized message from: 0', caplog)
    assert not log_has('Exception occurred within Telegram module', caplog)


async def test_authorized_only_unauthorized(default_conf, mocker, caplog) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    chat = Chat(0xdeadbeef, 0)
    message = Message(randint(1, 100), datetime.now(timezone.utc), chat)
    update = Update(randint(1, 100), message=message)

    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    patch_get_signal(bot)
    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 3735928559', caplog)
    assert log_has('Rejected unauthorized message from: 3735928559', caplog)
    assert not log_has('Exception occurred within Telegram module', caplog)


async def test_authorized_only_exception(default_conf, mocker, caplog, update) -> None:
    patch_exchange(mocker)

    default_conf['telegram']['enabled'] = False

    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)
    patch_get_signal(bot)

    await dummy.dummy_exception(update=update, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 0', caplog)
    assert not log_has('Rejected unauthorized message from: 0', caplog)
    assert log_has('Exception occurred within Telegram module', caplog)


async def test_telegram_status(default_conf, update, mocker) -> None:
    default_conf['telegram']['enabled'] = False

    status_table = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Telegram._status_table', status_table)

    mocker.patch.multiple(
        'freqtrade.rpc.rpc.RPC',
        _rpc_trade_status=MagicMock(return_value=[{
            'trade_id': 1,
            'pair': 'ETH/BTC',
            'base_currency': 'ETH',
            'quote_currency': 'BTC',
            'open_date': dt_now(),
            'close_date': None,
            'open_rate': 1.099e-05,
            'close_rate': None,
            'current_rate': 1.098e-05,
            'amount': 90.99181074,
            'stake_amount': 90.99181074,
            'max_stake_amount': 90.99181074,
            'buy_tag': None,
            'enter_tag': None,
            'close_profit_ratio': None,
            'profit': -0.0059,
            'profit_ratio': -0.0059,
            'profit_abs': -0.225,
            'realized_profit': 0.0,
            'total_profit_abs': -0.225,
            'initial_stop_loss_abs': 1.098e-05,
            'stop_loss_abs': 1.099e-05,
            'exit_order_status': None,
            'initial_stop_loss_ratio': -0.0005,
            'stoploss_current_dist': 1e-08,
            'stoploss_current_dist_ratio': -0.0002,
            'stop_loss_ratio': -0.0001,
            'open_order': '(limit buy rem=0.00000000)',
            'is_open': True,
            'is_short': False,
            'filled_entry_orders': [],
            'orders': []
        }]),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1

    context = MagicMock()
    # /status table
    context.args = ["table"]
    await telegram._status(update=update, context=context)
    assert status_table.call_count == 1


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_status_multi_entry(default_conf, update, mocker, fee) -> None:
    default_conf['telegram']['enabled'] = False
    default_conf['position_adjustment_enable'] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trades = Trade.get_open_trades()
    trade = trades[3]
    # Average may be empty on some exchanges
    trade.orders[0].average = 0
    trade.orders.append(Order(
        order_id='5412vbb',
        ft_order_side='buy',
        ft_pair=trade.pair,
        ft_is_open=False,
        ft_amount=trade.amount,
        ft_price=trade.open_rate,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="buy",
        price=trade.open_rate * 0.95,
        average=0,
        filled=trade.amount,
        remaining=0,
        cost=trade.amount,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    )
    trade.recalc_trade_from_orders()
    Trade.commit()

    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 4
    msg = msg_mock.call_args_list[3][0][0]
    assert re.search(r'Number of Entries.*2', msg)
    assert re.search(r'Number of Exits.*1', msg)
    assert re.search(r'Close Date:', msg) is None
    assert re.search(r'Close Profit:', msg) is None


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_status_closed_trade(default_conf, update, mocker, fee) -> None:
    default_conf['position_adjustment_enable'] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trade = Trade.get_trades([Trade.is_open.is_(False)]).first()
    context = MagicMock()
    context.args = [str(trade.id)]
    await telegram._status(update=update, context=context)
    assert msg_mock.call_count == 1
    msg = msg_mock.call_args_list[0][0][0]
    assert re.search(r'Close Date:', msg)
    assert re.search(r'Close Profit:', msg)


async def test_order_handle(default_conf, update, ticker, fee, mocker) -> None:
    default_conf['max_open_trades'] = 3
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _status_table=status_table,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.RUNNING
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()

    mocker.patch('freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH', 500)

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2"]
    await telegram._order(update=update, context=context)

    assert msg_mock.call_count == 1

    msg1 = msg_mock.call_args_list[0][0][0]

    assert 'Order List for Trade #*`2`' in msg1

    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH', 50)
    context = MagicMock()
    context.args = ["2"]
    await telegram._order(update=update, context=context)

    assert msg_mock.call_count == 2

    msg1 = msg_mock.call_args_list[0][0][0]
    msg2 = msg_mock.call_args_list[1][0][0]

    assert 'Order List for Trade #*`2`' in msg1
    assert '*Order List for Trade #*`2` - continued' in msg2


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_order_multi_entry(default_conf, update, mocker, fee) -> None:
    default_conf['telegram']['enabled'] = False
    default_conf['position_adjustment_enable'] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trades = Trade.get_open_trades()
    trade = trades[3]
    # Average may be empty on some exchanges
    trade.orders[0].average = 0
    trade.orders.append(Order(
        order_id='5412vbb',
        ft_order_side='buy',
        ft_pair=trade.pair,
        ft_is_open=False,
        ft_amount=trade.amount,
        ft_price=trade.open_rate,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="buy",
        price=trade.open_rate * 0.95,
        average=0,
        filled=trade.amount,
        remaining=0,
        cost=trade.amount,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    )
    trade.recalc_trade_from_orders()
    Trade.commit()

    await telegram._order(update=update, context=MagicMock())
    assert msg_mock.call_count == 4
    msg = msg_mock.call_args_list[3][0][0]
    assert re.search(r'from 1st entry rate', msg)
    assert re.search(r'Order Filled', msg)


async def test_status_handle(default_conf, update, ticker, fee, mocker) -> None:
    default_conf['max_open_trades'] = 3
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _status_table=status_table,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.STOPPED
    # Status is also enabled when stopped
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()
    # Trigger status while we have a fulfilled order for the open trade
    await telegram._status(update=update, context=MagicMock())

    # close_rate should not be included in the message as the trade is not closed
    # and no line should be empty
    lines = msg_mock.call_args_list[0][0][0].split('\n')
    assert '' not in lines[:-1]
    assert 'Close Rate' not in ''.join(lines)
    assert 'Close Profit' not in ''.join(lines)

    assert msg_mock.call_count == 3
    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'LTC/BTC' in msg_mock.call_args_list[1][0][0]

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2", "3"]

    await telegram._status(update=update, context=context)

    lines = msg_mock.call_args_list[0][0][0].split('\n')
    assert '' not in lines[:-1]
    assert 'Close Rate' not in ''.join(lines)
    assert 'Close Profit' not in ''.join(lines)

    assert msg_mock.call_count == 2
    assert 'LTC/BTC' in msg_mock.call_args_list[0][0][0]

    mocker.patch('freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH', 500)

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2"]
    await telegram._status(update=update, context=context)

    assert msg_mock.call_count == 1

    msg1 = msg_mock.call_args_list[0][0][0]

    assert 'Close Rate' not in msg1
    assert 'Trade ID:* `2`' in msg1


async def test_status_table_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    default_conf['stake_amount'] = 15.0

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.STOPPED
    # Status table is also enabled when stopped
    await telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    await telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()

    await telegram._status_table(update=update, context=MagicMock())

    text = re.sub('</?pre>', '', msg_mock.call_args_list[-1][0][0])
    line = text.split("\n")
    fields = re.sub('[ ]+', ' ', line[2].strip()).split(' ')

    assert int(fields[0]) == 1
    # assert 'L' in fields[1]
    assert 'ETH/BTC' in fields[1]
    assert msg_mock.call_count == 1


async def test_daily_handle(default_conf_usdt, update, ticker, fee, mocker, time_machine) -> None:
    mocker.patch(
        'freqtrade.rpc.rpc.CryptoToFiatConverter._find_price',
        return_value=1.1
    )
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)

    # Move date to within day
    time_machine.move_to('2022-06-11 08:00:00+00:00')
    # Create some test data
    create_mock_trades_usdt(fee)

    # Try valid data
    # /daily 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Daily Profit over the last 2 days</b>:" in msg_mock.call_args_list[0][0][0]
    assert 'Day ' in msg_mock.call_args_list[0][0][0]
    assert str(datetime.now(timezone.utc).date()) in msg_mock.call_args_list[0][0][0]
    assert '  6.83 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  7.51 USD' in msg_mock.call_args_list[0][0][0]
    assert '(2)' in msg_mock.call_args_list[0][0][0]
    assert '(2)  6.83 USDT  7.51 USD  0.64%' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Daily Profit over the last 7 days</b>:" in msg_mock.call_args_list[0][0][0]
    assert str(datetime.now(timezone.utc).date()) in msg_mock.call_args_list[0][0][0]
    assert str((datetime.now(timezone.utc) - timedelta(days=5)).date()
               ) in msg_mock.call_args_list[0][0][0]
    assert '  6.83 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  7.51 USD' in msg_mock.call_args_list[0][0][0]
    assert '(2)' in msg_mock.call_args_list[0][0][0]
    assert '(1)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()

    # /daily 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._daily(update=update, context=context)
    assert '  6.83 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  7.51 USD' in msg_mock.call_args_list[0][0][0]
    assert '(2)' in msg_mock.call_args_list[0][0][0]


async def test_daily_wrong_input(default_conf, update, ticker, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /daily -2
    context = MagicMock()
    context.args = ["-2"]
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /daily today
    context = MagicMock()
    context.args = ["today"]
    await telegram._daily(update=update, context=context)
    assert 'Daily Profit over the last 7 days</b>:' in msg_mock.call_args_list[0][0][0]


async def test_weekly_handle(default_conf_usdt, update, ticker, fee, mocker, time_machine) -> None:
    default_conf_usdt['max_open_trades'] = 1
    mocker.patch(
        'freqtrade.rpc.rpc.CryptoToFiatConverter._find_price',
        return_value=1.1
    )
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    # Move to saturday - so all trades are within that week
    time_machine.move_to('2022-06-11')
    create_mock_trades_usdt(fee)

    # Try valid data
    # /weekly 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Weekly Profit over the last 2 weeks (starting from Monday)</b>:" \
           in msg_mock.call_args_list[0][0][0]
    assert 'Monday ' in msg_mock.call_args_list[0][0][0]
    today = datetime.now(timezone.utc).date()
    first_iso_day_of_current_week = today - timedelta(days=today.weekday())
    assert str(first_iso_day_of_current_week) in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Weekly Profit over the last 8 weeks (starting from Monday)</b>:" \
           in msg_mock.call_args_list[0][0][0]
    assert 'Weekly' in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /weekly -3
    context = MagicMock()
    context.args = ["-3"]
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /weekly this week
    context = MagicMock()
    context.args = ["this week"]
    await telegram._weekly(update=update, context=context)
    assert (
        'Weekly Profit over the last 8 weeks (starting from Monday)</b>:'
        in msg_mock.call_args_list[0][0][0]
    )


async def test_monthly_handle(default_conf_usdt, update, ticker, fee, mocker, time_machine) -> None:
    default_conf_usdt['max_open_trades'] = 1
    mocker.patch(
        'freqtrade.rpc.rpc.CryptoToFiatConverter._find_price',
        return_value=1.1
    )
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    # Move to day within the month so all mock trades fall into this week.
    time_machine.move_to('2022-06-11')
    create_mock_trades_usdt(fee)

    # Try valid data
    # /monthly 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Monthly Profit over the last 2 months</b>:' in msg_mock.call_args_list[0][0][0]
    assert 'Month ' in msg_mock.call_args_list[0][0][0]
    today = datetime.now(timezone.utc).date()
    current_month = f"{today.year}-{today.month:02} "
    assert current_month in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    # Default to 6 months
    assert 'Monthly Profit over the last 6 months</b>:' in msg_mock.call_args_list[0][0][0]
    assert 'Month ' in msg_mock.call_args_list[0][0][0]
    assert current_month in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()

    # /monthly 12
    context = MagicMock()
    context.args = ["12"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Monthly Profit over the last 12 months</b>:' in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]

    # The one-digit months should contain a zero, Eg: September 2021 = "2021-09"
    # Since we loaded the last 12 months, any month should appear
    assert '-09' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /monthly -3
    context = MagicMock()
    context.args = ["-3"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /monthly february
    context = MagicMock()
    context.args = ["february"]
    await telegram._monthly(update=update, context=context)
    assert 'Monthly Profit over the last 6 months</b>:' in msg_mock.call_args_list[0][0][0]


async def test_telegram_profit_handle(
        default_conf_usdt, update, ticker_usdt, ticker_sell_up, fee,
        limit_sell_order_usdt, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    await telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'No trades yet.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()

    context = MagicMock()
    # Test with invalid 2nd argument (should silently pass)
    context.args = ["aaa"]
    await telegram._profit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'No closed trade' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    mocker.patch('freqtrade.wallets.Wallets.get_starting_balance', return_value=1000)
    assert ('∙ `0.298 USDT (0.50%) (0.03 \N{GREEK CAPITAL LETTER SIGMA}%)`'
            in msg_mock.call_args_list[-1][0][0])
    msg_mock.reset_mock()

    # Update the ticker with a market going up
    mocker.patch(f'{EXMS}.fetch_ticker', ticker_sell_up)
    # Simulate fulfilled LIMIT_SELL order for trade
    trade = Trade.session.scalars(select(Trade)).first()
    oobj = Order.parse_from_ccxt_object(
        limit_sell_order_usdt, limit_sell_order_usdt['symbol'], 'sell')
    trade.orders.append(oobj)
    trade.update_trade(oobj)

    trade.close_date = datetime.now(timezone.utc)
    trade.is_open = False
    Trade.commit()

    context.args = [3]
    await telegram._profit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert '*ROI:* Closed trades' in msg_mock.call_args_list[-1][0][0]
    assert ('∙ `5.685 USDT (9.45%) (0.57 \N{GREEK CAPITAL LETTER SIGMA}%)`'
            in msg_mock.call_args_list[-1][0][0])
    assert '∙ `6.253 USD`' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    assert ('∙ `5.685 USDT (9.45%) (0.57 \N{GREEK CAPITAL LETTER SIGMA}%)`'
            in msg_mock.call_args_list[-1][0][0])
    assert '∙ `6.253 USD`' in msg_mock.call_args_list[-1][0][0]

    assert '*Best Performing:* `ETH/USDT: 9.45%`' in msg_mock.call_args_list[-1][0][0]
    assert '*Max Drawdown:*' in msg_mock.call_args_list[-1][0][0]
    assert '*Profit factor:*' in msg_mock.call_args_list[-1][0][0]
    assert '*Winrate:*' in msg_mock.call_args_list[-1][0][0]
    assert '*Expectancy (Ratio):*' in msg_mock.call_args_list[-1][0][0]
    assert '*Trading volume:* `126 USDT`' in msg_mock.call_args_list[-1][0][0]


@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_stats(default_conf, update, ticker, fee, mocker, is_short) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._stats(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'No trades yet.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    create_mock_trades(fee, is_short=is_short)

    await telegram._stats(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Exit Reason' in msg_mock.call_args_list[-1][0][0]
    assert 'ROI' in msg_mock.call_args_list[-1][0][0]
    assert 'Avg. Duration' in msg_mock.call_args_list[-1][0][0]
    # Duration is not only N/A
    assert '0:19:00' in msg_mock.call_args_list[-1][0][0]
    assert 'N/A' in msg_mock.call_args_list[-1][0][0]
    msg_mock.reset_mock()


async def test_telegram_balance_handle(default_conf, update, mocker, rpc_balance, tickers) -> None:
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.get_balances', return_value=rpc_balance)
    mocker.patch(f'{EXMS}.get_tickers', tickers)
    mocker.patch(f'{EXMS}.get_valid_pair_combination', side_effect=lambda a, b: f"{a}/{b}")

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._balance(update=update, context=MagicMock())
    context = MagicMock()
    context.args = ["full"]
    await telegram._balance(update=update, context=context)
    result = msg_mock.call_args_list[0][0][0]
    result_full = msg_mock.call_args_list[1][0][0]
    assert msg_mock.call_count == 2
    assert '*BTC:*' in result
    assert '*ETH:*' not in result
    assert '*USDT:*' not in result
    assert '*EUR:*' not in result
    assert '*LTC:*' not in result

    assert '*LTC:*' in result_full
    assert '*XRP:*' not in result
    assert 'Balance:' in result
    assert 'Est. BTC:' in result
    assert 'BTC: 11' in result
    assert 'BTC: 12' in result_full
    assert "*3 Other Currencies (< 0.0001 BTC):*" in result
    assert 'BTC: 0.00000309' in result
    assert '*Estimated Value*:' in result_full
    assert '*Estimated Value (Bot managed assets only)*:' in result


async def test_balance_handle_empty_response(default_conf, update, mocker) -> None:
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.get_balances', return_value={})

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    freqtradebot.config['dry_run'] = False
    await telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert 'Starting capital: `0 BTC' in result


async def test_balance_handle_empty_response_dry(default_conf, update, mocker) -> None:
    mocker.patch(f'{EXMS}.get_balances', return_value={})

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert "*Warning:* Simulated balances in Dry Mode." in result
    assert "Starting capital: `1000 BTC`" in result


async def test_balance_handle_too_large_response(default_conf, update, mocker) -> None:
    balances = []
    for i in range(100):
        curr = choice(ascii_uppercase) + choice(ascii_uppercase) + choice(ascii_uppercase)
        balances.append({
            'currency': curr,
            'free': 1.0,
            'used': 0.5,
            'balance': i,
            'bot_owned': 0.5,
            'est_stake': 1,
            'est_stake_bot': 1,
            'stake': 'BTC',
            'is_position': False,
            'leverage': 1.0,
            'position': 0.0,
            'side': 'long',
            'is_bot_managed': True,
        })
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_balance', return_value={
        'currencies': balances,
        'total': 100.0,
        'total_bot': 100.0,
        'symbol': 100.0,
        'value': 1000.0,
        'value_bot': 1000.0,
        'starting_capital': 1000,
        'starting_capital_fiat': 1000,
    })

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._balance(update=update, context=MagicMock())
    assert msg_mock.call_count > 1
    # Test if wrap happens around 4000 -
    # and each single currency-output is around 120 characters long so we need
    # an offset to avoid random test failures
    assert len(msg_mock.call_args_list[0][0][0]) < 4096
    assert len(msg_mock.call_args_list[0][0][0]) > (4096 - 120)


async def test_start_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    await telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1


async def test_start_handle_already_running(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1
    assert 'already running' in msg_mock.call_args_list[0][0][0]


async def test_stop_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'stopping trader' in msg_mock.call_args_list[0][0][0]


async def test_stop_handle_already_stopped(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    await telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'already stopped' in msg_mock.call_args_list[0][0][0]


async def test_stopbuy_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    assert freqtradebot.config['max_open_trades'] != 0
    await telegram._stopentry(update=update, context=MagicMock())
    assert freqtradebot.config['max_open_trades'] == 0
    assert msg_mock.call_count == 1
    assert 'No more entries will occur from now. Run /reload_config to reset.' \
        in msg_mock.call_args_list[0][0][0]


async def test_reload_config_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._reload_config(update=update, context=MagicMock())
    assert freqtradebot.state == State.RELOAD_CONFIG
    assert msg_mock.call_count == 1
    assert 'Reloading config' in msg_mock.call_args_list[0][0][0]


async def test_telegram_forceexit_handle(default_conf, update, ticker, fee,
                                         ticker_sell_up, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    msg_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)

    # Create some test data
    freqtradebot.enter_positions()

    trade = Trade.session.scalars(select(Trade)).first()
    assert trade

    # Increase the price and sell it
    mocker.patch(f'{EXMS}.fetch_ticker', ticker_sell_up)

    # /forceexit 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._force_exit(update=update, context=context)

    assert msg_mock.call_count == 4
    last_msg = msg_mock.call_args_list[-2][0][0]
    assert {
        'type': RPCMessageType.EXIT,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'leverage': 1.0,
        'limit': 1.173e-05,
        'order_rate': 1.173e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.173e-05,
        'direction': 'Long',
        'profit_amount': 6.314e-05,
        'profit_ratio': 0.0629778,
        'stake_currency': 'BTC',
        'base_currency': 'ETH',
        'fiat_currency': 'USD',
        'buy_tag': ANY,
        'enter_tag': ANY,
        'sell_reason': ExitType.FORCE_EXIT.value,
        'exit_reason': ExitType.FORCE_EXIT.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'stake_amount': 0.0009999999999054,
        'sub_trade': False,
        'cumulative_profit': 0.0,
    } == last_msg


async def test_telegram_force_exit_down_handle(default_conf, update, ticker, fee,
                                               ticker_sell_down, mocker) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    msg_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)

    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)

    # Create some test data
    freqtradebot.enter_positions()

    # Decrease the price and sell it
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_sell_down
    )

    trade = Trade.session.scalars(select(Trade)).first()
    assert trade

    # /forceexit 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._force_exit(update=update, context=context)

    assert msg_mock.call_count == 4

    last_msg = msg_mock.call_args_list[-2][0][0]
    assert {
        'type': RPCMessageType.EXIT,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'leverage': 1.0,
        'limit': 1.043e-05,
        'order_rate': 1.043e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.043e-05,
        'direction': 'Long',
        'profit_amount': -5.497e-05,
        'profit_ratio': -0.05482878,
        'stake_currency': 'BTC',
        'base_currency': 'ETH',
        'fiat_currency': 'USD',
        'buy_tag': ANY,
        'enter_tag': ANY,
        'sell_reason': ExitType.FORCE_EXIT.value,
        'exit_reason': ExitType.FORCE_EXIT.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'stake_amount': 0.0009999999999054,
        'sub_trade': False,
        'cumulative_profit': 0.0,
    } == last_msg


async def test_forceexit_all_handle(default_conf, update, ticker, fee, mocker) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    msg_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    default_conf['max_open_trades'] = 4
    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)

    # Create some test data
    freqtradebot.enter_positions()
    msg_mock.reset_mock()

    # /forceexit all
    context = MagicMock()
    context.args = ["all"]
    await telegram._force_exit(update=update, context=context)

    # Called for each trade 2 times
    assert msg_mock.call_count == 8
    msg = msg_mock.call_args_list[0][0][0]
    assert {
        'type': RPCMessageType.EXIT,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'leverage': 1.0,
        'order_rate': 1.099e-05,
        'limit': 1.099e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.099e-05,
        'direction': 'Long',
        'profit_amount': -4.09e-06,
        'profit_ratio': -0.00408133,
        'stake_currency': 'BTC',
        'base_currency': 'ETH',
        'fiat_currency': 'USD',
        'buy_tag': ANY,
        'enter_tag': ANY,
        'sell_reason': ExitType.FORCE_EXIT.value,
        'exit_reason': ExitType.FORCE_EXIT.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'stake_amount': 0.0009999999999054,
        'sub_trade': False,
        'cumulative_profit': 0.0,
    } == msg


async def test_forceexit_handle_invalid(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    # Trader is not running
    freqtradebot.state = State.STOPPED
    # /forceexit 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]

    # Invalid argument
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /forceexit 123456
    context = MagicMock()
    context.args = ["123456"]
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'invalid argument' in msg_mock.call_args_list[0][0][0]


async def test_force_exit_no_pair(default_conf, update, ticker, fee, mocker) -> None:
    default_conf['max_open_trades'] = 4
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    femock = mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_exit')
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    # /forceexit
    context = MagicMock()
    context.args = []
    await telegram._force_exit(update=update, context=context)
    # No pair
    assert msg_mock.call_args_list[0][1]['msg'] == 'No open trade found.'

    # Create some test data
    freqtradebot.enter_positions()
    msg_mock.reset_mock()

    # /forceexit
    await telegram._force_exit(update=update, context=context)
    keyboard = msg_mock.call_args_list[0][1]['keyboard']
    # 4 pairs + cancel
    assert reduce(lambda acc, x: acc + len(x), keyboard, 0) == 5
    assert keyboard[-1][0].text == "Cancel"

    assert keyboard[1][0].callback_data == 'force_exit__2 '
    update = MagicMock()
    update.callback_query = AsyncMock()
    update.callback_query.data = keyboard[1][0].callback_data
    await telegram._force_exit_inline(update, None)
    assert update.callback_query.answer.call_count == 1
    assert update.callback_query.edit_message_text.call_count == 1
    assert femock.call_count == 1
    assert femock.call_args_list[0][0][0] == '2'

    # Retry exiting - but cancel instead
    update.callback_query.reset_mock()
    await telegram._force_exit(update=update, context=context)
    # Use cancel button
    update.callback_query.data = keyboard[-1][0].callback_data
    await telegram._force_exit_inline(update, None)
    query = update.callback_query
    assert query.answer.call_count == 1
    assert query.edit_message_text.call_count == 1
    assert query.edit_message_text.call_args_list[-1][1]['text'] == "Force exit canceled."


async def test_force_enter_handle(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_entry', fbuy_mock)

    telegram, freqtradebot, _ = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    # /forcelong ETH/BTC
    context = MagicMock()
    context.args = ["ETH/BTC"]
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)

    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == 'ETH/BTC'
    assert fbuy_mock.call_args_list[0][0][1] is None
    assert fbuy_mock.call_args_list[0][1]['order_side'] == SignalDirection.LONG

    # Reset and retry with specified price
    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_entry', fbuy_mock)
    # /forcelong ETH/BTC 0.055
    context = MagicMock()
    context.args = ["ETH/BTC", "0.055"]
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)

    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == 'ETH/BTC'
    assert isinstance(fbuy_mock.call_args_list[0][0][1], float)
    assert fbuy_mock.call_args_list[0][0][1] == 0.055


async def test_force_enter_handle_exception(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._force_enter(update=update, context=MagicMock(), order_side=SignalDirection.LONG)

    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][0][0] == 'Force_entry not enabled.'


async def test_force_enter_no_pair(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_entry', fbuy_mock)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    context = MagicMock()
    context.args = []
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)

    assert fbuy_mock.call_count == 0
    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][1]['msg'] == 'Which pair?'
    # assert msg_mock.call_args_list[0][1]['callback_query_handler'] == 'forcebuy'
    keyboard = msg_mock.call_args_list[0][1]['keyboard']
    # One additional button - cancel
    assert reduce(lambda acc, x: acc + len(x), keyboard, 0) == 5
    update = MagicMock()
    update.callback_query = AsyncMock()
    update.callback_query.data = 'force_enter__XRP/USDT_||_long'
    await telegram._force_enter_inline(update, None)
    assert fbuy_mock.call_count == 1

    fbuy_mock.reset_mock()
    update.callback_query = AsyncMock()
    update.callback_query.data = 'force_enter__cancel'
    await telegram._force_enter_inline(update, None)
    assert fbuy_mock.call_count == 0
    query = update.callback_query
    assert query.edit_message_text.call_count == 1
    assert query.edit_message_text.call_args_list[-1][1]['text'] == "Force enter canceled."


async def test_telegram_performance_handle(default_conf_usdt, update, ticker, fee, mocker) -> None:

    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)

    # Create some test data
    create_mock_trades_usdt(fee)

    await telegram._performance(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Performance' in msg_mock.call_args_list[0][0][0]
    assert '<code>XRP/USDT\t2.842 USDT (10.00%) (1)</code>' in msg_mock.call_args_list[0][0][0]


async def test_telegram_entry_tag_performance_handle(
        default_conf_usdt, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    create_mock_trades_usdt(fee)

    context = MagicMock()
    await telegram._enter_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Entry Tag Performance' in msg_mock.call_args_list[0][0][0]
    assert '<code>TEST1\t3.987 USDT (5.00%) (1)</code>' in msg_mock.call_args_list[0][0][0]

    context.args = ['XRP/USDT']
    await telegram._enter_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 2

    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_enter_tag_performance',
                 side_effect=RPCException('Error'))
    await telegram._enter_tag_performance(update=update, context=MagicMock())

    assert msg_mock.call_count == 1
    assert "Error" in msg_mock.call_args_list[0][0][0]


async def test_telegram_exit_reason_performance_handle(
        default_conf_usdt, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    create_mock_trades_usdt(fee)

    context = MagicMock()
    await telegram._exit_reason_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Exit Reason Performance' in msg_mock.call_args_list[0][0][0]
    assert '<code>roi\t2.842 USDT (10.00%) (1)</code>' in msg_mock.call_args_list[0][0][0]
    context.args = ['XRP/USDT']

    await telegram._exit_reason_performance(update=update, context=context)
    assert msg_mock.call_count == 2

    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_exit_reason_performance',
                 side_effect=RPCException('Error'))
    await telegram._exit_reason_performance(update=update, context=MagicMock())

    assert msg_mock.call_count == 1
    assert "Error" in msg_mock.call_args_list[0][0][0]


async def test_telegram_mix_tag_performance_handle(default_conf_usdt, update, ticker, fee,
                                                   mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    # Create some test data
    create_mock_trades_usdt(fee)

    context = MagicMock()
    await telegram._mix_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Mix Tag Performance' in msg_mock.call_args_list[0][0][0]
    assert ('<code>TEST3 roi\t2.842 USDT (10.00%) (1)</code>'
            in msg_mock.call_args_list[0][0][0])

    context.args = ['XRP/USDT']
    await telegram._mix_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 2

    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_mix_tag_performance',
                 side_effect=RPCException('Error'))
    await telegram._mix_tag_performance(update=update, context=MagicMock())

    assert msg_mock.call_count == 1
    assert "Error" in msg_mock.call_args_list[0][0][0]


async def test_count_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    freqtradebot.state = State.STOPPED
    await telegram._count(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING

    # Create some test data
    freqtradebot.enter_positions()
    msg_mock.reset_mock()
    await telegram._count(update=update, context=MagicMock())

    msg = ('<pre>  current    max    total stake\n---------  -----  -------------\n'
           '        1      {}          {}</pre>').format(
        default_conf['max_open_trades'],
        default_conf['stake_amount']
    )
    assert msg in msg_mock.call_args_list[0][0][0]


async def test_telegram_lock_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._locks(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'No active locks.' in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()

    PairLocks.lock_pair('ETH/BTC', dt_now() + timedelta(minutes=4), 'randreason')
    PairLocks.lock_pair('XRP/BTC', dt_now() + timedelta(minutes=20), 'deadbeef')

    await telegram._locks(update=update, context=MagicMock())

    assert 'Pair' in msg_mock.call_args_list[0][0][0]
    assert 'Until' in msg_mock.call_args_list[0][0][0]
    assert 'Reason\n' in msg_mock.call_args_list[0][0][0]
    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'XRP/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'deadbeef' in msg_mock.call_args_list[0][0][0]
    assert 'randreason' in msg_mock.call_args_list[0][0][0]

    context = MagicMock()
    context.args = ['XRP/BTC']
    msg_mock.reset_mock()
    await telegram._delete_locks(update=update, context=context)

    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'randreason' in msg_mock.call_args_list[0][0][0]
    assert 'XRP/BTC' not in msg_mock.call_args_list[0][0][0]
    assert 'deadbeef' not in msg_mock.call_args_list[0][0][0]


async def test_whitelist_static(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert ("Using whitelist `['StaticPairList']` with 4 pairs\n"
            "`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`" in msg_mock.call_args_list[0][0][0])

    context = MagicMock()
    context.args = ['sorted']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert ("Using whitelist `['StaticPairList']` with 4 pairs\n"
            "`ETH/BTC, LTC/BTC, NEO/BTC, XRP/BTC`" in msg_mock.call_args_list[0][0][0])

    context = MagicMock()
    context.args = ['baseonly']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert ("Using whitelist `['StaticPairList']` with 4 pairs\n"
            "`ETH, LTC, XRP, NEO`" in msg_mock.call_args_list[0][0][0])

    context = MagicMock()
    context.args = ['baseonly', 'sorted']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert ("Using whitelist `['StaticPairList']` with 4 pairs\n"
            "`ETH, LTC, NEO, XRP`" in msg_mock.call_args_list[0][0][0])


async def test_whitelist_dynamic(default_conf, update, mocker) -> None:
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    default_conf['pairlists'] = [{'method': 'VolumePairList',
                                  'number_assets': 4
                                  }]
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert ("Using whitelist `['VolumePairList']` with 4 pairs\n"
            "`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`" in msg_mock.call_args_list[0][0][0])

    context = MagicMock()
    context.args = ['sorted']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert ("Using whitelist `['VolumePairList']` with 4 pairs\n"
            "`ETH/BTC, LTC/BTC, NEO/BTC, XRP/BTC`" in msg_mock.call_args_list[0][0][0])

    context = MagicMock()
    context.args = ['baseonly']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert ("Using whitelist `['VolumePairList']` with 4 pairs\n"
            "`ETH, LTC, XRP, NEO`" in msg_mock.call_args_list[0][0][0])

    context = MagicMock()
    context.args = ['baseonly', 'sorted']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert ("Using whitelist `['VolumePairList']` with 4 pairs\n"
            "`ETH, LTC, NEO, XRP`" in msg_mock.call_args_list[0][0][0])


async def test_blacklist_static(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._blacklist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert ("Blacklist contains 2 pairs\n`DOGE/BTC, HOT/BTC`"
            in msg_mock.call_args_list[0][0][0])

    msg_mock.reset_mock()

    # /blacklist ETH/BTC
    context = MagicMock()
    context.args = ["ETH/BTC"]
    await telegram._blacklist(update=update, context=context)
    assert msg_mock.call_count == 1
    assert ("Blacklist contains 3 pairs\n`DOGE/BTC, HOT/BTC, ETH/BTC`"
            in msg_mock.call_args_list[0][0][0])
    assert freqtradebot.pairlists.blacklist == ["DOGE/BTC", "HOT/BTC", "ETH/BTC"]

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["XRP/.*"]
    await telegram._blacklist(update=update, context=context)
    assert msg_mock.call_count == 1

    assert ("Blacklist contains 4 pairs\n`DOGE/BTC, HOT/BTC, ETH/BTC, XRP/.*`"
            in msg_mock.call_args_list[0][0][0])
    assert freqtradebot.pairlists.blacklist == ["DOGE/BTC", "HOT/BTC", "ETH/BTC", "XRP/.*"]

    msg_mock.reset_mock()
    context.args = ["DOGE/BTC"]
    await telegram._blacklist_delete(update=update, context=context)
    assert msg_mock.call_count == 1
    assert ("Blacklist contains 3 pairs\n`HOT/BTC, ETH/BTC, XRP/.*`"
            in msg_mock.call_args_list[0][0][0])


async def test_telegram_logs(default_conf, update, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
    )
    setup_logging(default_conf)

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    context = MagicMock()
    context.args = []
    await telegram._logs(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "freqtrade\\.rpc\\.telegram" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    context.args = ["1"]
    await telegram._logs(update=update, context=context)
    assert msg_mock.call_count == 1

    msg_mock.reset_mock()
    # Test with changed MaxMessageLength
    mocker.patch('freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH', 200)
    context = MagicMock()
    context.args = []
    await telegram._logs(update=update, context=context)
    # Called at least 2 times. Exact times will change with unrelated changes to setup messages
    # Therefore we don't test for this explicitly.
    assert msg_mock.call_count >= 2


async def test_edge_disabled(default_conf, update, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "Edge is not enabled." in msg_mock.call_args_list[0][0][0]


async def test_edge_enabled(edge_conf, update, mocker) -> None:
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'E/F': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
        }
    ))

    telegram, _, msg_mock = get_telegram_testobject(mocker, edge_conf)

    await telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '<b>Edge only validated following pairs:</b>\n<pre>' in msg_mock.call_args_list[0][0][0]
    assert 'Pair      Winrate    Expectancy    Stoploss' in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()

    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={}))
    await telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '<b>Edge only validated following pairs:</b>' in msg_mock.call_args_list[0][0][0]
    assert 'Winrate' not in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize('is_short,regex_pattern',
                         [(True, r"just now[ ]*XRP\/BTC \(#3\)  -1.00% \("),
                          (False, r"just now[ ]*XRP\/BTC \(#3\)  1.00% \(")])
async def test_telegram_trades(mocker, update, default_conf, fee, is_short, regex_pattern):

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    context = MagicMock()
    context.args = []

    await telegram._trades(update=update, context=context)
    assert "<b>0 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    context.args = ['hello']
    await telegram._trades(update=update, context=context)
    assert "<b>0 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [5]
    await telegram._trades(update=update, context=context)
    msg_mock.call_count == 1
    assert "2 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "Profit (" in msg_mock.call_args_list[0][0][0]
    assert "Close Date" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" in msg_mock.call_args_list[0][0][0]
    assert bool(re.search(regex_pattern, msg_mock.call_args_list[0][0][0]))


@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_delete_trade(mocker, update, default_conf, fee, is_short):

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._delete_trade(update=update, context=context)
    assert "Trade-id not set." in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [1]
    await telegram._delete_trade(update=update, context=context)
    msg_mock.call_count == 1
    assert "Deleted trade 1." in msg_mock.call_args_list[0][0][0]
    assert "Please make sure to take care of this asset" in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_reload_trade_from_exchange(mocker, update, default_conf, fee, is_short):

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._reload_trade_from_exchange(update=update, context=context)
    assert "Trade-id not set." in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context.args = [5]

    await telegram._reload_trade_from_exchange(update=update, context=context)
    assert "Status: `Reloaded from orders from exchange`" in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_delete_open_order(mocker, update, default_conf, fee, is_short, ticker):

    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
    )
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._cancel_open_order(update=update, context=context)
    assert "Trade-id not set." in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [5]
    await telegram._cancel_open_order(update=update, context=context)
    assert "No open order for trade_id" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()

    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f'{EXMS}.fetch_order', return_value=trade.orders[-1].to_ccxt_object())
    context = MagicMock()
    context.args = [6]
    await telegram._cancel_open_order(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Open order canceled." in msg_mock.call_args_list[0][0][0]


async def test_help_handle(default_conf, update, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._help(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*/help:* `This help message`' in msg_mock.call_args_list[0][0][0]


async def test_version_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert f'*Version:* `{__version__}`' in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    freqtradebot.strategy.version = lambda: '1.1.1'

    await telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert f'*Version:* `{__version__}`' in msg_mock.call_args_list[0][0][0]
    assert '*Strategy version: * `1.1.1`' in msg_mock.call_args_list[0][0][0]


async def test_show_config_handle(default_conf, update, mocker) -> None:

    default_conf['runmode'] = RunMode.DRY_RUN

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Mode:* `{}`'.format('Dry-run') in msg_mock.call_args_list[0][0][0]
    assert '*Exchange:* `binance`' in msg_mock.call_args_list[0][0][0]
    assert f'*Strategy:* `{CURRENT_TEST_STRATEGY}`' in msg_mock.call_args_list[0][0][0]
    assert '*Stoploss:* `-0.1`' in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    freqtradebot.config['trailing_stop'] = True
    await telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Mode:* `{}`'.format('Dry-run') in msg_mock.call_args_list[0][0][0]
    assert '*Exchange:* `binance`' in msg_mock.call_args_list[0][0][0]
    assert f'*Strategy:* `{CURRENT_TEST_STRATEGY}`' in msg_mock.call_args_list[0][0][0]
    assert '*Initial Stoploss:* `-0.1`' in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize('message_type,enter,enter_signal,leverage', [
    (RPCMessageType.ENTRY, 'Long', 'long_signal_01', None),
    (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 1.0),
    (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 5.0),
    (RPCMessageType.ENTRY, 'Short', 'short_signal_01', 2.0)])
def test_send_msg_enter_notification(default_conf, mocker, caplog, message_type,
                                     enter, enter_signal, leverage) -> None:
    default_conf['telegram']['notification_settings']['show_candle'] = 'ohlc'
    df = DataFrame({
        'open': [1.1],
        'high': [2.2],
        'low': [1.0],
        'close': [1.5],
    })
    mocker.patch('freqtrade.data.dataprovider.DataProvider.get_analyzed_dataframe',
                 return_value=(df, 1))

    msg = {
        'type': message_type,
        'trade_id': 1,
        'enter_tag': enter_signal,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'leverage': leverage,
        'open_rate': 1.099e-05,
        'order_type': 'limit',
        'direction': enter,
        'stake_amount': 0.01465333,
        'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'current_rate': 1.099e-05,
        'amount': 1333.3333333333335,
        'analyzed_candle': {'open': 1.1, 'high': 2.2, 'low': 1.0, 'close': 1.5},
        'open_date': dt_now() + timedelta(hours=-1)
    }
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(msg)
    leverage_text = f'*Leverage:* `{leverage}`\n' if leverage and leverage != 1.0 else ''

    assert msg_mock.call_args[0][0] == (
        f'\N{LARGE BLUE CIRCLE} *Binance (dry):* {enter} ETH/BTC (#1)\n'
        '*Candle OHLC*: `1.1, 2.2, 1.0, 1.5`\n'
        f'*Enter Tag:* `{enter_signal}`\n'
        '*Amount:* `1333.33333333`\n'
        f'{leverage_text}'
        '*Open Rate:* `0.00001099`\n'
        '*Current Rate:* `0.00001099`\n'
        '*Total:* `(0.01465333 BTC, 180.895 USD)`'
    )

    freqtradebot.config['telegram']['notification_settings'] = {'buy': 'off'}
    caplog.clear()
    msg_mock.reset_mock()
    telegram.send_msg(msg)
    msg_mock.call_count == 0
    log_has("Notification 'buy' not sent.", caplog)

    freqtradebot.config['telegram']['notification_settings'] = {'buy': 'silent'}
    caplog.clear()
    msg_mock.reset_mock()

    telegram.send_msg(msg)
    msg_mock.call_count == 1
    msg_mock.call_args_list[0][1]['disable_notification'] is True


@pytest.mark.parametrize('message_type,enter_signal', [
    (RPCMessageType.ENTRY_CANCEL, 'long_signal_01'),
    (RPCMessageType.ENTRY_CANCEL, 'short_signal_01')])
def test_send_msg_enter_cancel_notification(
        default_conf, mocker, message_type, enter_signal) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': message_type,
        'enter_tag': enter_signal,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'reason': CANCEL_REASON['TIMEOUT']
    })
    assert (msg_mock.call_args[0][0] == '\N{WARNING SIGN} *Binance (dry):* '
            'Cancelling enter Order for ETH/BTC (#1). '
            'Reason: cancelled due to timeout.')


def test_send_msg_protection_notification(default_conf, mocker, time_machine) -> None:

    default_conf['telegram']['notification_settings']['protection_trigger'] = 'on'

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    time_machine.move_to("2021-09-01 05:00:00 +00:00")
    lock = PairLocks.lock_pair('ETH/BTC', dt_now() + timedelta(minutes=6), 'randreason')
    msg = {
        'type': RPCMessageType.PROTECTION_TRIGGER,
    }
    msg.update(lock.to_json())
    telegram.send_msg(msg)
    assert (msg_mock.call_args[0][0] == "*Protection* triggered due to randreason. "
            "`ETH/BTC` will be locked until `2021-09-01 05:10:00`.")

    msg_mock.reset_mock()
    # Test global protection

    msg = {
        'type': RPCMessageType.PROTECTION_TRIGGER_GLOBAL,
    }
    lock = PairLocks.lock_pair('*', dt_now() + timedelta(minutes=100), 'randreason')
    msg.update(lock.to_json())
    telegram.send_msg(msg)
    assert (msg_mock.call_args[0][0] == "*Protection* triggered due to randreason. "
            "*All pairs* will be locked until `2021-09-01 06:45:00`.")


@pytest.mark.parametrize('message_type,entered,enter_signal,leverage', [
    (RPCMessageType.ENTRY_FILL, 'Long', 'long_signal_01', 1.0),
    (RPCMessageType.ENTRY_FILL, 'Long', 'long_signal_02', 2.0),
    (RPCMessageType.ENTRY_FILL, 'Short', 'short_signal_01', 2.0),
])
def test_send_msg_entry_fill_notification(default_conf, mocker, message_type, entered,
                                          enter_signal, leverage) -> None:

    default_conf['telegram']['notification_settings']['entry_fill'] = 'on'
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': message_type,
        'trade_id': 1,
        'enter_tag': enter_signal,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'leverage': leverage,
        'stake_amount': 0.01465333,
        'direction': entered,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'open_rate': 1.099e-05,
        'amount': 1333.3333333333335,
        'open_date': dt_now() - timedelta(hours=1)
    })
    leverage_text = f'*Leverage:* `{leverage}`\n' if leverage != 1.0 else ''
    assert msg_mock.call_args[0][0] == (
        f'\N{CHECK MARK} *Binance (dry):* {entered}ed ETH/BTC (#1)\n'
        f'*Enter Tag:* `{enter_signal}`\n'
        '*Amount:* `1333.33333333`\n'
        f"{leverage_text}"
        '*Open Rate:* `0.00001099`\n'
        '*Total:* `(0.01465333 BTC, 180.895 USD)`'
    )

    msg_mock.reset_mock()
    telegram.send_msg({
        'type': message_type,
        'trade_id': 1,
        'enter_tag': enter_signal,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'leverage': leverage,
        'stake_amount': 0.01465333,
        'sub_trade': True,
        'direction': entered,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'open_rate': 1.099e-05,
        'amount': 1333.3333333333335,
        'open_date': dt_now() - timedelta(hours=1)
    })

    assert msg_mock.call_args[0][0] == (
        f'\N{CHECK MARK} *Binance (dry):* {entered}ed ETH/BTC (#1)\n'
        f'*Enter Tag:* `{enter_signal}`\n'
        '*Amount:* `1333.33333333`\n'
        f"{leverage_text}"
        '*Open Rate:* `0.00001099`\n'
        '*Total:* `(0.01465333 BTC, 180.895 USD)`'
    )


def test_send_msg_sell_notification(default_conf, mocker) -> None:

    with time_machine.travel("2022-09-01 05:00:00 +00:00", tick=False):
        telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

        old_convamount = telegram._rpc._fiat_converter.convert_amount
        telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
        telegram.send_msg({
            'type': RPCMessageType.EXIT,
            'trade_id': 1,
            'exchange': 'Binance',
            'pair': 'KEY/ETH',
            'leverage': 1.0,
            'direction': 'Long',
            'gain': 'loss',
            'order_rate': 3.201e-05,
            'amount': 1333.3333333333335,
            'order_type': 'market',
            'open_rate': 7.5e-05,
            'current_rate': 3.201e-05,
            'profit_amount': -0.05746268,
            'profit_ratio': -0.57405275,
            'stake_currency': 'ETH',
            'fiat_currency': 'USD',
            'enter_tag': 'buy_signal1',
            'exit_reason': ExitType.STOP_LOSS.value,
            'open_date': dt_now() - timedelta(hours=1),
            'close_date': dt_now(),
        })
        assert msg_mock.call_args[0][0] == (
            '\N{WARNING SIGN} *Binance (dry):* Exiting KEY/ETH (#1)\n'
            '*Unrealized Profit:* `-57.41% (loss: -0.05746268 ETH / -24.812 USD)`\n'
            '*Enter Tag:* `buy_signal1`\n'
            '*Exit Reason:* `stop_loss`\n'
            '*Direction:* `Long`\n'
            '*Amount:* `1333.33333333`\n'
            '*Open Rate:* `0.00007500`\n'
            '*Current Rate:* `0.00003201`\n'
            '*Exit Rate:* `0.00003201`\n'
            '*Duration:* `1:00:00 (60.0 min)`'
        )

        msg_mock.reset_mock()
        telegram.send_msg({
            'type': RPCMessageType.EXIT,
            'trade_id': 1,
            'exchange': 'Binance',
            'pair': 'KEY/ETH',
            'direction': 'Long',
            'gain': 'loss',
            'order_rate': 3.201e-05,
            'amount': 1333.3333333333335,
            'order_type': 'market',
            'open_rate': 7.5e-05,
            'current_rate': 3.201e-05,
            'cumulative_profit': -0.15746268,
            'profit_amount': -0.05746268,
            'profit_ratio': -0.57405275,
            'stake_currency': 'ETH',
            'fiat_currency': 'USD',
            'enter_tag': 'buy_signal1',
            'exit_reason': ExitType.STOP_LOSS.value,
            'open_date': dt_now() - timedelta(days=1, hours=2, minutes=30),
            'close_date': dt_now(),
            'stake_amount': 0.01,
            'sub_trade': True,
        })
        assert msg_mock.call_args[0][0] == (
            '\N{WARNING SIGN} *Binance (dry):* Partially exiting KEY/ETH (#1)\n'
            '*Unrealized Sub Profit:* `-57.41% (loss: -0.05746268 ETH / -24.812 USD)`\n'
            '*Cumulative Profit:* (`-0.15746268 ETH / -24.812 USD`)\n'
            '*Enter Tag:* `buy_signal1`\n'
            '*Exit Reason:* `stop_loss`\n'
            '*Direction:* `Long`\n'
            '*Amount:* `1333.33333333`\n'
            '*Open Rate:* `0.00007500`\n'
            '*Current Rate:* `0.00003201`\n'
            '*Exit Rate:* `0.00003201`\n'
            '*Remaining:* `(0.01 ETH, -24.812 USD)`'
            )

        msg_mock.reset_mock()
        telegram.send_msg({
            'type': RPCMessageType.EXIT,
            'trade_id': 1,
            'exchange': 'Binance',
            'pair': 'KEY/ETH',
            'direction': 'Long',
            'gain': 'loss',
            'order_rate': 3.201e-05,
            'amount': 1333.3333333333335,
            'order_type': 'market',
            'open_rate': 7.5e-05,
            'current_rate': 3.201e-05,
            'profit_amount': -0.05746268,
            'profit_ratio': -0.57405275,
            'stake_currency': 'ETH',
            'enter_tag': 'buy_signal1',
            'exit_reason': ExitType.STOP_LOSS.value,
            'open_date': dt_now() - timedelta(days=1, hours=2, minutes=30),
            'close_date': dt_now(),
        })
        assert msg_mock.call_args[0][0] == (
            '\N{WARNING SIGN} *Binance (dry):* Exiting KEY/ETH (#1)\n'
            '*Unrealized Profit:* `-57.41% (loss: -0.05746268 ETH)`\n'
            '*Enter Tag:* `buy_signal1`\n'
            '*Exit Reason:* `stop_loss`\n'
            '*Direction:* `Long`\n'
            '*Amount:* `1333.33333333`\n'
            '*Open Rate:* `0.00007500`\n'
            '*Current Rate:* `0.00003201`\n'
            '*Exit Rate:* `0.00003201`\n'
            '*Duration:* `1 day, 2:30:00 (1590.0 min)`'
        )
        # Reset singleton function to avoid random breaks
        telegram._rpc._fiat_converter.convert_amount = old_convamount


async def test_send_msg_sell_cancel_notification(default_conf, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    old_convamount = telegram._rpc._fiat_converter.convert_amount
    telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
    telegram.send_msg({
        'type': RPCMessageType.EXIT_CANCEL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'reason': 'Cancelled on exchange'
    })
    assert msg_mock.call_args[0][0] == (
        '\N{WARNING SIGN} *Binance (dry):* Cancelling exit Order for KEY/ETH (#1).'
        ' Reason: Cancelled on exchange.')

    msg_mock.reset_mock()
    # Test with live mode (no dry appendix)
    telegram._config['dry_run'] = False
    telegram.send_msg({
        'type': RPCMessageType.EXIT_CANCEL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'reason': 'timeout'
    })
    assert msg_mock.call_args[0][0] == (
        '\N{WARNING SIGN} *Binance:* Cancelling exit Order for KEY/ETH (#1). Reason: timeout.')
    # Reset singleton function to avoid random breaks
    telegram._rpc._fiat_converter.convert_amount = old_convamount


@pytest.mark.parametrize('direction,enter_signal,leverage', [
    ('Long', 'long_signal_01', None),
    ('Long', 'long_signal_01', 1.0),
    ('Long', 'long_signal_01', 5.0),
    ('Short', 'short_signal_01', 2.0)])
def test_send_msg_sell_fill_notification(default_conf, mocker, direction,
                                         enter_signal, leverage) -> None:

    default_conf['telegram']['notification_settings']['exit_fill'] = 'on'
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    with time_machine.travel("2022-09-01 05:00:00 +00:00", tick=False):
        telegram.send_msg({
            'type': RPCMessageType.EXIT_FILL,
            'trade_id': 1,
            'exchange': 'Binance',
            'pair': 'KEY/ETH',
            'leverage': leverage,
            'direction': direction,
            'gain': 'loss',
            'limit': 3.201e-05,
            'amount': 1333.3333333333335,
            'order_type': 'market',
            'open_rate': 7.5e-05,
            'close_rate': 3.201e-05,
            'profit_amount': -0.05746268,
            'profit_ratio': -0.57405275,
            'stake_currency': 'ETH',
            'enter_tag': enter_signal,
            'exit_reason': ExitType.STOP_LOSS.value,
            'open_date': dt_now() - timedelta(days=1, hours=2, minutes=30),
            'close_date': dt_now(),
        })

        leverage_text = f'*Leverage:* `{leverage}`\n' if leverage and leverage != 1.0 else ''
        assert msg_mock.call_args[0][0] == (
            '\N{WARNING SIGN} *Binance (dry):* Exited KEY/ETH (#1)\n'
            '*Profit:* `-57.41% (loss: -0.05746268 ETH)`\n'
            f'*Enter Tag:* `{enter_signal}`\n'
            '*Exit Reason:* `stop_loss`\n'
            f"*Direction:* `{direction}`\n"
            f"{leverage_text}"
            '*Amount:* `1333.33333333`\n'
            '*Open Rate:* `0.00007500`\n'
            '*Exit Rate:* `0.00003201`\n'
            '*Duration:* `1 day, 2:30:00 (1590.0 min)`'
        )


def test_send_msg_status_notification(default_conf, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': RPCMessageType.STATUS,
        'status': 'running'
    })
    assert msg_mock.call_args[0][0] == '*Status:* `running`'


async def test_warning_notification(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': RPCMessageType.WARNING,
        'status': 'message'
    })
    assert msg_mock.call_args[0][0] == '\N{WARNING SIGN} *Warning:* `message`'


def test_startup_notification(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': RPCMessageType.STARTUP,
        'status': '*Custom:* `Hello World`'
    })
    assert msg_mock.call_args[0][0] == '*Custom:* `Hello World`'


def test_send_msg_strategy_msg_notification(default_conf, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': RPCMessageType.STRATEGY_MSG,
        'msg': 'hello world, Test msg'
    })
    assert msg_mock.call_args[0][0] == 'hello world, Test msg'


def test_send_msg_unknown_type(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': None,
    })
    msg_mock.call_count == 0


@pytest.mark.parametrize('message_type,enter,enter_signal,leverage', [
    (RPCMessageType.ENTRY, 'Long', 'long_signal_01', None),
    (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 2.0),
    (RPCMessageType.ENTRY, 'Short', 'short_signal_01', 2.0)])
def test_send_msg_buy_notification_no_fiat(
        default_conf, mocker, message_type, enter, enter_signal, leverage) -> None:
    del default_conf['fiat_display_currency']
    default_conf['dry_run'] = False
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': message_type,
        'enter_tag': enter_signal,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'leverage': leverage,
        'open_rate': 1.099e-05,
        'order_type': 'limit',
        'direction': enter,
        'stake_amount': 0.01465333,
        'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC',
        'fiat_currency': None,
        'current_rate': 1.099e-05,
        'amount': 1333.3333333333335,
        'open_date': dt_now() - timedelta(hours=1)
    })

    leverage_text = f'*Leverage:* `{leverage}`\n' if leverage and leverage != 1.0 else ''
    assert msg_mock.call_args[0][0] == (
        f'\N{LARGE BLUE CIRCLE} *Binance:* {enter} ETH/BTC (#1)\n'
        f'*Enter Tag:* `{enter_signal}`\n'
        '*Amount:* `1333.33333333`\n'
        f'{leverage_text}'
        '*Open Rate:* `0.00001099`\n'
        '*Current Rate:* `0.00001099`\n'
        '*Total:* `(0.01465333 BTC)`'
    )


@pytest.mark.parametrize('direction,enter_signal,leverage', [
    ('Long', 'long_signal_01', None),
    ('Long', 'long_signal_01', 1.0),
    ('Long', 'long_signal_01', 5.0),
    ('Short', 'short_signal_01', 2.0),
])
def test_send_msg_sell_notification_no_fiat(
        default_conf, mocker, direction, enter_signal, leverage, time_machine) -> None:
    del default_conf['fiat_display_currency']
    time_machine.move_to('2022-05-02 00:00:00 +00:00', tick=False)
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': RPCMessageType.EXIT,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'gain': 'loss',
        'leverage': leverage,
        'direction': direction,
        'order_rate': 3.201e-05,
        'amount': 1333.3333333333335,
        'order_type': 'limit',
        'open_rate': 7.5e-05,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_ratio': -0.57405275,
        'stake_currency': 'ETH',
        'fiat_currency': 'USD',
        'enter_tag': enter_signal,
        'exit_reason': ExitType.STOP_LOSS.value,
        'open_date': dt_now() - timedelta(hours=2, minutes=35, seconds=3),
        'close_date': dt_now(),
    })

    leverage_text = f'*Leverage:* `{leverage}`\n' if leverage and leverage != 1.0 else ''
    assert msg_mock.call_args[0][0] == (
        '\N{WARNING SIGN} *Binance (dry):* Exiting KEY/ETH (#1)\n'
        '*Unrealized Profit:* `-57.41% (loss: -0.05746268 ETH)`\n'
        f'*Enter Tag:* `{enter_signal}`\n'
        '*Exit Reason:* `stop_loss`\n'
        f'*Direction:* `{direction}`\n'
        f'{leverage_text}'
        '*Amount:* `1333.33333333`\n'
        '*Open Rate:* `0.00007500`\n'
        '*Current Rate:* `0.00003201`\n'
        '*Exit Rate:* `0.00003201`\n'
        '*Duration:* `2:35:03 (155.1 min)`'
    )


@pytest.mark.parametrize('msg,expected', [
    ({'profit_percent': 20.1, 'exit_reason': 'roi'}, "\N{ROCKET}"),
    ({'profit_percent': 5.1, 'exit_reason': 'roi'}, "\N{ROCKET}"),
    ({'profit_percent': 2.56, 'exit_reason': 'roi'}, "\N{EIGHT SPOKED ASTERISK}"),
    ({'profit_percent': 1.0, 'exit_reason': 'roi'}, "\N{EIGHT SPOKED ASTERISK}"),
    ({'profit_percent': 0.0, 'exit_reason': 'roi'}, "\N{EIGHT SPOKED ASTERISK}"),
    ({'profit_percent': -5.0, 'exit_reason': 'stop_loss'}, "\N{WARNING SIGN}"),
    ({'profit_percent': -2.0, 'exit_reason': 'sell_signal'}, "\N{CROSS MARK}"),
])
def test__sell_emoji(default_conf, mocker, msg, expected):
    del default_conf['fiat_display_currency']

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)

    assert telegram._get_sell_emoji(msg) == expected


async def test_telegram__send_msg(default_conf, mocker, caplog) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    bot.send_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._app = MagicMock()
    telegram._app.bot = bot

    await telegram._send_msg('test')
    assert len(bot.method_calls) == 1

    # Test update
    query = MagicMock()
    await telegram._send_msg('test', callback_path="DeadBeef", query=query, reload_able=True)
    edit_message_text = telegram._app.bot.edit_message_text
    assert edit_message_text.call_count == 1
    assert "Updated: " in edit_message_text.call_args_list[0][1]['text']

    telegram._app.bot.edit_message_text = AsyncMock(side_effect=BadRequest("not modified"))
    await telegram._send_msg('test', callback_path="DeadBeef", query=query)
    assert telegram._app.bot.edit_message_text.call_count == 1
    assert not log_has_re(r"TelegramError: .*", caplog)

    telegram._app.bot.edit_message_text = AsyncMock(side_effect=BadRequest(""))
    await telegram._send_msg('test2', callback_path="DeadBeef", query=query)
    assert telegram._app.bot.edit_message_text.call_count == 1
    assert log_has_re(r"TelegramError: .*", caplog)

    telegram._app.bot.edit_message_text = AsyncMock(side_effect=TelegramError("DeadBEEF"))
    await telegram._send_msg('test3', callback_path="DeadBeef", query=query)

    assert log_has_re(r"TelegramError: DeadBEEF! Giving up.*", caplog)


async def test__send_msg_network_error(default_conf, mocker, caplog) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError('Oh snap'))
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._app = MagicMock()
    telegram._app.bot = bot

    telegram._config['telegram']['enabled'] = True
    await telegram._send_msg('test')

    # Bot should've tried to send it twice
    assert len(bot.method_calls) == 2
    assert log_has('Telegram NetworkError: Oh snap! Trying one more time.', caplog)


@pytest.mark.filterwarnings("ignore:.*ChatPermissions")
async def test__send_msg_keyboard(default_conf, mocker, caplog) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    bot.send_message = AsyncMock()
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)

    invalid_keys_list = [['/not_valid', '/profit'], ['/daily'], ['/alsoinvalid']]
    default_keys_list = [['/daily', '/profit', '/balance'],
                         ['/status', '/status table', '/performance'],
                         ['/count', '/start', '/stop', '/help']]
    default_keyboard = ReplyKeyboardMarkup(default_keys_list)

    custom_keys_list = [['/daily', '/stats', '/balance', '/profit', '/profit 5'],
                        ['/count', '/start', '/reload_config', '/help']]
    custom_keyboard = ReplyKeyboardMarkup(custom_keys_list)

    def init_telegram(freqtradebot):
        telegram = Telegram(rpc, default_conf)
        telegram._app = MagicMock()
        telegram._app.bot = bot
        return telegram

    # no keyboard in config -> default keyboard
    freqtradebot.config['telegram']['enabled'] = True
    telegram = init_telegram(freqtradebot)
    await telegram._send_msg('test')
    used_keyboard = bot.send_message.call_args[1]['reply_markup']
    assert used_keyboard == default_keyboard

    # invalid keyboard in config -> default keyboard
    freqtradebot.config['telegram']['enabled'] = True
    freqtradebot.config['telegram']['keyboard'] = invalid_keys_list
    err_msg = re.escape("config.telegram.keyboard: Invalid commands for custom "
                        "Telegram keyboard: ['/not_valid', '/alsoinvalid']"
                        "\nvalid commands are: ") + r"*"
    with pytest.raises(OperationalException, match=err_msg):
        telegram = init_telegram(freqtradebot)

    # valid keyboard in config -> custom keyboard
    freqtradebot.config['telegram']['enabled'] = True
    freqtradebot.config['telegram']['keyboard'] = custom_keys_list
    telegram = init_telegram(freqtradebot)
    await telegram._send_msg('test')
    used_keyboard = bot.send_message.call_args[1]['reply_markup']
    assert used_keyboard == custom_keyboard
    assert log_has("using custom keyboard from config.json: "
                   "[['/daily', '/stats', '/balance', '/profit', '/profit 5'], ['/count', "
                   "'/start', '/reload_config', '/help']]", caplog)


async def test_change_market_direction(default_conf, mocker, update) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.NONE
    context = MagicMock()
    context.args = ["long"]
    await telegram._changemarketdir(update, context)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.LONG
    context = MagicMock()
    context.args = ["invalid"]
    await telegram._changemarketdir(update, context)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.LONG
