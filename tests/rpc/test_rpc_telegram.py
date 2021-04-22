# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, unused-argument, invalid-name
# pragma pylint: disable=too-many-lines, too-many-arguments

import re
from datetime import datetime
from random import choice, randint
from string import ascii_uppercase
from unittest.mock import ANY, MagicMock

import arrow
import pytest
from telegram import Chat, Message, ReplyKeyboardMarkup, Update
from telegram.error import NetworkError

from freqtrade import __version__
from freqtrade.constants import CANCEL_REASON
from freqtrade.edge import PairInfo
from freqtrade.exceptions import OperationalException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.loggers import setup_logging
from freqtrade.persistence import PairLocks, Trade
from freqtrade.rpc import RPC, RPCMessageType
from freqtrade.rpc.telegram import Telegram, authorized_only
from freqtrade.state import RunMode, State
from freqtrade.strategy.interface import SellType
from tests.conftest import (create_mock_trades, get_patched_freqtradebot, log_has, patch_exchange,
                            patch_get_signal, patch_whitelist)


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


def get_telegram_testobject(mocker, default_conf, mock=True, ftbot=None):
    msg_mock = MagicMock()
    if mock:
        mocker.patch.multiple(
            'freqtrade.rpc.telegram.Telegram',
            _init=MagicMock(),
            _send_msg=msg_mock
        )
    if not ftbot:
        ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    telegram = Telegram(rpc, default_conf)

    return telegram, ftbot, msg_mock


def test_telegram__init__(default_conf, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    assert telegram._config == default_conf


def test_telegram_init(default_conf, mocker, caplog) -> None:
    start_polling = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock(return_value=start_polling))

    get_telegram_testobject(mocker, default_conf, mock=False)
    assert start_polling.call_count == 0

    # number of handles registered
    assert start_polling.dispatcher.add_handler.call_count > 0
    assert start_polling.start_polling.call_count == 1

    message_str = ("rpc.telegram is listening for following commands: [['status'], ['profit'], "
                   "['balance'], ['start'], ['stop'], ['forcesell'], ['forcebuy'], ['trades'], "
                   "['delete'], ['performance'], ['stats'], ['daily'], ['count'], ['locks'], "
                   "['unlock', 'delete_locks'], ['reload_config', 'reload_conf'], "
                   "['show_config', 'show_conf'], ['stopbuy'], "
                   "['whitelist'], ['blacklist'], ['logs'], ['edge'], ['help'], ['version']"
                   "]")

    assert log_has(message_str, caplog)


def test_cleanup(default_conf, mocker, ) -> None:
    updater_mock = MagicMock()
    updater_mock.stop = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Updater', updater_mock)

    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram.cleanup()
    assert telegram._updater.stop.call_count == 1


def test_authorized_only(default_conf, mocker, caplog, update) -> None:
    patch_exchange(mocker)

    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    patch_get_signal(bot, (True, False))
    dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state['called'] is True
    assert log_has('Executing handler: dummy_handler for chat_id: 0', caplog)
    assert not log_has('Rejected unauthorized message from: 0', caplog)
    assert not log_has('Exception occurred within Telegram module', caplog)


def test_authorized_only_unauthorized(default_conf, mocker, caplog) -> None:
    patch_exchange(mocker)
    chat = Chat(0xdeadbeef, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), datetime.utcnow(), chat)

    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    patch_get_signal(bot, (True, False))
    dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 3735928559', caplog)
    assert log_has('Rejected unauthorized message from: 3735928559', caplog)
    assert not log_has('Exception occurred within Telegram module', caplog)


def test_authorized_only_exception(default_conf, mocker, caplog, update) -> None:
    patch_exchange(mocker)

    default_conf['telegram']['enabled'] = False

    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)
    patch_get_signal(bot, (True, False))

    dummy.dummy_exception(update=update, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 0', caplog)
    assert not log_has('Rejected unauthorized message from: 0', caplog)
    assert log_has('Exception occurred within Telegram module', caplog)


def test_telegram_status(default_conf, update, mocker) -> None:
    update.message.chat.id = "123"
    default_conf['telegram']['enabled'] = False
    default_conf['telegram']['chat_id'] = "123"

    status_table = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Telegram._status_table', status_table)

    mocker.patch.multiple(
        'freqtrade.rpc.rpc.RPC',
        _rpc_trade_status=MagicMock(return_value=[{
            'trade_id': 1,
            'pair': 'ETH/BTC',
            'base_currency': 'BTC',
            'open_date': arrow.utcnow(),
            'close_date': None,
            'open_rate': 1.099e-05,
            'close_rate': None,
            'current_rate': 1.098e-05,
            'amount': 90.99181074,
            'stake_amount': 90.99181074,
            'close_profit_pct': None,
            'profit': -0.0059,
            'profit_pct': -0.59,
            'initial_stop_loss_abs': 1.098e-05,
            'stop_loss_abs': 1.099e-05,
            'sell_order_status': None,
            'initial_stop_loss_pct': -0.05,
            'stoploss_current_dist': 1e-08,
            'stoploss_current_dist_pct': -0.02,
            'stop_loss_pct': -0.01,
            'open_order': '(limit buy rem=0.00000000)',
            'is_open': True
        }]),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1

    context = MagicMock()
    # /status table
    context.args = ["table"]
    telegram._status(update=update, context=context)
    assert status_table.call_count == 1


def test_status_handle(default_conf, update, ticker, fee, mocker) -> None:
    default_conf['max_open_trades'] = 3
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    status_table = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _status_table=status_table,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot, (True, False))

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
    freqtradebot.enter_positions()
    # Trigger status while we have a fulfilled order for the open trade
    telegram._status(update=update, context=MagicMock())

    # close_rate should not be included in the message as the trade is not closed
    # and no line should be empty
    lines = msg_mock.call_args_list[0][0][0].split('\n')
    assert '' not in lines
    assert 'Close Rate' not in ''.join(lines)
    assert 'Close Profit' not in ''.join(lines)

    assert msg_mock.call_count == 3
    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'LTC/BTC' in msg_mock.call_args_list[1][0][0]

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2", "3"]

    telegram._status(update=update, context=context)

    lines = msg_mock.call_args_list[0][0][0].split('\n')
    assert '' not in lines
    assert 'Close Rate' not in ''.join(lines)
    assert 'Close Profit' not in ''.join(lines)

    assert msg_mock.call_count == 2
    assert 'LTC/BTC' in msg_mock.call_args_list[0][0][0]


def test_status_table_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )

    default_conf['stake_amount'] = 15.0

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot, (True, False))

    freqtradebot.state = State.STOPPED
    # Status table is also enabled when stopped
    telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()

    telegram._status_table(update=update, context=MagicMock())

    text = re.sub('</?pre>', '', msg_mock.call_args_list[-1][0][0])
    line = text.split("\n")
    fields = re.sub('[ ]+', ' ', line[2].strip()).split(' ')

    assert int(fields[0]) == 1
    assert 'ETH/BTC' in fields[1]
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
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot, (True, False))

    # Create some test data
    freqtradebot.enter_positions()
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
    context.args = []
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
    n = freqtradebot.enter_positions()
    assert n == 2

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
        fetch_ticker=ticker
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

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
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'No trades yet.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()
    trade = Trade.query.first()

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'No closed trade' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    assert ('∙ `-0.00000500 BTC (-0.50%) (-0.5 \N{GREEK CAPITAL LETTER SIGMA}%)`'
            in msg_mock.call_args_list[-1][0][0])
    msg_mock.reset_mock()

    # Update the ticker with a market going up
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', ticker_sell_up)
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*ROI:* Closed trades' in msg_mock.call_args_list[-1][0][0]
    assert ('∙ `0.00006217 BTC (6.20%) (6.2 \N{GREEK CAPITAL LETTER SIGMA}%)`'
            in msg_mock.call_args_list[-1][0][0])
    assert '∙ `0.933 USD`' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    assert ('∙ `0.00006217 BTC (6.20%) (6.2 \N{GREEK CAPITAL LETTER SIGMA}%)`'
            in msg_mock.call_args_list[-1][0][0])
    assert '∙ `0.933 USD`' in msg_mock.call_args_list[-1][0][0]

    assert '*Best Performing:* `ETH/BTC: 6.20%`' in msg_mock.call_args_list[-1][0][0]


def test_telegram_stats(default_conf, update, ticker, ticker_sell_up, fee,
                        limit_buy_order, limit_sell_order, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram._stats(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    # assert 'No trades yet.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    create_mock_trades(fee)

    telegram._stats(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Sell Reason' in msg_mock.call_args_list[-1][0][0]
    assert 'ROI' in msg_mock.call_args_list[-1][0][0]
    assert 'Avg. Duration' in msg_mock.call_args_list[-1][0][0]
    msg_mock.reset_mock()


def test_telegram_balance_handle(default_conf, update, mocker, rpc_balance, tickers) -> None:
    default_conf['dry_run'] = False
    mocker.patch('freqtrade.exchange.Exchange.get_balances', return_value=rpc_balance)
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)
    mocker.patch('freqtrade.exchange.Exchange.get_valid_pair_combination',
                 side_effect=lambda a, b: f"{a}/{b}")

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert '*BTC:*' in result
    assert '*ETH:*' not in result
    assert '*USDT:*' in result
    assert '*EUR:*' in result
    assert 'Balance:' in result
    assert 'Est. BTC:' in result
    assert 'BTC: 12.00000000' in result
    assert '*XRP:* not showing <0.0001 BTC amount' in result


def test_balance_handle_empty_response(default_conf, update, mocker) -> None:
    default_conf['dry_run'] = False
    mocker.patch('freqtrade.exchange.Exchange.get_balances', return_value={})

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    freqtradebot.config['dry_run'] = False
    telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert 'All balances are zero.' in result


def test_balance_handle_empty_response_dry(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_balances', return_value={})

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert "*Warning:* Simulated balances in Dry Mode." in result
    assert "Starting capital: `1000` BTC" in result


def test_balance_handle_too_large_response(default_conf, update, mocker) -> None:
    balances = []
    for i in range(100):
        curr = choice(ascii_uppercase) + choice(ascii_uppercase) + choice(ascii_uppercase)
        balances.append({
            'currency': curr,
            'free': 1.0,
            'used': 0.5,
            'balance': i,
            'est_stake': 1,
            'stake': 'BTC',
        })
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_balance', return_value={
        'currencies': balances,
        'total': 100.0,
        'symbol': 100.0,
        'value': 1000.0,
    })

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    telegram._balance(update=update, context=MagicMock())
    assert msg_mock.call_count > 1
    # Test if wrap happens around 4000 -
    # and each single currency-output is around 120 characters long so we need
    # an offset to avoid random test failures
    assert len(msg_mock.call_args_list[0][0][0]) < 4096
    assert len(msg_mock.call_args_list[0][0][0]) > (4096 - 120)


def test_start_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1


def test_start_handle_already_running(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1
    assert 'already running' in msg_mock.call_args_list[0][0][0]


def test_stop_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'stopping trader' in msg_mock.call_args_list[0][0][0]


def test_stop_handle_already_stopped(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'already stopped' in msg_mock.call_args_list[0][0][0]


def test_stopbuy_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    assert freqtradebot.config['max_open_trades'] != 0
    telegram._stopbuy(update=update, context=MagicMock())
    assert freqtradebot.config['max_open_trades'] == 0
    assert msg_mock.call_count == 1
    assert 'No more buy will occur from now. Run /reload_config to reset.' \
        in msg_mock.call_args_list[0][0][0]


def test_reload_config_handle(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    telegram._reload_config(update=update, context=MagicMock())
    assert freqtradebot.state == State.RELOAD_CONFIG
    assert msg_mock.call_count == 1
    assert 'Reloading config' in msg_mock.call_args_list[0][0][0]


def test_telegram_forcesell_handle(default_conf, update, ticker, fee,
                                   ticker_sell_up, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    msg_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    # Create some test data
    freqtradebot.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', ticker_sell_up)

    # /forcesell 1
    context = MagicMock()
    context.args = ["1"]
    telegram._forcesell(update=update, context=context)

    assert msg_mock.call_count == 4
    last_msg = msg_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'limit': 1.173e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.173e-05,
        'profit_amount': 6.314e-05,
        'profit_ratio': 0.0629778,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == last_msg


def test_telegram_forcesell_down_handle(default_conf, update, ticker, fee,
                                        ticker_sell_down, mocker) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    msg_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    # Create some test data
    freqtradebot.enter_positions()

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_down
    )

    trade = Trade.query.first()
    assert trade

    # /forcesell 1
    context = MagicMock()
    context.args = ["1"]
    telegram._forcesell(update=update, context=context)

    assert msg_mock.call_count == 4

    last_msg = msg_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.043e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.043e-05,
        'profit_amount': -5.497e-05,
        'profit_ratio': -0.05482878,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == last_msg


def test_forcesell_all_handle(default_conf, update, ticker, fee, mocker) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    msg_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    default_conf['max_open_trades'] = 4
    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    # Create some test data
    freqtradebot.enter_positions()
    msg_mock.reset_mock()

    # /forcesell all
    context = MagicMock()
    context.args = ["all"]
    telegram._forcesell(update=update, context=context)

    # Called for each trade 4 times
    assert msg_mock.call_count == 12
    msg = msg_mock.call_args_list[2][0][0]
    assert {
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.099e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.099e-05,
        'profit_amount': -4.09e-06,
        'profit_ratio': -0.00408133,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.FORCE_SELL.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == msg


def test_forcesell_handle_invalid(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

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
    assert "You must specify a trade-id or 'all'." in msg_mock.call_args_list[0][0][0]

    # Invalid argument
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /forcesell 123456
    context = MagicMock()
    context.args = ["123456"]
    telegram._forcesell(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'invalid argument' in msg_mock.call_args_list[0][0][0]


def test_forcebuy_handle(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.RPC._rpc_forcebuy', fbuy_mock)

    telegram, freqtradebot, _ = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

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


def test_forcebuy_handle_exception(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    update.message.text = '/forcebuy ETH/Nonepair'
    telegram._forcebuy(update=update, context=MagicMock())

    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][0][0] == 'Forcebuy not enabled.'


def test_performance_handle(default_conf, update, ticker, fee,
                            limit_buy_order, limit_sell_order, mocker) -> None:

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    # Create some test data
    freqtradebot.enter_positions()
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
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    freqtradebot.state = State.STOPPED
    telegram._count(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING

    # Create some test data
    freqtradebot.enter_positions()
    msg_mock.reset_mock()
    telegram._count(update=update, context=MagicMock())

    msg = ('<pre>  current    max    total stake\n---------  -----  -------------\n'
           '        1      {}          {}</pre>').format(
            default_conf['max_open_trades'],
            default_conf['stake_amount']
        )
    assert msg in msg_mock.call_args_list[0][0][0]


def test_telegram_lock_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))

    PairLocks.lock_pair('ETH/BTC', arrow.utcnow().shift(minutes=4).datetime, 'randreason')
    PairLocks.lock_pair('XRP/BTC', arrow.utcnow().shift(minutes=20).datetime, 'deadbeef')

    telegram._locks(update=update, context=MagicMock())

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
    telegram._delete_locks(update=update, context=context)

    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'randreason' in msg_mock.call_args_list[0][0][0]
    assert 'XRP/BTC' not in msg_mock.call_args_list[0][0][0]
    assert 'deadbeef' not in msg_mock.call_args_list[0][0][0]


def test_whitelist_static(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert ("Using whitelist `['StaticPairList']` with 4 pairs\n"
            "`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`" in msg_mock.call_args_list[0][0][0])


def test_whitelist_dynamic(default_conf, update, mocker) -> None:
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    default_conf['pairlists'] = [{'method': 'VolumePairList',
                                 'number_assets': 4
                                  }]
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert ("Using whitelist `['VolumePairList']` with 4 pairs\n"
            "`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`" in msg_mock.call_args_list[0][0][0])


def test_blacklist_static(default_conf, update, mocker) -> None:

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

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

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["XRP/.*"]
    telegram._blacklist(update=update, context=context)
    assert msg_mock.call_count == 1

    assert ("Blacklist contains 4 pairs\n`DOGE/BTC, HOT/BTC, ETH/BTC, XRP/.*`"
            in msg_mock.call_args_list[0][0][0])
    assert freqtradebot.pairlists.blacklist == ["DOGE/BTC", "HOT/BTC", "ETH/BTC", "XRP/.*"]


def test_telegram_logs(default_conf, update, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.rpc.telegram.Telegram',
        _init=MagicMock(),
    )
    setup_logging(default_conf)

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    context = MagicMock()
    context.args = []
    telegram._logs(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "freqtrade\\.rpc\\.telegram" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    context.args = ["1"]
    telegram._logs(update=update, context=context)
    assert msg_mock.call_count == 1

    msg_mock.reset_mock()
    # Test with changed MaxMessageLength
    mocker.patch('freqtrade.rpc.telegram.MAX_TELEGRAM_MESSAGE_LENGTH', 200)
    context = MagicMock()
    context.args = []
    telegram._logs(update=update, context=context)
    # Called at least 2 times. Exact times will change with unrelated changes to setup messages
    # Therefore we don't test for this explicitly.
    assert msg_mock.call_count >= 2


def test_edge_disabled(default_conf, update, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "Edge is not enabled." in msg_mock.call_args_list[0][0][0]


def test_edge_enabled(edge_conf, update, mocker) -> None:
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'E/F': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
        }
    ))

    telegram, _, msg_mock = get_telegram_testobject(mocker, edge_conf)

    telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '<b>Edge only validated following pairs:</b>\n<pre>' in msg_mock.call_args_list[0][0][0]
    assert 'Pair      Winrate    Expectancy    Stoploss' in msg_mock.call_args_list[0][0][0]


def test_telegram_trades(mocker, update, default_conf, fee):

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    context = MagicMock()
    context.args = []

    telegram._trades(update=update, context=context)
    assert "<b>0 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    context.args = ['hello']
    telegram._trades(update=update, context=context)
    assert "<b>0 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    create_mock_trades(fee)

    context = MagicMock()
    context.args = [5]
    telegram._trades(update=update, context=context)
    msg_mock.call_count == 1
    assert "2 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "Profit (" in msg_mock.call_args_list[0][0][0]
    assert "Close Date" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" in msg_mock.call_args_list[0][0][0]
    assert bool(re.search(r"just now[ ]*XRP\/BTC \(#3\)  1.00% \(",
                msg_mock.call_args_list[0][0][0]))


def test_telegram_delete_trade(mocker, update, default_conf, fee):

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    telegram._delete_trade(update=update, context=context)
    assert "Trade-id not set." in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee)

    context = MagicMock()
    context.args = [1]
    telegram._delete_trade(update=update, context=context)
    msg_mock.call_count == 1
    assert "Deleted trade 1." in msg_mock.call_args_list[0][0][0]
    assert "Please make sure to take care of this asset" in msg_mock.call_args_list[0][0][0]


def test_help_handle(default_conf, update, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram._help(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*/help:* `This help message`' in msg_mock.call_args_list[0][0][0]


def test_version_handle(default_conf, update, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Version:* `{}`'.format(__version__) in msg_mock.call_args_list[0][0][0]


def test_show_config_handle(default_conf, update, mocker) -> None:

    default_conf['runmode'] = RunMode.DRY_RUN

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Mode:* `{}`'.format('Dry-run') in msg_mock.call_args_list[0][0][0]
    assert '*Exchange:* `binance`' in msg_mock.call_args_list[0][0][0]
    assert '*Strategy:* `DefaultStrategy`' in msg_mock.call_args_list[0][0][0]
    assert '*Stoploss:* `-0.1`' in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    freqtradebot.config['trailing_stop'] = True
    telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Mode:* `{}`'.format('Dry-run') in msg_mock.call_args_list[0][0][0]
    assert '*Exchange:* `binance`' in msg_mock.call_args_list[0][0][0]
    assert '*Strategy:* `DefaultStrategy`' in msg_mock.call_args_list[0][0][0]
    assert '*Initial Stoploss:* `-0.1`' in msg_mock.call_args_list[0][0][0]


def test_send_msg_buy_notification(default_conf, mocker, caplog) -> None:

    msg = {
        'type': RPCMessageType.BUY,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'limit': 1.099e-05,
        'order_type': 'limit',
        'stake_amount': 0.001,
        'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'current_rate': 1.099e-05,
        'amount': 1333.3333333333335,
        'open_date': arrow.utcnow().shift(hours=-1)
    }
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(msg)
    assert msg_mock.call_args[0][0] \
        == '\N{LARGE BLUE CIRCLE} *Binance:* Buying ETH/BTC (#1)\n' \
           '*Amount:* `1333.33333333`\n' \
           '*Open Rate:* `0.00001099`\n' \
           '*Current Rate:* `0.00001099`\n' \
           '*Total:* `(0.00100000 BTC, 12.345 USD)`'

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


def test_send_msg_buy_cancel_notification(default_conf, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': RPCMessageType.BUY_CANCEL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'reason': CANCEL_REASON['TIMEOUT']
    })
    assert (msg_mock.call_args[0][0] == '\N{WARNING SIGN} *Binance:* '
            'Cancelling open buy Order for ETH/BTC (#1). '
            'Reason: cancelled due to timeout.')


def test_send_msg_buy_fill_notification(default_conf, mocker) -> None:

    default_conf['telegram']['notification_settings']['buy_fill'] = 'on'
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': RPCMessageType.BUY_FILL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'open_rate': 200,
        'stake_amount': 100,
        'amount': 0.5,
        'open_date': arrow.utcnow().datetime
    })
    assert (msg_mock.call_args[0][0] == '\N{LARGE CIRCLE} *Binance:* '
            'Buy order for ETH/USDT (#1) filled for 200.')


def test_send_msg_sell_notification(default_conf, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    old_convamount = telegram._rpc._fiat_converter.convert_amount
    telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
    telegram.send_msg({
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'gain': 'loss',
        'limit': 3.201e-05,
        'amount': 1333.3333333333335,
        'order_type': 'market',
        'open_rate': 7.5e-05,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_ratio': -0.57405275,
        'stake_currency': 'ETH',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value,
        'open_date': arrow.utcnow().shift(hours=-1),
        'close_date': arrow.utcnow(),
    })
    assert msg_mock.call_args[0][0] \
        == ('\N{WARNING SIGN} *Binance:* Selling KEY/ETH (#1)\n'
            '*Amount:* `1333.33333333`\n'
            '*Open Rate:* `0.00007500`\n'
            '*Current Rate:* `0.00003201`\n'
            '*Close Rate:* `0.00003201`\n'
            '*Sell Reason:* `stop_loss`\n'
            '*Duration:* `1:00:00 (60.0 min)`\n'
            '*Profit:* `-57.41%` `(loss: -0.05746268 ETH / -24.812 USD)`')

    msg_mock.reset_mock()
    telegram.send_msg({
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'gain': 'loss',
        'limit': 3.201e-05,
        'amount': 1333.3333333333335,
        'order_type': 'market',
        'open_rate': 7.5e-05,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_ratio': -0.57405275,
        'stake_currency': 'ETH',
        'sell_reason': SellType.STOP_LOSS.value,
        'open_date': arrow.utcnow().shift(days=-1, hours=-2, minutes=-30),
        'close_date': arrow.utcnow(),
    })
    assert msg_mock.call_args[0][0] \
        == ('\N{WARNING SIGN} *Binance:* Selling KEY/ETH (#1)\n'
            '*Amount:* `1333.33333333`\n'
            '*Open Rate:* `0.00007500`\n'
            '*Current Rate:* `0.00003201`\n'
            '*Close Rate:* `0.00003201`\n'
            '*Sell Reason:* `stop_loss`\n'
            '*Duration:* `1 day, 2:30:00 (1590.0 min)`\n'
            '*Profit:* `-57.41%`')
    # Reset singleton function to avoid random breaks
    telegram._rpc._fiat_converter.convert_amount = old_convamount


def test_send_msg_sell_cancel_notification(default_conf, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    old_convamount = telegram._rpc._fiat_converter.convert_amount
    telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
    telegram.send_msg({
        'type': RPCMessageType.SELL_CANCEL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'reason': 'Cancelled on exchange'
    })
    assert msg_mock.call_args[0][0] \
        == ('\N{WARNING SIGN} *Binance:* Cancelling open sell Order for KEY/ETH (#1).'
            ' Reason: Cancelled on exchange.')

    msg_mock.reset_mock()
    telegram.send_msg({
        'type': RPCMessageType.SELL_CANCEL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'reason': 'timeout'
    })
    assert msg_mock.call_args[0][0] \
        == ('\N{WARNING SIGN} *Binance:* Cancelling open sell Order for KEY/ETH (#1).'
            ' Reason: timeout.')
    # Reset singleton function to avoid random breaks
    telegram._rpc._fiat_converter.convert_amount = old_convamount


def test_send_msg_sell_fill_notification(default_conf, mocker) -> None:

    default_conf['telegram']['notification_settings']['sell_fill'] = 'on'
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': RPCMessageType.SELL_FILL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'gain': 'loss',
        'limit': 3.201e-05,
        'amount': 0.1,
        'order_type': 'market',
        'open_rate': 500,
        'close_rate': 550,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_ratio': -0.57405275,
        'stake_currency': 'ETH',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value,
        'open_date': arrow.utcnow().shift(hours=-1),
        'close_date': arrow.utcnow(),
    })
    assert msg_mock.call_args[0][0] \
        == ('\N{LARGE CIRCLE} *Binance:* Sell order for ETH/USDT (#1) filled for 550.')


def test_send_msg_status_notification(default_conf, mocker) -> None:

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': RPCMessageType.STATUS,
        'status': 'running'
    })
    assert msg_mock.call_args[0][0] == '*Status:* `running`'


def test_warning_notification(default_conf, mocker) -> None:
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


def test_send_msg_unknown_type(default_conf, mocker) -> None:
    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    with pytest.raises(NotImplementedError, match=r'Unknown message type: None'):
        telegram.send_msg({
            'type': None,
        })


def test_send_msg_buy_notification_no_fiat(default_conf, mocker) -> None:
    del default_conf['fiat_display_currency']
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': RPCMessageType.BUY,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'limit': 1.099e-05,
        'order_type': 'limit',
        'stake_amount': 0.001,
        'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC',
        'fiat_currency': None,
        'current_rate': 1.099e-05,
        'amount': 1333.3333333333335,
        'open_date': arrow.utcnow().shift(hours=-1)
    })
    assert msg_mock.call_args[0][0] == ('\N{LARGE BLUE CIRCLE} *Binance:* Buying ETH/BTC (#1)\n'
                                        '*Amount:* `1333.33333333`\n'
                                        '*Open Rate:* `0.00001099`\n'
                                        '*Current Rate:* `0.00001099`\n'
                                        '*Total:* `(0.00100000 BTC)`')


def test_send_msg_sell_notification_no_fiat(default_conf, mocker) -> None:
    del default_conf['fiat_display_currency']
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg({
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'KEY/ETH',
        'gain': 'loss',
        'limit': 3.201e-05,
        'amount': 1333.3333333333335,
        'order_type': 'limit',
        'open_rate': 7.5e-05,
        'current_rate': 3.201e-05,
        'profit_amount': -0.05746268,
        'profit_ratio': -0.57405275,
        'stake_currency': 'ETH',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value,
        'open_date': arrow.utcnow().shift(hours=-2, minutes=-35, seconds=-3),
        'close_date': arrow.utcnow(),
    })
    assert msg_mock.call_args[0][0] == ('\N{WARNING SIGN} *Binance:* Selling KEY/ETH (#1)\n'
                                        '*Amount:* `1333.33333333`\n'
                                        '*Open Rate:* `0.00007500`\n'
                                        '*Current Rate:* `0.00003201`\n'
                                        '*Close Rate:* `0.00003201`\n'
                                        '*Sell Reason:* `stop_loss`\n'
                                        '*Duration:* `2:35:03 (155.1 min)`\n'
                                        '*Profit:* `-57.41%`')


@pytest.mark.parametrize('msg,expected', [
    ({'profit_percent': 20.1, 'sell_reason': 'roi'}, "\N{ROCKET}"),
    ({'profit_percent': 5.1, 'sell_reason': 'roi'}, "\N{ROCKET}"),
    ({'profit_percent': 2.56, 'sell_reason': 'roi'}, "\N{EIGHT SPOKED ASTERISK}"),
    ({'profit_percent': 1.0, 'sell_reason': 'roi'}, "\N{EIGHT SPOKED ASTERISK}"),
    ({'profit_percent': 0.0, 'sell_reason': 'roi'}, "\N{EIGHT SPOKED ASTERISK}"),
    ({'profit_percent': -5.0, 'sell_reason': 'stop_loss'}, "\N{WARNING SIGN}"),
    ({'profit_percent': -2.0, 'sell_reason': 'sell_signal'}, "\N{CROSS MARK}"),
])
def test__sell_emoji(default_conf, mocker, msg, expected):
    del default_conf['fiat_display_currency']

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)

    assert telegram._get_sell_emoji(msg) == expected


def test__send_msg(default_conf, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._updater = MagicMock()
    telegram._updater.bot = bot

    telegram._config['telegram']['enabled'] = True
    telegram._send_msg('test')
    assert len(bot.method_calls) == 1


def test__send_msg_network_error(default_conf, mocker, caplog) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError('Oh snap'))
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._updater = MagicMock()
    telegram._updater.bot = bot

    telegram._config['telegram']['enabled'] = True
    telegram._send_msg('test')

    # Bot should've tried to send it twice
    assert len(bot.method_calls) == 2
    assert log_has('Telegram NetworkError: Oh snap! Trying one more time.', caplog)


def test__send_msg_keyboard(default_conf, mocker, caplog) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot = MagicMock()
    bot.send_message = MagicMock()
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)

    invalid_keys_list = [['/not_valid', '/profit'], ['/daily'], ['/alsoinvalid']]
    default_keys_list = [['/daily', '/profit', '/balance'],
                         ['/status', '/status table', '/performance'],
                         ['/count', '/start', '/stop', '/help']]
    default_keyboard = ReplyKeyboardMarkup(default_keys_list)

    custom_keys_list = [['/daily', '/stats', '/balance', '/profit'],
                        ['/count', '/start', '/reload_config', '/help']]
    custom_keyboard = ReplyKeyboardMarkup(custom_keys_list)

    def init_telegram(freqtradebot):
        telegram = Telegram(rpc, default_conf)
        telegram._updater = MagicMock()
        telegram._updater.bot = bot
        return telegram

    # no keyboard in config -> default keyboard
    freqtradebot.config['telegram']['enabled'] = True
    telegram = init_telegram(freqtradebot)
    telegram._send_msg('test')
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
    telegram._send_msg('test')
    used_keyboard = bot.send_message.call_args[1]['reply_markup']
    assert used_keyboard == custom_keyboard
    assert log_has("using custom keyboard from config.json: "
                   "[['/daily', '/stats', '/balance', '/profit'], ['/count', "
                   "'/start', '/reload_config', '/help']]", caplog)
