# pragma pylint: disable=missing-docstring, too-many-arguments, too-many-ancestors, C0103
# pragma pylint: disable=unused-argument
import re
from datetime import datetime
from random import randint
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from telegram import Update, Message, Chat
from telegram.error import NetworkError

from freqtrade import __version__
from freqtrade.main import init, create_trade
from freqtrade.misc import update_state, State, get_state
from freqtrade.persistence import Trade
from freqtrade.rpc import telegram
from freqtrade.rpc.telegram import authorized_only, is_enabled, send_msg, _status, _status_table, \
    _profit, _forcesell, _performance, _daily, _count, _start, _stop, _balance, _version, _help

import freqtrade.rpc.telegram as tg


def test_is_enabled(default_conf, mocker):
    mocker.patch.dict('freqtrade.rpc.telegram._CONF', default_conf)
    default_conf['telegram']['enabled'] = False
    assert is_enabled() is False


def test_init_disabled(default_conf, mocker):
    mocker.patch.dict('freqtrade.rpc.telegram._CONF', default_conf)
    default_conf['telegram']['enabled'] = False
    telegram.init(default_conf)


def test_authorized_only(default_conf, mocker):
    mocker.patch.dict('freqtrade.rpc.telegram._CONF', default_conf)

    chat = Chat(0, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), chat)
    state = {'called': False}

    @authorized_only
    def dummy_handler(*args, **kwargs) -> None:
        state['called'] = True

    dummy_handler(MagicMock(), update)
    assert state['called'] is True


def test_authorized_only_unauthorized(default_conf, mocker):
    mocker.patch.dict('freqtrade.rpc.telegram._CONF', default_conf)

    chat = Chat(0xdeadbeef, 0)
    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), chat)
    state = {'called': False}

    @authorized_only
    def dummy_handler(*args, **kwargs) -> None:
        state['called'] = True

    dummy_handler(MagicMock(), update)
    assert state['called'] is False


def test_authorized_only_exception(default_conf, mocker):
    mocker.patch.dict('freqtrade.rpc.telegram._CONF', default_conf)

    update = Update(randint(1, 100))
    update.message = Message(randint(1, 100), 0, datetime.utcnow(), Chat(0, 0))

    @authorized_only
    def dummy_handler(*args, **kwargs) -> None:
        raise Exception('test')

    dummy_handler(MagicMock(), update)


def test_status_handle(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    msg_mock = MagicMock()
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    init(default_conf, create_engine('sqlite://'))

    update_state(State.STOPPED)
    _status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'trader is not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    update_state(State.RUNNING)
    _status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    create_trade(0.001, int(default_conf['ticker_interval']))
    # Trigger status while we have a fulfilled order for the open trade
    _status(bot=MagicMock(), update=update)

    assert msg_mock.call_count == 1
    assert '[BTC_ETH]' in msg_mock.call_args_list[0][0][0]


def test_status_table_handle(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    msg_mock = MagicMock()
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple(
        'freqtrade.rpc.telegram',
        _CONF=default_conf,
        init=MagicMock(),
        send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_order_id'))
    init(default_conf, create_engine('sqlite://'))
    update_state(State.STOPPED)
    _status_table(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'trader is not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    update_state(State.RUNNING)
    _status_table(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no active order' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    create_trade(15.0, int(default_conf['ticker_interval']))

    _status_table(bot=MagicMock(), update=update)

    text = re.sub('</?pre>', '', msg_mock.call_args_list[-1][0][0])
    line = text.split("\n")
    fields = re.sub('[ ]+', ' ', line[2].strip()).split(' ')

    assert int(fields[0]) == 1
    assert fields[1] == 'BTC_ETH'
    assert msg_mock.call_count == 1


def test_profit_handle(
        default_conf, update, ticker, ticker_sell_up, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    msg_mock = MagicMock()
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch.multiple('freqtrade.fiat_convert.Market',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}))
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    init(default_conf, create_engine('sqlite://'))

    _profit(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no closed trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    create_trade(0.001, int(default_conf['ticker_interval']))
    trade = Trade.query.first()

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    _profit(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'no closed trade' in msg_mock.call_args_list[-1][0][0]
    msg_mock.reset_mock()

    # Update the ticker with a market going up
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker_sell_up)
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    _profit(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert '*ROI:* Close trades' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.00006217 BTC (6.20%)`' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.933 USD`' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.00006217 BTC (6.20%)`' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `0.933 USD`' in msg_mock.call_args_list[-1][0][0]

    assert '*Best Performing:* `BTC_ETH: 6.20%`' in msg_mock.call_args_list[-1][0][0]


def test_forcesell_handle(default_conf, update, ticker, ticker_sell_up, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    rpc_mock = mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    init(default_conf, create_engine('sqlite://'))

    # Create some test data
    create_trade(0.001, int(default_conf['ticker_interval']))

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker_sell_up)

    update.message.text = '/forcesell 1'
    _forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 2
    assert 'Selling' in rpc_mock.call_args_list[-1][0][0]
    assert '[BTC_ETH]' in rpc_mock.call_args_list[-1][0][0]
    assert 'Amount' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001172' in rpc_mock.call_args_list[-1][0][0]
    assert 'profit: 6.11%, 0.00006126' in rpc_mock.call_args_list[-1][0][0]
    assert '0.919 USD' in rpc_mock.call_args_list[-1][0][0]


def test_forcesell_down_handle(default_conf, update, ticker, ticker_sell_down, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    rpc_mock = mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    init(default_conf, create_engine('sqlite://'))

    # Create some test data
    create_trade(0.001, int(default_conf['ticker_interval']))

    # Decrease the price and sell it
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker_sell_down)

    trade = Trade.query.first()
    assert trade

    update.message.text = '/forcesell 1'
    _forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 2
    assert 'Selling' in rpc_mock.call_args_list[-1][0][0]
    assert '[BTC_ETH]' in rpc_mock.call_args_list[-1][0][0]
    assert 'Amount' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001044' in rpc_mock.call_args_list[-1][0][0]
    assert 'loss: -5.48%, -0.00005492' in rpc_mock.call_args_list[-1][0][0]
    assert '-0.824 USD' in rpc_mock.call_args_list[-1][0][0]


def test_forcesell_all_handle(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    rpc_mock = mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    init(default_conf, create_engine('sqlite://'))

    # Create some test data
    for _ in range(4):
        create_trade(0.001, int(default_conf['ticker_interval']))
    rpc_mock.reset_mock()

    update.message.text = '/forcesell all'
    _forcesell(bot=MagicMock(), update=update)

    assert rpc_mock.call_count == 4
    for args in rpc_mock.call_args_list:
        assert '0.00001098' in args[0][0]
        assert 'loss: -0.59%, -0.00000591 BTC' in args[0][0]
        assert '-0.089 USD' in args[0][0]


def test_forcesell_handle_invalid(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, True))
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock())
    init(default_conf, create_engine('sqlite://'))

    # Trader is not running
    update_state(State.STOPPED)
    update.message.text = '/forcesell 1'
    _forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]

    # No argument
    msg_mock.reset_mock()
    update_state(State.RUNNING)
    update.message.text = '/forcesell'
    _forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'Invalid argument' in msg_mock.call_args_list[0][0][0]

    # Invalid argument
    msg_mock.reset_mock()
    update_state(State.RUNNING)
    update.message.text = '/forcesell 123456'
    _forcesell(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'Invalid argument.' in msg_mock.call_args_list[0][0][0]


def test_performance_handle(
        default_conf, update, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    msg_mock = MagicMock()
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    init(default_conf, create_engine('sqlite://'))

    # Create some test data
    create_trade(0.001, int(default_conf['ticker_interval']))
    trade = Trade.query.first()
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False
    _performance(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'Performance' in msg_mock.call_args_list[0][0][0]
    assert '<code>BTC_ETH\t6.20% (1)</code>' in msg_mock.call_args_list[0][0][0]


def test_daily_handle(default_conf, update, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    msg_mock = MagicMock()
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch.multiple('freqtrade.fiat_convert.Market',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}))
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    init(default_conf, create_engine('sqlite://'))

    # Create some test data
    create_trade(0.001, int(default_conf['ticker_interval']))
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
    _daily(bot=MagicMock(), update=update)
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
    create_trade(0.001, int(default_conf['ticker_interval']))
    create_trade(0.001, int(default_conf['ticker_interval']))

    trades = Trade.query.all()
    for trade in trades:
        trade.update(limit_buy_order)
        trade.update(limit_sell_order)
        trade.close_date = datetime.utcnow()
        trade.is_open = False

    update.message.text = '/daily 1'

    _daily(bot=MagicMock(), update=update)
    assert str('  0.00018651 BTC') in msg_mock.call_args_list[0][0][0]
    assert str('  2.798 USD') in msg_mock.call_args_list[0][0][0]
    assert str('  3 trades') in msg_mock.call_args_list[0][0][0]


def test_daily_wrong_input(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    msg_mock = MagicMock()
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch.multiple('freqtrade.fiat_convert.Market',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}))
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    init(default_conf, create_engine('sqlite://'))

    # Try invalid data
    msg_mock.reset_mock()
    update_state(State.RUNNING)
    update.message.text = '/daily -2'
    _daily(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    update_state(State.RUNNING)
    update.message.text = '/daily today'
    _daily(bot=MagicMock(), update=update)
    assert str('Daily Profit over the last 7 days') in msg_mock.call_args_list[0][0][0]


def test_count_handle(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.rpc.telegram',
        _CONF=default_conf,
        init=MagicMock(),
        send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_order_id'))
    init(default_conf, create_engine('sqlite://'))
    update_state(State.STOPPED)
    _count(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    update_state(State.RUNNING)

    # Create some test data
    create_trade(0.001, int(default_conf['ticker_interval']))
    msg_mock.reset_mock()
    _count(bot=MagicMock(), update=update)

    msg = '<pre>  current    max\n---------  -----\n        1      {}</pre>'.format(
        default_conf['max_open_trades']
    )
    assert msg in msg_mock.call_args_list[0][0][0]


def test_performance_handle_invalid(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, True))
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock())
    init(default_conf, create_engine('sqlite://'))

    # Trader is not running
    update_state(State.STOPPED)
    _performance(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]


def test_start_handle(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          _CONF=default_conf,
                          init=MagicMock())
    init(default_conf, create_engine('sqlite://'))
    update_state(State.STOPPED)
    assert get_state() == State.STOPPED
    _start(bot=MagicMock(), update=update)
    assert get_state() == State.RUNNING
    assert msg_mock.call_count == 0


def test_start_handle_already_running(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          _CONF=default_conf,
                          init=MagicMock())
    init(default_conf, create_engine('sqlite://'))
    update_state(State.RUNNING)
    assert get_state() == State.RUNNING
    _start(bot=MagicMock(), update=update)
    assert get_state() == State.RUNNING
    assert msg_mock.call_count == 1
    assert 'already running' in msg_mock.call_args_list[0][0][0]


def test_stop_handle(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          _CONF=default_conf,
                          init=MagicMock())
    init(default_conf, create_engine('sqlite://'))
    update_state(State.RUNNING)
    assert get_state() == State.RUNNING
    _stop(bot=MagicMock(), update=update)
    assert get_state() == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'Stopping trader' in msg_mock.call_args_list[0][0][0]


def test_stop_handle_already_stopped(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          _CONF=default_conf,
                          init=MagicMock())
    init(default_conf, create_engine('sqlite://'))
    update_state(State.STOPPED)
    assert get_state() == State.STOPPED
    _stop(bot=MagicMock(), update=update)
    assert get_state() == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'already stopped' in msg_mock.call_args_list[0][0][0]


def test_telegram_balance_handle(default_conf, update, mocker):
    mock_balance = [{
        'Currency': 'BTC',
        'Balance': 10.0,
        'Available': 12.0,
        'Pending': 0.0,
        'CryptoAddress': 'XXXX',
    }, {
        'Currency': 'ETH',
        'Balance': 0.0,
        'Available': 0.0,
        'Pending': 0.0,
        'CryptoAddress': 'XXXX',
    }, {
        'Currency': 'USDT',
        'Balance': 10000.0,
        'Available': 0.0,
        'Pending': 0.0,
        'CryptoAddress': 'XXXX',
    }, {
        'Currency': 'LTC',
        'Balance': 10.0,
        'Available': 10.0,
        'Pending': 0.0,
        'CryptoAddress': 'XXXX',
    }]

    def mock_ticker(symbol, refresh):
        if symbol == 'USDT_BTC':
            return {
                'bid': 10000.00,
                'ask': 10000.00,
                'last': 10000.00,
            }
        else:
            return {
                'bid': 0.1,
                'ask': 0.1,
                'last': 0.1,
            }

    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_balances=MagicMock(return_value=mock_balance))
    mocker.patch.multiple('freqtrade.fiat_convert.Market',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}))
    mocker.patch('freqtrade.main.exchange.get_ticker', side_effect=mock_ticker)

    _balance(bot=MagicMock(), update=update)
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert '*Currency*: BTC' in result
    assert '*Currency*: ETH' not in result
    assert '*Currency*: USDT' in result
    assert 'Balance' in result
    assert 'Est. BTC' in result
    assert '*BTC*:  12.00000000' in result


def test_zero_balance_handle(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_balances=MagicMock(return_value=[]))
    _balance(bot=MagicMock(), update=update)
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert '`All balances are zero.`' in result


def test_help_handle(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)

    _help(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert '*/help:* `This help message`' in msg_mock.call_args_list[0][0][0]


def test_version_handle(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)

    _version(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 1
    assert '*Version:* `{}`'.format(__version__) in msg_mock.call_args_list[0][0][0]


def test_send_msg(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock())
    bot = MagicMock()
    send_msg('test', bot)
    assert not bot.method_calls
    bot.reset_mock()

    default_conf['telegram']['enabled'] = True
    send_msg('test', bot)
    assert len(bot.method_calls) == 1


def test_send_msg_network_error(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock())
    default_conf['telegram']['enabled'] = True
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError('Oh snap'))
    send_msg('test', bot)

    # Bot should've tried to send it twice
    assert len(bot.method_calls) == 2


def test_init(default_conf, mocker):
    start_polling = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          # mock telegram.ext.Updater
                          Updater=MagicMock(return_value=start_polling))
    # not enabled
    tg.init(default_conf)
    assert start_polling.call_count == 0
    # number of handles registered
    assert start_polling.dispatcher.add_handler.call_count == 11
    assert start_polling.start_polling.call_count == 1

    # enabled
    default_conf['telegram'] = {}
    default_conf['telegram']['enabled'] = True
    default_conf['telegram']['token'] = ''
    tg.init(default_conf)


def test_cleanup(default_conf, mocker):
    default_conf['telegram'] = {}
    default_conf['telegram']['enabled'] = False
    updater_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          _UPDATER=updater_mock)
    # not enabled
    tg.cleanup()
    assert updater_mock.stop.call_count == 0

    # enabled
    default_conf['telegram']['enabled'] = True
    tg.cleanup()
    assert updater_mock.stop.call_count == 1


def test_status(default_conf, update, mocker):
    update.message.chat.id = 123
    default_conf['telegram'] = {}
    default_conf['telegram']['chat_id'] = 123
    mocker.patch('telegram.update', MagicMock())
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock())
    msg_mock = MagicMock()
    status_table = MagicMock()
    mocker.patch.multiple('freqtrade.rpc',
                          send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          rpc_trade_status=MagicMock(return_value=(False, [1, 2, 3])),
                          _status_table=status_table,
                          send_msg=msg_mock)
    _status(bot=MagicMock(), update=update)
    assert msg_mock.call_count == 3
    update.message.text = MagicMock()
    update.message.text.replace = MagicMock(return_value='table 2 3')
    _status(bot=MagicMock(), update=update)
    assert status_table.call_count == 1
