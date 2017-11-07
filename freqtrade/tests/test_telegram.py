# pragma pylint: disable=missing-docstring, too-many-arguments, too-many-ancestors
import re
from datetime import datetime
from unittest.mock import MagicMock

from telegram import Bot

from freqtrade.main import init, create_trade
from freqtrade.misc import update_state, State, get_state
from freqtrade.persistence import Trade
from freqtrade.rpc.telegram import (
    _status, _status_table, _profit, _forcesell, _performance, _count, _start, _stop, _balance
)


class MagicBot(MagicMock, Bot):
    pass


def test_status_handle(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    init(default_conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade
    Trade.session.add(trade)
    Trade.session.flush()

    # Trigger status while we have a fulfilled order for the open trade
    _status(bot=MagicBot(), update=update)

    assert msg_mock.call_count == 2
    assert '[BTC_ETH]' in msg_mock.call_args_list[-1][0][0]


def test_status_table_handle(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.main.telegram',
        _CONF=default_conf,
        init=MagicMock(),
        send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_order_id'))
    init(default_conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade
    Trade.session.add(trade)
    Trade.session.flush()

    _status_table(bot=MagicBot(), update=update)

    text = re.sub('</?pre>', '', msg_mock.call_args_list[-1][0][0])
    line = text.split("\n")
    fields = re.sub('[ ]+', ' ', line[2].strip()).split(' ')

    assert int(fields[0]) == 1
    assert fields[1] == 'BTC_ETH'
    assert msg_mock.call_count == 2


def test_profit_handle(default_conf, update, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    init(default_conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)
    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False
    Trade.session.add(trade)
    Trade.session.flush()

    _profit(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 2
    assert '*ROI:* `1.50701325 (10.05%)`' in msg_mock.call_args_list[-1][0][0]
    assert 'Best Performing:* `BTC_ETH: 10.05%`' in msg_mock.call_args_list[-1][0][0]


def test_forcesell_handle(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    init(default_conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade

    Trade.session.add(trade)
    Trade.session.flush()

    update.message.text = '/forcesell 1'
    _forcesell(bot=MagicBot(), update=update)

    assert msg_mock.call_count == 2
    assert 'Selling [BTC/ETH]' in msg_mock.call_args_list[-1][0][0]
    assert '0.07256061 (profit: ~-0.64%)' in msg_mock.call_args_list[-1][0][0]


def test_performance_handle(default_conf, update, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    init(default_conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False
    Trade.session.add(trade)
    Trade.session.flush()

    _performance(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 2
    assert 'Performance' in msg_mock.call_args_list[-1][0][0]
    assert '<code>BTC_ETH\t10.05%</code>' in msg_mock.call_args_list[-1][0][0]


def test_count_handle(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.main.telegram',
        _CONF=default_conf,
        init=MagicMock(),
        send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_order_id'))
    init(default_conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    trade2 = create_trade(15.0)
    assert trade
    assert trade2
    Trade.session.add(trade)
    Trade.session.add(trade2)
    Trade.session.flush()

    _count(bot=MagicBot(), update=update)
    line = msg_mock.call_args_list[-1][0][0].split("\n")
    assert line[2] == '{}/{}'.format(2, default_conf['max_open_trades'])


def test_start_handle(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          _CONF=default_conf,
                          init=MagicMock())
    init(default_conf, 'sqlite://')

    update_state(State.STOPPED)
    assert get_state() == State.STOPPED
    _start(bot=MagicBot(), update=update)
    assert get_state() == State.RUNNING
    assert msg_mock.call_count == 0


def test_stop_handle(default_conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          _CONF=default_conf,
                          init=MagicMock())
    init(default_conf, 'sqlite://')

    update_state(State.RUNNING)
    assert get_state() == State.RUNNING
    _stop(bot=MagicBot(), update=update)
    assert get_state() == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'Stopping trader' in msg_mock.call_args_list[0][0][0]


def test_balance_handle(default_conf, update, mocker):
    mock_balance = [{
        'Currency': 'BTC',
        'Balance': 10.0,
        'Available': 12.0,
        'Pending': 0.0,
        'CryptoAddress': 'XXXX'}]
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_balances=MagicMock(return_value=mock_balance))

    _balance(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 1
    assert '*Currency*: BTC' in msg_mock.call_args_list[0][0][0]
    assert 'Balance' in msg_mock.call_args_list[0][0][0]
