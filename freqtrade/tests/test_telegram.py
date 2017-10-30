# pragma pylint: disable=missing-docstring
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from jsonschema import validate
from telegram import Bot, Update, Message, Chat

from freqtrade.main import init, create_trade
from freqtrade.misc import update_state, State, get_state, CONF_SCHEMA
from freqtrade.persistence import Trade
from freqtrade.rpc.telegram import _status, _profit, _forcesell, _performance, _start, _stop, _balance


@pytest.fixture
def conf():
    configuration = {
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
        "exchange": {
            "name": "bittrex",
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
        },
        "initial_state": "running"
    }
    validate(configuration, CONF_SCHEMA)
    return configuration

@pytest.fixture
def update():
    _update = Update(0)
    _update.message = Message(0, 0, datetime.utcnow(), Chat(0, 0))
    return _update


class MagicBot(MagicMock, Bot):
    pass


def test_status_handle(conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram', _CONF=conf, init=MagicMock(), send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.07256061,
                              'ask': 0.072661,
                              'last': 0.07256061
                          }),
                          buy=MagicMock(return_value='mocked_order_id'))
    init(conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade
    Trade.session.add(trade)
    Trade.session.flush()

    _status(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 2
    assert '[BTC_ETH]' in msg_mock.call_args_list[-1][0][0]


def test_profit_handle(conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram', _CONF=conf, init=MagicMock(), send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.07256061,
                              'ask': 0.072661,
                              'last': 0.07256061
                          }),
                          buy=MagicMock(return_value='mocked_order_id'))
    init(conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade
    trade.close_rate = 0.07256061
    trade.close_profit = 100.00
    trade.close_date = datetime.utcnow()
    trade.open_order_id = None
    trade.is_open = False
    Trade.session.add(trade)
    Trade.session.flush()

    _profit(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 2
    assert '(100.00%)' in msg_mock.call_args_list[-1][0][0]


def test_forcesell_handle(conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram', _CONF=conf, init=MagicMock(), send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.07256061,
                              'ask': 0.072661,
                              'last': 0.07256061
                          }),
                          buy=MagicMock(return_value='mocked_order_id'))
    init(conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade
    Trade.session.add(trade)
    Trade.session.flush()

    update.message.text = '/forcesell 1'
    _forcesell(bot=MagicBot(), update=update)

    assert msg_mock.call_count == 2
    assert 'Selling [BTC/ETH]' in msg_mock.call_args_list[-1][0][0]
    assert '0.072561' in msg_mock.call_args_list[-1][0][0]


def test_performance_handle(conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram', _CONF=conf, init=MagicMock(), send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.07256061,
                              'ask': 0.072661,
                              'last': 0.07256061
                          }),
                          buy=MagicMock(return_value='mocked_order_id'))
    init(conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade
    trade.close_rate = 0.07256061
    trade.close_profit = 100.00
    trade.close_date = datetime.utcnow()
    trade.open_order_id = None
    trade.is_open = False
    Trade.session.add(trade)
    Trade.session.flush()

    _performance(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 2
    assert 'Performance' in msg_mock.call_args_list[-1][0][0]
    assert 'BTC_ETH	100.00%' in msg_mock.call_args_list[-1][0][0]


def test_start_handle(conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram', _CONF=conf, init=MagicMock(), send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange', _CONF=conf, init=MagicMock())
    init(conf, 'sqlite://')

    update_state(State.STOPPED)
    assert get_state() == State.STOPPED
    _start(bot=MagicBot(), update=update)
    assert get_state() == State.RUNNING
    assert msg_mock.call_count == 0


def test_stop_handle(conf, update, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram', _CONF=conf, init=MagicMock(), send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange', _CONF=conf, init=MagicMock())
    init(conf, 'sqlite://')

    update_state(State.RUNNING)
    assert get_state() == State.RUNNING
    _stop(bot=MagicBot(), update=update)
    assert get_state() == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'Stopping trader' in msg_mock.call_args_list[0][0][0]


def test_balance_handle(conf, update, mocker):
    mock_balance = [{
        'Currency': 'BTC',
        'Balance': 10.0,
        'Available': 12.0,
        'Pending': 0.0,
        'CryptoAddress': 'XXXX'}]
    mocker.patch.dict('freqtrade.main._CONF', conf)
    msg_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.telegram', _CONF=conf, init=MagicMock(), send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_balances=MagicMock(return_value=mock_balance))

    _balance(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 1
    assert '*Currency*: BTC' in msg_mock.call_args_list[0][0][0]
    assert 'Balance' in msg_mock.call_args_list[0][0][0]
