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

    # Trigger status while we don't know the open_rate yet
    _status(bot=MagicBot(), update=update)

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update({
        'id': 'mocked_limit_buy',
        'type': 'LIMIT_BUY',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.07256060,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    })
    Trade.session.flush()

    # Trigger status while we have a fulfilled order for the open trade
    _status(bot=MagicBot(), update=update)

    assert msg_mock.call_count == 3
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
                          buy=MagicMock(return_value='mocked_limit_buy'))
    init(conf, 'sqlite://')

    # Create some test data
    trade = create_trade(15.0)
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update({
        'id': 'mocked_limit_buy',
        'type': 'LIMIT_BUY',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.07256061,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    })
    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update({
        'id': 'mocked_limit_sell',
        'type': 'LIMIT_SELL',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.0802134,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    })

    trade.close_date = datetime.utcnow()
    trade.open_order_id = None
    trade.is_open = False
    Trade.session.add(trade)
    Trade.session.flush()

    _profit(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 2
    assert '*ROI:* `1.507013 (10.05%)`' in msg_mock.call_args_list[-1][0][0]
    assert 'Best Performing:* `BTC_ETH: 10.05%`' in msg_mock.call_args_list[-1][0][0]


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

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update({
        'id': 'mocked_limit_buy',
        'type': 'LIMIT_BUY',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.07256060,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    })

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

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update({
        'id': 'mocked_limit_buy',
        'type': 'LIMIT_BUY',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.07256061,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    })

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update({
        'id': 'mocked_limit_sell',
        'type': 'LIMIT_SELL',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.0802134,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    })

    trade.close_date = datetime.utcnow()
    trade.is_open = False
    Trade.session.add(trade)
    Trade.session.flush()

    _performance(bot=MagicBot(), update=update)
    assert msg_mock.call_count == 2
    assert 'Performance' in msg_mock.call_args_list[-1][0][0]
    assert '<code>BTC_ETH\t10.05%</code>' in msg_mock.call_args_list[-1][0][0]


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
