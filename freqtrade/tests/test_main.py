# pragma pylint: disable=missing-docstring
import copy
from unittest.mock import MagicMock

import pytest
import requests
from sqlalchemy import create_engine

from freqtrade.exchange import Exchanges
from freqtrade.main import create_trade, handle_trade, close_trade_if_fulfilled, init, \
    get_target_bid, _process
from freqtrade.misc import get_state, State, FreqtradeException
from freqtrade.persistence import Trade


def test_process_trade_creation(default_conf, ticker, health, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_wallet_health=health,
                          buy=MagicMock(return_value='mocked_limit_buy'))
    init(default_conf, create_engine('sqlite://'))

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 0

    result = _process()
    assert result is True

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert trade.stake_amount == default_conf['stake_amount']
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == Exchanges.BITTREX.name
    assert trade.open_rate == 0.072661
    assert trade.amount == 0.6864067381401302


def test_process_exchange_failures(default_conf, ticker, health, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    sleep_mock = mocker.patch('time.sleep', side_effect=lambda _: None)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_wallet_health=health,
                          buy=MagicMock(side_effect=requests.exceptions.RequestException))
    init(default_conf, create_engine('sqlite://'))
    result = _process()
    assert result is False
    assert sleep_mock.has_calls()


def test_process_runtime_error(default_conf, ticker, health, mocker):
    msg_mock = MagicMock()
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=msg_mock)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_wallet_health=health,
                          buy=MagicMock(side_effect=RuntimeError))
    init(default_conf, create_engine('sqlite://'))
    assert get_state() == State.RUNNING

    result = _process()
    assert result is False
    assert get_state() == State.STOPPED
    assert 'RuntimeError' in msg_mock.call_args_list[-1][0][0]


def test_process_trade_handling(default_conf, ticker, limit_buy_order, health, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_wallet_health=health,
                          buy=MagicMock(return_value='mocked_limit_buy'),
                          get_order=MagicMock(return_value=limit_buy_order))
    init(default_conf, create_engine('sqlite://'))

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 0
    result = _process()
    assert result is True
    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1

    result = _process()
    assert result is False


def test_create_trade(default_conf, ticker, limit_buy_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))
    # Save state of current whitelist
    whitelist = copy.deepcopy(default_conf['exchange']['pair_whitelist'])

    init(default_conf, create_engine('sqlite://'))
    trade = create_trade(15.0)
    Trade.session.add(trade)
    Trade.session.flush()
    assert trade is not None
    assert trade.stake_amount == 15.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == Exchanges.BITTREX.name

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    assert trade.open_rate == 0.07256061
    assert trade.amount == 206.43811673387373

    assert whitelist == default_conf['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(default_conf, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'),
                          get_balance=MagicMock(return_value=default_conf['stake_amount'] * 0.5))
    with pytest.raises(FreqtradeException, match=r'.*stake amount.*'):
        create_trade(default_conf['stake_amount'])


def test_create_trade_no_pairs(default_conf, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))

    with pytest.raises(FreqtradeException, match=r'.*No pair in whitelist.*'):
        conf = copy.deepcopy(default_conf)
        conf['exchange']['pair_whitelist'] = []
        mocker.patch.dict('freqtrade.main._CONF', conf)
        create_trade(default_conf['stake_amount'])


def test_handle_trade(default_conf, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.17256061,
                              'ask': 0.172661,
                              'last': 0.17256061
                          }),
                          buy=MagicMock(return_value='mocked_limit_buy'),
                          sell=MagicMock(return_value='mocked_limit_sell'))
    init(default_conf, create_engine('sqlite://'))
    trade = create_trade(15.0)
    trade.update(limit_buy_order)
    Trade.session.add(trade)
    Trade.session.flush()
    trade = Trade.query.filter(Trade.is_open.is_(True)).first()
    assert trade

    handle_trade(trade)
    assert trade.open_order_id == 'mocked_limit_sell'
    assert close_trade_if_fulfilled(trade) is False

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    assert trade.close_rate == 0.0802134
    assert trade.close_profit == 0.10046755
    assert trade.close_date is not None


def test_close_trade(default_conf, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))

    # Create trade and sell it
    init(default_conf, create_engine('sqlite://'))
    trade = create_trade(15.0)
    trade.update(limit_buy_order)
    trade.update(limit_sell_order)

    Trade.session.add(trade)
    Trade.session.flush()
    trade = Trade.query.filter(Trade.is_open.is_(True)).first()
    assert trade

    # Simulate that there is no open order
    trade.open_order_id = None

    closed = close_trade_if_fulfilled(trade)
    assert closed
    assert not trade.is_open
    with pytest.raises(ValueError, match=r'.*closed trade.*'):
        handle_trade(trade)


def test_balance_fully_ask_side(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 0.0}})
    assert get_target_bid({'ask': 20, 'last': 10}) == 20


def test_balance_fully_last_side(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}})
    assert get_target_bid({'ask': 20, 'last': 10}) == 10


def test_balance_bigger_last_ask(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}})
    assert get_target_bid({'ask': 5, 'last': 10}) == 5
