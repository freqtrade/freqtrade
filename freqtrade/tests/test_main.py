# pragma pylint: disable=missing-docstring
import copy
from datetime import datetime
from unittest.mock import MagicMock, call

import pytest
from jsonschema import validate

from freqtrade.exchange import Exchanges
from freqtrade.main import create_trade, handle_trade, close_trade_if_fulfilled, init, \
    get_target_bid
from freqtrade.misc import CONF_SCHEMA
from freqtrade.persistence import Trade


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
                "BTC_ETH",
                "BTC_TKN",
                "BTC_TRST",
                "BTC_SWT",
            ]
        },
        "telegram": {
            "enabled": True,
            "token": "token",
            "chat_id": "chat_id"
        }
    }
    validate(configuration, CONF_SCHEMA)
    return configuration


def test_create_trade(conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    buy_signal = mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.07256061,
                              'ask': 0.072661,
                              'last': 0.07256061
                          }),
                          buy=MagicMock(return_value='mocked_limit_buy'))
    # Save state of current whitelist
    whitelist = copy.deepcopy(conf['exchange']['pair_whitelist'])

    init(conf, 'sqlite://')
    for _ in ['BTC_ETH', 'BTC_TKN', 'BTC_TRST', 'BTC_SWT']:
        trade = create_trade(15.0)
        Trade.session.add(trade)
        Trade.session.flush()
        assert trade is not None
        assert trade.stake_amount == 15.0
        assert trade.is_open
        assert trade.open_date is not None
        assert trade.exchange == Exchanges.BITTREX.name

        # Simulate fulfilled LIMIT_BUY order for trade
        trade.update({
            'id': 'mocked_limit_buy',
            'type': 'LIMIT_BUY',
            'pair': 'mocked',
            'opened': datetime.utcnow(),
            'rate': 0.072661,
            'amount': 206.43811673387373,
            'remaining': 0.0,
            'closed': datetime.utcnow(),
        })

        assert trade.open_rate == 0.072661
        assert trade.amount == 206.43811673387373

        assert whitelist == conf['exchange']['pair_whitelist']

    buy_signal.assert_has_calls(
        [call('BTC_ETH'), call('BTC_TKN'), call('BTC_TRST'), call('BTC_SWT')]
    )


def test_handle_trade(conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.17256061,
                              'ask': 0.172661,
                              'last': 0.17256061
                          }),
                          sell=MagicMock(return_value='mocked_limit_sell'))
    trade = Trade.query.filter(Trade.is_open.is_(True)).first()
    assert trade

    handle_trade(trade)
    assert trade.open_order_id == 'mocked_limit_sell'

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update({
        'id': 'mocked_sell_limit',
        'type': 'LIMIT_SELL',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.17256061,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    })

    assert trade.close_rate == 0.17256061
    assert trade.close_profit == 1.3698725
    assert trade.close_date is not None


def test_close_trade(conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', conf)
    trade = Trade.query.filter(Trade.is_open.is_(True)).first()
    assert trade

    # Simulate that there is no open order
    trade.open_order_id = None

    closed = close_trade_if_fulfilled(trade)
    assert closed
    assert not trade.is_open


def test_balance_fully_ask_side(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 0.0}})
    assert get_target_bid({'ask': 20, 'last': 10}) == 20


def test_balance_fully_last_side(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}})
    assert get_target_bid({'ask': 20, 'last': 10}) == 10


def test_balance_when_last_bigger_than_ask(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}})
    assert get_target_bid({'ask': 5, 'last': 10}) == 5
