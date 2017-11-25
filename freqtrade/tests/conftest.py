# pragma pylint: disable=missing-docstring
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from jsonschema import validate
from telegram import Message, Chat, Update

from freqtrade.misc import CONF_SCHEMA


@pytest.fixture(scope="module")
def default_conf():
    """ Returns validated configuration suitable for most tests """
    configuration = {
        "max_open_trades": 3,
        "stake_currency": "BTC",
        "stake_amount": 0.004,
        "dry_run": True,
        "minimal_roi": {
	"30":  0.0,
        "25":  0.015,
        "20":  0.020,
        "15":  0.025,
        "10":  0.030,
        "5":  0.035,
        "0":  0.045
        },
        "stoploss": -0.03,
        "bid_strategy": {
            "ask_last_balance": 0.0
        },
        "exchange": {
            "name": "bittrex",
            "enabled": True,
            "key": "key",
            "secret": "secret",
	"pair_whitelist": ["BTC_EDG", "BTC_ETC", "BTC_MTL", "BTC_OK", "BTC_PAY", "BTC_PIVX", "BTC_SNT", "BTC_XZC", "BTC_VTC", "BTC_XLM", "BTC_SWT",
        "BTC_MER", "BTC_FTC", "BTC_INCNT", "BTC_TIX", "BTC_RCN", "BTC_RLC", "BTC_TKN", "BTC_TRST", "BTC_MLN", "BTC_TIME", "BTC_LUN", "BTC_WAVES"]
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


@pytest.fixture(scope="module")
def backtest_conf():
    return {
        "max_open_trades": 3,
        "stake_currency": "BTC",
        "stake_amount": 0.004,
        "minimal_roi": {
	"30":  0.0,
        "25":  0.015,
        "20":  0.020,
        "15":  0.025,
        "10":  0.030,
        "5":  0.035,
        "0":  0.045
        },
        "stoploss": -0.03
    }


@pytest.fixture
def update():
    _update = Update(0)
    _update.message = Message(0, 0, datetime.utcnow(), Chat(0, 0))
    return _update


@pytest.fixture
def ticker():
    return MagicMock(return_value={
        'bid': 0.07256061,
        'ask': 0.072661,
        'last': 0.07256061,
    })


@pytest.fixture
def health():
    return MagicMock(return_value=[{
        'Currency': 'BTC',
        'IsActive': True,
        'LastChecked': '2017-11-13T20:15:00.00',
        'Notice': None
    }, {
        'Currency': 'ETH',
        'IsActive': True,
        'LastChecked': '2017-11-13T20:15:00.00',
        'Notice': None
    }, {
        'Currency': 'TRST',
        'IsActive': True,
        'LastChecked': '2017-11-13T20:15:00.00',
        'Notice': None
    }, {
        'Currency': 'SWT',
        'IsActive': True,
        'LastChecked': '2017-11-13T20:15:00.00',
        'Notice': None
    }, {
        'Currency': 'BCC',
        'IsActive': False,
        'LastChecked': '2017-11-13T20:15:00.00',
        'Notice': None
    }])


@pytest.fixture
def limit_buy_order():
    return {
        'id': 'mocked_limit_buy',
        'type': 'LIMIT_BUY',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.07256061,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    }


@pytest.fixture
def limit_sell_order():
    return {
        'id': 'mocked_limit_sell',
        'type': 'LIMIT_SELL',
        'pair': 'mocked',
        'opened': datetime.utcnow(),
        'rate': 0.0802134,
        'amount': 206.43811673387373,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    }
