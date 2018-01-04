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
        "max_open_trades": 1,
        "stake_currency": "BTC",
        "stake_amount": 0.001,
        "fiat_display_currency": "USD",
        "dry_run": True,
        "minimal_roi": {
            "40":  0.0,
            "30":  0.01,
            "20":  0.02,
            "0":  0.04
        },
        "stoploss": -0.10,
        "unfilledtimeout": 600,
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
                "BTC_BCC"
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


@pytest.fixture
def ticker():
    return MagicMock(return_value={
        'bid': 0.00001098,
        'ask': 0.00001099,
        'last': 0.00001098,
    })


@pytest.fixture
def ticker_sell_up():
    return MagicMock(return_value={
        'bid': 0.00001172,
        'ask': 0.00001173,
        'last': 0.00001172,
    })


@pytest.fixture
def ticker_sell_down():
    return MagicMock(return_value={
        'bid': 0.00001044,
        'ask': 0.00001043,
        'last': 0.00001044,
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
        'rate': 0.00001099,
        'amount': 90.99181073,
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
        'rate': 0.00001173,
        'amount': 90.99181073,
        'remaining': 0.0,
        'closed': datetime.utcnow(),
    }


@pytest.fixture
def ticker_history():
    return [
        {
            "O": 8.794e-05,
            "H": 8.948e-05,
            "L": 8.794e-05,
            "C": 8.88e-05,
            "V": 991.09056638,
            "T": "2017-11-26T08:50:00",
            "BV": 0.0877869
        },
        {
            "O": 8.88e-05,
            "H": 8.942e-05,
            "L": 8.88e-05,
            "C": 8.893e-05,
            "V": 658.77935965,
            "T": "2017-11-26T08:55:00",
            "BV": 0.05874751
        },
        {
            "O": 8.891e-05,
            "H": 8.893e-05,
            "L": 8.875e-05,
            "C": 8.877e-05,
            "V": 7920.73570705,
            "T": "2017-11-26T09:00:00",
            "BV": 0.7039405
        }
    ]
