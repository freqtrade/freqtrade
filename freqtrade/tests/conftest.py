# pragma pylint: disable=missing-docstring
from datetime import datetime
from unittest.mock import MagicMock
from functools import reduce

import json
import arrow
import pytest
from jsonschema import validate
from telegram import Chat, Message, Update
from freqtrade.analyze import parse_ticker_dataframe
from freqtrade.strategy.strategy import Strategy

from freqtrade.misc import CONF_SCHEMA


def log_has(line, logs):
    # caplog mocker returns log as a tuple: ('freqtrade.analyze', logging.WARNING, 'foobar')
    # and we want to match line against foobar in the tuple
    return reduce(lambda a, b: a or b,
                  filter(lambda x: x[2] == line, logs),
                  False)


@pytest.fixture(scope="module")
def default_conf():
    """ Returns validated configuration suitable for most tests """
    configuration = {
        "max_open_trades": 1,
        "stake_currency": "BTC",
        "stake_amount": 0.001,
        "fiat_display_currency": "USD",
        "ticker_interval": 5,
        "dry_run": True,
        "minimal_roi": {
            "40": 0.0,
            "30": 0.01,
            "20": 0.02,
            "0": 0.04
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
def ticker_usdt():
    return MagicMock(return_value={
        'bid': 10000.00,
        'ask': 10000.00,
        'last': 10000.00,
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
        'opened': str(arrow.utcnow().datetime),
        'rate': 0.00001099,
        'amount': 90.99181073,
        'remaining': 0.0,
        'closed': str(arrow.utcnow().datetime),
    }


@pytest.fixture
def limit_buy_order_old():
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'LIMIT_BUY',
        'pair': 'BTC_ETH',
        'opened': str(arrow.utcnow().shift(minutes=-601).datetime),
        'rate': 0.00001099,
        'amount': 90.99181073,
        'remaining': 90.99181073,
    }


@pytest.fixture
def limit_sell_order_old():
    return {
        'id': 'mocked_limit_sell_old',
        'type': 'LIMIT_SELL',
        'pair': 'BTC_ETH',
        'opened': str(arrow.utcnow().shift(minutes=-601).datetime),
        'rate': 0.00001099,
        'amount': 90.99181073,
        'remaining': 90.99181073,
    }


@pytest.fixture
def limit_buy_order_old_partial():
    return {
        'id': 'mocked_limit_buy_old_partial',
        'type': 'LIMIT_BUY',
        'pair': 'BTC_ETH',
        'opened': str(arrow.utcnow().shift(minutes=-601).datetime),
        'rate': 0.00001099,
        'amount': 90.99181073,
        'remaining': 67.99181073,
    }


@pytest.fixture
def limit_sell_order():
    return {
        'id': 'mocked_limit_sell',
        'type': 'LIMIT_SELL',
        'pair': 'mocked',
        'opened': str(arrow.utcnow().datetime),
        'rate': 0.00001173,
        'amount': 90.99181073,
        'remaining': 0.0,
        'closed': str(arrow.utcnow().datetime),
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


@pytest.fixture
def ticker_history_without_bv():
    return [
        {
            "O": 8.794e-05,
            "H": 8.948e-05,
            "L": 8.794e-05,
            "C": 8.88e-05,
            "V": 991.09056638,
            "T": "2017-11-26T08:50:00"
        },
        {
            "O": 8.88e-05,
            "H": 8.942e-05,
            "L": 8.88e-05,
            "C": 8.893e-05,
            "V": 658.77935965,
            "T": "2017-11-26T08:55:00"
        },
        {
            "O": 8.891e-05,
            "H": 8.893e-05,
            "L": 8.875e-05,
            "C": 8.877e-05,
            "V": 7920.73570705,
            "T": "2017-11-26T09:00:00"
        }
    ]


@pytest.fixture
def result():
    with open('freqtrade/tests/testdata/BTC_ETH-1.json') as data_file:
        return parse_ticker_dataframe(json.load(data_file))


@pytest.fixture
def default_strategy():
    strategy = Strategy()
    strategy.init({'strategy': 'default_strategy'})
    return strategy
