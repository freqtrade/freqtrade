# pragma pylint: disable=missing-docstring
import json
import logging
from datetime import datetime
from functools import reduce
from unittest.mock import MagicMock

import arrow
import pytest
from jsonschema import validate
from sqlalchemy import create_engine
from telegram import Chat, Message, Update

from freqtrade.analyze import Analyze
from freqtrade.constants import Constants
from freqtrade.freqtradebot import FreqtradeBot

logging.getLogger('').setLevel(logging.INFO)


def log_has(line, logs):
    # caplog mocker returns log as a tuple: ('freqtrade.analyze', logging.WARNING, 'foobar')
    # and we want to match line against foobar in the tuple
    return reduce(lambda a, b: a or b,
                  filter(lambda x: x[2] == line, logs),
                  False)


# Functions for recurrent object patching
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot:
    """
    This function patch _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: None
    """
    mocker.patch('freqtrade.fiat_convert.Market', {'price_usd': 12345.0})
    mocker.patch('freqtrade.freqtradebot.Analyze', MagicMock())
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    mocker.patch('freqtrade.freqtradebot.persistence.init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.RPCManager._init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.RPCManager.send_msg', MagicMock())
    mocker.patch('freqtrade.freqtradebot.Analyze.get_signal', MagicMock())

    return FreqtradeBot(config, create_engine('sqlite://'))


@pytest.fixture(scope="module")
def default_conf():
    """ Returns validated configuration suitable for most tests """
    configuration = {
        "max_open_trades": 1,
        "stake_currency": "BTC",
        "stake_amount": 0.001,
        "fiat_display_currency": "USD",
        "ticker_interval": '5m',
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
                "ETH/BTC",
                "TKN/BTC",
                "TRST/BTC",
                "SWT/BTC",
                "BCC/BTC"
            ]
        },
        "telegram": {
            "enabled": True,
            "token": "token",
            "chat_id": "0"
        },
        "initial_state": "running",
        "loglevel": logging.DEBUG
    }
    validate(configuration, Constants.CONF_SCHEMA)
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
        'type': 'limit',
        'side': 'buy',
        'pair': 'mocked',
        'datetime': arrow.utcnow().isoformat(),
        'price': 0.00001099,
        'amount': 90.99181073,
        'remaining': 0.0,
        'status': 'closed'
    }


@pytest.fixture
def limit_buy_order_old():
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'limit',
        'side': 'buy',
        'pair': 'mocked',
        'datetime': str(arrow.utcnow().shift(minutes=-601).datetime),
        'price': 0.00001099,
        'amount': 90.99181073,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_sell_order_old():
    return {
        'id': 'mocked_limit_sell_old',
        'type': 'limit',
        'side': 'sell',
        'pair': 'ETH/BTC',
        'datetime': arrow.utcnow().shift(minutes=-601).isoformat(),
        'price': 0.00001099,
        'amount': 90.99181073,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_buy_order_old_partial():
    return {
        'id': 'mocked_limit_buy_old_partial',
        'type': 'limit',
        'side': 'buy',
        'pair': 'ETH/BTC',
        'datetime': arrow.utcnow().shift(minutes=-601).isoformat(),
        'price': 0.00001099,
        'amount': 90.99181073,
        'remaining': 67.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_sell_order():
    return {
        'id': 'mocked_limit_sell',
        'type': 'limit',
        'side': 'sell',
        'pair': 'mocked',
        'datetime': arrow.utcnow().isoformat(),
        'price': 0.00001173,
        'amount': 90.99181073,
        'remaining': 0.0,
        'status': 'closed'
    }


@pytest.fixture
def ticker_history_api():
    return [
        [
            1511686200000,  # unix timestamp ms
            8.794e-05,  # open
            8.948e-05,  # high
            8.794e-05,  # low
            8.88e-05,  # close
            0.0877869,  # volume (in quote currency)
        ],
        [
            1511686500000,
            8.88e-05,
            8.942e-05,
            8.88e-05,
            8.893e-05,
            0.05874751,
        ],
        [
            1511686800,
            8.891e-05,
            8.893e-05,
            8.875e-05,
            8.877e-05,
            0.7039405
        ]
    ]


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
    with open('freqtrade/tests/testdata/UNITTEST_BTC-1m.json') as data_file:
        return Analyze.parse_ticker_dataframe(json.load(data_file))


# FIX:
# Create an fixture/function
# that inserts a trade of some type and open-status
# return the open-order-id
# See tests in rpc/main that could use this
