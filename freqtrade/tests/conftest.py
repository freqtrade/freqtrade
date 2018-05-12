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
from freqtrade import constants
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


@pytest.fixture(scope="function")
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
        "initial_state": "running",
        "loglevel": logging.DEBUG
    }
    validate(configuration, constants.CONF_SCHEMA)
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


# FIX: Perhaps change result fixture to use BTC_UNITEST instead?
@pytest.fixture
def result():
    with open('freqtrade/tests/testdata/BTC_ETH-1.json') as data_file:
        return Analyze.parse_ticker_dataframe(json.load(data_file))


# FIX:
# Create an fixture/function
# that inserts a trade of some type and open-status
# return the open-order-id
# See tests in rpc/main that could use this


@pytest.fixture
def get_market_summaries_data():
    """
    This fixture is a real result from exchange.get_market_summaries() but reduced to only
    8 entries. 4 BTC, 4 USTD
    :return: JSON market summaries
    """
    return [
        {
            'Ask': 1.316e-05,
            'BaseVolume': 5.72599471,
            'Bid': 1.3e-05,
            'Created': '2014-04-14T00:00:00',
            'High': 1.414e-05,
            'Last': 1.298e-05,
            'Low': 1.282e-05,
            'MarketName': 'BTC-XWC',
            'OpenBuyOrders': 2000,
            'OpenSellOrders': 1484,
            'PrevDay': 1.376e-05,
            'TimeStamp': '2018-02-05T01:32:40.493',
            'Volume': 424041.21418375
        },
        {
            'Ask': 0.00627051,
            'BaseVolume': 93.23302388,
            'Bid': 0.00618192,
            'Created': '2016-10-20T04:48:30.387',
            'High': 0.00669897,
            'Last': 0.00618192,
            'Low': 0.006,
            'MarketName': 'BTC-XZC',
            'OpenBuyOrders': 343,
            'OpenSellOrders': 2037,
            'PrevDay': 0.00668229,
            'TimeStamp': '2018-02-05T01:32:43.383',
            'Volume': 14863.60730702
        },
        {
            'Ask': 0.01137247,
            'BaseVolume': 383.55922657,
            'Bid': 0.01136006,
            'Created': '2016-11-15T20:29:59.73',
            'High': 0.012,
            'Last': 0.01137247,
            'Low': 0.01119883,
            'MarketName': 'BTC-ZCL',
            'OpenBuyOrders': 1332,
            'OpenSellOrders': 5317,
            'PrevDay': 0.01179603,
            'TimeStamp': '2018-02-05T01:32:42.773',
            'Volume': 33308.07358285
        },
        {
            'Ask': 0.04155821,
            'BaseVolume': 274.75369074,
            'Bid': 0.04130002,
            'Created': '2016-10-28T17:13:10.833',
            'High': 0.04354429,
            'Last': 0.041585,
            'Low': 0.0413,
            'MarketName': 'BTC-ZEC',
            'OpenBuyOrders': 863,
            'OpenSellOrders': 5579,
            'PrevDay': 0.0429,
            'TimeStamp': '2018-02-05T01:32:43.21',
            'Volume': 6479.84033259
        },
        {
            'Ask': 210.99999999,
            'BaseVolume': 615132.70989532,
            'Bid': 210.05503736,
            'Created': '2017-07-21T01:08:49.397',
            'High': 257.396,
            'Last': 211.0,
            'Low': 209.05333589,
            'MarketName': 'USDT-XMR',
            'OpenBuyOrders': 180,
            'OpenSellOrders': 1203,
            'PrevDay': 247.93528899,
            'TimeStamp': '2018-02-05T01:32:43.117',
            'Volume': 2688.17410793
        },
        {
            'Ask': 0.79589979,
            'BaseVolume': 9349557.01853031,
            'Bid': 0.789226,
            'Created': '2017-07-14T17:10:10.737',
            'High': 0.977,
            'Last': 0.79589979,
            'Low': 0.781,
            'MarketName': 'USDT-XRP',
            'OpenBuyOrders': 1075,
            'OpenSellOrders': 6508,
            'PrevDay': 0.93300218,
            'TimeStamp': '2018-02-05T01:32:42.383',
            'Volume': 10801663.00788851
        },
        {
            'Ask': 0.05154982,
            'BaseVolume': 2311087.71232136,
            'Bid': 0.05040107,
            'Created': '2017-12-29T19:29:18.357',
            'High': 0.06668561,
            'Last': 0.0508,
            'Low': 0.05006731,
            'MarketName': 'USDT-XVG',
            'OpenBuyOrders': 655,
            'OpenSellOrders': 5544,
            'PrevDay': 0.0627,
            'TimeStamp': '2018-02-05T01:32:41.507',
            'Volume': 40031424.2152716
        },
        {
            'Ask': 332.65500022,
            'BaseVolume': 562911.87455665,
            'Bid': 330.00000001,
            'Created': '2017-07-14T17:10:10.673',
            'High': 401.59999999,
            'Last': 332.65500019,
            'Low': 330.0,
            'MarketName': 'USDT-ZEC',
            'OpenBuyOrders': 161,
            'OpenSellOrders': 1731,
            'PrevDay': 391.42,
            'TimeStamp': '2018-02-05T01:32:42.947',
            'Volume': 1571.09647946
        }
    ]
