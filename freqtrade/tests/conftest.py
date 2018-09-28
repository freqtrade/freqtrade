# pragma pylint: disable=missing-docstring
import json
import logging
from datetime import datetime
from functools import reduce
from typing import Dict, Optional
from unittest.mock import MagicMock, PropertyMock

import arrow
import pytest
from telegram import Chat, Message, Update

from freqtrade.exchange.exchange_helpers import parse_ticker_dataframe
from freqtrade.exchange import Exchange
from freqtrade.freqtradebot import FreqtradeBot

logging.getLogger('').setLevel(logging.INFO)


def log_has(line, logs):
    # caplog mocker returns log as a tuple: ('freqtrade.something', logging.WARNING, 'foobar')
    # and we want to match line against foobar in the tuple
    return reduce(lambda a, b: a or b,
                  filter(lambda x: x[2] == line, logs),
                  False)


def patch_exchange(mocker, api_mock=None) -> None:
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value="Bittrex"))
    mocker.patch('freqtrade.exchange.Exchange.id', PropertyMock(return_value="bittrex"))
    if api_mock:
        mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    else:
        mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock())


def get_patched_exchange(mocker, config, api_mock=None) -> Exchange:
    patch_exchange(mocker, api_mock)
    exchange = Exchange(config)
    return exchange


# Functions for recurrent object patching
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot:
    """
    This function patch _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: None
    """
    # mocker.patch('freqtrade.fiat_convert.Market', {'price_usd': 12345.0})
    patch_coinmarketcap(mocker, {'price_usd': 12345.0})
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    mocker.patch('freqtrade.freqtradebot.persistence.init', MagicMock())
    patch_exchange(mocker, None)
    mocker.patch('freqtrade.freqtradebot.RPCManager._init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.RPCManager.send_msg', MagicMock())

    return FreqtradeBot(config)


def patch_coinmarketcap(mocker, value: Optional[Dict[str, float]] = None) -> None:
    """
    Mocker to coinmarketcap to speed up tests
    :param mocker: mocker to patch coinmarketcap class
    :return: None
    """

    tickermock = MagicMock(return_value={'price_usd': 12345.0})
    listmock = MagicMock(return_value={'data': [{'id': 1, 'name': 'Bitcoin', 'symbol': 'BTC',
                                                 'website_slug': 'bitcoin'},
                                                {'id': 1027, 'name': 'Ethereum', 'symbol': 'ETH',
                                                 'website_slug': 'ethereum'}
                                                ]})
    mocker.patch.multiple(
        'freqtrade.fiat_convert.Market',
        ticker=tickermock,
        listings=listmock,

    )


@pytest.fixture(scope="function")
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
        "unfilledtimeout": {
            "buy": 10,
            "sell": 30
        },
        "bid_strategy": {
            "ask_last_balance": 0.0,
            "use_order_book": False,
            "order_book_top": 1,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1
            }
        },
        "ask_strategy": {
            "use_order_book": False,
            "order_book_min": 1,
            "order_book_max": 1
        },
        "exchange": {
            "name": "bittrex",
            "enabled": True,
            "key": "key",
            "secret": "secret",
            "pair_whitelist": [
                "ETH/BTC",
                "LTC/BTC",
                "XRP/BTC",
                "NEO/BTC"
            ]
        },
        "edge": {
            "enabled": False,
            "process_throttle_secs": 1800,
            "calculate_since_number_of_days": 14,
            "total_capital_in_stake_currency": 0.5,
            "allowed_risk": 0.01,
            "stoploss_range_min": -0.01,
            "stoploss_range_max": -0.1,
            "stoploss_range_step": -0.001,
            "maximum_winrate": 0.80,
            "minimum_expectancy": 0.20,
            "min_trade_number": 15,
            "max_trade_duration_minute": 1440,
            "remove_pumps": True,
            "minimum_delta": 1
        },
        "telegram": {
            "enabled": True,
            "token": "token",
            "chat_id": "0"
        },
        "initial_state": "running",
        "db_url": "sqlite://",
        "loglevel": logging.DEBUG,
    }
    return configuration


@pytest.fixture
def update():
    _update = Update(0)
    _update.message = Message(0, 0, datetime.utcnow(), Chat(0, 0))
    return _update


@pytest.fixture
def fee():
    return MagicMock(return_value=0.0025)


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
def markets():
    return MagicMock(return_value=[
        {
            'id': 'ethbtc',
            'symbol': 'ETH/BTC',
            'base': 'ETH',
            'quote': 'BTC',
            'active': True,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': {
                    'min': 1,
                    'max': 500000,
                },
            },
            'info': '',
        },
        {
            'id': 'tknbtc',
            'symbol': 'TKN/BTC',
            'base': 'TKN',
            'quote': 'BTC',
            'active': True,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': {
                    'min': 1,
                    'max': 500000,
                },
            },
            'info': '',
        },
        {
            'id': 'blkbtc',
            'symbol': 'BLK/BTC',
            'base': 'BLK',
            'quote': 'BTC',
            'active': True,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': {
                    'min': 1,
                    'max': 500000,
                },
            },
            'info': '',
        },
        {
            'id': 'ltcbtc',
            'symbol': 'LTC/BTC',
            'base': 'LTC',
            'quote': 'BTC',
            'active': False,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': {
                    'min': 1,
                    'max': 500000,
                },
            },
            'info': '',
        },
        {
            'id': 'xrpbtc',
            'symbol': 'XRP/BTC',
            'base': 'XRP',
            'quote': 'BTC',
            'active': False,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': {
                    'min': 1,
                    'max': 500000,
                },
            },
            'info': '',
        },
        {
            'id': 'neobtc',
            'symbol': 'NEO/BTC',
            'base': 'NEO',
            'quote': 'BTC',
            'active': False,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': {
                    'min': 1,
                    'max': 500000,
                },
            },
            'info': '',
        }
    ])


@pytest.fixture
def markets_empty():
    return MagicMock(return_value=[])


@pytest.fixture(scope='function')
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
def order_book_l2():
    return MagicMock(return_value={
        'bids': [
            [0.043936, 10.442],
            [0.043935, 31.865],
            [0.043933, 11.212],
            [0.043928, 0.088],
            [0.043925, 10.0],
            [0.043921, 10.0],
            [0.04392, 37.64],
            [0.043899, 0.066],
            [0.043885, 0.676],
            [0.04387, 22.758]
        ],
        'asks': [
            [0.043949, 0.346],
            [0.04395, 0.608],
            [0.043951, 3.948],
            [0.043954, 0.288],
            [0.043958, 9.277],
            [0.043995, 1.566],
            [0.044, 0.588],
            [0.044002, 0.992],
            [0.044003, 0.095],
            [0.04402, 37.64]
        ],
        'timestamp': None,
        'datetime': None,
        'nonce': 288004540
    })


@pytest.fixture
def ticker_history():
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
            1511686800000,
            8.891e-05,
            8.893e-05,
            8.875e-05,
            8.877e-05,
            0.7039405
        ]
    ]


@pytest.fixture
def tickers():
    return MagicMock(return_value={
        'ETH/BTC': {
            'symbol': 'ETH/BTC',
            'timestamp': 1522014806207,
            'datetime': '2018-03-25T21:53:26.207Z',
            'high': 0.061697,
            'low': 0.060531,
            'bid': 0.061588,
            'bidVolume': 3.321,
            'ask': 0.061655,
            'askVolume': 0.212,
            'vwap': 0.06105296,
            'open': 0.060809,
            'close': 0.060761,
            'first': None,
            'last': 0.061588,
            'change': 1.281,
            'percentage': None,
            'average': None,
            'baseVolume': 111649.001,
            'quoteVolume': 6816.50176926,
            'info': {}
        },
        'TKN/BTC': {
            'symbol': 'TKN/BTC',
            'timestamp': 1522014806169,
            'datetime': '2018-03-25T21:53:26.169Z',
            'high': 0.01885,
            'low': 0.018497,
            'bid': 0.018799,
            'bidVolume': 8.38,
            'ask': 0.018802,
            'askVolume': 15.0,
            'vwap': 0.01869197,
            'open': 0.018585,
            'close': 0.018573,
            'baseVolume': 81058.66,
            'quoteVolume': 2247.48374509,
        },
        'BLK/BTC': {
            'symbol': 'BLK/BTC',
            'timestamp': 1522014806072,
            'datetime': '2018-03-25T21:53:26.720Z',
            'high': 0.007745,
            'low': 0.007512,
            'bid': 0.007729,
            'bidVolume': 0.01,
            'ask': 0.007743,
            'askVolume': 21.37,
            'vwap': 0.00761466,
            'open': 0.007653,
            'close': 0.007652,
            'first': None,
            'last': 0.007743,
            'change': 1.176,
            'percentage': None,
            'average': None,
            'baseVolume': 295152.26,
            'quoteVolume': 1515.14631229,
            'info': {}
        },
        'LTC/BTC': {
            'symbol': 'LTC/BTC',
            'timestamp': 1523787258992,
            'datetime': '2018-04-15T10:14:19.992Z',
            'high': 0.015978,
            'low': 0.0157,
            'bid': 0.015954,
            'bidVolume': 12.83,
            'ask': 0.015957,
            'askVolume': 0.49,
            'vwap': 0.01581636,
            'open': 0.015823,
            'close': 0.01582,
            'first': None,
            'last': 0.015951,
            'change': 0.809,
            'percentage': None,
            'average': None,
            'baseVolume': 88620.68,
            'quoteVolume': 1401.65697943,
            'info': {}
        },
        'ETH/USDT': {
            'symbol': 'ETH/USDT',
            'timestamp': 1522014804118,
            'datetime': '2018-03-25T21:53:24.118Z',
            'high': 530.88,
            'low': 512.0,
            'bid': 529.73,
            'bidVolume': 0.2,
            'ask': 530.21,
            'askVolume': 0.2464,
            'vwap': 521.02438405,
            'open': 527.27,
            'close': 528.42,
            'first': None,
            'last': 530.21,
            'change': 0.558,
            'percentage': None,
            'average': None,
            'baseVolume': 72300.0659,
            'quoteVolume': 37670097.3022171,
            'info': {}
        },
        'TKN/USDT': {
            'symbol': 'TKN/USDT',
            'timestamp': 1522014806198,
            'datetime': '2018-03-25T21:53:26.198Z',
            'high': 8718.0,
            'low': 8365.77,
            'bid': 8603.64,
            'bidVolume': 0.15846,
            'ask': 8603.67,
            'askVolume': 0.069147,
            'vwap': 8536.35621697,
            'open': 8680.0,
            'close': 8680.0,
            'first': None,
            'last': 8603.67,
            'change': -0.879,
            'percentage': None,
            'average': None,
            'baseVolume': 30414.604298,
            'quoteVolume': 259629896.48584127,
            'info': {}
        },
        'BLK/USDT': {
            'symbol': 'BLK/USDT',
            'timestamp': 1522014806145,
            'datetime': '2018-03-25T21:53:26.145Z',
            'high': 66.95,
            'low': 63.38,
            'bid': 66.473,
            'bidVolume': 4.968,
            'ask': 66.54,
            'askVolume': 2.704,
            'vwap': 65.0526901,
            'open': 66.43,
            'close': 66.383,
            'first': None,
            'last': 66.5,
            'change': 0.105,
            'percentage': None,
            'average': None,
            'baseVolume': 294106.204,
            'quoteVolume': 19132399.743954,
            'info': {}
        },
        'LTC/USDT': {
            'symbol': 'LTC/USDT',
            'timestamp': 1523787257812,
            'datetime': '2018-04-15T10:14:18.812Z',
            'high': 129.94,
            'low': 124.0,
            'bid': 129.28,
            'bidVolume': 0.03201,
            'ask': 129.52,
            'askVolume': 0.14529,
            'vwap': 126.92838682,
            'open': 127.0,
            'close': 127.1,
            'first': None,
            'last': 129.28,
            'change': 1.795,
            'percentage': None,
            'average': None,
            'baseVolume': 59698.79897,
            'quoteVolume': 29132399.743954,
            'info': {}
        }
    })


@pytest.fixture
def result():
    with open('freqtrade/tests/testdata/UNITTEST_BTC-1m.json') as data_file:
        return parse_ticker_dataframe(json.load(data_file))

# FIX:
# Create an fixture/function
# that inserts a trade of some type and open-status
# return the open-order-id
# See tests in rpc/main that could use this


@pytest.fixture(scope="function")
def trades_for_order():
    return [{'info': {'id': 34567,
                      'orderId': 123456,
                      'price': '0.24544100',
                      'qty': '8.00000000',
                      'commission': '0.00800000',
                      'commissionAsset': 'LTC',
                      'time': 1521663363189,
                      'isBuyer': True,
                      'isMaker': False,
                      'isBestMatch': True},
             'timestamp': 1521663363189,
             'datetime': '2018-03-21T20:16:03.189Z',
             'symbol': 'LTC/ETH',
             'id': '34567',
             'order': '123456',
             'type': None,
             'side': 'buy',
             'price': 0.245441,
             'cost': 1.963528,
             'amount': 8.0,
             'fee': {'cost': 0.008, 'currency': 'LTC'}}]


@pytest.fixture(scope="function")
def trades_for_order2():
    return [{'info': {'id': 34567,
                      'orderId': 123456,
                      'price': '0.24544100',
                      'qty': '8.00000000',
                      'commission': '0.00800000',
                      'commissionAsset': 'LTC',
                      'time': 1521663363189,
                      'isBuyer': True,
                      'isMaker': False,
                      'isBestMatch': True},
             'timestamp': 1521663363189,
             'datetime': '2018-03-21T20:16:03.189Z',
             'symbol': 'LTC/ETH',
             'id': '34567',
             'order': '123456',
             'type': None,
             'side': 'buy',
             'price': 0.245441,
             'cost': 1.963528,
             'amount': 4.0,
             'fee': {'cost': 0.004, 'currency': 'LTC'}},
            {'info': {'id': 34567,
                      'orderId': 123456,
                      'price': '0.24544100',
                      'qty': '8.00000000',
                      'commission': '0.00800000',
                      'commissionAsset': 'LTC',
                      'time': 1521663363189,
                      'isBuyer': True,
                      'isMaker': False,
                      'isBestMatch': True},
             'timestamp': 1521663363189,
             'datetime': '2018-03-21T20:16:03.189Z',
             'symbol': 'LTC/ETH',
             'id': '34567',
             'order': '123456',
             'type': None,
             'side': 'buy',
             'price': 0.245441,
             'cost': 1.963528,
             'amount': 4.0,
             'fee': {'cost': 0.004, 'currency': 'LTC'}}]


@pytest.fixture
def buy_order_fee():
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'limit',
        'side': 'buy',
        'pair': 'mocked',
        'datetime': str(arrow.utcnow().shift(minutes=-601).datetime),
        'price': 0.245441,
        'amount': 8.0,
        'remaining': 90.99181073,
        'status': 'closed',
        'fee': None
    }
