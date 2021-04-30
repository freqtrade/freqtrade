# pragma pylint: disable=missing-docstring
import json
import logging
import re
from copy import deepcopy
from datetime import datetime
from functools import reduce
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock

import arrow
import numpy as np
import pytest
from telegram import Chat, Message, Update

from freqtrade import constants
from freqtrade.commands import Arguments
from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.edge import Edge, PairInfo
from freqtrade.exchange import Exchange
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import LocalTrade, Trade, init_db
from freqtrade.resolvers import ExchangeResolver
from freqtrade.worker import Worker
from tests.conftest_trades import (mock_trade_1, mock_trade_2, mock_trade_3, mock_trade_4,
                                   mock_trade_5, mock_trade_6)


logging.getLogger('').setLevel(logging.INFO)


# Do not mask numpy errors as warnings that no one read, raise the exсeption
np.seterr(all='raise')


def pytest_addoption(parser):
    parser.addoption('--longrun', action='store_true', dest="longrun",
                     default=False, help="Enable long-run tests (ccxt compat)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "longrun: mark test that is running slowly and should not be run regularily"
    )
    if not config.option.longrun:
        setattr(config.option, 'markexpr', 'not longrun')


def log_has(line, logs):
    # caplog mocker returns log as a tuple: ('freqtrade.something', logging.WARNING, 'foobar')
    # and we want to match line against foobar in the tuple
    return reduce(lambda a, b: a or b,
                  filter(lambda x: x[2] == line, logs.record_tuples),
                  False)


def log_has_re(line, logs):
    return reduce(lambda a, b: a or b,
                  filter(lambda x: re.match(line, x[2]), logs.record_tuples),
                  False)


def get_args(args):
    return Arguments(args).get_parsed_arg()


# Source: https://stackoverflow.com/questions/29881236/how-to-mock-asyncio-coroutines
def get_mock_coro(return_value):
    async def mock_coro(*args, **kwargs):
        return return_value

    return Mock(wraps=mock_coro)


def patched_configuration_load_config_file(mocker, config) -> None:
    mocker.patch(
        'freqtrade.configuration.configuration.load_config_file',
        lambda *args, **kwargs: config
    )


def patch_exchange(mocker, api_mock=None, id='binance', mock_markets=True) -> None:
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange.validate_ordertypes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange.id', PropertyMock(return_value=id))
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value=id.title()))
    mocker.patch('freqtrade.exchange.Exchange.precisionMode', PropertyMock(return_value=2))
    if mock_markets:
        mocker.patch('freqtrade.exchange.Exchange.markets',
                     PropertyMock(return_value=get_markets()))

    if api_mock:
        mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    else:
        mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock())


def get_patched_exchange(mocker, config, api_mock=None, id='binance',
                         mock_markets=True) -> Exchange:
    patch_exchange(mocker, api_mock, id, mock_markets)
    config['exchange']['name'] = id
    try:
        exchange = ExchangeResolver.load_exchange(id, config)
    except ImportError:
        exchange = Exchange(config)
    return exchange


def patch_wallet(mocker, free=999.9) -> None:
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(
        return_value=free
    ))


def patch_whitelist(mocker, conf) -> None:
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._refresh_active_whitelist',
                 MagicMock(return_value=conf['exchange']['pair_whitelist']))


def patch_edge(mocker) -> None:
    # "ETH/BTC",
    # "LTC/BTC",
    # "XRP/BTC",
    # "NEO/BTC"

    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'NEO/BTC': PairInfo(-0.20, 0.66, 3.71, 0.50, 1.71, 10, 25),
            'LTC/BTC': PairInfo(-0.21, 0.66, 3.71, 0.50, 1.71, 11, 20),
        }
    ))
    mocker.patch('freqtrade.edge.Edge.calculate', MagicMock(return_value=True))


def get_patched_edge(mocker, config) -> Edge:
    patch_edge(mocker)
    edge = Edge(config)
    return edge

# Functions for recurrent object patching


def patch_freqtradebot(mocker, config) -> None:
    """
    This function patch _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: None
    """
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    init_db(config['db_url'])
    patch_exchange(mocker)
    mocker.patch('freqtrade.freqtradebot.RPCManager._init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.RPCManager.send_msg', MagicMock())
    patch_whitelist(mocker, config)


def get_patched_freqtradebot(mocker, config) -> FreqtradeBot:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: FreqtradeBot
    """
    patch_freqtradebot(mocker, config)
    config['datadir'] = Path(config['datadir'])
    return FreqtradeBot(config)


def get_patched_worker(mocker, config) -> Worker:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: Worker
    """
    patch_freqtradebot(mocker, config)
    return Worker(args=None, config=config)


def patch_get_signal(freqtrade: FreqtradeBot, value=(True, False)) -> None:
    """
    :param mocker: mocker to patch IStrategy class
    :param value: which value IStrategy.get_signal() must return
    :return: None
    """
    freqtrade.strategy.get_signal = lambda e, s, x: value
    freqtrade.exchange.refresh_latest_ohlcv = lambda p: None


def create_mock_trades(fee, use_db: bool = True):
    """
    Create some fake trades ...
    """
    def add_trade(trade):
        if use_db:
            Trade.query.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    # Simulate dry_run entries
    trade = mock_trade_1(fee)
    add_trade(trade)

    trade = mock_trade_2(fee)
    add_trade(trade)

    trade = mock_trade_3(fee)
    add_trade(trade)

    trade = mock_trade_4(fee)
    add_trade(trade)

    trade = mock_trade_5(fee)
    add_trade(trade)

    trade = mock_trade_6(fee)
    add_trade(trade)

    if use_db:
        Trade.query.session.flush()


@pytest.fixture(autouse=True)
def patch_coingekko(mocker) -> None:
    """
    Mocker to coingekko to speed up tests
    :param mocker: mocker to patch coingekko class
    :return: None
    """

    tickermock = MagicMock(return_value={'bitcoin': {'usd': 12345.0}, 'ethereum': {'usd': 12345.0}})
    listmock = MagicMock(return_value=[{'id': 'bitcoin', 'name': 'Bitcoin', 'symbol': 'btc',
                                        'website_slug': 'bitcoin'},
                                       {'id': 'ethereum', 'name': 'Ethereum', 'symbol': 'eth',
                                        'website_slug': 'ethereum'}
                                       ])
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.CoinGeckoAPI',
        get_price=tickermock,
        get_coins_list=listmock,

    )


@pytest.fixture(scope='function')
def init_persistence(default_conf):
    init_db(default_conf['db_url'], default_conf['dry_run'])


@pytest.fixture(scope="function")
def default_conf(testdatadir):
    return get_default_conf(testdatadir)


def get_default_conf(testdatadir):
    """ Returns validated configuration suitable for most tests """
    configuration = {
        "max_open_trades": 1,
        "stake_currency": "BTC",
        "stake_amount": 0.001,
        "fiat_display_currency": "USD",
        "timeframe": '5m',
        "dry_run": True,
        "cancel_open_orders_on_exit": False,
        "minimal_roi": {
            "40": 0.0,
            "30": 0.01,
            "20": 0.02,
            "0": 0.04
        },
        "dry_run_wallet": 1000,
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
            "name": "binance",
            "enabled": True,
            "key": "key",
            "secret": "secret",
            "pair_whitelist": [
                "ETH/BTC",
                "LTC/BTC",
                "XRP/BTC",
                "NEO/BTC"
            ],
            "pair_blacklist": [
                "DOGE/BTC",
                "HOT/BTC",
            ]
        },
        "pairlists": [
            {"method": "StaticPairList"}
        ],
        "telegram": {
            "enabled": True,
            "token": "token",
            "chat_id": "0",
            "notification_settings": {},
        },
        "datadir": str(testdatadir),
        "initial_state": "running",
        "db_url": "sqlite://",
        "user_data_dir": Path("user_data"),
        "verbosity": 3,
        "strategy_path": str(Path(__file__).parent / "strategy" / "strats"),
        "strategy": "DefaultStrategy",
        "internals": {},
    }
    return configuration


@pytest.fixture
def update():
    _update = Update(0)
    _update.message = Message(0, datetime.utcnow(), Chat(0, 0))
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
    return get_markets()


def get_markets():
    return {
        'ETH/BTC': {
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
                    'min': 0.0001,
                    'max': 500000,
                },
            },
            'info': {},
        },
        'TKN/BTC': {
            'id': 'tknbtc',
            'symbol': 'TKN/BTC',
            'base': 'TKN',
            'quote': 'BTC',
            # According to ccxt, markets without active item set are also active
            # 'active': True,
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
                    'min': 0.0001,
                    'max': 500000,
                },
            },
            'info': {},
        },
        'BLK/BTC': {
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
                    'min': 0.0001,
                    'max': 500000,
                },
            },
            'info': {},
        },
        'LTC/BTC': {
            'id': 'ltcbtc',
            'symbol': 'LTC/BTC',
            'base': 'LTC',
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
                    'min': 0.0001,
                    'max': 500000,
                },
            },
            'info': {},
        },
        'XRP/BTC': {
            'id': 'xrpbtc',
            'symbol': 'XRP/BTC',
            'base': 'XRP',
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
                    'min': 0.0001,
                    'max': 500000,
                },
            },
            'info': {},
        },
        'NEO/BTC': {
            'id': 'neobtc',
            'symbol': 'NEO/BTC',
            'base': 'NEO',
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
                    'min': 0.0001,
                    'max': 500000,
                },
            },
            'info': {},
        },
        'BTT/BTC': {
            'id': 'BTTBTC',
            'symbol': 'BTT/BTC',
            'base': 'BTT',
            'quote': 'BTC',
            'active': False,
            'precision': {
                'base': 8,
                'quote': 8,
                'amount': 0,
                'price': 8
            },
            'limits': {
                'amount': {
                    'min': 1.0,
                    'max': 90000000.0
                },
                'price': {
                    'min': None,
                    'max': None
                },
                'cost': {
                    'min': 0.0001,
                    'max': None
                }
            },
            'info': {},
        },
        'ETH/USDT': {
            'id': 'USDT-ETH',
            'symbol': 'ETH/USDT',
            'base': 'ETH',
            'quote': 'USDT',
            'precision': {
                'amount': 8,
                'price': 8
            },
            'limits': {
                'amount': {
                    'min': 0.02214286,
                    'max': None
                },
                'price': {
                    'min': 1e-08,
                    'max': None
                }
            },
            'active': True,
            'info': {},
        },
        'LTC/USDT': {
            'id': 'USDT-LTC',
            'symbol': 'LTC/USDT',
            'base': 'LTC',
            'quote': 'USDT',
            'active': False,
            'precision': {
                'amount': 8,
                'price': 8
            },
            'limits': {
                'amount': {
                    'min': 0.06646786,
                    'max': None
                },
                'price': {
                    'min': 1e-08,
                    'max': None
                }
            },
            'info': {},
        },
        'LTC/USD': {
            'id': 'USD-LTC',
            'symbol': 'LTC/USD',
            'base': 'LTC',
            'quote': 'USD',
            'active': True,
            'precision': {
                'amount': 8,
                'price': 8
            },
            'limits': {
                'amount': {
                    'min': 0.06646786,
                    'max': None
                },
                'price': {
                    'min': 1e-08,
                    'max': None
                }
            },
            'info': {},
        },
        'XLTCUSDT': {
            'id': 'xLTCUSDT',
            'symbol': 'XLTCUSDT',
            'base': 'LTC',
            'quote': 'USDT',
            'active': True,
            'precision': {
                'amount': 8,
                'price': 8
            },
            'limits': {
                'amount': {
                    'min': 0.06646786,
                    'max': None
                },
                'price': {
                    'min': 1e-08,
                    'max': None
                }
            },
            'info': {},
        },
        'LTC/ETH': {
            'id': 'LTCETH',
            'symbol': 'LTC/ETH',
            'base': 'LTC',
            'quote': 'ETH',
            'active': True,
            'precision': {
                'base': 8,
                'quote': 8,
                'amount': 3,
                'price': 5
            },
            'limits': {
                'amount': {
                    'min': 0.001,
                    'max': 10000000.0
                },
                'price': {
                    'min': 1e-05,
                    'max': 1000.0
                },
                'cost': {
                    'min': 0.01,
                    'max': None
                }
            },
        },
    }


@pytest.fixture
def shitcoinmarkets(markets):
    """
    Fixture with shitcoin markets - used to test filters in pairlists
    """
    shitmarkets = deepcopy(markets)
    shitmarkets.update({
        'HOT/BTC': {
            'id': 'HOTBTC',
            'symbol': 'HOT/BTC',
            'base': 'HOT',
            'quote': 'BTC',
            'active': True,
            'precision': {
                'base': 8,
                'quote': 8,
                'amount': 0,
                'price': 8
            },
            'limits': {
                'amount': {
                    'min': 1.0,
                    'max': 90000000.0
                },
                'price': {
                    'min': None,
                    'max': None
                },
                'cost': {
                    'min': 0.001,
                    'max': None
                }
            },
            'info': {},
        },
        'FUEL/BTC': {
            'id': 'FUELBTC',
            'symbol': 'FUEL/BTC',
            'base': 'FUEL',
            'quote': 'BTC',
            'active': True,
            'precision': {
                'base': 8,
                'quote': 8,
                'amount': 0,
                'price': 8
            },
            'limits': {
                'amount': {
                    'min': 1.0,
                    'max': 90000000.0
                },
                'price': {
                    'min': 1e-08,
                    'max': 1000.0
                },
                'cost': {
                    'min': 0.001,
                    'max': None
                }
            },
            'info': {},
        },
        'NANO/USDT': {
            "percentage": True,
            "tierBased": False,
            "taker": 0.001,
            "maker": 0.001,
            "precision": {
                "base": 8,
                "quote": 8,
                "amount": 2,
                "price": 4
            },
            "limits": {
            },
            "id": "NANOUSDT",
            "symbol": "NANO/USDT",
            "base": "NANO",
            "quote": "USDT",
            "baseId": "NANO",
            "quoteId": "USDT",
            "info": {},
            "type": "spot",
            "spot": True,
            "future": False,
            "active": True
        },
        'ADAHALF/USDT': {
            "percentage": True,
            "tierBased": False,
            "taker": 0.001,
            "maker": 0.001,
            "precision": {
                "base": 8,
                "quote": 8,
                "amount": 2,
                "price": 4
            },
            "limits": {
            },
            "id": "ADAHALFUSDT",
            "symbol": "ADAHALF/USDT",
            "base": "ADAHALF",
            "quote": "USDT",
            "baseId": "ADAHALF",
            "quoteId": "USDT",
            "info": {},
            "type": "spot",
            "spot": True,
            "future": False,
            "active": True
        },
        'ADADOUBLE/USDT': {
            "percentage": True,
            "tierBased": False,
            "taker": 0.001,
            "maker": 0.001,
            "precision": {
                "base": 8,
                "quote": 8,
                "amount": 2,
                "price": 4
            },
            "limits": {
            },
            "id": "ADADOUBLEUSDT",
            "symbol": "ADADOUBLE/USDT",
            "base": "ADADOUBLE",
            "quote": "USDT",
            "baseId": "ADADOUBLE",
            "quoteId": "USDT",
            "info": {},
            "type": "spot",
            "spot": True,
            "future": False,
            "active": True
        },
        })
    return shitmarkets


@pytest.fixture
def markets_empty():
    return MagicMock(return_value=[])


@pytest.fixture(scope='function')
def limit_buy_order_open():
    return {
        'id': 'mocked_limit_buy',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'datetime': arrow.utcnow().isoformat(),
        'timestamp': arrow.utcnow().int_timestamp,
        'price': 0.00001099,
        'amount': 90.99181073,
        'filled': 0.0,
        'cost': 0.0009999,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture(scope='function')
def limit_buy_order(limit_buy_order_open):
    order = deepcopy(limit_buy_order_open)
    order['status'] = 'closed'
    order['filled'] = order['amount']
    order['remaining'] = 0.0
    return order


@pytest.fixture(scope='function')
def market_buy_order():
    return {
        'id': 'mocked_market_buy',
        'type': 'market',
        'side': 'buy',
        'symbol': 'mocked',
        'datetime': arrow.utcnow().isoformat(),
        'price': 0.00004099,
        'amount': 91.99181073,
        'filled': 91.99181073,
        'remaining': 0.0,
        'status': 'closed'
    }


@pytest.fixture
def market_sell_order():
    return {
        'id': 'mocked_limit_sell',
        'type': 'market',
        'side': 'sell',
        'symbol': 'mocked',
        'datetime': arrow.utcnow().isoformat(),
        'price': 0.00004173,
        'amount': 91.99181073,
        'filled': 91.99181073,
        'remaining': 0.0,
        'status': 'closed'
    }


@pytest.fixture
def limit_buy_order_old():
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'datetime': str(arrow.utcnow().shift(minutes=-601).datetime),
        'price': 0.00001099,
        'amount': 90.99181073,
        'filled': 0.0,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_sell_order_old():
    return {
        'id': 'mocked_limit_sell_old',
        'type': 'limit',
        'side': 'sell',
        'symbol': 'ETH/BTC',
        'datetime': arrow.utcnow().shift(minutes=-601).isoformat(),
        'price': 0.00001099,
        'amount': 90.99181073,
        'filled': 0.0,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_buy_order_old_partial():
    return {
        'id': 'mocked_limit_buy_old_partial',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'ETH/BTC',
        'datetime': arrow.utcnow().shift(minutes=-601).isoformat(),
        'price': 0.00001099,
        'amount': 90.99181073,
        'filled': 23.0,
        'remaining': 67.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_buy_order_old_partial_canceled(limit_buy_order_old_partial):
    res = deepcopy(limit_buy_order_old_partial)
    res['status'] = 'canceled'
    res['fee'] = {'cost': 0.023, 'currency': 'ETH'}
    return res


@pytest.fixture(scope='function')
def limit_buy_order_canceled_empty(request):
    # Indirect fixture
    # Documentation:
    # https://docs.pytest.org/en/latest/example/parametrize.html#apply-indirect-on-particular-arguments

    exchange_name = request.param
    if exchange_name == 'ftx':
        return {
            'info': {},
            'id': '1234512345',
            'clientOrderId': None,
            'timestamp': arrow.utcnow().shift(minutes=-601).int_timestamp,
            'datetime': arrow.utcnow().shift(minutes=-601).isoformat(),
            'lastTradeTimestamp': None,
            'symbol': 'LTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'price': 34.3225,
            'amount': 0.55,
            'cost': 0.0,
            'average': None,
            'filled': 0.0,
            'remaining': 0.0,
            'status': 'closed',
            'fee': None,
            'trades': None
        }
    elif exchange_name == 'kraken':
        return {
            'info': {},
            'id': 'AZNPFF-4AC4N-7MKTAT',
            'clientOrderId': None,
            'timestamp': arrow.utcnow().shift(minutes=-601).int_timestamp,
            'datetime': arrow.utcnow().shift(minutes=-601).isoformat(),
            'lastTradeTimestamp': None,
            'status': 'canceled',
            'symbol': 'LTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'price': 34.3225,
            'cost': 0.0,
            'amount': 0.55,
            'filled': 0.0,
            'average': 0.0,
            'remaining': 0.55,
            'fee': {'cost': 0.0, 'rate': None, 'currency': 'USDT'},
            'trades': []
        }
    elif exchange_name == 'binance':
        return {
            'info': {},
            'id': '1234512345',
            'clientOrderId': 'alb1234123',
            'timestamp': arrow.utcnow().shift(minutes=-601).int_timestamp,
            'datetime': arrow.utcnow().shift(minutes=-601).isoformat(),
            'lastTradeTimestamp': None,
            'symbol': 'LTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'price': 0.016804,
            'amount': 0.55,
            'cost': 0.0,
            'average': None,
            'filled': 0.0,
            'remaining': 0.55,
            'status': 'canceled',
            'fee': None,
            'trades': None
        }
    else:
        return {
            'info': {},
            'id': '1234512345',
            'clientOrderId': 'alb1234123',
            'timestamp': arrow.utcnow().shift(minutes=-601).int_timestamp,
            'datetime': arrow.utcnow().shift(minutes=-601).isoformat(),
            'lastTradeTimestamp': None,
            'symbol': 'LTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'price': 0.016804,
            'amount': 0.55,
            'cost': 0.0,
            'average': None,
            'filled': 0.0,
            'remaining': 0.55,
            'status': 'canceled',
            'fee': None,
            'trades': None
        }


@pytest.fixture
def limit_sell_order_open():
    return {
        'id': 'mocked_limit_sell',
        'type': 'limit',
        'side': 'sell',
        'pair': 'mocked',
        'datetime': arrow.utcnow().isoformat(),
        'timestamp': arrow.utcnow().int_timestamp,
        'price': 0.00001173,
        'amount': 90.99181073,
        'filled': 0.0,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_sell_order(limit_sell_order_open):
    order = deepcopy(limit_sell_order_open)
    order['remaining'] = 0.0
    order['filled'] = order['amount']
    order['status'] = 'closed'
    return order


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
def ohlcv_history_list():
    return [
        [
            1511686200000,  # unix timestamp ms
            8.794e-05,      # open
            8.948e-05,      # high
            8.794e-05,      # low
            8.88e-05,       # close
            0.0877869,      # volume (in quote currency)
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
def ohlcv_history(ohlcv_history_list):
    return ohlcv_to_dataframe(ohlcv_history_list, "5m", pair="UNITTEST/BTC",
                              fill_missing=True, drop_incomplete=False)


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
            'last': 0.018799,
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
        'BTT/BTC': {
            'symbol': 'BTT/BTC',
            'timestamp': 1550936557206,
            'datetime': '2019-02-23T15:42:37.206Z',
            'high': 0.00000026,
            'low': 0.00000024,
            'bid': 0.00000024,
            'bidVolume': 2446894197.0,
            'ask': 0.00000025,
            'askVolume': 2447913837.0,
            'vwap': 0.00000025,
            'open': 0.00000026,
            'close': 0.00000024,
            'last': 0.00000024,
            'previousClose': 0.00000026,
            'change': -0.00000002,
            'percentage': -7.692,
            'average': None,
            'baseVolume': 4886464537.0,
            'quoteVolume': 1215.14489611,
            'info': {}
        },
        'HOT/BTC': {
            'symbol': 'HOT/BTC',
            'timestamp': 1572273518661,
            'datetime': '2019-10-28T14:38:38.661Z',
            'high': 0.00000011,
            'low': 0.00000009,
            'bid': 0.0000001,
            'bidVolume': 1476027288.0,
            'ask': 0.00000011,
            'askVolume': 820153831.0,
            'vwap': 0.0000001,
            'open': 0.00000009,
            'close': 0.00000011,
            'last': 0.00000011,
            'previousClose': 0.00000009,
            'change': 0.00000002,
            'percentage': 22.222,
            'average': None,
            'baseVolume': 1442290324.0,
            'quoteVolume': 143.78311994,
            'info': {}
        },
        'FUEL/BTC': {
            'symbol': 'FUEL/BTC',
            'timestamp': 1572340250771,
            'datetime': '2019-10-29T09:10:50.771Z',
            'high': 0.00000040,
            'low': 0.00000035,
            'bid': 0.00000036,
            'bidVolume': 8932318.0,
            'ask': 0.00000037,
            'askVolume': 10140774.0,
            'vwap': 0.00000037,
            'open': 0.00000039,
            'close': 0.00000037,
            'last': 0.00000037,
            'previousClose': 0.00000038,
            'change': -0.00000002,
            'percentage': -5.128,
            'average': None,
            'baseVolume': 168927742.0,
            'quoteVolume': 62.68220262,
            'info': {}
        },
        'BTC/USDT': {
            'symbol': 'BTC/USDT',
            'timestamp': 1573758371399,
            'datetime': '2019-11-14T19:06:11.399Z',
            'high': 8800.0,
            'low': 8582.6,
            'bid': 8648.16,
            'bidVolume': 0.238771,
            'ask': 8648.72,
            'askVolume': 0.016253,
            'vwap': 8683.13647806,
            'open': 8759.7,
            'close': 8648.72,
            'last': 8648.72,
            'previousClose': 8759.67,
            'change': -110.98,
            'percentage': -1.267,
            'average': None,
            'baseVolume': 35025.943355,
            'quoteVolume': 304135046.4242901,
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
        },
        'XRP/BTC': {
            'symbol': 'XRP/BTC',
            'timestamp': 1573758257534,
            'datetime': '2019-11-14T19:04:17.534Z',
            'high': 3.126e-05,
            'low': 3.061e-05,
            'bid': 3.093e-05,
            'bidVolume': 27901.0,
            'ask': 3.095e-05,
            'askVolume': 10551.0,
            'vwap': 3.091e-05,
            'open': 3.119e-05,
            'close': 3.094e-05,
            'last': 3.094e-05,
            'previousClose': 3.117e-05,
            'change': -2.5e-07,
            'percentage': -0.802,
            'average': None,
            'baseVolume': 37334921.0,
            'quoteVolume': 1154.19266394,
            'info': {}
        },
        "NANO/USDT": {
            "symbol": "NANO/USDT",
            "timestamp": 1580469388244,
            "datetime": "2020-01-31T11:16:28.244Z",
            "high": 0.7519,
            "low": 0.7154,
            "bid": 0.7305,
            "bidVolume": 300.3,
            "ask": 0.7342,
            "askVolume": 15.14,
            "vwap": 0.73645591,
            "open": 0.7154,
            "close": 0.7342,
            "last": 0.7342,
            "previousClose": 0.7189,
            "change": 0.0188,
            "percentage": 2.628,
            "average": None,
            "baseVolume": 439472.44,
            "quoteVolume": 323652.075405,
            "info": {}
        },
        # Example of leveraged pair with incomplete info
        "ADAHALF/USDT": {
            "symbol": "ADAHALF/USDT",
            "timestamp": 1580469388244,
            "datetime": "2020-01-31T11:16:28.244Z",
            "high": None,
            "low": None,
            "bid": 0.7305,
            "bidVolume": None,
            "ask": 0.7342,
            "askVolume": None,
            "vwap": None,
            "open": None,
            "close": None,
            "last": None,
            "previousClose": None,
            "change": None,
            "percentage": 2.628,
            "average": None,
            "baseVolume": 0.0,
            "quoteVolume": 0.0,
            "info": {}
        },
        "ADADOUBLE/USDT": {
            "symbol": "ADADOUBLE/USDT",
            "timestamp": 1580469388244,
            "datetime": "2020-01-31T11:16:28.244Z",
            "high": None,
            "low": None,
            "bid": 0.7305,
            "bidVolume": None,
            "ask": 0.7342,
            "askVolume": None,
            "vwap": None,
            "open": None,
            "close": None,
            "last": 0,
            "previousClose": None,
            "change": None,
            "percentage": 2.628,
            "average": None,
            "baseVolume": 0.0,
            "quoteVolume": 0.0,
            "info": {}
        },
    })


@pytest.fixture
def result(testdatadir):
    with (testdatadir / 'UNITTEST_BTC-1m.json').open('r') as data_file:
        return ohlcv_to_dataframe(json.load(data_file), '1m', pair="UNITTEST/BTC",
                                  fill_missing=True)


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
def trades_history():
    return [[1565798389463, '12618132aa9', None, 'buy', 0.019627, 0.04, 0.00078508],
            [1565798399629, '1261813bb30', None, 'buy', 0.019627, 0.244, 0.004788987999999999],
            [1565798399752, '1261813cc31', None, 'sell', 0.019626, 0.011, 0.00021588599999999999],
            [1565798399862, '126181cc332', None, 'sell', 0.019626, 0.011, 0.00021588599999999999],
            [1565798399872, '1261aa81333', None, 'sell', 0.019626, 0.011, 0.00021588599999999999]]


@pytest.fixture(scope="function")
def fetch_trades_result():
    return [{'info': {'a': 126181329,
                      'p': '0.01962700',
                      'q': '0.04000000',
                      'f': 138604155,
                      'l': 138604155,
                      'T': 1565798399463,
                      'm': False,
                      'M': True},
             'timestamp': 1565798399463,
             'datetime': '2019-08-14T15:59:59.463Z',
             'symbol': 'ETH/BTC',
             'id': '126181329',
             'order': None,
             'type': None,
             'takerOrMaker': None,
             'side': 'buy',
             'price': 0.019627,
             'amount': 0.04,
             'cost': 0.00078508,
             'fee': None},
            {'info': {'a': 126181330,
                      'p': '0.01962700',
                      'q': '0.24400000',
                      'f': 138604156,
                      'l': 138604156,
                      'T': 1565798399629,
                      'm': False,
                      'M': True},
             'timestamp': 1565798399629,
             'datetime': '2019-08-14T15:59:59.629Z',
             'symbol': 'ETH/BTC',
             'id': '126181330',
             'order': None,
             'type': None,
             'takerOrMaker': None,
             'side': 'buy',
             'price': 0.019627,
             'amount': 0.244,
             'cost': 0.004788987999999999,
             'fee': None},
            {'info': {'a': 126181331,
                      'p': '0.01962600',
                      'q': '0.01100000',
                      'f': 138604157,
                      'l': 138604157,
                      'T': 1565798399752,
                      'm': True,
                      'M': True},
             'timestamp': 1565798399752,
             'datetime': '2019-08-14T15:59:59.752Z',
             'symbol': 'ETH/BTC',
             'id': '126181331',
             'order': None,
             'type': None,
             'takerOrMaker': None,
             'side': 'sell',
             'price': 0.019626,
             'amount': 0.011,
             'cost': 0.00021588599999999999,
             'fee': None},
            {'info': {'a': 126181332,
                      'p': '0.01962600',
                      'q': '0.01100000',
                      'f': 138604158,
                      'l': 138604158,
                      'T': 1565798399862,
                      'm': True,
                      'M': True},
             'timestamp': 1565798399862,
             'datetime': '2019-08-14T15:59:59.862Z',
             'symbol': 'ETH/BTC',
             'id': '126181332',
             'order': None,
             'type': None,
             'takerOrMaker': None,
             'side': 'sell',
             'price': 0.019626,
             'amount': 0.011,
             'cost': 0.00021588599999999999,
             'fee': None},
            {'info': {'a': 126181333,
                      'p': '0.01952600',
                      'q': '0.01200000',
                      'f': 138604158,
                      'l': 138604158,
                      'T': 1565798399872,
                      'm': True,
                      'M': True},
             'timestamp': 1565798399872,
             'datetime': '2019-08-14T15:59:59.872Z',
             'symbol': 'ETH/BTC',
             'id': '126181333',
             'order': None,
             'type': None,
             'takerOrMaker': None,
             'side': 'sell',
             'price': 0.019626,
             'amount': 0.011,
             'cost': 0.00021588599999999999,
             'fee': None}]


@pytest.fixture(scope="function")
def trades_for_order2():
    return [{'info': {},
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
            {'info': {},
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


@pytest.fixture(scope="function")
def trades_for_order3(trades_for_order2):
    # Different fee currencies for each trade
    trades_for_order = deepcopy(trades_for_order2)
    trades_for_order[0]['fee'] = {'cost': 0.02, 'currency': 'BNB'}
    return trades_for_order


@pytest.fixture
def buy_order_fee():
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'datetime': str(arrow.utcnow().shift(minutes=-601).datetime),
        'price': 0.245441,
        'amount': 8.0,
        'cost': 1.963528,
        'remaining': 90.99181073,
        'status': 'closed',
        'fee': None
    }


@pytest.fixture(scope="function")
def edge_conf(default_conf):
    conf = deepcopy(default_conf)
    conf['max_open_trades'] = -1
    conf['tradable_balance_ratio'] = 0.5
    conf['stake_amount'] = constants.UNLIMITED_STAKE_AMOUNT
    conf['edge'] = {
        "enabled": True,
        "process_throttle_secs": 1800,
        "calculate_since_number_of_days": 14,
        "allowed_risk": 0.01,
        "stoploss_range_min": -0.01,
        "stoploss_range_max": -0.1,
        "stoploss_range_step": -0.01,
        "maximum_winrate": 0.80,
        "minimum_expectancy": 0.20,
        "min_trade_number": 15,
        "max_trade_duration_minute": 1440,
        "remove_pumps": False
    }

    return conf


@pytest.fixture
def rpc_balance():
    return {
        'BTC': {
            'total': 12.0,
            'free': 12.0,
            'used': 0.0
        },
        'ETH': {
            'total': 0.0,
            'free': 0.0,
            'used': 0.0
        },
        'USDT': {
            'total': 10000.0,
            'free': 10000.0,
            'used': 0.0
        },
        'LTC': {
            'total': 10.0,
            'free': 10.0,
            'used': 0.0
        },
        'XRP': {
            'total': 0.1,
            'free': 0.01,
            'used': 0.0
            },
        'EUR': {
            'total': 10.0,
            'free': 10.0,
            'used': 0.0
        },
    }


@pytest.fixture
def testdatadir() -> Path:
    """Return the path where testdata files are stored"""
    return (Path(__file__).parent / "testdata").resolve()


@pytest.fixture(scope="function")
def import_fails() -> None:
    # Source of this test-method:
    # https://stackoverflow.com/questions/2481511/mocking-importerror-in-python
    import builtins
    realimport = builtins.__import__

    def mockedimport(name, *args, **kwargs):
        if name in ["filelock", 'systemd.journal', 'uvloop']:
            raise ImportError(f"No module named '{name}'")
        return realimport(name, *args, **kwargs)

    builtins.__import__ = mockedimport

    # Run test - then cleanup
    yield

    # restore previous importfunction
    builtins.__import__ = realimport


@pytest.fixture(scope="function")
def open_trade():
    return Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='binance',
        open_order_id='123456789',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=True
    )


@pytest.fixture
def hyperopt_results():
    return [
        {
            'loss': 0.4366182531161519,
            'params_dict': {
                'mfi-value': 15, 'fastd-value': 20, 'adx-value': 25, 'rsi-value': 28, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 88, 'sell-fastd-value': 97, 'sell-adx-value': 51, 'sell-rsi-value': 67, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper', 'roi_t1': 1190, 'roi_t2': 541, 'roi_t3': 408, 'roi_p1': 0.026035863879169705, 'roi_p2': 0.12508730043628782, 'roi_p3': 0.27766427921605896, 'stoploss': -0.2562930402099556},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 15, 'fastd-value': 20, 'adx-value': 25, 'rsi-value': 28, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 88, 'sell-fastd-value': 97, 'sell-adx-value': 51, 'sell-rsi-value': 67, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper'}, 'roi': {0: 0.4287874435315165, 408: 0.15112316431545753, 949: 0.026035863879169705, 2139: 0}, 'stoploss': {'stoploss': -0.2562930402099556}},  # noqa: E501
            'results_metrics': {'trade_count': 2, 'avg_profit': -1.254995, 'median_profit': -1.2222, 'total_profit': -0.00125625, 'profit': -2.50999, 'duration': 3930.0},  # noqa: E501
            'results_explanation': '     2 trades. Avg profit  -1.25%. Total profit -0.00125625 BTC (  -2.51Σ%). Avg duration 3930.0 min.',  # noqa: E501
            'total_profit': -0.00125625,
            'current_epoch': 1,
            'is_initial_point': True,
            'is_best': True
        }, {
            'loss': 20.0,
            'params_dict': {
                'mfi-value': 17, 'fastd-value': 38, 'adx-value': 48, 'rsi-value': 22, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 96, 'sell-fastd-value': 68, 'sell-adx-value': 63, 'sell-rsi-value': 81, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal', 'roi_t1': 334, 'roi_t2': 683, 'roi_t3': 140, 'roi_p1': 0.06403981740598495, 'roi_p2': 0.055519840060645045, 'roi_p3': 0.3253712811342459, 'stoploss': -0.338070047333259},  # noqa: E501
            'params_details': {
                'buy': {'mfi-value': 17, 'fastd-value': 38, 'adx-value': 48, 'rsi-value': 22, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'macd_cross_signal'},  # noqa: E501
                'sell': {'sell-mfi-value': 96, 'sell-fastd-value': 68, 'sell-adx-value': 63, 'sell-rsi-value': 81, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal'},  # noqa: E501
                'roi': {0: 0.4449309386008759, 140: 0.11955965746663, 823: 0.06403981740598495, 1157: 0},  # noqa: E501
                'stoploss': {'stoploss': -0.338070047333259}},
            'results_metrics': {'trade_count': 1, 'avg_profit': 0.12357, 'median_profit': -1.2222, 'total_profit': 6.185e-05, 'profit': 0.12357, 'duration': 1200.0},  # noqa: E501
            'results_explanation': '     1 trades. Avg profit   0.12%. Total profit  0.00006185 BTC (   0.12Σ%). Avg duration 1200.0 min.',  # noqa: E501
            'total_profit': 6.185e-05,
            'current_epoch': 2,
            'is_initial_point': True,
            'is_best': False
        }, {
            'loss': 14.241196856510731,
            'params_dict': {'mfi-value': 25, 'fastd-value': 16, 'adx-value': 29, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 98, 'sell-fastd-value': 72, 'sell-adx-value': 51, 'sell-rsi-value': 82, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 889, 'roi_t2': 533, 'roi_t3': 263, 'roi_p1': 0.04759065393663096, 'roi_p2': 0.1488819964638463, 'roi_p3': 0.4102801822104605, 'stoploss': -0.05394588767607611},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 25, 'fastd-value': 16, 'adx-value': 29, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 98, 'sell-fastd-value': 72, 'sell-adx-value': 51, 'sell-rsi-value': 82, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.6067528326109377, 263: 0.19647265040047726, 796: 0.04759065393663096, 1685: 0}, 'stoploss': {'stoploss': -0.05394588767607611}},  # noqa: E501
            'results_metrics': {'trade_count': 621, 'avg_profit': -0.43883302093397747, 'median_profit': -1.2222, 'total_profit': -0.13639474, 'profit': -272.515306, 'duration': 1691.207729468599},  # noqa: E501
            'results_explanation': '   621 trades. Avg profit  -0.44%. Total profit -0.13639474 BTC (-272.52Σ%). Avg duration 1691.2 min.',  # noqa: E501
            'total_profit': -0.13639474,
            'current_epoch': 3,
            'is_initial_point': True,
            'is_best': False
        }, {
            'loss': 100000,
            'params_dict': {'mfi-value': 13, 'fastd-value': 35, 'adx-value': 39, 'rsi-value': 29, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': True, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 87, 'sell-fastd-value': 54, 'sell-adx-value': 63, 'sell-rsi-value': 93, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper', 'roi_t1': 1402, 'roi_t2': 676, 'roi_t3': 215, 'roi_p1': 0.06264755784937427, 'roi_p2': 0.14258587851894644, 'roi_p3': 0.20671291201040828, 'stoploss': -0.11818343570194478},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 13, 'fastd-value': 35, 'adx-value': 39, 'rsi-value': 29, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': True, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 87, 'sell-fastd-value': 54, 'sell-adx-value': 63, 'sell-rsi-value': 93, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper'}, 'roi': {0: 0.411946348378729, 215: 0.2052334363683207, 891: 0.06264755784937427, 2293: 0}, 'stoploss': {'stoploss': -0.11818343570194478}},  # noqa: E501
            'results_metrics': {'trade_count': 0, 'avg_profit': None, 'median_profit': None, 'total_profit': 0, 'profit': 0.0, 'duration': None},  # noqa: E501
            'results_explanation': '     0 trades. Avg profit    nan%. Total profit  0.00000000 BTC (   0.00Σ%). Avg duration   nan min.',  # noqa: E501
            'total_profit': 0, 'current_epoch': 4, 'is_initial_point': True, 'is_best': False
        }, {
            'loss': 0.22195522184191518,
            'params_dict': {'mfi-value': 17, 'fastd-value': 21, 'adx-value': 38, 'rsi-value': 33, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': False, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 87, 'sell-fastd-value': 82, 'sell-adx-value': 78, 'sell-rsi-value': 69, 'sell-mfi-enabled': True, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': False, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 1269, 'roi_t2': 601, 'roi_t3': 444, 'roi_p1': 0.07280999507931168, 'roi_p2': 0.08946698095898986, 'roi_p3': 0.1454876733325284, 'stoploss': -0.18181041180901014},   # noqa: E501
            'params_details': {'buy': {'mfi-value': 17, 'fastd-value': 21, 'adx-value': 38, 'rsi-value': 33, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': False, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 87, 'sell-fastd-value': 82, 'sell-adx-value': 78, 'sell-rsi-value': 69, 'sell-mfi-enabled': True, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': False, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.3077646493708299, 444: 0.16227697603830155, 1045: 0.07280999507931168, 2314: 0}, 'stoploss': {'stoploss': -0.18181041180901014}},  # noqa: E501
            'results_metrics': {'trade_count': 14, 'avg_profit': -0.3539515, 'median_profit': -1.2222, 'total_profit': -0.002480140000000001, 'profit': -4.955321, 'duration': 3402.8571428571427},  # noqa: E501
            'results_explanation': '    14 trades. Avg profit  -0.35%. Total profit -0.00248014 BTC (  -4.96Σ%). Avg duration 3402.9 min.',  # noqa: E501
            'total_profit': -0.002480140000000001,
            'current_epoch': 5,
            'is_initial_point': True,
            'is_best': True
        }, {
            'loss': 0.545315889154162,
            'params_dict': {'mfi-value': 22, 'fastd-value': 43, 'adx-value': 46, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'bb_lower', 'sell-mfi-value': 87, 'sell-fastd-value': 65, 'sell-adx-value': 94, 'sell-rsi-value': 63, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 319, 'roi_t2': 556, 'roi_t3': 216, 'roi_p1': 0.06251955472249589, 'roi_p2': 0.11659519602202795, 'roi_p3': 0.0953744132197762, 'stoploss': -0.024551752215582423},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 22, 'fastd-value': 43, 'adx-value': 46, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'bb_lower'}, 'sell': {'sell-mfi-value': 87, 'sell-fastd-value': 65, 'sell-adx-value': 94, 'sell-rsi-value': 63, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.2744891639643, 216: 0.17911475074452382, 772: 0.06251955472249589, 1091: 0}, 'stoploss': {'stoploss': -0.024551752215582423}},  # noqa: E501
            'results_metrics': {'trade_count': 39, 'avg_profit': -0.21400679487179478, 'median_profit': -1.2222, 'total_profit': -0.0041773, 'profit': -8.346264999999997, 'duration': 636.9230769230769},  # noqa: E501
            'results_explanation': '    39 trades. Avg profit  -0.21%. Total profit -0.00417730 BTC (  -8.35Σ%). Avg duration 636.9 min.',  # noqa: E501
            'total_profit': -0.0041773,
            'current_epoch': 6,
            'is_initial_point': True,
            'is_best': False
        }, {
            'loss': 4.713497421432944,
            'params_dict': {'mfi-value': 13, 'fastd-value': 41, 'adx-value': 21, 'rsi-value': 29, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'bb_lower', 'sell-mfi-value': 99, 'sell-fastd-value': 60, 'sell-adx-value': 81, 'sell-rsi-value': 69, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': False, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 771, 'roi_t2': 620, 'roi_t3': 145, 'roi_p1': 0.0586919200378493, 'roi_p2': 0.04984118697312542, 'roi_p3': 0.37521058680247044, 'stoploss': -0.14613268022709905},  # noqa: E501
            'params_details': {
                'buy': {'mfi-value': 13, 'fastd-value': 41, 'adx-value': 21, 'rsi-value': 29, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'bb_lower'}, 'sell': {'sell-mfi-value': 99, 'sell-fastd-value': 60, 'sell-adx-value': 81, 'sell-rsi-value': 69, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': False, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.4837436938134452, 145: 0.10853310701097472, 765: 0.0586919200378493, 1536: 0},  # noqa: E501
                'stoploss': {'stoploss': -0.14613268022709905}},  # noqa: E501
            'results_metrics': {'trade_count': 318, 'avg_profit': -0.39833954716981146, 'median_profit': -1.2222, 'total_profit': -0.06339929, 'profit': -126.67197600000004, 'duration': 3140.377358490566},  # noqa: E501
            'results_explanation': '   318 trades. Avg profit  -0.40%. Total profit -0.06339929 BTC (-126.67Σ%). Avg duration 3140.4 min.',  # noqa: E501
            'total_profit': -0.06339929,
            'current_epoch': 7,
            'is_initial_point': True,
            'is_best': False
        }, {
            'loss': 20.0,  # noqa: E501
            'params_dict': {'mfi-value': 24, 'fastd-value': 43, 'adx-value': 33, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'sar_reversal', 'sell-mfi-value': 89, 'sell-fastd-value': 74, 'sell-adx-value': 70, 'sell-rsi-value': 70, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': False, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal', 'roi_t1': 1149, 'roi_t2': 375, 'roi_t3': 289, 'roi_p1': 0.05571820757172588, 'roi_p2': 0.0606240398618907, 'roi_p3': 0.1729012220156157, 'stoploss': -0.1588514289110401},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 24, 'fastd-value': 43, 'adx-value': 33, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'sar_reversal'}, 'sell': {'sell-mfi-value': 89, 'sell-fastd-value': 74, 'sell-adx-value': 70, 'sell-rsi-value': 70, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': False, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal'}, 'roi': {0: 0.2892434694492323, 289: 0.11634224743361658, 664: 0.05571820757172588, 1813: 0}, 'stoploss': {'stoploss': -0.1588514289110401}},  # noqa: E501
            'results_metrics': {'trade_count': 1, 'avg_profit': 0.0, 'median_profit': 0.0, 'total_profit': 0.0, 'profit': 0.0, 'duration': 5340.0},  # noqa: E501
            'results_explanation': '     1 trades. Avg profit   0.00%. Total profit  0.00000000 BTC (   0.00Σ%). Avg duration 5340.0 min.',  # noqa: E501
            'total_profit': 0.0,
            'current_epoch': 8,
            'is_initial_point': True,
            'is_best': False
        }, {
            'loss': 2.4731817780991223,
            'params_dict': {'mfi-value': 22, 'fastd-value': 20, 'adx-value': 29, 'rsi-value': 40, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'sar_reversal', 'sell-mfi-value': 97, 'sell-fastd-value': 65, 'sell-adx-value': 81, 'sell-rsi-value': 64, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper', 'roi_t1': 1012, 'roi_t2': 584, 'roi_t3': 422, 'roi_p1': 0.036764323603472565, 'roi_p2': 0.10335480573205287, 'roi_p3': 0.10322347377503042, 'stoploss': -0.2780610808108503},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 22, 'fastd-value': 20, 'adx-value': 29, 'rsi-value': 40, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'sar_reversal'}, 'sell': {'sell-mfi-value': 97, 'sell-fastd-value': 65, 'sell-adx-value': 81, 'sell-rsi-value': 64, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper'}, 'roi': {0: 0.2433426031105559, 422: 0.14011912933552545, 1006: 0.036764323603472565, 2018: 0}, 'stoploss': {'stoploss': -0.2780610808108503}},  # noqa: E501
            'results_metrics': {'trade_count': 229, 'avg_profit': -0.38433433624454144, 'median_profit': -1.2222, 'total_profit': -0.044050070000000004, 'profit': -88.01256299999999, 'duration': 6505.676855895196},  # noqa: E501
            'results_explanation': '   229 trades. Avg profit  -0.38%. Total profit -0.04405007 BTC ( -88.01Σ%). Avg duration 6505.7 min.',  # noqa: E501
            'total_profit': -0.044050070000000004,  # noqa: E501
            'current_epoch': 9,
            'is_initial_point': True,
            'is_best': False
        }, {
            'loss': -0.2604606005845212,  # noqa: E501
            'params_dict': {'mfi-value': 23, 'fastd-value': 24, 'adx-value': 22, 'rsi-value': 24, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': True, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 97, 'sell-fastd-value': 70, 'sell-adx-value': 64, 'sell-rsi-value': 80, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal', 'roi_t1': 792, 'roi_t2': 464, 'roi_t3': 215, 'roi_p1': 0.04594053535385903, 'roi_p2': 0.09623192684243963, 'roi_p3': 0.04428219070850663, 'stoploss': -0.16992287161634415},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 23, 'fastd-value': 24, 'adx-value': 22, 'rsi-value': 24, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': True, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 97, 'sell-fastd-value': 70, 'sell-adx-value': 64, 'sell-rsi-value': 80, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal'}, 'roi': {0: 0.18645465290480528, 215: 0.14217246219629864, 679: 0.04594053535385903, 1471: 0}, 'stoploss': {'stoploss': -0.16992287161634415}},  # noqa: E501
            'results_metrics': {'trade_count': 4, 'avg_profit': 0.1080385, 'median_profit': -1.2222, 'total_profit': 0.00021629, 'profit': 0.432154, 'duration': 2850.0},  # noqa: E501
            'results_explanation': '     4 trades. Avg profit   0.11%. Total profit  0.00021629 BTC (   0.43Σ%). Avg duration 2850.0 min.',  # noqa: E501
            'total_profit': 0.00021629,
            'current_epoch': 10,
            'is_initial_point': True,
            'is_best': True
        }, {
            'loss': 4.876465945994304,  # noqa: E501
            'params_dict': {'mfi-value': 20, 'fastd-value': 32, 'adx-value': 49, 'rsi-value': 23, 'mfi-enabled': True, 'fastd-enabled': True, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'bb_lower', 'sell-mfi-value': 75, 'sell-fastd-value': 56, 'sell-adx-value': 61, 'sell-rsi-value': 62, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 579, 'roi_t2': 614, 'roi_t3': 273, 'roi_p1': 0.05307643172744114, 'roi_p2': 0.1352282078262871, 'roi_p3': 0.1913307406325751, 'stoploss': -0.25728526022513887},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 20, 'fastd-value': 32, 'adx-value': 49, 'rsi-value': 23, 'mfi-enabled': True, 'fastd-enabled': True, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'bb_lower'}, 'sell': {'sell-mfi-value': 75, 'sell-fastd-value': 56, 'sell-adx-value': 61, 'sell-rsi-value': 62, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.3796353801863034, 273: 0.18830463955372825, 887: 0.05307643172744114, 1466: 0}, 'stoploss': {'stoploss': -0.25728526022513887}},  # noqa: E501
            'results_metrics': {'trade_count': 117, 'avg_profit': -1.2698609145299145, 'median_profit': -1.2222, 'total_profit': -0.07436117, 'profit': -148.573727, 'duration': 4282.5641025641025},  # noqa: E501
            'results_explanation': '   117 trades. Avg profit  -1.27%. Total profit -0.07436117 BTC (-148.57Σ%). Avg duration 4282.6 min.',  # noqa: E501
            'total_profit': -0.07436117,
            'current_epoch': 11,
            'is_initial_point': True,
            'is_best': False
        }, {
            'loss': 100000,
            'params_dict': {'mfi-value': 10, 'fastd-value': 36, 'adx-value': 31, 'rsi-value': 22, 'mfi-enabled': True, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': False, 'trigger': 'sar_reversal', 'sell-mfi-value': 80, 'sell-fastd-value': 71, 'sell-adx-value': 60, 'sell-rsi-value': 85, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper', 'roi_t1': 1156, 'roi_t2': 581, 'roi_t3': 408, 'roi_p1': 0.06860454019988212, 'roi_p2': 0.12473718444931989, 'roi_p3': 0.2896360635226823, 'stoploss': -0.30889015124682806},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 10, 'fastd-value': 36, 'adx-value': 31, 'rsi-value': 22, 'mfi-enabled': True, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': False, 'trigger': 'sar_reversal'}, 'sell': {'sell-mfi-value': 80, 'sell-fastd-value': 71, 'sell-adx-value': 60, 'sell-rsi-value': 85, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper'}, 'roi': {0: 0.4829777881718843, 408: 0.19334172464920202, 989: 0.06860454019988212, 2145: 0}, 'stoploss': {'stoploss': -0.30889015124682806}},  # noqa: E501
            'results_metrics': {'trade_count': 0, 'avg_profit': None, 'median_profit': None, 'total_profit': 0, 'profit': 0.0, 'duration': None},  # noqa: E501
            'results_explanation': '     0 trades. Avg profit    nan%. Total profit  0.00000000 BTC (   0.00Σ%). Avg duration   nan min.',  # noqa: E501
            'total_profit': 0,
            'current_epoch': 12,
            'is_initial_point': True,
            'is_best': False
            }
    ]
