# pragma pylint: disable=missing-docstring
import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, Mock, PropertyMock

import numpy as np
import pandas as pd
import pytest
from xdist.scheduler.loadscope import LoadScopeScheduling

from freqtrade import constants
from freqtrade.commands import Arguments
from freqtrade.data.converter import ohlcv_to_dataframe, trades_list_to_df
from freqtrade.edge import PairInfo
from freqtrade.enums import CandleType, MarginMode, RunMode, SignalDirection, TradingMode
from freqtrade.exchange import Exchange, timeframe_to_minutes, timeframe_to_seconds
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from freqtrade.resolvers import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from freqtrade.worker import Worker
from tests.conftest_trades import (leverage_trade, mock_trade_1, mock_trade_2, mock_trade_3,
                                   mock_trade_4, mock_trade_5, mock_trade_6, short_trade)
from tests.conftest_trades_usdt import (mock_trade_usdt_1, mock_trade_usdt_2, mock_trade_usdt_3,
                                        mock_trade_usdt_4, mock_trade_usdt_5, mock_trade_usdt_6,
                                        mock_trade_usdt_7)


logging.getLogger('').setLevel(logging.INFO)


# Do not mask numpy errors as warnings that no one read, raise the exсeption
np.seterr(all='raise')

CURRENT_TEST_STRATEGY = 'StrategyTestV3'
TRADE_SIDES = ('long', 'short')
EXMS = 'freqtrade.exchange.exchange.Exchange'


def pytest_addoption(parser):
    parser.addoption('--longrun', action='store_true', dest="longrun",
                     default=False, help="Enable long-run tests (ccxt compat)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "longrun: mark test that is running slowly and should not be run regularily"
    )
    if not config.option.longrun:
        setattr(config.option, 'markexpr', 'not longrun')


class FixtureScheduler(LoadScopeScheduling):
    # Based on the suggestion in
    # https://github.com/pytest-dev/pytest-xdist/issues/18

    def _split_scope(self, nodeid):
        if 'exchange_online' in nodeid:
            try:
                # Extract exchange ID from nodeid
                exchange_id = nodeid.split('[')[1].split('-')[0].rstrip(']')
                return exchange_id
            except Exception as e:
                print(e)
                pass

        return nodeid


def pytest_xdist_make_scheduler(config, log):
    return FixtureScheduler(config, log)


def log_has(line, logs):
    """Check if line is found on some caplog's message."""
    return any(line == message for message in logs.messages)


def log_has_when(line, logs, when):
    """Check if line is found in caplog's messages during a specified stage"""
    return any(line == message.message for message in logs.get_records(when))


def log_has_re(line, logs):
    """Check if line matches some caplog's message."""
    return any(re.match(line, message) for message in logs.messages)


def num_log_has(line, logs):
    """Check how many times line is found in caplog's messages."""
    return sum(line == message for message in logs.messages)


def num_log_has_re(line, logs):
    """Check how many times line matches caplog's messages."""
    return sum(bool(re.match(line, message)) for message in logs.messages)


def get_args(args):
    return Arguments(args).get_parsed_arg()


def generate_trades_history(n_rows, start_date: Optional[datetime] = None, days=5):
    np.random.seed(42)
    if not start_date:
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

        # Generate random data
    end_date = start_date + timedelta(days=days)
    _start_timestamp = start_date.timestamp()
    _end_timestamp = pd.to_datetime(end_date).timestamp()

    random_timestamps_in_seconds = np.random.uniform(_start_timestamp, _end_timestamp, n_rows)
    timestamp = pd.to_datetime(random_timestamps_in_seconds, unit='s')

    id = [
        f'a{np.random.randint(1e6, 1e7 - 1)}cd{np.random.randint(100, 999)}'
        for _ in range(n_rows)
    ]

    side = np.random.choice(['buy', 'sell'], n_rows)

    # Initial price and subsequent changes
    initial_price = 0.019626
    price_changes = np.random.normal(0, initial_price * 0.05, n_rows)
    price = np.cumsum(np.concatenate(([initial_price], price_changes)))[:n_rows]

    amount = np.random.uniform(0.011, 20, n_rows)
    cost = price * amount

    # Create DataFrame
    df = pd.DataFrame({'timestamp': timestamp, 'id': id, 'type': None, 'side': side,
                       'price': price, 'amount': amount, 'cost': cost})
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    assert list(df.columns) == constants.DEFAULT_TRADES_COLUMNS + ['date']
    return df


def generate_test_data(timeframe: str, size: int, start: str = '2020-07-05'):
    np.random.seed(42)

    base = np.random.normal(20, 2, size=size)
    if timeframe == '1y':
        date = pd.date_range(start, periods=size, freq='1YS', tz='UTC')
    elif timeframe == '1M':
        date = pd.date_range(start, periods=size, freq='1MS', tz='UTC')
    elif timeframe == '3M':
        date = pd.date_range(start, periods=size, freq='3MS', tz='UTC')
    elif timeframe == '1w' or timeframe == '7d':
        date = pd.date_range(start, periods=size, freq='1W-MON', tz='UTC')
    else:
        tf_mins = timeframe_to_minutes(timeframe)
        if tf_mins >= 1:
            date = pd.date_range(start, periods=size, freq=f'{tf_mins}min', tz='UTC')
        else:
            tf_secs = timeframe_to_seconds(timeframe)
            date = pd.date_range(start, periods=size, freq=f'{tf_secs}s', tz='UTC')
    df = pd.DataFrame({
        'date': date,
        'open': base,
        'high': base + np.random.normal(2, 1, size=size),
        'low': base - np.random.normal(2, 1, size=size),
        'close': base + np.random.normal(0, 1, size=size),
        'volume': np.random.normal(200, size=size)
    }
    )
    df = df.dropna()
    return df


def generate_test_data_raw(timeframe: str, size: int, start: str = '2020-07-05'):
    """ Generates data in the ohlcv format used by ccxt """
    df = generate_test_data(timeframe, size, start)
    df['date'] = df.loc[:, 'date'].view(np.int64) // 1000 // 1000
    return list(list(x) for x in zip(*(df[x].values.tolist() for x in df.columns)))


# Source: https://stackoverflow.com/questions/29881236/how-to-mock-asyncio-coroutines
# TODO: This should be replaced with AsyncMock once support for python 3.7 is dropped.
def get_mock_coro(return_value=None, side_effect=None):
    async def mock_coro(*args, **kwargs):
        if side_effect:
            if isinstance(side_effect, list):
                effect = side_effect.pop(0)
            else:
                effect = side_effect
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(*args, **kwargs)
            return effect
        else:
            return return_value

    return Mock(wraps=mock_coro)


def patched_configuration_load_config_file(mocker, config) -> None:
    mocker.patch(
        'freqtrade.configuration.load_config.load_config_file',
        lambda *args, **kwargs: config
    )


def patch_exchange(
    mocker,
    api_mock=None,
    id='binance',
    mock_markets=True,
    mock_supported_modes=True
) -> None:
    mocker.patch(f'{EXMS}._load_async_markets', return_value={})
    mocker.patch(f'{EXMS}.validate_config', MagicMock())
    mocker.patch(f'{EXMS}.validate_timeframes', MagicMock())
    mocker.patch(f'{EXMS}.id', PropertyMock(return_value=id))
    mocker.patch(f'{EXMS}.name', PropertyMock(return_value=id.title()))
    mocker.patch(f'{EXMS}.precisionMode', PropertyMock(return_value=2))

    if mock_markets:
        if isinstance(mock_markets, bool):
            mock_markets = get_markets()
        mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=mock_markets))

    if mock_supported_modes:
        mocker.patch(
            f'freqtrade.exchange.{id}.{id.capitalize()}._supported_trading_mode_margin_pairs',
            PropertyMock(return_value=[
                (TradingMode.MARGIN, MarginMode.CROSS),
                (TradingMode.MARGIN, MarginMode.ISOLATED),
                (TradingMode.FUTURES, MarginMode.CROSS),
                (TradingMode.FUTURES, MarginMode.ISOLATED)
            ])
        )

    if api_mock:
        mocker.patch(f'{EXMS}._init_ccxt', return_value=api_mock)
    else:
        mocker.patch(f'{EXMS}._init_ccxt', MagicMock())
        mocker.patch(f'{EXMS}.timeframes', PropertyMock(
                return_value=['5m', '15m', '1h', '1d']))


def get_patched_exchange(mocker, config, api_mock=None, id='binance',
                         mock_markets=True, mock_supported_modes=True) -> Exchange:
    patch_exchange(mocker, api_mock, id, mock_markets, mock_supported_modes)
    config['exchange']['name'] = id
    try:
        exchange = ExchangeResolver.load_exchange(config, load_leverage_tiers=True)
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


# Functions for recurrent object patching


def patch_freqtradebot(mocker, config) -> None:
    """
    This function patch _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: None
    """
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    patch_exchange(mocker)
    mocker.patch('freqtrade.freqtradebot.RPCManager._init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.RPCManager.send_msg', MagicMock())
    patch_whitelist(mocker, config)
    mocker.patch('freqtrade.freqtradebot.ExternalMessageConsumer')
    mocker.patch('freqtrade.configuration.config_validation._validate_consumers')


def get_patched_freqtradebot(mocker, config) -> FreqtradeBot:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: FreqtradeBot
    """
    patch_freqtradebot(mocker, config)
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


def patch_get_signal(
    freqtrade: FreqtradeBot,
    enter_long=True,
    exit_long=False,
    enter_short=False,
    exit_short=False,
    enter_tag: Optional[str] = None,
    exit_tag: Optional[str] = None,
) -> None:
    """
    :param mocker: mocker to patch IStrategy class
    :return: None
    """
    # returns (Signal-direction, signaname)
    def patched_get_entry_signal(*args, **kwargs):
        direction = None
        if enter_long and not any([exit_long, enter_short]):
            direction = SignalDirection.LONG
        if enter_short and not any([exit_short, enter_long]):
            direction = SignalDirection.SHORT

        return direction, enter_tag

    freqtrade.strategy.get_entry_signal = patched_get_entry_signal

    def patched_get_exit_signal(pair, timeframe, dataframe, is_short):
        if is_short:
            return enter_short, exit_short, exit_tag
        else:
            return enter_long, exit_long, exit_tag

    # returns (enter, exit)
    freqtrade.strategy.get_exit_signal = patched_get_exit_signal

    freqtrade.exchange.refresh_latest_ohlcv = lambda p: None


def create_mock_trades(fee, is_short: Optional[bool] = False, use_db: bool = True):
    """
    Create some fake trades ...
    :param is_short: Optional bool, None creates a mix of long and short trades.
    """
    def add_trade(trade):
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)
    is_short1 = is_short if is_short is not None else True
    is_short2 = is_short if is_short is not None else False
    # Simulate dry_run entries
    trade = mock_trade_1(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_2(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_3(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_4(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_5(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_6(fee, is_short1)
    add_trade(trade)

    if use_db:
        Trade.commit()


def create_mock_trades_with_leverage(fee, use_db: bool = True):
    """
    Create some fake trades ...
    """
    if use_db:
        Trade.session.rollback()

    def add_trade(trade):
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    # Simulate dry_run entries
    trade = mock_trade_1(fee, False)
    add_trade(trade)

    trade = mock_trade_2(fee, False)
    add_trade(trade)

    trade = mock_trade_3(fee, False)
    add_trade(trade)

    trade = mock_trade_4(fee, False)
    add_trade(trade)

    trade = mock_trade_5(fee, False)
    add_trade(trade)

    trade = mock_trade_6(fee, False)
    add_trade(trade)

    trade = short_trade(fee)
    add_trade(trade)

    trade = leverage_trade(fee)
    add_trade(trade)

    if use_db:
        Trade.session.flush()


def create_mock_trades_usdt(fee, is_short: Optional[bool] = False, use_db: bool = True):
    """
    Create some fake trades ...
    """
    def add_trade(trade):
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    is_short1 = is_short if is_short is not None else True
    is_short2 = is_short if is_short is not None else False

    # Simulate dry_run entries
    trade = mock_trade_usdt_1(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_usdt_2(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_usdt_3(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_usdt_4(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_usdt_5(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_usdt_6(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_usdt_7(fee, is_short1)
    add_trade(trade)
    if use_db:
        Trade.commit()


@pytest.fixture(autouse=True)
def patch_gc(mocker) -> None:
    mocker.patch("freqtrade.main.gc_set_threshold")


@pytest.fixture(autouse=True)
def user_dir(mocker, tmp_path) -> Path:
    user_dir = tmp_path / "user_data"
    mocker.patch('freqtrade.configuration.configuration.create_userdata_dir',
                 return_value=user_dir)
    return user_dir


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
    init_db(default_conf['db_url'])


@pytest.fixture(scope="function")
def default_conf(testdatadir):
    return get_default_conf(testdatadir)


@pytest.fixture(scope="function")
def default_conf_usdt(testdatadir):
    return get_default_conf_usdt(testdatadir)


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
            "entry": 10,
            "exit": 30
        },
        "entry_pricing": {
            "price_last_balance": 0.0,
            "use_order_book": False,
            "order_book_top": 1,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1
            }
        },
        "exit_pricing": {
            "use_order_book": False,
            "order_book_top": 1,
        },
        "exchange": {
            "name": "binance",
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
            "enabled": False,
            "token": "token",
            "chat_id": "0",
            "notification_settings": {},
        },
        "datadir": Path(testdatadir),
        "initial_state": "running",
        "db_url": "sqlite://",
        "user_data_dir": Path("user_data"),
        "verbosity": 3,
        "strategy_path": str(Path(__file__).parent / "strategy" / "strats"),
        "strategy": CURRENT_TEST_STRATEGY,
        "disableparamexport": True,
        "internals": {},
        "export": "none",
        "dataformat_ohlcv": "feather",
        "runmode": "dry_run",
        "candle_type_def": CandleType.SPOT,
    }
    return configuration


def get_default_conf_usdt(testdatadir):
    configuration = get_default_conf(testdatadir)
    configuration.update({
        "stake_amount": 60.0,
        "stake_currency": "USDT",
        "exchange": {
            "name": "binance",
            "enabled": True,
            "key": "key",
            "secret": "secret",
            "pair_whitelist": [
                "ETH/USDT",
                "LTC/USDT",
                "XRP/USDT",
                "NEO/USDT",
                "TKN/USDT",
            ],
            "pair_blacklist": [
                "DOGE/USDT",
                "HOT/USDT",
            ]
        },
    })
    return configuration


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
def ticker_usdt():
    return MagicMock(return_value={
        'bid': 2.0,
        'ask': 2.02,
        'last': 2.0,
    })


@pytest.fixture
def ticker_usdt_sell_up():
    return MagicMock(return_value={
        'bid': 2.2,
        'ask': 2.3,
        'last': 2.2,
    })


@pytest.fixture
def ticker_usdt_sell_down():
    return MagicMock(return_value={
        'bid': 2.01,
        'ask': 2.0,
        'last': 2.01,
    })


@pytest.fixture
def markets():
    return get_markets()


def get_markets():
    # See get_markets_static() for immutable markets and do not modify them unless absolutely
    # necessary!
    return {
        'ETH/BTC': {
            'id': 'ethbtc',
            'symbol': 'ETH/BTC',
            'base': 'ETH',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'contractSize': None,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 100000000,
                },
                'price': {
                    'min': None,
                    'max': 500000,
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
                'leverage': {
                    'min': 1.0,
                    'max': 2.0
                }
            },
        },
        'TKN/BTC': {
            'id': 'tknbtc',
            'symbol': 'TKN/BTC',
            'base': 'TKN',
            'quote': 'BTC',
            # According to ccxt, markets without active item set are also active
            # 'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'contractSize': None,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 100000000,
                },
                'price': {
                    'min': None,
                    'max': 500000,
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
                'leverage': {
                    'min': 1.0,
                    'max': 5.0
                }
            },
        },
        'BLK/BTC': {
            'id': 'blkbtc',
            'symbol': 'BLK/BTC',
            'base': 'BLK',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'contractSize': None,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': {
                    'min': None,
                    'max': 500000,
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
                'leverage': {
                    'min': 1.0,
                    'max': 3.0
                },
            },
        },
        'LTC/BTC': {
            'id': 'ltcbtc',
            'symbol': 'LTC/BTC',
            'base': 'LTC',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'contractSize': None,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 100000000,
                },
                'price': {
                    'min': None,
                    'max': 500000,
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
                'leverage': {
                    'min': None,
                    'max': None
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
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'contractSize': None,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 100000000,
                },
                'price': {
                    'min': None,
                    'max': 500000,
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
                'leverage': {
                    'min': None,
                    'max': None,
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
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'contractSize': None,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 100000000,
                },
                'price': {
                    'min': None,
                    'max': 500000,
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
                'leverage': {
                    'min': None,
                    'max': None,
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
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
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
                },
                'leverage': {
                    'min': None,
                    'max': None,
                },
            },
            'info': {},
        },
        'ETH/USDT': {
            'id': 'USDT-ETH',
            'symbol': 'ETH/USDT',
            'base': 'ETH',
            'quote': 'USDT',
            'settle': None,
            'baseId': 'ETH',
            'quoteId': 'USDT',
            'settleId': None,
            'type': 'spot',
            'spot': True,
            'margin': True,
            'swap': True,
            'future': True,
            'option': False,
            'active': True,
            'contract': None,
            'linear': None,
            'inverse': None,
            'taker': 0.0006,
            'maker': 0.0002,
            'contractSize': None,
            'expiry': None,
            'expiryDateTime': None,
            'strike': None,
            'optionType': None,
            'precision': {
                'amount': 8,
                'price': 8,
            },
            'limits': {
                'leverage': {
                    'min': 1,
                    'max': 100,
                },
                'amount': {
                    'min': 0.02214286,
                    'max': None,
                },
                'price': {
                    'min': 1e-08,
                    'max': None,
                },
                'cost': {
                    'min': None,
                    'max': None,
                },
            },
            'info': {
                'maintenance_rate': '0.005',
            },
        },
        'BTC/USDT': {
            'id': 'USDT-BTC',
            'symbol': 'BTC/USDT',
            'base': 'BTC',
            'quote': 'USDT',
            'settle': None,
            'baseId': 'BTC',
            'quoteId': 'USDT',
            'settleId': None,
            'type': 'spot',
            'spot': True,
            'margin': True,
            'swap': False,
            'future': False,
            'option': False,
            'active': True,
            'contract': None,
            'linear': None,
            'inverse': None,
            'taker': 0.0006,
            'maker': 0.0002,
            'contractSize': None,
            'expiry': None,
            'expiryDateTime': None,
            'strike': None,
            'optionType': None,
            'precision': {
                'amount': 4,
                'price': 4,
            },
            'limits': {
                'leverage': {
                    'min': 1,
                    'max': 100,
                },
                'amount': {
                    'min': 0.000221,
                    'max': None,
                },
                'price': {
                    'min': 1e-02,
                    'max': None,
                },
                'cost': {
                    'min': None,
                    'max': None,
                },
            },
            'info': {
                'maintenance_rate': '0.005',
            },
        },
        'LTC/USDT': {
            'id': 'USDT-LTC',
            'symbol': 'LTC/USDT',
            'base': 'LTC',
            'quote': 'USDT',
            'active': False,
            'spot': True,
            'future': True,
            'swap': True,
            'margin': True,
            'linear': None,
            'inverse': False,
            'type': 'spot',
            'contractSize': None,
            'taker': 0.0006,
            'maker': 0.0002,
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
                },
                'leverage': {
                    'min': None,
                    'max': None,
                },
                'cost': {
                    'min': None,
                    'max': None,
                },
            },
            'info': {},
        },
        'XRP/USDT': {
            'id': 'xrpusdt',
            'symbol': 'XRP/USDT',
            'base': 'XRP',
            'quote': 'USDT',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'taker': 0.0006,
            'maker': 0.0002,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'contractSize': None,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': {
                    'min': None,
                    'max': 500000,
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
            },
            'info': {},
        },
        'NEO/USDT': {
            'id': 'neousdt',
            'symbol': 'NEO/USDT',
            'base': 'NEO',
            'quote': 'USDT',
            'settle': '',
            'baseId': 'NEO',
            'quoteId': 'USDT',
            'settleId': '',
            'type': 'spot',
            'spot': True,
            'margin': True,
            'swap': False,
            'futures': False,
            'option': False,
            'active': True,
            'contract': False,
            'linear': None,
            'inverse': None,
            'taker': 0.0006,
            'maker': 0.0002,
            'contractSize': None,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'tierBased': None,
            'percentage': None,
            'lot': 0.00000001,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'limits': {
                "leverage": {
                    'min': 1,
                    'max': 10
                },
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': {
                    'min': None,
                    'max': 500000,
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
            },
            'info': {},
        },
        'TKN/USDT': {
            'id': 'tknusdt',
            'symbol': 'TKN/USDT',
            'base': 'TKN',
            'quote': 'USDT',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
            'taker': 0.0006,
            'maker': 0.0002,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,
            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 100000000000,
                },
                'price': {
                    'min': None,
                    'max': 500000
                },
                'cost': {
                    'min': 0.0001,
                    'max': 500000,
                },
                'leverage': {
                    'min': None,
                    'max': None,
                },
            },
            'info': {},
        },
        'LTC/USD': {
            'id': 'USD-LTC',
            'symbol': 'LTC/USD',
            'base': 'LTC',
            'quote': 'USD',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
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
                },
                'leverage': {
                    'min': None,
                    'max': None,
                },
                'cost': {
                    'min': None,
                    'max': None,
                },
            },
            'info': {},
        },
        'XLTCUSDT': {
            'id': 'xLTCUSDT',
            'symbol': 'XLTCUSDT',
            'base': 'LTC',
            'quote': 'USDT',
            'active': True,
            'spot': False,
            'type': 'swap',
            'contractSize': 0.01,
            'swap': False,
            'linear': False,
            'taker': 0.0006,
            'maker': 0.0002,
            'precision': {
                'amount': 8,
                'price': 8
            },
            'limits': {
                'leverage': {
                    'min': None,
                    'max': None,
                },
                'amount': {
                    'min': 0.06646786,
                    'max': None
                },
                'price': {
                    'min': 1e-08,
                    'max': None
                },
                'cost': {
                    'min': None,
                    'max': None,
                },
            },
            'info': {},
        },
        'LTC/ETH': {
            'id': 'LTCETH',
            'symbol': 'LTC/ETH',
            'base': 'LTC',
            'quote': 'ETH',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
            'precision': {
                'base': 8,
                'quote': 8,
                'amount': 3,
                'price': 5
            },
            'limits': {
                'leverage': {
                    'min': None,
                    'max': None,
                },
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
            'info': {
            }
        },
        'ETH/USDT:USDT': {
            'id': 'ETH_USDT',
            'symbol': 'ETH/USDT:USDT',
            'base': 'ETH',
            'quote': 'USDT',
            'settle': 'USDT',
            'baseId': 'ETH',
            'quoteId': 'USDT',
            'settleId': 'USDT',
            'type': 'swap',
            'spot': False,
            'margin': False,
            'swap': True,
            'future': True,  # Binance mode ...
            'option': False,
            'contract': True,
            'linear': True,
            'inverse': False,
            'tierBased': False,
            'percentage': True,
            'taker': 0.0006,
            'maker': 0.0002,
            'contractSize': 10,
            'active': True,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'limits': {
                'leverage': {
                    'min': 1,
                    'max': 100
                },
                'amount': {
                    'min': 1,
                    'max': 300000
                },
                'price': {
                    'min': None,
                    'max': None,
                },
                'cost': {
                    'min': None,
                    'max': None,
                }
            },
            'precision': {
                'price': 0.05,
                'amount': 1
            },
            'info': {}
        },
        'ADA/USDT:USDT': {
            'limits': {
                'leverage': {
                    'min': 1,
                    'max': 20,
                },
                'amount': {
                    'min': 1,
                    'max': 1000000,
                },
                'price': {
                    'min': 0.52981,
                    'max': 1.58943,
                },
                'cost': {
                    'min': None,
                    'max': None,
                }
            },
            'precision': {
                'amount': 1,
                'price': 0.00001
            },
            'tierBased': True,
            'percentage': True,
            'taker': 0.0000075,
            'maker': -0.0000025,
            'feeSide': 'get',
            'tiers': {
                'maker': [
                    [0, 0.002],       [1.5, 0.00185],
                    [3, 0.00175],     [6, 0.00165],
                    [12.5, 0.00155],  [25, 0.00145],
                    [75, 0.00135],    [200, 0.00125],
                    [500, 0.00115],   [1250, 0.00105],
                    [2500, 0.00095],  [3000, 0.00085],
                    [6000, 0.00075],  [11000, 0.00065],
                    [20000, 0.00055], [40000, 0.00055],
                    [75000, 0.00055]
                ],
                'taker': [
                    [0, 0.002],       [1.5, 0.00195],
                    [3, 0.00185],     [6, 0.00175],
                    [12.5, 0.00165],  [25, 0.00155],
                    [75, 0.00145],    [200, 0.00135],
                    [500, 0.00125],   [1250, 0.00115],
                    [2500, 0.00105],  [3000, 0.00095],
                    [6000, 0.00085],  [11000, 0.00075],
                    [20000, 0.00065], [40000, 0.00065],
                    [75000, 0.00065]
                ]
            },
            'id': 'ADA_USDT',
            'symbol': 'ADA/USDT:USDT',
            'base': 'ADA',
            'quote': 'USDT',
            'settle': 'USDT',
            'baseId': 'ADA',
            'quoteId': 'USDT',
            'settleId': 'usdt',
            'type': 'swap',
            'spot': False,
            'margin': False,
            'swap': True,
            'future': True,  # Binance mode ...
            'option': False,
            'active': True,
            'contract': True,
            'linear': True,
            'inverse': False,
            'contractSize': 0.01,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'info': {}
        },
        'SOL/BUSD:BUSD': {
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': 1, 'max': 1000000},
                'price': {'min': 0.04, 'max': 100000},
                'cost': {'min': 5, 'max': None},
                'market': {'min': 1, 'max': 1500}
            },
            'precision': {'amount': 0, 'price': 2, 'base': 8, 'quote': 8},
            'tierBased': False,
            'percentage': True,
            'taker': 0.0004,
            'maker': 0.0002,
            'feeSide': 'get',
            'id': 'SOLBUSD',
            'lowercaseId': 'solbusd',
            'symbol': 'SOL/BUSD',
            'base': 'SOL',
            'quote': 'BUSD',
            'settle': 'BUSD',
            'baseId': 'SOL',
            'quoteId': 'BUSD',
            'settleId': 'BUSD',
            'type': 'future',
            'spot': False,
            'margin': False,
            'future': True,
            'delivery': False,
            'option': False,
            'active': True,
            'contract': True,
            'linear': True,
            'inverse': False,
            'contractSize': 1,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'info': {
                'symbol': 'SOLBUSD',
                'pair': 'SOLBUSD',
                'contractType': 'PERPETUAL',
                'deliveryDate': '4133404800000',
                'onboardDate': '1630566000000',
                'status': 'TRADING',
                'maintMarginPercent': '2.5000',
                'requiredMarginPercent': '5.0000',
                'baseAsset': 'SOL',
                'quoteAsset': 'BUSD',
                'marginAsset': 'BUSD',
                'pricePrecision': '4',
                'quantityPrecision': '0',
                'baseAssetPrecision': '8',
                'quotePrecision': '8',
                'underlyingType': 'COIN',
                'underlyingSubType': [],
                'settlePlan': '0',
                'triggerProtect': '0.0500',
                'liquidationFee': '0.005000',
                'marketTakeBound': '0.05',
                'filters': [
                    {
                        'minPrice': '0.0400',
                        'maxPrice': '100000',
                        'filterType': 'PRICE_FILTER',
                        'tickSize': '0.0100'
                    },
                    {
                        'stepSize': '1',
                        'filterType': 'LOT_SIZE',
                        'maxQty': '1000000',
                        'minQty': '1'
                    },
                    {
                        'stepSize': '1',
                        'filterType': 'MARKET_LOT_SIZE',
                        'maxQty': '1500',
                        'minQty': '1'
                    },
                    {'limit': '200', 'filterType': 'MAX_NUM_ORDERS'},
                    {'limit': '10', 'filterType': 'MAX_NUM_ALGO_ORDERS'},
                    {'notional': '5', 'filterType': 'MIN_NOTIONAL'},
                    {
                        'multiplierDown': '0.9500',
                        'multiplierUp': '1.0500',
                        'multiplierDecimal': '4',
                        'filterType': 'PERCENT_PRICE'
                    }
                ],
                'orderTypes': [
                    'LIMIT',
                    'MARKET',
                    'STOP',
                    'STOP_MARKET',
                    'TAKE_PROFIT',
                    'TAKE_PROFIT_MARKET',
                    'TRAILING_STOP_MARKET'
                ],
                'timeInForce': ['GTC', 'IOC', 'FOK', 'GTX']
            }
        },
    }


@pytest.fixture
def markets_static():
    # These markets are used in some tests that would need adaptation should anything change in
    # market list. Do not modify this list without a good reason! Do not modify market parameters
    # of listed pairs in get_markets() without a good reason either!
    static_markets = ['BLK/BTC', 'BTT/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD',
                      'LTC/USDT', 'NEO/BTC', 'TKN/BTC', 'XLTCUSDT', 'XRP/BTC',
                      'ADA/USDT:USDT', 'ETH/USDT:USDT',
                      ]
    all_markets = get_markets()
    return {m: all_markets[m] for m in static_markets}


@pytest.fixture
def shitcoinmarkets(markets_static):
    """
    Fixture with shitcoin markets - used to test filters in pairlists
    """
    shitmarkets = deepcopy(markets_static)
    shitmarkets.update({
        'HOT/BTC': {
            'id': 'HOTBTC',
            'symbol': 'HOT/BTC',
            'base': 'HOT',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'type': 'spot',
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
            'spot': True,
            'type': 'spot',
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
                'leverage': {
                    'min': None,
                    'max': None,
                },
                'amount': {
                    'min': None,
                    'max': None,
                },
                'price': {
                    'min': None,
                    'max': None,
                },
                'cost': {
                    'min': None,
                    'max': None,
                },
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
                'leverage': {
                    'min': None,
                    'max': None,
                },
                'amount': {
                    'min': None,
                    'max': None,
                },
                'price': {
                    'min': None,
                    'max': None,
                },
                'cost': {
                    'min': None,
                    'max': None,
                },
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
                'leverage': {
                    'min': None,
                    'max': None,
                },
                'amount': {
                    'min': None,
                    'max': None,
                },
                'price': {
                    'min': None,
                    'max': None,
                },
                'cost': {
                    'min': None,
                    'max': None,
                },
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
        'timestamp': dt_ts(),
        'datetime': dt_now().isoformat(),
        'price': 0.00001099,
        'average': 0.00001099,
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


@pytest.fixture
def limit_buy_order_old():
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
        'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
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
        'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
        'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
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
        'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
        'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
        'price': 0.00001099,
        'amount': 90.99181073,
        'filled': 23.0,
        'cost': 90.99181073 * 23.0,
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
    if exchange_name == 'kraken':
        return {
            'info': {},
            'id': 'AZNPFF-4AC4N-7MKTAT',
            'clientOrderId': None,
            'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
            'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
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
            'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
            'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
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
            'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
            'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
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
        'symbol': 'mocked',
        'datetime': dt_now().isoformat(),
        'timestamp': dt_ts(),
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
def order_book_l2_usd():
    return MagicMock(return_value={
        'symbol': 'LTC/USDT',
        'bids': [
            [25.563, 49.269],
            [25.562, 83.0],
            [25.56, 106.0],
            [25.559, 15.381],
            [25.558, 29.299],
            [25.557, 34.624],
            [25.556, 10.0],
            [25.555, 14.684],
            [25.554, 45.91],
            [25.553, 50.0]
        ],
        'asks': [
            [25.566, 14.27],
            [25.567, 48.484],
            [25.568, 92.349],
            [25.572, 31.48],
            [25.573, 23.0],
            [25.574, 20.0],
            [25.575, 89.606],
            [25.576, 262.016],
            [25.577, 178.557],
            [25.578, 78.614]
        ],
        'timestamp': None,
        'datetime': None,
        'nonce': 2372149736
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
            'datetime': '2018-03-25T21:53:26.072Z',
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
def dataframe_1m(testdatadir):
    with (testdatadir / 'UNITTEST_BTC-1m.json').open('r') as data_file:
        return ohlcv_to_dataframe(json.load(data_file), '1m', pair="UNITTEST/BTC",
                                  fill_missing=True)


@pytest.fixture(scope="function")
def trades_for_order():
    return [{
        'info': {
            'id': 34567,
            'orderId': 123456,
            'price': '2.0',
            'qty': '8.00000000',
            'commission': '0.00800000',
            'commissionAsset': 'LTC',
            'time': 1521663363189,
            'isBuyer': True,
            'isMaker': False,
            'isBestMatch': True
        },
        'timestamp': 1521663363189,
        'datetime': '2018-03-21T20:16:03.189Z',
        'symbol': 'LTC/USDT',
        'id': '34567',
        'order': '123456',
        'type': None,
        'side': 'buy',
        'price': 2.0,
        'cost': 16.0,
        'amount': 8.0,
        'fee': {
            'cost': 0.008,
            'currency': 'LTC'
        }
    }]


@pytest.fixture(scope="function")
def trades_history():
    return [[1565798389463, '12618132aa9', None, 'buy', 0.019627, 0.04, 0.00078508],
            [1565798399629, '1261813bb30', None, 'buy', 0.019627, 0.244, 0.004788987999999999],
            [1565798399752, '1261813cc31', None, 'sell', 0.019626, 0.011, 0.00021588599999999999],
            [1565798399862, '126181cc332', None, 'sell', 0.019626, 0.011, 0.00021588599999999999],
            [1565798399862, '126181cc333', None, 'sell', 0.019626, 0.012, 0.00021588599999999999],
            [1565798399872, '1261aa81334', None, 'sell', 0.019626, 0.011, 0.00021588599999999999]]


@pytest.fixture(scope="function")
def trades_history_df(trades_history):
    trades = trades_list_to_df(trades_history)
    trades['date'] = pd.to_datetime(trades['timestamp'], unit='ms', utc=True)
    return trades


@pytest.fixture(scope="function")
def fetch_trades_result():
    return [{'info': ['0.01962700', '0.04000000', '1565798399.4631551', 'b', 'm', '', '126181329'],
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
            {'info': ['0.01962700', '0.24400000', '1565798399.6291551', 'b', 'm', '', '126181330'],
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
            {'info': ['0.01962600', '0.01100000', '1565798399.7521551', 's', 'm', '', '126181331'],
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
            {'info': ['0.01962600', '0.01100000', '1565798399.8621551', 's', 'm', '', '126181332'],
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
            {'info': ['0.01952600', '0.01200000', '1565798399.8721551', 's', 'm', '', '126181333',
                      1565798399872512133],
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


@pytest.fixture
def buy_order_fee():
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
        'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
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
    conf['runmode'] = RunMode.DRY_RUN
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
        if name in ["filelock", 'cysystemd.journal', 'uvloop']:
            raise ImportError(f"No module named '{name}'")
        return realimport(name, *args, **kwargs)

    builtins.__import__ = mockedimport

    # Run test - then cleanup
    yield

    # restore previous importfunction
    builtins.__import__ = realimport


@pytest.fixture(scope="function")
def open_trade():
    trade = Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='binance',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=dt_now() - timedelta(minutes=601),
        is_open=True
    )
    trade.orders = [
        Order(
            ft_order_side='buy',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789',
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        )
    ]
    return trade


@pytest.fixture(scope="function")
def open_trade_usdt():
    trade = Trade(
        pair='ADA/USDT',
        open_rate=2.0,
        exchange='binance',
        amount=30.0,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=60.0,
        open_date=dt_now() - timedelta(minutes=601),
        is_open=True
    )
    trade.orders = [
        Order(
            ft_order_side='buy',
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789',
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        ),
        Order(
            ft_order_side='exit',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789_exit',
            status="open",
            symbol=trade.pair,
            order_type="limit",
            side="sell",
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        )
    ]
    return trade


@pytest.fixture
def saved_hyperopt_results():
    hyperopt_res = [
        {
            'loss': 0.4366182531161519,
            'params_dict': {
                'mfi-value': 15, 'fastd-value': 20, 'adx-value': 25, 'rsi-value': 28, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 88, 'sell-fastd-value': 97, 'sell-adx-value': 51, 'sell-rsi-value': 67, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper', 'roi_t1': 1190, 'roi_t2': 541, 'roi_t3': 408, 'roi_p1': 0.026035863879169705, 'roi_p2': 0.12508730043628782, 'roi_p3': 0.27766427921605896, 'stoploss': -0.2562930402099556},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 15, 'fastd-value': 20, 'adx-value': 25, 'rsi-value': 28, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 88, 'sell-fastd-value': 97, 'sell-adx-value': 51, 'sell-rsi-value': 67, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper'}, 'roi': {0: 0.4287874435315165, 408: 0.15112316431545753, 949: 0.026035863879169705, 2139: 0}, 'stoploss': {'stoploss': -0.2562930402099556}},  # noqa: E501
            'results_metrics': {'total_trades': 2, 'trade_count_long': 2, 'trade_count_short': 0, 'wins': 0, 'draws': 0, 'losses': 2, 'profit_mean': -0.01254995, 'profit_median': -0.012222, 'profit_total': -0.00125625,  'profit_total_abs': -2.50999, 'max_drawdown': 0.23, 'max_drawdown_abs': -0.00125625,  'holding_avg': timedelta(minutes=3930.0), 'stake_currency': 'BTC', 'strategy_name': 'SampleStrategy'},  # noqa: E501
            'results_explanation': '     2 trades. Avg profit  -1.25%. Total profit -0.00125625 BTC (  -2.51Σ%). Avg duration 3930.0 min.',  # noqa: E501
            'total_profit': -0.00125625,
            'current_epoch': 1,
            'is_initial_point': True,
            'is_random': False,
            'is_best': True,

        }, {
            'loss': 20.0,
            'params_dict': {
                'mfi-value': 17, 'fastd-value': 38, 'adx-value': 48, 'rsi-value': 22, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 96, 'sell-fastd-value': 68, 'sell-adx-value': 63, 'sell-rsi-value': 81, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal', 'roi_t1': 334, 'roi_t2': 683, 'roi_t3': 140, 'roi_p1': 0.06403981740598495, 'roi_p2': 0.055519840060645045, 'roi_p3': 0.3253712811342459, 'stoploss': -0.338070047333259},  # noqa: E501
            'params_details': {
                'buy': {'mfi-value': 17, 'fastd-value': 38, 'adx-value': 48, 'rsi-value': 22, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'macd_cross_signal'},  # noqa: E501
                'sell': {'sell-mfi-value': 96, 'sell-fastd-value': 68, 'sell-adx-value': 63, 'sell-rsi-value': 81, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal'},  # noqa: E501
                'roi': {0: 0.4449309386008759, 140: 0.11955965746663, 823: 0.06403981740598495, 1157: 0},  # noqa: E501
                'stoploss': {'stoploss': -0.338070047333259}},
            'results_metrics': {'total_trades': 1, 'trade_count_long': 1, 'trade_count_short': 0, 'wins': 0, 'draws': 0, 'losses': 1, 'profit_mean': 0.012357, 'profit_median': -0.012222, 'profit_total': 6.185e-05, 'profit_total_abs': 0.12357, 'max_drawdown': 0.23, 'max_drawdown_abs': -0.00125625, 'holding_avg': timedelta(minutes=1200.0)},  # noqa: E501
            'results_explanation': '     1 trades. Avg profit   0.12%. Total profit  0.00006185 BTC (   0.12Σ%). Avg duration 1200.0 min.',  # noqa: E501
            'total_profit': 6.185e-05,
            'current_epoch': 2,
            'is_initial_point': True,
            'is_random': False,
            'is_best': False
        }, {
            'loss': 14.241196856510731,
            'params_dict': {'mfi-value': 25, 'fastd-value': 16, 'adx-value': 29, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 98, 'sell-fastd-value': 72, 'sell-adx-value': 51, 'sell-rsi-value': 82, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 889, 'roi_t2': 533, 'roi_t3': 263, 'roi_p1': 0.04759065393663096, 'roi_p2': 0.1488819964638463, 'roi_p3': 0.4102801822104605, 'stoploss': -0.05394588767607611},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 25, 'fastd-value': 16, 'adx-value': 29, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 98, 'sell-fastd-value': 72, 'sell-adx-value': 51, 'sell-rsi-value': 82, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.6067528326109377, 263: 0.19647265040047726, 796: 0.04759065393663096, 1685: 0}, 'stoploss': {'stoploss': -0.05394588767607611}},  # noqa: E501
            'results_metrics': {'total_trades': 621, 'trade_count_long': 621, 'trade_count_short': 0, 'wins': 320, 'draws': 0, 'losses': 301, 'profit_mean': -0.043883302093397747, 'profit_median': -0.012222, 'profit_total': -0.13639474, 'profit_total_abs': -272.515306, 'max_drawdown': 0.25, 'max_drawdown_abs': -272.515306, 'holding_avg': timedelta(minutes=1691.207729468599)},  # noqa: E501
            'results_explanation': '   621 trades. Avg profit  -0.44%. Total profit -0.13639474 BTC (-272.52Σ%). Avg duration 1691.2 min.',  # noqa: E501
            'total_profit': -0.13639474,
            'current_epoch': 3,
            'is_initial_point': True,
            'is_random': False,
            'is_best': False
        }, {
            'loss': 100000,
            'params_dict': {'mfi-value': 13, 'fastd-value': 35, 'adx-value': 39, 'rsi-value': 29, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': True, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 87, 'sell-fastd-value': 54, 'sell-adx-value': 63, 'sell-rsi-value': 93, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper', 'roi_t1': 1402, 'roi_t2': 676, 'roi_t3': 215, 'roi_p1': 0.06264755784937427, 'roi_p2': 0.14258587851894644, 'roi_p3': 0.20671291201040828, 'stoploss': -0.11818343570194478},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 13, 'fastd-value': 35, 'adx-value': 39, 'rsi-value': 29, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': True, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 87, 'sell-fastd-value': 54, 'sell-adx-value': 63, 'sell-rsi-value': 93, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper'}, 'roi': {0: 0.411946348378729, 215: 0.2052334363683207, 891: 0.06264755784937427, 2293: 0}, 'stoploss': {'stoploss': -0.11818343570194478}},  # noqa: E501
            'results_metrics': {'total_trades': 0, 'trade_count_long': 0, 'trade_count_short': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'profit_mean': None, 'profit_median': None, 'profit_total': 0, 'profit': 0.0, 'holding_avg': timedelta()},  # noqa: E501
            'results_explanation': '     0 trades. Avg profit    nan%. Total profit  0.00000000 BTC (   0.00Σ%). Avg duration   nan min.',  # noqa: E501
            'total_profit': 0, 'current_epoch': 4, 'is_initial_point': True, 'is_random': False, 'is_best': False  # noqa: E501
        }, {
            'loss': 0.22195522184191518,
            'params_dict': {'mfi-value': 17, 'fastd-value': 21, 'adx-value': 38, 'rsi-value': 33, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': False, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 87, 'sell-fastd-value': 82, 'sell-adx-value': 78, 'sell-rsi-value': 69, 'sell-mfi-enabled': True, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': False, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 1269, 'roi_t2': 601, 'roi_t3': 444, 'roi_p1': 0.07280999507931168, 'roi_p2': 0.08946698095898986, 'roi_p3': 0.1454876733325284, 'stoploss': -0.18181041180901014},   # noqa: E501
            'params_details': {'buy': {'mfi-value': 17, 'fastd-value': 21, 'adx-value': 38, 'rsi-value': 33, 'mfi-enabled': True, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': False, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 87, 'sell-fastd-value': 82, 'sell-adx-value': 78, 'sell-rsi-value': 69, 'sell-mfi-enabled': True, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': False, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.3077646493708299, 444: 0.16227697603830155, 1045: 0.07280999507931168, 2314: 0}, 'stoploss': {'stoploss': -0.18181041180901014}},  # noqa: E501
            'results_metrics': {'total_trades': 14, 'trade_count_long': 14, 'trade_count_short': 0, 'wins': 6, 'draws': 0, 'losses': 8, 'profit_mean': -0.003539515, 'profit_median': -0.012222, 'profit_total': -0.002480140000000001, 'profit_total_abs': -4.955321, 'max_drawdown': 0.34, 'max_drawdown_abs': -4.955321, 'holding_avg': timedelta(minutes=3402.8571428571427)},  # noqa: E501
            'results_explanation': '    14 trades. Avg profit  -0.35%. Total profit -0.00248014 BTC (  -4.96Σ%). Avg duration 3402.9 min.',  # noqa: E501
            'total_profit': -0.002480140000000001,
            'current_epoch': 5,
            'is_initial_point': True,
            'is_random': False,
            'is_best': True
        }, {
            'loss': 0.545315889154162,
            'params_dict': {'mfi-value': 22, 'fastd-value': 43, 'adx-value': 46, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'bb_lower', 'sell-mfi-value': 87, 'sell-fastd-value': 65, 'sell-adx-value': 94, 'sell-rsi-value': 63, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 319, 'roi_t2': 556, 'roi_t3': 216, 'roi_p1': 0.06251955472249589, 'roi_p2': 0.11659519602202795, 'roi_p3': 0.0953744132197762, 'stoploss': -0.024551752215582423},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 22, 'fastd-value': 43, 'adx-value': 46, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'bb_lower'}, 'sell': {'sell-mfi-value': 87, 'sell-fastd-value': 65, 'sell-adx-value': 94, 'sell-rsi-value': 63, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.2744891639643, 216: 0.17911475074452382, 772: 0.06251955472249589, 1091: 0}, 'stoploss': {'stoploss': -0.024551752215582423}},  # noqa: E501
            'results_metrics': {'total_trades': 39, 'trade_count_long': 39, 'trade_count_short': 0, 'wins': 20, 'draws': 0, 'losses': 19, 'profit_mean': -0.0021400679487179478, 'profit_median': -0.012222, 'profit_total': -0.0041773, 'profit_total_abs': -8.346264999999997, 'max_drawdown': 0.45, 'max_drawdown_abs': -4.955321, 'holding_avg': timedelta(minutes=636.9230769230769)},  # noqa: E501
            'results_explanation': '    39 trades. Avg profit  -0.21%. Total profit -0.00417730 BTC (  -8.35Σ%). Avg duration 636.9 min.',  # noqa: E501
            'total_profit': -0.0041773,
            'current_epoch': 6,
            'is_initial_point': True,
            'is_random': False,
            'is_best': False
        }, {
            'loss': 4.713497421432944,
            'params_dict': {'mfi-value': 13, 'fastd-value': 41, 'adx-value': 21, 'rsi-value': 29, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'bb_lower', 'sell-mfi-value': 99, 'sell-fastd-value': 60, 'sell-adx-value': 81, 'sell-rsi-value': 69, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': False, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 771, 'roi_t2': 620, 'roi_t3': 145, 'roi_p1': 0.0586919200378493, 'roi_p2': 0.04984118697312542, 'roi_p3': 0.37521058680247044, 'stoploss': -0.14613268022709905},  # noqa: E501
            'params_details': {
                'buy': {'mfi-value': 13, 'fastd-value': 41, 'adx-value': 21, 'rsi-value': 29, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'bb_lower'}, 'sell': {'sell-mfi-value': 99, 'sell-fastd-value': 60, 'sell-adx-value': 81, 'sell-rsi-value': 69, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': False, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.4837436938134452, 145: 0.10853310701097472, 765: 0.0586919200378493, 1536: 0},  # noqa: E501
                'stoploss': {'stoploss': -0.14613268022709905}},  # noqa: E501
            'results_metrics': {'total_trades': 318, 'trade_count_long': 318, 'trade_count_short': 0, 'wins': 100, 'draws': 0, 'losses': 218, 'profit_mean': -0.0039833954716981146, 'profit_median': -0.012222, 'profit_total': -0.06339929, 'profit_total_abs': -126.67197600000004, 'max_drawdown': 0.50, 'max_drawdown_abs': -200.955321, 'holding_avg': timedelta(minutes=3140.377358490566)},  # noqa: E501
            'results_explanation': '   318 trades. Avg profit  -0.40%. Total profit -0.06339929 BTC (-126.67Σ%). Avg duration 3140.4 min.',  # noqa: E501
            'total_profit': -0.06339929,
            'current_epoch': 7,
            'is_initial_point': True,
            'is_random': False,
            'is_best': False
        }, {
            'loss': 20.0,  # noqa: E501
            'params_dict': {'mfi-value': 24, 'fastd-value': 43, 'adx-value': 33, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'sar_reversal', 'sell-mfi-value': 89, 'sell-fastd-value': 74, 'sell-adx-value': 70, 'sell-rsi-value': 70, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': False, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal', 'roi_t1': 1149, 'roi_t2': 375, 'roi_t3': 289, 'roi_p1': 0.05571820757172588, 'roi_p2': 0.0606240398618907, 'roi_p3': 0.1729012220156157, 'stoploss': -0.1588514289110401},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 24, 'fastd-value': 43, 'adx-value': 33, 'rsi-value': 20, 'mfi-enabled': False, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': True, 'trigger': 'sar_reversal'}, 'sell': {'sell-mfi-value': 89, 'sell-fastd-value': 74, 'sell-adx-value': 70, 'sell-rsi-value': 70, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': False, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal'}, 'roi': {0: 0.2892434694492323, 289: 0.11634224743361658, 664: 0.05571820757172588, 1813: 0}, 'stoploss': {'stoploss': -0.1588514289110401}},  # noqa: E501
            'results_metrics': {'total_trades': 1, 'trade_count_long': 1, 'trade_count_short': 0, 'wins': 0, 'draws': 1, 'losses': 0, 'profit_mean': 0.0, 'profit_median': 0.0, 'profit_total': 0.0, 'profit_total_abs': 0.0, 'max_drawdown': 0.0, 'max_drawdown_abs': 0.52, 'holding_avg': timedelta(minutes=5340.0)},  # noqa: E501
            'results_explanation': '     1 trades. Avg profit   0.00%. Total profit  0.00000000 BTC (   0.00Σ%). Avg duration 5340.0 min.',  # noqa: E501
            'total_profit': 0.0,
            'current_epoch': 8,
            'is_initial_point': True,
            'is_random': False,
            'is_best': False
        }, {
            'loss': 2.4731817780991223,
            'params_dict': {'mfi-value': 22, 'fastd-value': 20, 'adx-value': 29, 'rsi-value': 40, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'sar_reversal', 'sell-mfi-value': 97, 'sell-fastd-value': 65, 'sell-adx-value': 81, 'sell-rsi-value': 64, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper', 'roi_t1': 1012, 'roi_t2': 584, 'roi_t3': 422, 'roi_p1': 0.036764323603472565, 'roi_p2': 0.10335480573205287, 'roi_p3': 0.10322347377503042, 'stoploss': -0.2780610808108503},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 22, 'fastd-value': 20, 'adx-value': 29, 'rsi-value': 40, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'sar_reversal'}, 'sell': {'sell-mfi-value': 97, 'sell-fastd-value': 65, 'sell-adx-value': 81, 'sell-rsi-value': 64, 'sell-mfi-enabled': True, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper'}, 'roi': {0: 0.2433426031105559, 422: 0.14011912933552545, 1006: 0.036764323603472565, 2018: 0}, 'stoploss': {'stoploss': -0.2780610808108503}},  # noqa: E501
            'results_metrics': {'total_trades': 229, 'trade_count_long': 229, 'trade_count_short': 0, 'wins': 150, 'draws': 0, 'losses': 79, 'profit_mean': -0.0038433433624454144, 'profit_median': -0.012222, 'profit_total': -0.044050070000000004, 'profit_total_abs': -88.01256299999999, 'max_drawdown': 0.41, 'max_drawdown_abs': -150.955321, 'holding_avg': timedelta(minutes=6505.676855895196)},  # noqa: E501
            'results_explanation': '   229 trades. Avg profit  -0.38%. Total profit -0.04405007 BTC ( -88.01Σ%). Avg duration 6505.7 min.',  # noqa: E501
            'total_profit': -0.044050070000000004,  # noqa: E501
            'current_epoch': 9,
            'is_initial_point': True,
            'is_random': False,
            'is_best': False
        }, {
            'loss': -0.2604606005845212,  # noqa: E501
            'params_dict': {'mfi-value': 23, 'fastd-value': 24, 'adx-value': 22, 'rsi-value': 24, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': True, 'trigger': 'macd_cross_signal', 'sell-mfi-value': 97, 'sell-fastd-value': 70, 'sell-adx-value': 64, 'sell-rsi-value': 80, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal', 'roi_t1': 792, 'roi_t2': 464, 'roi_t3': 215, 'roi_p1': 0.04594053535385903, 'roi_p2': 0.09623192684243963, 'roi_p3': 0.04428219070850663, 'stoploss': -0.16992287161634415},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 23, 'fastd-value': 24, 'adx-value': 22, 'rsi-value': 24, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'rsi-enabled': True, 'trigger': 'macd_cross_signal'}, 'sell': {'sell-mfi-value': 97, 'sell-fastd-value': 70, 'sell-adx-value': 64, 'sell-rsi-value': 80, 'sell-mfi-enabled': False, 'sell-fastd-enabled': True, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-sar_reversal'}, 'roi': {0: 0.18645465290480528, 215: 0.14217246219629864, 679: 0.04594053535385903, 1471: 0}, 'stoploss': {'stoploss': -0.16992287161634415}},  # noqa: E501
            'results_metrics': {'total_trades': 4, 'trade_count_long': 4, 'trade_count_short': 0, 'wins': 0, 'draws': 0, 'losses': 4, 'profit_mean': 0.001080385, 'profit_median': -0.012222, 'profit_total': 0.00021629, 'profit_total_abs': 0.432154, 'max_drawdown': 0.13, 'max_drawdown_abs': -4.955321, 'holding_avg': timedelta(minutes=2850.0)},  # noqa: E501
            'results_explanation': '     4 trades. Avg profit   0.11%. Total profit  0.00021629 BTC (   0.43Σ%). Avg duration 2850.0 min.',  # noqa: E501
            'total_profit': 0.00021629,
            'current_epoch': 10,
            'is_initial_point': True,
            'is_random': False,
            'is_best': True
        }, {
            'loss': 4.876465945994304,  # noqa: E501
            'params_dict': {'mfi-value': 20, 'fastd-value': 32, 'adx-value': 49, 'rsi-value': 23, 'mfi-enabled': True, 'fastd-enabled': True, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'bb_lower', 'sell-mfi-value': 75, 'sell-fastd-value': 56, 'sell-adx-value': 61, 'sell-rsi-value': 62, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal', 'roi_t1': 579, 'roi_t2': 614, 'roi_t3': 273, 'roi_p1': 0.05307643172744114, 'roi_p2': 0.1352282078262871, 'roi_p3': 0.1913307406325751, 'stoploss': -0.25728526022513887},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 20, 'fastd-value': 32, 'adx-value': 49, 'rsi-value': 23, 'mfi-enabled': True, 'fastd-enabled': True, 'adx-enabled': False, 'rsi-enabled': False, 'trigger': 'bb_lower'}, 'sell': {'sell-mfi-value': 75, 'sell-fastd-value': 56, 'sell-adx-value': 61, 'sell-rsi-value': 62, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-macd_cross_signal'}, 'roi': {0: 0.3796353801863034, 273: 0.18830463955372825, 887: 0.05307643172744114, 1466: 0}, 'stoploss': {'stoploss': -0.25728526022513887}},  # noqa: E501
            # New Hyperopt mode!
            'results_metrics': {'total_trades': 117, 'trade_count_long': 117, 'trade_count_short': 0, 'wins': 67, 'draws': 0, 'losses': 50, 'profit_mean': -0.012698609145299145, 'profit_median': -0.012222, 'profit_total': -0.07436117, 'profit_total_abs': -148.573727, 'max_drawdown': 0.52, 'max_drawdown_abs': -224.955321, 'holding_avg': timedelta(minutes=4282.5641025641025)},  # noqa: E501
            'results_explanation': '   117 trades. Avg profit  -1.27%. Total profit -0.07436117 BTC (-148.57Σ%). Avg duration 4282.6 min.',  # noqa: E501
            'total_profit': -0.07436117,
            'current_epoch': 11,
            'is_initial_point': True,
            'is_random': False,
            'is_best': False
        }, {
            'loss': 100000,
            'params_dict': {'mfi-value': 10, 'fastd-value': 36, 'adx-value': 31, 'rsi-value': 22, 'mfi-enabled': True, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': False, 'trigger': 'sar_reversal', 'sell-mfi-value': 80, 'sell-fastd-value': 71, 'sell-adx-value': 60, 'sell-rsi-value': 85, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper', 'roi_t1': 1156, 'roi_t2': 581, 'roi_t3': 408, 'roi_p1': 0.06860454019988212, 'roi_p2': 0.12473718444931989, 'roi_p3': 0.2896360635226823, 'stoploss': -0.30889015124682806},  # noqa: E501
            'params_details': {'buy': {'mfi-value': 10, 'fastd-value': 36, 'adx-value': 31, 'rsi-value': 22, 'mfi-enabled': True, 'fastd-enabled': True, 'adx-enabled': True, 'rsi-enabled': False, 'trigger': 'sar_reversal'}, 'sell': {'sell-mfi-value': 80, 'sell-fastd-value': 71, 'sell-adx-value': 60, 'sell-rsi-value': 85, 'sell-mfi-enabled': False, 'sell-fastd-enabled': False, 'sell-adx-enabled': True, 'sell-rsi-enabled': True, 'sell-trigger': 'sell-bb_upper'}, 'roi': {0: 0.4829777881718843, 408: 0.19334172464920202, 989: 0.06860454019988212, 2145: 0}, 'stoploss': {'stoploss': -0.30889015124682806}},  # noqa: E501
            'results_metrics': {'total_trades': 0, 'trade_count_long': 0, 'trade_count_short': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'profit_mean': None, 'profit_median': None, 'profit_total': 0, 'profit_total_abs': 0.0, 'max_drawdown': 0.0, 'max_drawdown_abs': 0.0, 'holding_avg': timedelta()},  # noqa: E501
            'results_explanation': '     0 trades. Avg profit    nan%. Total profit  0.00000000 BTC (   0.00Σ%). Avg duration   nan min.',  # noqa: E501
            'total_profit': 0,
            'current_epoch': 12,
            'is_initial_point': True,
            'is_random': False,
            'is_best': False
            }
    ]

    for res in hyperopt_res:
        res['results_metrics']['holding_avg_s'] = res['results_metrics']['holding_avg'
                                                                         ].total_seconds()

    return hyperopt_res


@pytest.fixture(scope='function')
def limit_buy_order_usdt_open():
    return {
        'id': 'mocked_limit_buy_usdt',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'datetime': dt_now().isoformat(),
        'timestamp': dt_ts(),
        'price': 2.00,
        'average': 2.00,
        'amount': 30.0,
        'filled': 0.0,
        'cost': 60.0,
        'remaining': 30.0,
        'status': 'open'
    }


@pytest.fixture(scope='function')
def limit_buy_order_usdt(limit_buy_order_usdt_open):
    order = deepcopy(limit_buy_order_usdt_open)
    order['status'] = 'closed'
    order['filled'] = order['amount']
    order['remaining'] = 0.0
    return order


@pytest.fixture
def limit_sell_order_usdt_open():
    return {
        'id': 'mocked_limit_sell_usdt',
        'type': 'limit',
        'side': 'sell',
        'symbol': 'mocked',
        'datetime': dt_now().isoformat(),
        'timestamp': dt_ts(),
        'price': 2.20,
        'amount': 30.0,
        'cost': 66.0,
        'filled': 0.0,
        'remaining': 30.0,
        'status': 'open'
    }


@pytest.fixture
def limit_sell_order_usdt(limit_sell_order_usdt_open):
    order = deepcopy(limit_sell_order_usdt_open)
    order['remaining'] = 0.0
    order['filled'] = order['amount']
    order['status'] = 'closed'
    return order


@pytest.fixture(scope='function')
def market_buy_order_usdt():
    return {
        'id': 'mocked_market_buy',
        'type': 'market',
        'side': 'buy',
        'symbol': 'mocked',
        'timestamp': dt_ts(),
        'datetime': dt_now().isoformat(),
        'price': 2.00,
        'amount': 30.0,
        'filled': 30.0,
        'remaining': 0.0,
        'status': 'closed'
    }


@pytest.fixture
def market_buy_order_usdt_doublefee(market_buy_order_usdt):
    order = deepcopy(market_buy_order_usdt)
    order['fee'] = None
    # Market orders filled with 2 trades can have fees in different currencies
    # assuming the account runs out of BNB.
    order['fees'] = [
        {'cost': 0.00025125, 'currency': 'BNB'},
        {'cost': 0.05030681, 'currency': 'USDT'},
    ]
    order['trades'] = [{
        'timestamp': None,
        'datetime': None,
        'symbol': 'ETH/USDT',
        'id': None,
        'order': '123',
        'type': 'market',
        'side': 'sell',
        'takerOrMaker': None,
        'price': 2.01,
        'amount': 25.0,
        'cost': 50.25,
        'fee': {'cost': 0.00025125, 'currency': 'BNB'}
    }, {
        'timestamp': None,
        'datetime': None,
        'symbol': 'ETH/USDT',
        'id': None,
        'order': '123',
        'type': 'market',
        'side': 'sell',
        'takerOrMaker': None,
        'price': 2.0,
        'amount': 5,
        'cost': 10,
        'fee': {'cost': 0.0100306, 'currency': 'USDT'}
    }]
    return order


@pytest.fixture
def market_sell_order_usdt():
    return {
        'id': 'mocked_limit_sell',
        'type': 'market',
        'side': 'sell',
        'symbol': 'mocked',
        'timestamp': dt_ts(),
        'datetime': dt_now().isoformat(),
        'price': 2.20,
        'amount': 30.0,
        'filled': 30.0,
        'remaining': 0.0,
        'status': 'closed'
    }


@pytest.fixture(scope='function')
def limit_order(limit_buy_order_usdt, limit_sell_order_usdt):
    return {
        'buy': limit_buy_order_usdt,
        'sell': limit_sell_order_usdt
    }


@pytest.fixture(scope='function')
def limit_order_open(limit_buy_order_usdt_open, limit_sell_order_usdt_open):
    return {
        'buy': limit_buy_order_usdt_open,
        'sell': limit_sell_order_usdt_open
    }


@pytest.fixture(scope='function')
def mark_ohlcv():
    return [
        [1630454400000, 2.77, 2.77, 2.73, 2.73, 0],
        [1630458000000, 2.73, 2.76, 2.72, 2.74, 0],
        [1630461600000, 2.74, 2.76, 2.74, 2.76, 0],
        [1630465200000, 2.76, 2.76, 2.74, 2.76, 0],
        [1630468800000, 2.76, 2.77, 2.75, 2.77, 0],
        [1630472400000, 2.77, 2.79, 2.75, 2.78, 0],
        [1630476000000, 2.78, 2.80, 2.77, 2.77, 0],
        [1630479600000, 2.78, 2.79, 2.77, 2.77, 0],
        [1630483200000, 2.77, 2.79, 2.77, 2.78, 0],
        [1630486800000, 2.77, 2.84, 2.77, 2.84, 0],
        [1630490400000, 2.84, 2.85, 2.81, 2.81, 0],
        [1630494000000, 2.81, 2.83, 2.81, 2.81, 0],
        [1630497600000, 2.81, 2.84, 2.81, 2.82, 0],
        [1630501200000, 2.82, 2.83, 2.81, 2.81, 0],
    ]


@pytest.fixture(scope='function')
def funding_rate_history_hourly():
    return [
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000008,
            "timestamp": 1630454400000,
            "datetime": "2021-09-01T00:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000004,
            "timestamp": 1630458000000,
            "datetime": "2021-09-01T01:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000012,
            "timestamp": 1630461600000,
            "datetime": "2021-09-01T02:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000003,
            "timestamp": 1630465200000,
            "datetime": "2021-09-01T03:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000007,
            "timestamp": 1630468800000,
            "datetime": "2021-09-01T04:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000003,
            "timestamp": 1630472400000,
            "datetime": "2021-09-01T05:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000019,
            "timestamp": 1630476000000,
            "datetime": "2021-09-01T06:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000003,
            "timestamp": 1630479600000,
            "datetime": "2021-09-01T07:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000003,
            "timestamp": 1630483200000,
            "datetime": "2021-09-01T08:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0,
            "timestamp": 1630486800000,
            "datetime": "2021-09-01T09:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000013,
            "timestamp": 1630490400000,
            "datetime": "2021-09-01T10:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000077,
            "timestamp": 1630494000000,
            "datetime": "2021-09-01T11:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000072,
            "timestamp": 1630497600000,
            "datetime": "2021-09-01T12:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000097,
            "timestamp": 1630501200000,
            "datetime": "2021-09-01T13:00:00.000Z"
        },
    ]


@pytest.fixture(scope='function')
def funding_rate_history_octohourly():
    return [
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000008,
            "timestamp": 1630454400000,
            "datetime": "2021-09-01T00:00:00.000Z"
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000003,
            "timestamp": 1630483200000,
            "datetime": "2021-09-01T08:00:00.000Z"
        }
    ]


@pytest.fixture(scope='function')
def leverage_tiers():
    return {
        "1000SHIB/USDT:USDT": [
            {
                'minNotional': 0,
                'maxNotional': 50000,
                'maintenanceMarginRate': 0.01,
                'maxLeverage': 50,
                'maintAmt': 0.0
            },
            {
                'minNotional': 50000,
                'maxNotional': 150000,
                'maintenanceMarginRate': 0.025,
                'maxLeverage': 20,
                'maintAmt': 750.0
            },
            {
                'minNotional': 150000,
                'maxNotional': 250000,
                'maintenanceMarginRate': 0.05,
                'maxLeverage': 10,
                'maintAmt': 4500.0
            },
            {
                'minNotional': 250000,
                'maxNotional': 500000,
                'maintenanceMarginRate': 0.1,
                'maxLeverage': 5,
                'maintAmt': 17000.0
            },
            {
                'minNotional': 500000,
                'maxNotional': 1000000,
                'maintenanceMarginRate': 0.125,
                'maxLeverage': 4,
                'maintAmt': 29500.0
            },
            {
                'minNotional': 1000000,
                'maxNotional': 2000000,
                'maintenanceMarginRate': 0.25,
                'maxLeverage': 2,
                'maintAmt': 154500.0
            },
            {
                'minNotional': 2000000,
                'maxNotional': 30000000,
                'maintenanceMarginRate': 0.5,
                'maxLeverage': 1,
                'maintAmt': 654500.0
            },
        ],
        "1INCH/USDT:USDT": [
            {
                'minNotional': 0,
                'maxNotional': 5000,
                'maintenanceMarginRate': 0.012,
                'maxLeverage': 50,
                'maintAmt': 0.0
            },
            {
                'minNotional': 5000,
                'maxNotional': 25000,
                'maintenanceMarginRate': 0.025,
                'maxLeverage': 20,
                'maintAmt': 65.0
            },
            {
                'minNotional': 25000,
                'maxNotional': 100000,
                'maintenanceMarginRate': 0.05,
                'maxLeverage': 10,
                'maintAmt': 690.0
            },
            {
                'minNotional': 100000,
                'maxNotional': 250000,
                'maintenanceMarginRate': 0.1,
                'maxLeverage': 5,
                'maintAmt': 5690.0
            },
            {
                'minNotional': 250000,
                'maxNotional': 1000000,
                'maintenanceMarginRate': 0.125,
                'maxLeverage': 2,
                'maintAmt': 11940.0
            },
            {
                'minNotional': 1000000,
                'maxNotional': 100000000,
                'maintenanceMarginRate': 0.5,
                'maxLeverage': 1,
                'maintAmt': 386940.0
            },
        ],
        "AAVE/USDT:USDT": [
            {
                'minNotional': 0,
                'maxNotional': 5000,
                'maintenanceMarginRate': 0.01,
                'maxLeverage': 50,
                'maintAmt': 0.0
            },
            {
                'minNotional': 5000,
                'maxNotional': 25000,
                'maintenanceMarginRate': 0.02,
                'maxLeverage': 25,
                'maintAmt': 75.0
            },
            {
                'minNotional': 25000,
                'maxNotional': 100000,
                'maintenanceMarginRate': 0.05,
                'maxLeverage': 10,
                'maintAmt': 700.0
            },
            {
                'minNotional': 100000,
                'maxNotional': 250000,
                'maintenanceMarginRate': 0.1,
                'maxLeverage': 5,
                'maintAmt': 5700.0
            },
            {
                'minNotional': 250000,
                'maxNotional': 1000000,
                'maintenanceMarginRate': 0.125,
                'maxLeverage': 2,
                'maintAmt': 11950.0
            },
            {
                'minNotional': 10000000,
                'maxNotional': 50000000,
                'maintenanceMarginRate': 0.5,
                'maxLeverage': 1,
                'maintAmt': 386950.0
            },
        ],
        "ADA/BUSD:BUSD": [
            {
                "minNotional": 0,
                "maxNotional": 100000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 0.0
            },
            {
                "minNotional": 100000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 2500.0
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 27500.0
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 77500.0
            },
            {
                "minNotional": 2000000,
                "maxNotional": 5000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 277500.0
            },
            {
                "minNotional": 5000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 1527500.0
            },
        ],
        'BNB/BUSD:BUSD': [
            {
                "minNotional": 0,       # stake(before leverage) = 0
                "maxNotional": 100000,  # max stake(before leverage) = 5000
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 0.0
            },
            {
                "minNotional": 100000,  # stake = 10000.0
                "maxNotional": 500000,  # max_stake = 50000.0
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 2500.0
            },
            {
                "minNotional": 500000,   # stake = 100000.0
                "maxNotional": 1000000,  # max_stake = 200000.0
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 27500.0
            },
            {
                "minNotional": 1000000,  # stake = 333333.3333333333
                "maxNotional": 2000000,  # max_stake = 666666.6666666666
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 77500.0
            },
            {
                "minNotional": 2000000,  # stake = 1000000.0
                "maxNotional": 5000000,  # max_stake = 2500000.0
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 277500.0
            },
            {
                "minNotional": 5000000,   # stake = 5000000.0
                "maxNotional": 30000000,  # max_stake = 30000000.0
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 1527500.0
            }
        ],
        'BNB/USDT:USDT': [
            {
                "minNotional": 0,      # stake = 0.0
                "maxNotional": 10000,  # max_stake = 133.33333333333334
                "maintenanceMarginRate": 0.0065,
                "maxLeverage": 75,
                "maintAmt": 0.0
            },
            {
                "minNotional": 10000,  # stake = 200.0
                "maxNotional": 50000,  # max_stake = 1000.0
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 35.0
            },
            {
                "minNotional": 50000,   # stake = 2000.0
                "maxNotional": 250000,  # max_stake = 10000.0
                "maintenanceMarginRate": 0.02,
                "maxLeverage": 25,
                "maintAmt": 535.0
            },
            {
                "minNotional": 250000,   # stake = 25000.0
                "maxNotional": 1000000,  # max_stake = 100000.0
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 8035.0
            },
            {
                "minNotional": 1000000,  # stake = 200000.0
                "maxNotional": 2000000,  # max_stake = 400000.0
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 58035.0
            },
            {
                "minNotional": 2000000,  # stake = 500000.0
                "maxNotional": 5000000,  # max_stake = 1250000.0
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "maintAmt": 108035.0
            },
            {
                "minNotional": 5000000,   # stake = 1666666.6666666667
                "maxNotional": 10000000,  # max_stake = 3333333.3333333335
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 233035.0
            },
            {
                "minNotional": 10000000,  # stake = 5000000.0
                "maxNotional": 20000000,  # max_stake = 10000000.0
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 1233035.0
            },
            {
                "minNotional": 20000000,  # stake = 20000000.0
                "maxNotional": 50000000,  # max_stake = 50000000.0
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 6233035.0
            },
        ],
        'BTC/USDT:USDT': [
            {
                "minNotional": 0,      # stake = 0.0
                "maxNotional": 50000,  # max_stake = 400.0
                "maintenanceMarginRate": 0.004,
                "maxLeverage": 125,
                "maintAmt": 0.0
            },
            {
                "minNotional": 50000,   # stake = 500.0
                "maxNotional": 250000,  # max_stake = 2500.0
                "maintenanceMarginRate": 0.005,
                "maxLeverage": 100,
                "maintAmt": 50.0
            },
            {
                "minNotional": 250000,   # stake = 5000.0
                "maxNotional": 1000000,  # max_stake = 20000.0
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 1300.0
            },
            {
                "minNotional": 1000000,  # stake = 50000.0
                "maxNotional": 7500000,  # max_stake = 375000.0
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 16300.0
            },
            {
                "minNotional": 7500000,   # stake = 750000.0
                "maxNotional": 40000000,  # max_stake = 4000000.0
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 203800.0
            },
            {
                "minNotional": 40000000,   # stake = 8000000.0
                "maxNotional": 100000000,  # max_stake = 20000000.0
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 2203800.0
            },
            {
                "minNotional": 100000000,  # stake = 25000000.0
                "maxNotional": 200000000,  # max_stake = 50000000.0
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "maintAmt": 4703800.0
            },
            {
                "minNotional": 200000000,  # stake = 66666666.666666664
                "maxNotional": 400000000,  # max_stake = 133333333.33333333
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 9703800.0
            },
            {
                "minNotional": 400000000,  # stake = 200000000.0
                "maxNotional": 600000000,  # max_stake = 300000000.0
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 4.97038E7
            },
            {
                "minNotional": 600000000,   # stake = 600000000.0
                "maxNotional": 1000000000,  # max_stake = 1000000000.0
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 1.997038E8
            },
        ],
        "ZEC/USDT:USDT": [
            {
                'minNotional': 0,
                'maxNotional': 50000,
                'maintenanceMarginRate': 0.01,
                'maxLeverage': 50,
                'maintAmt': 0.0
            },
            {
                'minNotional': 50000,
                'maxNotional': 150000,
                'maintenanceMarginRate': 0.025,
                'maxLeverage': 20,
                'maintAmt': 750.0
            },
            {
                'minNotional': 150000,
                'maxNotional': 250000,
                'maintenanceMarginRate': 0.05,
                'maxLeverage': 10,
                'maintAmt': 4500.0
            },
            {
                'minNotional': 250000,
                'maxNotional': 500000,
                'maintenanceMarginRate': 0.1,
                'maxLeverage': 5,
                'maintAmt': 17000.0
            },
            {
                'minNotional': 500000,
                'maxNotional': 1000000,
                'maintenanceMarginRate': 0.125,
                'maxLeverage': 4,
                'maintAmt': 29500.0
            },
            {
                'minNotional': 1000000,
                'maxNotional': 2000000,
                'maintenanceMarginRate': 0.25,
                'maxLeverage': 2,
                'maintAmt': 154500.0
            },
            {
                'minNotional': 2000000,
                'maxNotional': 30000000,
                'maintenanceMarginRate': 0.5,
                'maxLeverage': 1,
                'maintAmt': 654500.0
            },
        ]
    }
