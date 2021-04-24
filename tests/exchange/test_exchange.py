import copy
import logging
from datetime import datetime, timedelta, timezone
from math import isclose
from random import randint
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import arrow
import ccxt
import pytest
from pandas import DataFrame

from freqtrade.exceptions import (DDosProtection, DependencyException, InvalidOrderException,
                                  OperationalException, TemporaryError)
from freqtrade.exchange import Binance, Bittrex, Exchange, Kraken
from freqtrade.exchange.common import (API_FETCH_ORDER_RETRY_COUNT, API_RETRY_COUNT,
                                       calculate_backoff)
from freqtrade.exchange.exchange import (market_is_active, timeframe_to_minutes, timeframe_to_msecs,
                                         timeframe_to_next_date, timeframe_to_prev_date,
                                         timeframe_to_seconds)
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from tests.conftest import get_mock_coro, get_patched_exchange, log_has, log_has_re


# Make sure to always keep one exchange here which is NOT subclassed!!
EXCHANGES = ['bittrex', 'binance', 'kraken', 'ftx']


def ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           fun, mock_ccxt_fun, retries=API_RETRY_COUNT + 1, **kwargs):

    with patch('freqtrade.exchange.common.time.sleep'):
        with pytest.raises(DDosProtection):
            api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.DDoSProtection("DDos"))
            exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
            getattr(exchange, fun)(**kwargs)
        assert api_mock.__dict__[mock_ccxt_fun].call_count == retries

    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.NetworkError("DeaDBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == retries

    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1


async def async_ccxt_exception(mocker, default_conf, api_mock, fun, mock_ccxt_fun,
                               retries=API_RETRY_COUNT + 1, **kwargs):

    with patch('freqtrade.exchange.common.asyncio.sleep', get_mock_coro(None)):
        with pytest.raises(DDosProtection):
            api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.DDoSProtection("Dooh"))
            exchange = get_patched_exchange(mocker, default_conf, api_mock)
            await getattr(exchange, fun)(**kwargs)
        assert api_mock.__dict__[mock_ccxt_fun].call_count == retries

    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.NetworkError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == retries

    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1


def test_init(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Instance is running with dry_run enabled', caplog)


def test_init_ccxt_kwargs(default_conf, mocker, caplog):
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    caplog.set_level(logging.INFO)
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_async_config'] = {'aiohttp_trust_env': True, 'asyncio_loop': True}
    ex = Exchange(conf)
    assert log_has(
        "Applying additional ccxt config: {'aiohttp_trust_env': True, 'asyncio_loop': True}",
        caplog)
    assert ex._api_async.aiohttp_trust_env
    assert not ex._api.aiohttp_trust_env

    # Reset logging and config
    caplog.clear()
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_config'] = {'TestKWARG': 11}
    conf['exchange']['ccxt_sync_config'] = {'TestKWARG44': 11}
    conf['exchange']['ccxt_async_config'] = {'asyncio_loop': True}
    asynclogmsg = "Applying additional ccxt config: {'TestKWARG': 11, 'asyncio_loop': True}"
    ex = Exchange(conf)
    assert not ex._api_async.aiohttp_trust_env
    assert hasattr(ex._api, 'TestKWARG')
    assert ex._api.TestKWARG == 11
    # ccxt_config is assigned to both sync and async
    assert not hasattr(ex._api_async, 'TestKWARG44')

    assert hasattr(ex._api_async, 'TestKWARG')
    assert log_has("Applying additional ccxt config: {'TestKWARG': 11, 'TestKWARG44': 11}", caplog)
    assert log_has(asynclogmsg, caplog)


def test_destroy(default_conf, mocker, caplog):
    caplog.set_level(logging.DEBUG)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Exchange object destroyed, closing async loop', caplog)


def test_init_exception(default_conf, mocker):
    default_conf['exchange']['name'] = 'wrong_exchange_name'

    with pytest.raises(OperationalException,
                       match=f"Exchange {default_conf['exchange']['name']} is not supported"):
        Exchange(default_conf)

    default_conf['exchange']['name'] = 'binance'
    with pytest.raises(OperationalException,
                       match=f"Exchange {default_conf['exchange']['name']} is not supported"):
        mocker.patch("ccxt.binance", MagicMock(side_effect=AttributeError))
        Exchange(default_conf)

    with pytest.raises(OperationalException,
                       match=r"Initialization of ccxt failed. Reason: DeadBeef"):
        mocker.patch("ccxt.binance", MagicMock(side_effect=ccxt.BaseError("DeadBeef")))
        Exchange(default_conf)


def test_exchange_resolver(default_conf, mocker, caplog):
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=MagicMock()))
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')

    exchange = ExchangeResolver.load_exchange('huobi', default_conf)
    assert isinstance(exchange, Exchange)
    assert log_has_re(r"No .* specific subclass found. Using the generic class instead.", caplog)
    caplog.clear()

    exchange = ExchangeResolver.load_exchange('Bittrex', default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Bittrex)
    assert not log_has_re(r"No .* specific subclass found. Using the generic class instead.",
                          caplog)
    caplog.clear()

    exchange = ExchangeResolver.load_exchange('kraken', default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Kraken)
    assert not isinstance(exchange, Binance)
    assert not log_has_re(r"No .* specific subclass found. Using the generic class instead.",
                          caplog)

    exchange = ExchangeResolver.load_exchange('binance', default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Binance)
    assert not isinstance(exchange, Kraken)

    assert not log_has_re(r"No .* specific subclass found. Using the generic class instead.",
                          caplog)

    # Test mapping
    exchange = ExchangeResolver.load_exchange('binanceus', default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Binance)
    assert not isinstance(exchange, Kraken)


def test_validate_order_time_in_force(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    # explicitly test bittrex, exchanges implementing other policies need seperate tests
    ex = get_patched_exchange(mocker, default_conf, id="bittrex")
    tif = {
        "buy": "gtc",
        "sell": "gtc",
    }

    ex.validate_order_time_in_force(tif)
    tif2 = {
        "buy": "fok",
        "sell": "ioc",
    }
    with pytest.raises(OperationalException, match=r"Time in force.*not supported for .*"):
        ex.validate_order_time_in_force(tif2)

    # Patch to see if this will pass if the values are in the ft dict
    ex._ft_has.update({"order_time_in_force": ["gtc", "fok", "ioc"]})
    ex.validate_order_time_in_force(tif2)


@pytest.mark.parametrize("amount,precision_mode,precision,expected", [
    (2.34559, 2, 4, 2.3455),
    (2.34559, 2, 5, 2.34559),
    (2.34559, 2, 3, 2.345),
    (2.9999, 2, 3, 2.999),
    (2.9909, 2, 3, 2.990),
    # Tests for Tick-size
    (2.34559, 4, 0.0001, 2.3455),
    (2.34559, 4, 0.00001, 2.34559),
    (2.34559, 4, 0.001, 2.345),
    (2.9999, 4, 0.001, 2.999),
    (2.9909, 4, 0.001, 2.990),
    (2.9909, 4, 0.005, 2.990),
    (2.9999, 4, 0.005, 2.995),
])
def test_amount_to_precision(default_conf, mocker, amount, precision_mode, precision, expected):
    '''
    Test rounds down
    '''

    markets = PropertyMock(return_value={'ETH/BTC': {'precision': {'amount': precision}}})

    exchange = get_patched_exchange(mocker, default_conf, id="binance")
    # digits counting mode
    # DECIMAL_PLACES = 2
    # SIGNIFICANT_DIGITS = 3
    # TICK_SIZE = 4
    mocker.patch('freqtrade.exchange.Exchange.precisionMode',
                 PropertyMock(return_value=precision_mode))
    mocker.patch('freqtrade.exchange.Exchange.markets', markets)

    pair = 'ETH/BTC'
    assert exchange.amount_to_precision(pair, amount) == expected


@pytest.mark.parametrize("price,precision_mode,precision,expected", [
    (2.34559, 2, 4, 2.3456),
    (2.34559, 2, 5, 2.34559),
    (2.34559, 2, 3, 2.346),
    (2.9999, 2, 3, 3.000),
    (2.9909, 2, 3, 2.991),
    # Tests for Tick_size
    (2.34559, 4, 0.0001, 2.3456),
    (2.34559, 4, 0.00001, 2.34559),
    (2.34559, 4, 0.001, 2.346),
    (2.9999, 4, 0.001, 3.000),
    (2.9909, 4, 0.001, 2.991),
    (2.9909, 4, 0.005, 2.995),
    (2.9973, 4, 0.005, 3.0),
    (2.9977, 4, 0.005, 3.0),
    (234.43, 4, 0.5, 234.5),
    (234.53, 4, 0.5, 235.0),
    (0.891534, 4, 0.0001, 0.8916),

])
def test_price_to_precision(default_conf, mocker, price, precision_mode, precision, expected):
    '''
    Test price to precision
    '''
    markets = PropertyMock(return_value={'ETH/BTC': {'precision': {'price': precision}}})

    exchange = get_patched_exchange(mocker, default_conf, id="binance")
    mocker.patch('freqtrade.exchange.Exchange.markets', markets)
    # digits counting mode
    # DECIMAL_PLACES = 2
    # SIGNIFICANT_DIGITS = 3
    # TICK_SIZE = 4
    mocker.patch('freqtrade.exchange.Exchange.precisionMode',
                 PropertyMock(return_value=precision_mode))

    pair = 'ETH/BTC'
    assert pytest.approx(exchange.price_to_precision(pair, price)) == expected


@pytest.mark.parametrize("price,precision_mode,precision,expected", [
    (2.34559, 2, 4, 0.0001),
    (2.34559, 2, 5, 0.00001),
    (2.34559, 2, 3, 0.001),
    (2.9999, 2, 3, 0.001),
    (200.0511, 2, 3, 0.001),
    # Tests for Tick_size
    (2.34559, 4, 0.0001, 0.0001),
    (2.34559, 4, 0.00001, 0.00001),
    (2.34559, 4, 0.0025, 0.0025),
    (2.9909, 4, 0.0025, 0.0025),
    (234.43, 4, 0.5, 0.5),
    (234.43, 4, 0.0025, 0.0025),
    (234.43, 4, 0.00013, 0.00013),

])
def test_price_get_one_pip(default_conf, mocker, price, precision_mode, precision, expected):
    markets = PropertyMock(return_value={'ETH/BTC': {'precision': {'price': precision}}})
    exchange = get_patched_exchange(mocker, default_conf, id="binance")
    mocker.patch('freqtrade.exchange.Exchange.markets', markets)
    mocker.patch('freqtrade.exchange.Exchange.precisionMode',
                 PropertyMock(return_value=precision_mode))
    pair = 'ETH/BTC'
    assert pytest.approx(exchange.price_get_one_pip(pair, price)) == expected


def test_get_min_pair_stake_amount(mocker, default_conf) -> None:

    exchange = get_patched_exchange(mocker, default_conf, id="binance")
    stoploss = -0.05
    markets = {'ETH/BTC': {'symbol': 'ETH/BTC'}}

    # no pair found
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    with pytest.raises(ValueError, match=r'.*get market information.*'):
        exchange.get_min_pair_stake_amount('BNB/BTC', 1, stoploss)

    # no 'limits' section
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    assert result is None

    # empty 'limits' section
    markets["ETH/BTC"]["limits"] = {}
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    assert result is None

    # no cost Min
    markets["ETH/BTC"]["limits"] = {
        'cost': {"min": None},
        'amount': {}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    assert result is None

    # no amount Min
    markets["ETH/BTC"]["limits"] = {
        'cost': {},
        'amount': {"min": None}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    assert result is None

    # empty 'cost'/'amount' section
    markets["ETH/BTC"]["limits"] = {
        'cost': {},
        'amount': {}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    assert result is None

    # min cost is set
    markets["ETH/BTC"]["limits"] = {
        'cost': {'min': 2},
        'amount': {}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    assert isclose(result, 2 * (1+0.05) / (1-abs(stoploss)))

    # min amount is set
    markets["ETH/BTC"]["limits"] = {
        'cost': {},
        'amount': {'min': 2}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss)
    assert isclose(result, 2 * 2 * (1+0.05) / (1-abs(stoploss)))

    # min amount and cost are set (cost is minimal)
    markets["ETH/BTC"]["limits"] = {
        'cost': {'min': 2},
        'amount': {'min': 2}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss)
    assert isclose(result, max(2, 2 * 2) * (1+0.05) / (1-abs(stoploss)))

    # min amount and cost are set (amount is minial)
    markets["ETH/BTC"]["limits"] = {
        'cost': {'min': 8},
        'amount': {'min': 2}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss)
    assert isclose(result, max(8, 2 * 2) * (1+0.05) / (1-abs(stoploss)))

    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, -0.4)
    assert isclose(result, max(8, 2 * 2) * 1.5)

    # Really big stoploss
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, -1)
    assert isclose(result, max(8, 2 * 2) * 1.5)


def test_get_min_pair_stake_amount_real_data(mocker, default_conf) -> None:
    exchange = get_patched_exchange(mocker, default_conf, id="binance")
    stoploss = -0.05
    markets = {'ETH/BTC': {'symbol': 'ETH/BTC'}}

    # Real Binance data
    markets["ETH/BTC"]["limits"] = {
        'cost': {'min': 0.0001},
        'amount': {'min': 0.001}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 0.020405, stoploss)
    assert round(result, 8) == round(
        max(0.0001, 0.001 * 0.020405) * (1+0.05) / (1-abs(stoploss)),
        8
    )


def test_set_sandbox(default_conf, mocker):
    """
    Test working scenario
    """
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'LTC/BTC': '', 'XRP/BTC': '', 'NEO/BTC': ''
    })
    url_mock = PropertyMock(return_value={'test': "api-public.sandbox.gdax.com",
                                          'api': 'https://api.gdax.com'})
    type(api_mock).urls = url_mock
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    liveurl = exchange._api.urls['api']
    default_conf['exchange']['sandbox'] = True
    exchange.set_sandbox(exchange._api, default_conf['exchange'], 'Logname')
    assert exchange._api.urls['api'] != liveurl


def test_set_sandbox_exception(default_conf, mocker):
    """
    Test Fail scenario
    """
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'LTC/BTC': '', 'XRP/BTC': '', 'NEO/BTC': ''
    })
    url_mock = PropertyMock(return_value={'api': 'https://api.gdax.com'})
    type(api_mock).urls = url_mock

    with pytest.raises(OperationalException, match=r'does not provide a sandbox api'):
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        default_conf['exchange']['sandbox'] = True
        exchange.set_sandbox(exchange._api, default_conf['exchange'], 'Logname')


def test__load_async_markets(default_conf, mocker, caplog):
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt')
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    exchange = Exchange(default_conf)
    exchange._api_async.load_markets = get_mock_coro(None)
    exchange._load_async_markets()
    assert exchange._api_async.load_markets.call_count == 1
    caplog.set_level(logging.DEBUG)

    exchange._api_async.load_markets = Mock(side_effect=ccxt.BaseError("deadbeef"))
    exchange._load_async_markets()

    assert log_has('Could not load async markets. Reason: deadbeef', caplog)


def test__load_markets(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(side_effect=ccxt.BaseError("SomeError"))
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    Exchange(default_conf)
    assert log_has('Unable to initialize markets.', caplog)

    expected_return = {'ETH/BTC': 'available'}
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value=expected_return)
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    default_conf['exchange']['pair_whitelist'] = ['ETH/BTC']
    ex = Exchange(default_conf)

    assert ex.markets == expected_return


def test_reload_markets(default_conf, mocker, caplog):
    caplog.set_level(logging.DEBUG)
    initial_markets = {'ETH/BTC': {}}
    updated_markets = {'ETH/BTC': {}, "LTC/BTC": {}}

    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value=initial_markets)
    default_conf['exchange']['markets_refresh_interval'] = 10
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="binance",
                                    mock_markets=False)
    exchange._load_async_markets = MagicMock()
    exchange._last_markets_refresh = arrow.utcnow().int_timestamp

    assert exchange.markets == initial_markets

    # less than 10 minutes have passed, no reload
    exchange.reload_markets()
    assert exchange.markets == initial_markets
    assert exchange._load_async_markets.call_count == 0

    api_mock.load_markets = MagicMock(return_value=updated_markets)
    # more than 10 minutes have passed, reload is executed
    exchange._last_markets_refresh = arrow.utcnow().int_timestamp - 15 * 60
    exchange.reload_markets()
    assert exchange.markets == updated_markets
    assert exchange._load_async_markets.call_count == 1
    assert log_has('Performing scheduled market reload..', caplog)


def test_reload_markets_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.DEBUG)

    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(side_effect=ccxt.NetworkError("LoadError"))
    default_conf['exchange']['markets_refresh_interval'] = 10
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="binance")

    # less than 10 minutes have passed, no reload
    exchange.reload_markets()
    assert exchange._last_markets_refresh == 0
    assert log_has_re(r"Could not reload markets.*", caplog)


@pytest.mark.parametrize("stake_currency", ['ETH', 'BTC', 'USDT'])
def test_validate_stake_currency(default_conf, stake_currency, mocker, caplog):
    default_conf['stake_currency'] = stake_currency
    api_mock = MagicMock()
    type(api_mock).load_markets = MagicMock(return_value={
        'ETH/BTC': {'quote': 'BTC'}, 'LTC/BTC': {'quote': 'BTC'},
        'XRP/ETH': {'quote': 'ETH'}, 'NEO/USDT': {'quote': 'USDT'},
    })
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    Exchange(default_conf)


def test_validate_stake_currency_error(default_conf, mocker, caplog):
    default_conf['stake_currency'] = 'XRP'
    api_mock = MagicMock()
    type(api_mock).load_markets = MagicMock(return_value={
        'ETH/BTC': {'quote': 'BTC'}, 'LTC/BTC': {'quote': 'BTC'},
        'XRP/ETH': {'quote': 'ETH'}, 'NEO/USDT': {'quote': 'USDT'},
    })
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    with pytest.raises(OperationalException,
                       match=r'XRP is not available as stake on .*'
                       'Available currencies are: BTC, ETH, USDT'):
        Exchange(default_conf)


def test_get_quote_currencies(default_conf, mocker):
    ex = get_patched_exchange(mocker, default_conf)

    assert set(ex.get_quote_currencies()) == set(['USD', 'ETH', 'BTC', 'USDT'])


@pytest.mark.parametrize('pair,expected', [
    ('XRP/BTC', 'BTC'),
    ('LTC/USD', 'USD'),
    ('ETH/USDT', 'USDT'),
    ('XLTCUSDT', 'USDT'),
    ('XRP/NOCURRENCY', ''),
])
def test_get_pair_quote_currency(default_conf, mocker, pair, expected):
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.get_pair_quote_currency(pair) == expected


@pytest.mark.parametrize('pair,expected', [
    ('XRP/BTC', 'XRP'),
    ('LTC/USD', 'LTC'),
    ('ETH/USDT', 'ETH'),
    ('XLTCUSDT', 'LTC'),
    ('XRP/NOCURRENCY', ''),
])
def test_get_pair_base_currency(default_conf, mocker, pair, expected):
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.get_pair_base_currency(pair) == expected


def test_validate_pairs(default_conf, mocker):  # test exchange.validate_pairs directly
    api_mock = MagicMock()
    type(api_mock).load_markets = MagicMock(return_value={
        'ETH/BTC': {'quote': 'BTC'},
        'LTC/BTC': {'quote': 'BTC'},
        'XRP/BTC': {'quote': 'BTC'},
        'NEO/BTC': {'quote': 'BTC'},
    })
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    Exchange(default_conf)


def test_validate_pairs_not_available(default_conf, mocker):
    api_mock = MagicMock()
    type(api_mock).markets = PropertyMock(return_value={
        'XRP/BTC': {'inactive': True, 'base': 'XRP', 'quote': 'BTC'}
    })
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')

    with pytest.raises(OperationalException, match=r'not available'):
        Exchange(default_conf)


def test_validate_pairs_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value='Binance'))

    type(api_mock).markets = PropertyMock(return_value={})
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', api_mock)
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')

    with pytest.raises(OperationalException, match=r'Pair ETH/BTC is not available on Binance'):
        Exchange(default_conf)

    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value={}))
    Exchange(default_conf)
    assert log_has('Unable to validate pairs (assuming they are correct).', caplog)


def test_validate_pairs_restricted(default_conf, mocker, caplog):
    api_mock = MagicMock()
    type(api_mock).load_markets = MagicMock(return_value={
        'ETH/BTC': {'quote': 'BTC'}, 'LTC/BTC': {'quote': 'BTC'},
        'XRP/BTC': {'quote': 'BTC', 'info': {'IsRestricted': True}},
        'NEO/BTC': {'quote': 'BTC', 'info': 'TestString'},  # info can also be a string ...
    })
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')

    Exchange(default_conf)
    assert log_has("Pair XRP/BTC is restricted for some users on this exchange."
                   "Please check if you are impacted by this restriction "
                   "on the exchange and eventually remove XRP/BTC from your whitelist.", caplog)


def test_validate_pairs_stakecompatibility(default_conf, mocker, caplog):
    api_mock = MagicMock()
    type(api_mock).load_markets = MagicMock(return_value={
        'ETH/BTC': {'quote': 'BTC'}, 'LTC/BTC': {'quote': 'BTC'},
        'XRP/BTC': {'quote': 'BTC'}, 'NEO/BTC': {'quote': 'BTC'},
        'HELLO-WORLD': {'quote': 'BTC'},
    })
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')

    Exchange(default_conf)


def test_validate_pairs_stakecompatibility_downloaddata(default_conf, mocker, caplog):
    api_mock = MagicMock()
    default_conf['stake_currency'] = ''
    type(api_mock).load_markets = MagicMock(return_value={
        'ETH/BTC': {'quote': 'BTC'}, 'LTC/BTC': {'quote': 'BTC'},
        'XRP/BTC': {'quote': 'BTC'}, 'NEO/BTC': {'quote': 'BTC'},
        'HELLO-WORLD': {'quote': 'BTC'},
    })
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')

    Exchange(default_conf)
    assert type(api_mock).load_markets.call_count == 1


def test_validate_pairs_stakecompatibility_fail(default_conf, mocker, caplog):
    default_conf['exchange']['pair_whitelist'].append('HELLO-WORLD')
    api_mock = MagicMock()
    type(api_mock).load_markets = MagicMock(return_value={
        'ETH/BTC': {'quote': 'BTC'}, 'LTC/BTC': {'quote': 'BTC'},
        'XRP/BTC': {'quote': 'BTC'}, 'NEO/BTC': {'quote': 'BTC'},
        'HELLO-WORLD': {'quote': 'USDT'},
    })
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')

    with pytest.raises(OperationalException, match=r"Stake-currency 'BTC' not compatible with.*"):
        Exchange(default_conf)


@pytest.mark.parametrize("timeframe", [
    ('5m'), ("1m"), ("15m"), ("1h")
])
def test_validate_timeframes(default_conf, mocker, timeframe):
    default_conf["timeframe"] = timeframe
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={'1m': '1m',
                                            '5m': '5m',
                                            '15m': '15m',
                                            '1h': '1h'})
    type(api_mock).timeframes = timeframes

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    Exchange(default_conf)


def test_validate_timeframes_failed(default_conf, mocker):
    default_conf["timeframe"] = "3m"
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={'15s': '15s',
                                            '1m': '1m',
                                            '5m': '5m',
                                            '15m': '15m',
                                            '1h': '1h'})
    type(api_mock).timeframes = timeframes

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs', MagicMock())
    with pytest.raises(OperationalException,
                       match=r"Invalid timeframe '3m'. This exchange supports.*"):
        Exchange(default_conf)
    default_conf["timeframe"] = "15s"

    with pytest.raises(OperationalException,
                       match=r"Timeframes < 1m are currently not supported by Freqtrade."):
        Exchange(default_conf)


def test_validate_timeframes_emulated_ohlcv_1(default_conf, mocker):
    default_conf["timeframe"] = "3m"
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock

    # delete timeframes so magicmock does not autocreate it
    del api_mock.timeframes

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    with pytest.raises(OperationalException,
                       match=r'The ccxt library does not provide the list of timeframes '
                             r'for the exchange ".*" and this exchange '
                             r'is therefore not supported. *'):
        Exchange(default_conf)


def test_validate_timeframes_emulated_ohlcvi_2(default_conf, mocker):
    default_conf["timeframe"] = "3m"
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock

    # delete timeframes so magicmock does not autocreate it
    del api_mock.timeframes

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange._load_markets',
                 MagicMock(return_value={'timeframes': None}))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    with pytest.raises(OperationalException,
                       match=r'The ccxt library does not provide the list of timeframes '
                             r'for the exchange ".*" and this exchange '
                             r'is therefore not supported. *'):
        Exchange(default_conf)


def test_validate_timeframes_not_in_config(default_conf, mocker):
    del default_conf["timeframe"]
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={'1m': '1m',
                                            '5m': '5m',
                                            '15m': '15m',
                                            '1h': '1h'})
    type(api_mock).timeframes = timeframes

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    Exchange(default_conf)


def test_validate_order_types(default_conf, mocker):
    api_mock = MagicMock()

    type(api_mock).has = PropertyMock(return_value={'createMarketOrder': True})
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')
    mocker.patch('freqtrade.exchange.Exchange.name', 'Bittrex')

    default_conf['order_types'] = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    Exchange(default_conf)

    type(api_mock).has = PropertyMock(return_value={'createMarketOrder': False})
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))

    default_conf['order_types'] = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    with pytest.raises(OperationalException,
                       match=r'Exchange .* does not support market orders.'):
        Exchange(default_conf)

    default_conf['order_types'] = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }
    with pytest.raises(OperationalException,
                       match=r'On exchange stoploss is not supported for .*'):
        Exchange(default_conf)


def test_validate_order_types_not_in_config(default_conf, mocker):
    api_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')

    conf = copy.deepcopy(default_conf)
    Exchange(conf)


def test_validate_required_startup_candles(default_conf, mocker, caplog):
    api_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value='Binance'))

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', api_mock)
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes')
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets')
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs')
    mocker.patch('freqtrade.exchange.Exchange.validate_stakecurrency')

    default_conf['startup_candle_count'] = 20
    ex = Exchange(default_conf)
    assert ex
    default_conf['startup_candle_count'] = 600

    with pytest.raises(OperationalException, match=r'This strategy requires 600.*'):
        Exchange(default_conf)


def test_exchange_has(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf)
    assert not exchange.exchange_has('ASDFASDF')
    api_mock = MagicMock()

    type(api_mock).has = PropertyMock(return_value={'deadbeef': True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert exchange.exchange_has("deadbeef")

    type(api_mock).has = PropertyMock(return_value={'deadbeef': False})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert not exchange.exchange_has("deadbeef")


@pytest.mark.parametrize("side", [
    ("buy"),
    ("sell")
])
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_create_dry_run_order(default_conf, mocker, side, exchange_name):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)

    order = exchange.create_dry_run_order(
        pair='ETH/BTC', ordertype='limit', side=side, amount=1, rate=200)
    assert 'id' in order
    assert f'dry_run_{side}_' in order["id"]
    assert order["side"] == side
    assert order["type"] == "limit"
    assert order["symbol"] == "ETH/BTC"


@pytest.mark.parametrize("side", [
    ("buy"),
    ("sell")
])
@pytest.mark.parametrize("ordertype,rate,marketprice", [
    ("market", None, None),
    ("market", 200, True),
    ("limit", 200, None),
    ("stop_loss_limit", 200, None)
])
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_create_order(default_conf, mocker, side, ordertype, rate, marketprice, exchange_name):
    api_mock = MagicMock()
    order_id = 'test_prod_{}_{}'.format(side, randint(0, 10 ** 6))
    api_mock.options = {} if not marketprice else {"createMarketBuyOrderRequiresPrice": True}
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False
    mocker.patch('freqtrade.exchange.Exchange.amount_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.price_to_precision', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    order = exchange.create_order(
        pair='ETH/BTC', ordertype=ordertype, side=side, amount=1, rate=200)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == ordertype
    assert api_mock.create_order.call_args[0][2] == side
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] is rate


def test_buy_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf)

    order = exchange.buy(pair='ETH/BTC', ordertype='limit',
                         amount=1, rate=200, time_in_force='gtc')
    assert 'id' in order
    assert 'dry_run_buy_' in order['id']


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_buy_prod(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))
    order_type = 'market'
    time_in_force = 'gtc'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False
    mocker.patch('freqtrade.exchange.Exchange.amount_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.price_to_precision', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    order = exchange.buy(pair='ETH/BTC', ordertype=order_type,
                         amount=1, rate=200, time_in_force=time_in_force)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] is None

    api_mock.create_order.reset_mock()
    order_type = 'limit'
    order = exchange.buy(
        pair='ETH/BTC',
        ordertype=order_type,
        amount=1,
        rate=200,
        time_in_force=time_in_force)
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("Not enough funds"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.buy(pair='ETH/BTC', ordertype=order_type,
                     amount=1, rate=200, time_in_force=time_in_force)

    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder("Order not found"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.buy(pair='ETH/BTC', ordertype='limit',
                     amount=1, rate=200, time_in_force=time_in_force)

    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder("Order not found"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.buy(pair='ETH/BTC', ordertype='market',
                     amount=1, rate=200, time_in_force=time_in_force)

    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError("Network disconnect"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.buy(pair='ETH/BTC', ordertype=order_type,
                     amount=1, rate=200, time_in_force=time_in_force)

    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError("Unknown error"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.buy(pair='ETH/BTC', ordertype=order_type,
                     amount=1, rate=200, time_in_force=time_in_force)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_buy_considers_time_in_force(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False
    mocker.patch('freqtrade.exchange.Exchange.amount_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.price_to_precision', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    order_type = 'limit'
    time_in_force = 'ioc'

    order = exchange.buy(pair='ETH/BTC', ordertype=order_type,
                         amount=1, rate=200, time_in_force=time_in_force)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200
    assert "timeInForce" in api_mock.create_order.call_args[0][5]
    assert api_mock.create_order.call_args[0][5]["timeInForce"] == time_in_force

    order_type = 'market'
    time_in_force = 'ioc'

    order = exchange.buy(pair='ETH/BTC', ordertype=order_type,
                         amount=1, rate=200, time_in_force=time_in_force)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] is None
    # Market orders should not send timeInForce!!
    assert "timeInForce" not in api_mock.create_order.call_args[0][5]


def test_sell_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf)

    order = exchange.sell(pair='ETH/BTC', ordertype='limit', amount=1, rate=200)
    assert 'id' in order
    assert 'dry_run_sell_' in order['id']


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_sell_prod(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = 'test_prod_sell_{}'.format(randint(0, 10 ** 6))
    order_type = 'market'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False

    mocker.patch('freqtrade.exchange.Exchange.amount_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.price_to_precision', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    order = exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'sell'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] is None

    api_mock.create_order.reset_mock()
    order_type = 'limit'
    order = exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'sell'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("0 balance"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder("Order not found"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.sell(pair='ETH/BTC', ordertype='limit', amount=1, rate=200)

    # Market orders don't require price, so the behaviour is slightly different
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder("Order not found"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.sell(pair='ETH/BTC', ordertype='market', amount=1, rate=200)

    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError("No Connection"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_sell_considers_time_in_force(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = 'test_prod_sell_{}'.format(randint(0, 10 ** 6))
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    api_mock.options = {}
    default_conf['dry_run'] = False
    mocker.patch('freqtrade.exchange.Exchange.amount_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.price_to_precision', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    order_type = 'limit'
    time_in_force = 'ioc'

    order = exchange.sell(pair='ETH/BTC', ordertype=order_type,
                          amount=1, rate=200, time_in_force=time_in_force)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'sell'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200
    assert "timeInForce" in api_mock.create_order.call_args[0][5]
    assert api_mock.create_order.call_args[0][5]["timeInForce"] == time_in_force

    order_type = 'market'
    time_in_force = 'ioc'
    order = exchange.sell(pair='ETH/BTC', ordertype=order_type,
                          amount=1, rate=200, time_in_force=time_in_force)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'sell'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] is None
    # Market orders should not send timeInForce!!
    assert "timeInForce" not in api_mock.create_order.call_args[0][5]


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_balance_prod(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={'BTC': {'free': 123.4, 'total': 123.4}})
    default_conf['dry_run'] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    assert exchange.get_balance(currency='BTC') == 123.4

    with pytest.raises(OperationalException):
        api_mock.fetch_balance = MagicMock(side_effect=ccxt.BaseError("Unknown error"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

        exchange.get_balance(currency='BTC')

    with pytest.raises(TemporaryError, match=r'.*balance due to malformed exchange response:.*'):
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        mocker.patch('freqtrade.exchange.Exchange.get_balances', MagicMock(return_value={}))
        mocker.patch('freqtrade.exchange.Kraken.get_balances', MagicMock(return_value={}))
        exchange.get_balance(currency='BTC')


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_balances_prod(default_conf, mocker, exchange_name):
    balance_item = {
        'free': 10.0,
        'total': 10.0,
        'used': 0.0
    }

    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={
        '1ST': balance_item,
        '2ST': balance_item,
        '3ST': balance_item
    })
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    assert len(exchange.get_balances()) == 3
    assert exchange.get_balances()['1ST']['free'] == 10.0
    assert exchange.get_balances()['1ST']['total'] == 10.0
    assert exchange.get_balances()['1ST']['used'] == 0.0

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           "get_balances", "fetch_balance")


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_tickers(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    tick = {'ETH/BTC': {
        'symbol': 'ETH/BTC',
        'bid': 0.5,
        'ask': 1,
        'last': 42,
    }, 'BCH/BTC': {
        'symbol': 'BCH/BTC',
        'bid': 0.6,
        'ask': 0.5,
        'last': 41,
    }
    }
    api_mock.fetch_tickers = MagicMock(return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    # retrieve original ticker
    tickers = exchange.get_tickers()

    assert 'ETH/BTC' in tickers
    assert 'BCH/BTC' in tickers
    assert tickers['ETH/BTC']['bid'] == 0.5
    assert tickers['ETH/BTC']['ask'] == 1
    assert tickers['BCH/BTC']['bid'] == 0.6
    assert tickers['BCH/BTC']['ask'] == 0.5
    assert api_mock.fetch_tickers.call_count == 1

    api_mock.fetch_tickers.reset_mock()

    # Cached ticker should not call api again
    tickers2 = exchange.get_tickers(cached=True)
    assert tickers2 == tickers
    assert api_mock.fetch_tickers.call_count == 0
    tickers2 = exchange.get_tickers(cached=False)
    assert api_mock.fetch_tickers.call_count == 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           "get_tickers", "fetch_tickers")

    with pytest.raises(OperationalException):
        api_mock.fetch_tickers = MagicMock(side_effect=ccxt.NotSupported("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.get_tickers()

    api_mock.fetch_tickers = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    exchange.get_tickers()


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_ticker(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    tick = {
        'symbol': 'ETH/BTC',
        'bid': 0.00001098,
        'ask': 0.00001099,
        'last': 0.0001,
    }
    api_mock.fetch_ticker = MagicMock(return_value=tick)
    api_mock.markets = {'ETH/BTC': {'active': True}}
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    # retrieve original ticker
    ticker = exchange.fetch_ticker(pair='ETH/BTC')

    assert ticker['bid'] == 0.00001098
    assert ticker['ask'] == 0.00001099

    # change the ticker
    tick = {
        'symbol': 'ETH/BTC',
        'bid': 0.5,
        'ask': 1,
        'last': 42,
    }
    api_mock.fetch_ticker = MagicMock(return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    # if not caching the result we should get the same ticker
    # if not fetching a new result we should get the cached ticker
    ticker = exchange.fetch_ticker(pair='ETH/BTC')

    assert api_mock.fetch_ticker.call_count == 1
    assert ticker['bid'] == 0.5
    assert ticker['ask'] == 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           "fetch_ticker", "fetch_ticker",
                           pair='ETH/BTC')

    api_mock.fetch_ticker = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    exchange.fetch_ticker(pair='ETH/BTC')

    with pytest.raises(DependencyException, match=r'Pair XRP/ETH not available'):
        exchange.fetch_ticker(pair='XRP/ETH')


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_historic_ohlcv(default_conf, mocker, caplog, exchange_name):
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    ohlcv = [
        [
            arrow.utcnow().int_timestamp * 1000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]
    pair = 'ETH/BTC'

    async def mock_candle_hist(pair, timeframe, since_ms):
        return pair, timeframe, ohlcv

    exchange._async_get_candle_history = Mock(wraps=mock_candle_hist)
    # one_call calculation * 1.8 should do 2 calls

    since = 5 * 60 * exchange.ohlcv_candle_limit('5m') * 1.8
    ret = exchange.get_historic_ohlcv(pair, "5m", int((
        arrow.utcnow().int_timestamp - since) * 1000))

    assert exchange._async_get_candle_history.call_count == 2
    # Returns twice the above OHLCV data
    assert len(ret) == 2

    caplog.clear()

    async def mock_get_candle_hist_error(pair, *args, **kwargs):
        raise TimeoutError()

    exchange._async_get_candle_history = MagicMock(side_effect=mock_get_candle_hist_error)
    ret = exchange.get_historic_ohlcv(pair, "5m", int(
        (arrow.utcnow().int_timestamp - since) * 1000))
    assert log_has_re(r"Async code raised an exception: .*", caplog)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_historic_ohlcv_as_df(default_conf, mocker, exchange_name):
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    ohlcv = [
        [
            arrow.utcnow().int_timestamp * 1000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ],
        [
            arrow.utcnow().shift(minutes=5).int_timestamp * 1000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ],
        [
            arrow.utcnow().shift(minutes=10).int_timestamp * 1000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]
    pair = 'ETH/BTC'

    async def mock_candle_hist(pair, timeframe, since_ms):
        return pair, timeframe, ohlcv

    exchange._async_get_candle_history = Mock(wraps=mock_candle_hist)
    # one_call calculation * 1.8 should do 2 calls

    since = 5 * 60 * exchange.ohlcv_candle_limit('5m') * 1.8
    ret = exchange.get_historic_ohlcv_as_df(pair, "5m", int((
        arrow.utcnow().int_timestamp - since) * 1000))

    assert exchange._async_get_candle_history.call_count == 2
    # Returns twice the above OHLCV data
    assert len(ret) == 2
    assert isinstance(ret, DataFrame)
    assert 'date' in ret.columns
    assert 'open' in ret.columns
    assert 'close' in ret.columns
    assert 'high' in ret.columns


def test_refresh_latest_ohlcv(mocker, default_conf, caplog) -> None:
    ohlcv = [
        [
            (arrow.utcnow().int_timestamp - 1) * 1000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ],
        [
            arrow.utcnow().int_timestamp * 1000,  # unix timestamp ms
            3,  # open
            1,  # high
            4,  # low
            6,  # close
            5,  # volume (in quote currency)
        ]
    ]

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)

    pairs = [('IOTA/ETH', '5m'), ('XRP/ETH', '5m')]
    # empty dicts
    assert not exchange._klines
    exchange.refresh_latest_ohlcv(pairs, cache=False)
    # No caching
    assert not exchange._klines
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    exchange._api_async.fetch_ohlcv.reset_mock()

    exchange.refresh_latest_ohlcv(pairs)

    assert log_has(f'Refreshing candle (OHLCV) data for {len(pairs)} pairs', caplog)
    assert exchange._klines
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    for pair in pairs:
        assert isinstance(exchange.klines(pair), DataFrame)
        assert len(exchange.klines(pair)) > 0

        # klines function should return a different object on each call
        # if copy is "True"
        assert exchange.klines(pair) is not exchange.klines(pair)
        assert exchange.klines(pair) is not exchange.klines(pair, copy=True)
        assert exchange.klines(pair, copy=True) is not exchange.klines(pair, copy=True)
        assert exchange.klines(pair, copy=False) is exchange.klines(pair, copy=False)

    # test caching
    exchange.refresh_latest_ohlcv([('IOTA/ETH', '5m'), ('XRP/ETH', '5m')])

    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert log_has(f"Using cached candle (OHLCV) data for pair {pairs[0][0]}, "
                   f"timeframe {pairs[0][1]} ...",
                   caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_get_candle_history(default_conf, mocker, caplog, exchange_name):
    ohlcv = [
        [
            arrow.utcnow().int_timestamp * 1000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    # Monkey-patch async function
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)

    pair = 'ETH/BTC'
    res = await exchange._async_get_candle_history(pair, "5m")
    assert type(res) is tuple
    assert len(res) == 3
    assert res[0] == pair
    assert res[1] == "5m"
    assert res[2] == ohlcv
    assert exchange._api_async.fetch_ohlcv.call_count == 1
    assert not log_has(f"Using cached candle (OHLCV) data for {pair} ...", caplog)

    # exchange = Exchange(default_conf)
    await async_ccxt_exception(mocker, default_conf, MagicMock(),
                               "_async_get_candle_history", "fetch_ohlcv",
                               pair='ABCD/BTC', timeframe=default_conf['timeframe'])

    api_mock = MagicMock()
    with pytest.raises(OperationalException,
                       match=r'Could not fetch historical candle \(OHLCV\) data.*'):
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.BaseError("Unknown error"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        await exchange._async_get_candle_history(pair, "5m",
                                                 (arrow.utcnow().int_timestamp - 2000) * 1000)

    with pytest.raises(OperationalException, match=r'Exchange.* does not support fetching '
                                                   r'historical candle \(OHLCV\) data\..*'):
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.NotSupported("Not supported"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        await exchange._async_get_candle_history(pair, "5m",
                                                 (arrow.utcnow().int_timestamp - 2000) * 1000)


@pytest.mark.asyncio
async def test__async_get_candle_history_empty(default_conf, mocker, caplog):
    """ Test empty exchange result """
    ohlcv = []

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    # Monkey-patch async function
    exchange._api_async.fetch_ohlcv = get_mock_coro([])

    exchange = Exchange(default_conf)
    pair = 'ETH/BTC'
    res = await exchange._async_get_candle_history(pair, "5m")
    assert type(res) is tuple
    assert len(res) == 3
    assert res[0] == pair
    assert res[1] == "5m"
    assert res[2] == ohlcv
    assert exchange._api_async.fetch_ohlcv.call_count == 1


def test_refresh_latest_ohlcv_inv_result(default_conf, mocker, caplog):

    async def mock_get_candle_hist(pair, *args, **kwargs):
        if pair == 'ETH/BTC':
            return [[]]
        else:
            raise TypeError()

    exchange = get_patched_exchange(mocker, default_conf)

    # Monkey-patch async function with empty result
    exchange._api_async.fetch_ohlcv = MagicMock(side_effect=mock_get_candle_hist)

    pairs = [("ETH/BTC", "5m"), ("XRP/BTC", "5m")]
    res = exchange.refresh_latest_ohlcv(pairs)
    assert exchange._klines
    assert exchange._api_async.fetch_ohlcv.call_count == 2

    assert type(res) is dict
    assert len(res) == 1
    # Test that each is in list at least once as order is not guaranteed
    assert log_has("Error loading ETH/BTC. Result was [[]].", caplog)
    assert log_has("Async code raised an exception: TypeError", caplog)


def test_get_next_limit_in_list():
    limit_range = [5, 10, 20, 50, 100, 500, 1000]
    assert Exchange.get_next_limit_in_list(1, limit_range) == 5
    assert Exchange.get_next_limit_in_list(5, limit_range) == 5
    assert Exchange.get_next_limit_in_list(6, limit_range) == 10
    assert Exchange.get_next_limit_in_list(9, limit_range) == 10
    assert Exchange.get_next_limit_in_list(10, limit_range) == 10
    assert Exchange.get_next_limit_in_list(11, limit_range) == 20
    assert Exchange.get_next_limit_in_list(19, limit_range) == 20
    assert Exchange.get_next_limit_in_list(21, limit_range) == 50
    assert Exchange.get_next_limit_in_list(51, limit_range) == 100
    assert Exchange.get_next_limit_in_list(1000, limit_range) == 1000
    # Going over the limit ...
    assert Exchange.get_next_limit_in_list(1001, limit_range) == 1000
    assert Exchange.get_next_limit_in_list(2000, limit_range) == 1000
    # Without required range
    assert Exchange.get_next_limit_in_list(2000, limit_range, False) is None
    assert Exchange.get_next_limit_in_list(15, limit_range, False) == 20

    assert Exchange.get_next_limit_in_list(21, None) == 21
    assert Exchange.get_next_limit_in_list(100, None) == 100
    assert Exchange.get_next_limit_in_list(1000, None) == 1000


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_l2_order_book(default_conf, mocker, order_book_l2, exchange_name):
    default_conf['exchange']['name'] = exchange_name
    api_mock = MagicMock()

    api_mock.fetch_l2_order_book = order_book_l2
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    order_book = exchange.fetch_l2_order_book(pair='ETH/BTC', limit=10)
    assert 'bids' in order_book
    assert 'asks' in order_book
    assert len(order_book['bids']) == 10
    assert len(order_book['asks']) == 10
    assert api_mock.fetch_l2_order_book.call_args_list[0][0][0] == 'ETH/BTC'

    for val in [1, 5, 10, 12, 20, 50, 100]:
        api_mock.fetch_l2_order_book.reset_mock()

        order_book = exchange.fetch_l2_order_book(pair='ETH/BTC', limit=val)
        assert api_mock.fetch_l2_order_book.call_args_list[0][0][0] == 'ETH/BTC'
        # Not all exchanges support all limits for orderbook
        if not exchange._ft_has['l2_limit_range'] or val in exchange._ft_has['l2_limit_range']:
            assert api_mock.fetch_l2_order_book.call_args_list[0][0][1] == val
        else:
            next_limit = exchange.get_next_limit_in_list(val, exchange._ft_has['l2_limit_range'])
            assert api_mock.fetch_l2_order_book.call_args_list[0][0][1] == next_limit


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_l2_order_book_exception(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    with pytest.raises(OperationalException):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NotSupported("Not supported"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.fetch_l2_order_book(pair='ETH/BTC', limit=50)
    with pytest.raises(TemporaryError):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NetworkError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.fetch_l2_order_book(pair='ETH/BTC', limit=50)
    with pytest.raises(OperationalException):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.fetch_l2_order_book(pair='ETH/BTC', limit=50)


def make_fetch_ohlcv_mock(data):
    def fetch_ohlcv_mock(pair, timeframe, since):
        if since:
            assert since > data[-1][0]
            return []
        return data
    return fetch_ohlcv_mock


@pytest.mark.parametrize("exchange_name", EXCHANGES)
@pytest.mark.asyncio
async def test___async_get_candle_history_sort(default_conf, mocker, exchange_name):
    def sort_data(data, key):
        return sorted(data, key=key)

    # GDAX use-case (real data from GDAX)
    # This OHLCV data is ordered DESC (newest first, oldest last)
    ohlcv = [
        [1527833100000, 0.07666, 0.07671, 0.07666, 0.07668, 16.65244264],
        [1527832800000, 0.07662, 0.07666, 0.07662, 0.07666, 1.30051526],
        [1527832500000, 0.07656, 0.07661, 0.07656, 0.07661, 12.034778840000001],
        [1527832200000, 0.07658, 0.07658, 0.07655, 0.07656, 0.59780186],
        [1527831900000, 0.07658, 0.07658, 0.07658, 0.07658, 1.76278136],
        [1527831600000, 0.07658, 0.07658, 0.07658, 0.07658, 2.22646521],
        [1527831300000, 0.07655, 0.07657, 0.07655, 0.07657, 1.1753],
        [1527831000000, 0.07654, 0.07654, 0.07651, 0.07651, 0.8073060299999999],
        [1527830700000, 0.07652, 0.07652, 0.07651, 0.07652, 10.04822687],
        [1527830400000, 0.07649, 0.07651, 0.07649, 0.07651, 2.5734867]
    ]
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    sort_mock = mocker.patch('freqtrade.exchange.exchange.sorted', MagicMock(side_effect=sort_data))
    # Test the OHLCV data sort
    res = await exchange._async_get_candle_history('ETH/BTC', default_conf['timeframe'])
    assert res[0] == 'ETH/BTC'
    res_ohlcv = res[2]

    assert sort_mock.call_count == 1
    assert res_ohlcv[0][0] == 1527830400000
    assert res_ohlcv[0][1] == 0.07649
    assert res_ohlcv[0][2] == 0.07651
    assert res_ohlcv[0][3] == 0.07649
    assert res_ohlcv[0][4] == 0.07651
    assert res_ohlcv[0][5] == 2.5734867

    assert res_ohlcv[9][0] == 1527833100000
    assert res_ohlcv[9][1] == 0.07666
    assert res_ohlcv[9][2] == 0.07671
    assert res_ohlcv[9][3] == 0.07666
    assert res_ohlcv[9][4] == 0.07668
    assert res_ohlcv[9][5] == 16.65244264

    # Bittrex use-case (real data from Bittrex)
    # This OHLCV data is ordered ASC (oldest first, newest last)
    ohlcv = [
        [1527827700000, 0.07659999, 0.0766, 0.07627, 0.07657998, 1.85216924],
        [1527828000000, 0.07657995, 0.07657995, 0.0763, 0.0763, 26.04051037],
        [1527828300000, 0.0763, 0.07659998, 0.0763, 0.0764, 10.36434124],
        [1527828600000, 0.0764, 0.0766, 0.0764, 0.0766, 5.71044773],
        [1527828900000, 0.0764, 0.07666998, 0.0764, 0.07666998, 47.48888565],
        [1527829200000, 0.0765, 0.07672999, 0.0765, 0.07672999, 3.37640326],
        [1527829500000, 0.0766, 0.07675, 0.0765, 0.07675, 8.36203831],
        [1527829800000, 0.07675, 0.07677999, 0.07620002, 0.076695, 119.22963884],
        [1527830100000, 0.076695, 0.07671, 0.07624171, 0.07671, 1.80689244],
        [1527830400000, 0.07671, 0.07674399, 0.07629216, 0.07655213, 2.31452783]
    ]
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    # Reset sort mock
    sort_mock = mocker.patch('freqtrade.exchange.sorted', MagicMock(side_effect=sort_data))
    # Test the OHLCV data sort
    res = await exchange._async_get_candle_history('ETH/BTC', default_conf['timeframe'])
    assert res[0] == 'ETH/BTC'
    assert res[1] == default_conf['timeframe']
    res_ohlcv = res[2]
    # Sorted not called again - data is already in order
    assert sort_mock.call_count == 0
    assert res_ohlcv[0][0] == 1527827700000
    assert res_ohlcv[0][1] == 0.07659999
    assert res_ohlcv[0][2] == 0.0766
    assert res_ohlcv[0][3] == 0.07627
    assert res_ohlcv[0][4] == 0.07657998
    assert res_ohlcv[0][5] == 1.85216924

    assert res_ohlcv[9][0] == 1527830400000
    assert res_ohlcv[9][1] == 0.07671
    assert res_ohlcv[9][2] == 0.07674399
    assert res_ohlcv[9][3] == 0.07629216
    assert res_ohlcv[9][4] == 0.07655213
    assert res_ohlcv[9][5] == 2.31452783


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_fetch_trades(default_conf, mocker, caplog, exchange_name,
                                   fetch_trades_result):

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    # Monkey-patch async function
    exchange._api_async.fetch_trades = get_mock_coro(fetch_trades_result)

    pair = 'ETH/BTC'
    res = await exchange._async_fetch_trades(pair, since=None, params=None)
    assert type(res) is list
    assert isinstance(res[0], list)
    assert isinstance(res[1], list)

    assert exchange._api_async.fetch_trades.call_count == 1
    assert exchange._api_async.fetch_trades.call_args[0][0] == pair
    assert exchange._api_async.fetch_trades.call_args[1]['limit'] == 1000

    assert log_has_re(f"Fetching trades for pair {pair}, since .*", caplog)
    caplog.clear()
    exchange._api_async.fetch_trades.reset_mock()
    res = await exchange._async_fetch_trades(pair, since=None, params={'from': '123'})
    assert exchange._api_async.fetch_trades.call_count == 1
    assert exchange._api_async.fetch_trades.call_args[0][0] == pair
    assert exchange._api_async.fetch_trades.call_args[1]['limit'] == 1000
    assert exchange._api_async.fetch_trades.call_args[1]['params'] == {'from': '123'}
    assert log_has_re(f"Fetching trades for pair {pair}, params: .*", caplog)

    exchange = Exchange(default_conf)
    await async_ccxt_exception(mocker, default_conf, MagicMock(),
                               "_async_fetch_trades", "fetch_trades",
                               pair='ABCD/BTC', since=None)

    api_mock = MagicMock()
    with pytest.raises(OperationalException, match=r'Could not fetch trade data*'):
        api_mock.fetch_trades = MagicMock(side_effect=ccxt.BaseError("Unknown error"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        await exchange._async_fetch_trades(pair, since=(arrow.utcnow().int_timestamp - 2000) * 1000)

    with pytest.raises(OperationalException, match=r'Exchange.* does not support fetching '
                                                   r'historical trade data\..*'):
        api_mock.fetch_trades = MagicMock(side_effect=ccxt.NotSupported("Not supported"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        await exchange._async_fetch_trades(pair, since=(arrow.utcnow().int_timestamp - 2000) * 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_get_trade_history_id(default_conf, mocker, exchange_name,
                                           fetch_trades_result):

    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    pagination_arg = exchange._trades_pagination_arg

    async def mock_get_trade_hist(pair, *args, **kwargs):
        if 'since' in kwargs:
            # Return first 3
            return fetch_trades_result[:-2]
        elif kwargs.get('params', {}).get(pagination_arg) == fetch_trades_result[-3]['id']:
            # Return 2
            return fetch_trades_result[-3:-1]
        else:
            # Return last 2
            return fetch_trades_result[-2:]
    # Monkey-patch async function
    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)

    pair = 'ETH/BTC'
    ret = await exchange._async_get_trade_history_id(pair,
                                                     since=fetch_trades_result[0]['timestamp'],
                                                     until=fetch_trades_result[-1]['timestamp'] - 1)
    assert type(ret) is tuple
    assert ret[0] == pair
    assert type(ret[1]) is list
    assert len(ret[1]) == len(fetch_trades_result)
    assert exchange._api_async.fetch_trades.call_count == 3
    fetch_trades_cal = exchange._api_async.fetch_trades.call_args_list
    # first call (using since, not fromId)
    assert fetch_trades_cal[0][0][0] == pair
    assert fetch_trades_cal[0][1]['since'] == fetch_trades_result[0]['timestamp']

    # 2nd call
    assert fetch_trades_cal[1][0][0] == pair
    assert 'params' in fetch_trades_cal[1][1]
    assert exchange._ft_has['trades_pagination_arg'] in fetch_trades_cal[1][1]['params']


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_get_trade_history_time(default_conf, mocker, caplog, exchange_name,
                                             fetch_trades_result):

    caplog.set_level(logging.DEBUG)

    async def mock_get_trade_hist(pair, *args, **kwargs):
        if kwargs['since'] == fetch_trades_result[0]['timestamp']:
            return fetch_trades_result[:-1]
        else:
            return fetch_trades_result[-1:]

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    # Monkey-patch async function
    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    pair = 'ETH/BTC'
    ret = await exchange._async_get_trade_history_time(pair,
                                                       since=fetch_trades_result[0]['timestamp'],
                                                       until=fetch_trades_result[-1]['timestamp']-1)
    assert type(ret) is tuple
    assert ret[0] == pair
    assert type(ret[1]) is list
    assert len(ret[1]) == len(fetch_trades_result)
    assert exchange._api_async.fetch_trades.call_count == 2
    fetch_trades_cal = exchange._api_async.fetch_trades.call_args_list
    # first call (using since, not fromId)
    assert fetch_trades_cal[0][0][0] == pair
    assert fetch_trades_cal[0][1]['since'] == fetch_trades_result[0]['timestamp']

    # 2nd call
    assert fetch_trades_cal[1][0][0] == pair
    assert fetch_trades_cal[1][1]['since'] == fetch_trades_result[-2]['timestamp']
    assert log_has_re(r"Stopping because until was reached.*", caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_get_trade_history_time_empty(default_conf, mocker, caplog, exchange_name,
                                                   trades_history):

    caplog.set_level(logging.DEBUG)

    async def mock_get_trade_hist(pair, *args, **kwargs):
        if kwargs['since'] == trades_history[0][0]:
            return trades_history[:-1]
        else:
            return []

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    # Monkey-patch async function
    exchange._async_fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    pair = 'ETH/BTC'
    ret = await exchange._async_get_trade_history_time(pair, since=trades_history[0][0],
                                                       until=trades_history[-1][0]-1)
    assert type(ret) is tuple
    assert ret[0] == pair
    assert type(ret[1]) is list
    assert len(ret[1]) == len(trades_history) - 1
    assert exchange._async_fetch_trades.call_count == 2
    fetch_trades_cal = exchange._async_fetch_trades.call_args_list
    # first call (using since, not fromId)
    assert fetch_trades_cal[0][0][0] == pair
    assert fetch_trades_cal[0][1]['since'] == trades_history[0][0]


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_historic_trades(default_conf, mocker, caplog, exchange_name, trades_history):
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)

    pair = 'ETH/BTC'

    exchange._async_get_trade_history_id = get_mock_coro((pair, trades_history))
    exchange._async_get_trade_history_time = get_mock_coro((pair, trades_history))
    ret = exchange.get_historic_trades(pair, since=trades_history[0][0],
                                       until=trades_history[-1][0])

    # Depending on the exchange, one or the other method should be called
    assert sum([exchange._async_get_trade_history_id.call_count,
                exchange._async_get_trade_history_time.call_count]) == 1

    assert len(ret) == 2
    assert ret[0] == pair
    assert len(ret[1]) == len(trades_history)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_historic_trades_notsupported(default_conf, mocker, caplog, exchange_name,
                                          trades_history):
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', return_value=False)
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)

    pair = 'ETH/BTC'

    with pytest.raises(OperationalException,
                       match="This exchange does not suport downloading Trades."):
        exchange.get_historic_trades(pair, since=trades_history[0][0],
                                     until=trades_history[-1][0])


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_order_dry_run(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    assert exchange.cancel_order(order_id='123', pair='TKN/BTC') == {}
    assert exchange.cancel_stoploss_order(order_id='123', pair='TKN/BTC') == {}

    order = exchange.buy('ETH/BTC', 'limit', 5, 0.55, 'gtc')

    cancel_order = exchange.cancel_order(order_id=order['id'], pair='ETH/BTC')
    assert order['id'] == cancel_order['id']
    assert order['amount'] == cancel_order['amount']
    assert order['symbol'] == cancel_order['symbol']
    assert cancel_order['status'] == 'canceled'


@pytest.mark.parametrize("exchange_name", EXCHANGES)
@pytest.mark.parametrize("order,result", [
    ({'status': 'closed', 'filled': 10}, False),
    ({'status': 'closed', 'filled': 0.0}, True),
    ({'status': 'canceled', 'filled': 0.0}, True),
    ({'status': 'canceled', 'filled': 10.0}, False),
    ({'status': 'unknown', 'filled': 10.0}, False),
    ({'result': 'testest123'}, False),
    ])
def test_check_order_canceled_empty(mocker, default_conf, exchange_name, order, result):
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    assert exchange.check_order_canceled_empty(order) == result


@pytest.mark.parametrize("exchange_name", EXCHANGES)
@pytest.mark.parametrize("order,result", [
    ({'status': 'closed', 'amount': 10, 'fee': {}}, True),
    ({'status': 'closed', 'amount': 0.0, 'fee': {}}, True),
    ({'status': 'canceled', 'amount': 0.0, 'fee': {}}, True),
    ({'status': 'canceled', 'amount': 10.0}, False),
    ({'amount': 10.0, 'fee': {}}, False),
    ({'result': 'testest123'}, False),
    ('hello_world', False),
])
def test_is_cancel_order_result_suitable(mocker, default_conf, exchange_name, order, result):
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    assert exchange.is_cancel_order_result_suitable(order) == result


@pytest.mark.parametrize("exchange_name", EXCHANGES)
@pytest.mark.parametrize("corder,call_corder,call_forder", [
    ({'status': 'closed', 'amount': 10, 'fee': {}}, 1, 0),
    ({'amount': 10, 'fee': {}}, 1, 1),
])
def test_cancel_order_with_result(default_conf, mocker, exchange_name, corder,
                                  call_corder, call_forder):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value=corder)
    api_mock.fetch_order = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    res = exchange.cancel_order_with_result('1234', 'ETH/BTC', 1234)
    assert isinstance(res, dict)
    assert api_mock.cancel_order.call_count == call_corder
    assert api_mock.fetch_order.call_count == call_forder


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_order_with_result_error(default_conf, mocker, exchange_name, caplog):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("Did not find order"))
    api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder("Did not find order"))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    res = exchange.cancel_order_with_result('1234', 'ETH/BTC', 1541)
    assert isinstance(res, dict)
    assert log_has("Could not cancel order 1234 for ETH/BTC.", caplog)
    assert log_has("Could not fetch cancelled order 1234.", caplog)
    assert res['amount'] == 1541


# Ensure that if not dry_run, we should call API
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_order(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value={'id': '123'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    assert exchange.cancel_order(order_id='_', pair='TKN/BTC') == {'id': '123'}

    with pytest.raises(InvalidOrderException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("Did not find order"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           "cancel_order", "cancel_order",
                           order_id='_', pair='TKN/BTC')


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_stoploss_order(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value={'id': '123'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    assert exchange.cancel_stoploss_order(order_id='_', pair='TKN/BTC') == {'id': '123'}

    with pytest.raises(InvalidOrderException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("Did not find order"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.cancel_stoploss_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           "cancel_stoploss_order", "cancel_order",
                           order_id='_', pair='TKN/BTC')


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_order(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = True
    order = MagicMock()
    order.myid = 123
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    exchange._dry_run_open_orders['X'] = order
    assert exchange.fetch_order('X', 'TKN/BTC').myid == 123

    with pytest.raises(InvalidOrderException, match=r'Tried to get an invalid dry-run-order.*'):
        exchange.fetch_order('Y', 'TKN/BTC')

    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value=456)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    assert exchange.fetch_order('X', 'TKN/BTC') == 456

    with pytest.raises(InvalidOrderException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder("Order not found"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.fetch_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == 1

    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound("Order not found"))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    with patch('freqtrade.exchange.common.time.sleep') as tm:
        with pytest.raises(InvalidOrderException):
            exchange.fetch_order(order_id='_', pair='TKN/BTC')
        # Ensure backoff is called
        assert tm.call_args_list[0][0][0] == 1
        assert tm.call_args_list[1][0][0] == 2
        if API_FETCH_ORDER_RETRY_COUNT > 2:
            assert tm.call_args_list[2][0][0] == 5
        if API_FETCH_ORDER_RETRY_COUNT > 3:
            assert tm.call_args_list[3][0][0] == 10
    assert api_mock.fetch_order.call_count == API_FETCH_ORDER_RETRY_COUNT + 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           'fetch_order', 'fetch_order', retries=API_FETCH_ORDER_RETRY_COUNT + 1,
                           order_id='_', pair='TKN/BTC')


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_stoploss_order(default_conf, mocker, exchange_name):
    # Don't test FTX here - that needs a seperate test
    if exchange_name == 'ftx':
        return
    default_conf['dry_run'] = True
    order = MagicMock()
    order.myid = 123
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    exchange._dry_run_open_orders['X'] = order
    assert exchange.fetch_stoploss_order('X', 'TKN/BTC').myid == 123

    with pytest.raises(InvalidOrderException, match=r'Tried to get an invalid dry-run-order.*'):
        exchange.fetch_stoploss_order('Y', 'TKN/BTC')

    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value=456)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    assert exchange.fetch_stoploss_order('X', 'TKN/BTC') == 456

    with pytest.raises(InvalidOrderException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder("Order not found"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
        exchange.fetch_stoploss_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           'fetch_stoploss_order', 'fetch_order',
                           retries=API_FETCH_ORDER_RETRY_COUNT + 1,
                           order_id='_', pair='TKN/BTC')


def test_fetch_order_or_stoploss_order(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, id='binance')
    fetch_order_mock = MagicMock()
    fetch_stoploss_order_mock = MagicMock()
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          fetch_order=fetch_order_mock,
                          fetch_stoploss_order=fetch_stoploss_order_mock,
                          )

    exchange.fetch_order_or_stoploss_order('1234', 'ETH/BTC', False)
    assert fetch_order_mock.call_count == 1
    assert fetch_order_mock.call_args_list[0][0][0] == '1234'
    assert fetch_order_mock.call_args_list[0][0][1] == 'ETH/BTC'
    assert fetch_stoploss_order_mock.call_count == 0

    fetch_order_mock.reset_mock()
    fetch_stoploss_order_mock.reset_mock()

    exchange.fetch_order_or_stoploss_order('1234', 'ETH/BTC', True)
    assert fetch_order_mock.call_count == 0
    assert fetch_stoploss_order_mock.call_count == 1
    assert fetch_stoploss_order_mock.call_args_list[0][0][0] == '1234'
    assert fetch_stoploss_order_mock.call_args_list[0][0][1] == 'ETH/BTC'


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_name(default_conf, mocker, exchange_name):
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)

    assert exchange.name == exchange_name.title()
    assert exchange.id == exchange_name


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_trades_for_order(default_conf, mocker, exchange_name):

    order_id = 'ABCD-ABCD'
    since = datetime(2018, 5, 5, 0, 0, 0)
    default_conf["dry_run"] = False
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', return_value=True)
    api_mock = MagicMock()

    api_mock.fetch_my_trades = MagicMock(return_value=[{'id': 'TTR67E-3PFBD-76IISV',
                                                        'order': 'ABCD-ABCD',
                                                        'info': {'pair': 'XLTCZBTC',
                                                                 'time': 1519860024.4388,
                                                                 'type': 'buy',
                                                                 'ordertype': 'limit',
                                                                 'price': '20.00000',
                                                                 'cost': '38.62000',
                                                                 'fee': '0.06179',
                                                                 'vol': '5',
                                                                 'id': 'ABCD-ABCD'},
                                                        'timestamp': 1519860024438,
                                                        'datetime': '2018-02-28T23:20:24.438Z',
                                                        'symbol': 'LTC/BTC',
                                                        'type': 'limit',
                                                        'side': 'buy',
                                                        'price': 165.0,
                                                        'amount': 0.2340606,
                                                        'fee': {'cost': 0.06179, 'currency': 'BTC'}
                                                        }])
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    orders = exchange.get_trades_for_order(order_id, 'LTC/BTC', since)
    assert len(orders) == 1
    assert orders[0]['price'] == 165
    assert api_mock.fetch_my_trades.call_count == 1
    # since argument should be
    assert isinstance(api_mock.fetch_my_trades.call_args[0][1], int)
    assert api_mock.fetch_my_trades.call_args[0][0] == 'LTC/BTC'
    # Same test twice, hardcoded number and doing the same calculation
    assert api_mock.fetch_my_trades.call_args[0][1] == 1525478395000
    assert api_mock.fetch_my_trades.call_args[0][1] == int(since.replace(
        tzinfo=timezone.utc).timestamp() - 5) * 1000

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           'get_trades_for_order', 'fetch_my_trades',
                           order_id=order_id, pair='LTC/BTC', since=since)

    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=False))
    assert exchange.get_trades_for_order(order_id, 'LTC/BTC', since) == []


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_fee(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    api_mock.calculate_fee = MagicMock(return_value={
        'type': 'taker',
        'currency': 'BTC',
        'rate': 0.025,
        'cost': 0.05
    })
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)
    exchange._config.pop('fee', None)

    assert exchange.get_fee('ETH/BTC') == 0.025
    assert api_mock.calculate_fee.call_count == 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name,
                           'get_fee', 'calculate_fee', symbol="ETH/BTC")

    api_mock.calculate_fee.reset_mock()
    exchange._config['fee'] = 0.001

    assert exchange.get_fee('ETH/BTC') == 0.001
    assert api_mock.calculate_fee.call_count == 0


def test_stoploss_order_unsupported_exchange(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, id='bittrex')
    with pytest.raises(OperationalException, match=r"stoploss is not implemented .*"):
        exchange.stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={})

    with pytest.raises(OperationalException, match=r"stoploss is not implemented .*"):
        exchange.stoploss_adjust(1, {})


def test_merge_ft_has_dict(default_conf, mocker):
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          _init_ccxt=MagicMock(return_value=MagicMock()),
                          _load_async_markets=MagicMock(),
                          validate_pairs=MagicMock(),
                          validate_timeframes=MagicMock(),
                          validate_stakecurrency=MagicMock()
                          )
    ex = Exchange(default_conf)
    assert ex._ft_has == Exchange._ft_has_default

    ex = Kraken(default_conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert ex._ft_has['trades_pagination'] == 'id'
    assert ex._ft_has['trades_pagination_arg'] == 'since'

    # Binance defines different values
    ex = Binance(default_conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert ex._ft_has['stoploss_on_exchange']
    assert ex._ft_has['order_time_in_force'] == ['gtc', 'fok', 'ioc']
    assert ex._ft_has['trades_pagination'] == 'id'
    assert ex._ft_has['trades_pagination_arg'] == 'fromId'

    conf = copy.deepcopy(default_conf)
    conf['exchange']['_ft_has_params'] = {"DeadBeef": 20,
                                          "stoploss_on_exchange": False}
    # Use settings from configuration (overriding stoploss_on_exchange)
    ex = Binance(conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert not ex._ft_has['stoploss_on_exchange']
    assert ex._ft_has['DeadBeef'] == 20


def test_get_valid_pair_combination(default_conf, mocker, markets):
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          _init_ccxt=MagicMock(return_value=MagicMock()),
                          _load_async_markets=MagicMock(),
                          validate_pairs=MagicMock(),
                          validate_timeframes=MagicMock(),
                          markets=PropertyMock(return_value=markets))
    ex = Exchange(default_conf)

    assert ex.get_valid_pair_combination("ETH", "BTC") == "ETH/BTC"
    assert ex.get_valid_pair_combination("BTC", "ETH") == "ETH/BTC"
    with pytest.raises(DependencyException, match=r"Could not combine.* to get a valid pair."):
        ex.get_valid_pair_combination("NOPAIR", "ETH")


@pytest.mark.parametrize(
    "base_currencies, quote_currencies, pairs_only, active_only, expected_keys", [
        # Testing markets (in conftest.py):
        # 'BLK/BTC':  'active': True
        # 'BTT/BTC':  'active': True
        # 'ETH/BTC':  'active': True
        # 'ETH/USDT': 'active': True
        # 'LTC/BTC':  'active': False
        # 'LTC/ETH':  'active': True
        # 'LTC/USD':  'active': True
        # 'LTC/USDT': 'active': True
        # 'NEO/BTC':  'active': False
        # 'TKN/BTC':  'active'  not set
        # 'XLTCUSDT': 'active': True, not a pair
        # 'XRP/BTC':  'active': False
        # all markets
        ([], [], False, False,
         ['BLK/BTC', 'BTT/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD',
          'LTC/USDT', 'NEO/BTC', 'TKN/BTC', 'XLTCUSDT', 'XRP/BTC']),
        # active markets
        ([], [], False, True,
         ['BLK/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'NEO/BTC',
          'TKN/BTC', 'XLTCUSDT', 'XRP/BTC']),
        # all pairs
        ([], [], True, False,
         ['BLK/BTC', 'BTT/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD',
          'LTC/USDT', 'NEO/BTC', 'TKN/BTC', 'XRP/BTC']),
        # active pairs
        ([], [], True, True,
         ['BLK/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'NEO/BTC',
          'TKN/BTC', 'XRP/BTC']),
        # all markets, base=ETH, LTC
        (['ETH', 'LTC'], [], False, False,
         ['ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT', 'XLTCUSDT']),
        # all markets, base=LTC
        (['LTC'], [], False, False,
         ['LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT', 'XLTCUSDT']),
        # all markets, quote=USDT
        ([], ['USDT'], False, False,
         ['ETH/USDT', 'LTC/USDT', 'XLTCUSDT']),
        # all markets, quote=USDT, USD
        ([], ['USDT', 'USD'], False, False,
         ['ETH/USDT', 'LTC/USD', 'LTC/USDT', 'XLTCUSDT']),
        # all markets, base=LTC, quote=USDT
        (['LTC'], ['USDT'], False, False,
         ['LTC/USDT', 'XLTCUSDT']),
        # all pairs, base=LTC, quote=USDT
        (['LTC'], ['USDT'], True, False,
         ['LTC/USDT']),
        # all markets, base=LTC, quote=USDT, NONEXISTENT
        (['LTC'], ['USDT', 'NONEXISTENT'], False, False,
         ['LTC/USDT', 'XLTCUSDT']),
        # all markets, base=LTC, quote=NONEXISTENT
        (['LTC'], ['NONEXISTENT'], False, False,
         []),
    ])
def test_get_markets(default_conf, mocker, markets,
                     base_currencies, quote_currencies, pairs_only, active_only,
                     expected_keys):
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          _init_ccxt=MagicMock(return_value=MagicMock()),
                          _load_async_markets=MagicMock(),
                          validate_pairs=MagicMock(),
                          validate_timeframes=MagicMock(),
                          markets=PropertyMock(return_value=markets))
    ex = Exchange(default_conf)
    pairs = ex.get_markets(base_currencies, quote_currencies, pairs_only, active_only)
    assert sorted(pairs.keys()) == sorted(expected_keys)


def test_get_markets_error(default_conf, mocker):
    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=None))
    with pytest.raises(OperationalException, match="Markets were not loaded."):
        ex.get_markets('LTC', 'USDT', True, False)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_ohlcv_candle_limit(default_conf, mocker, exchange_name):
    exchange = get_patched_exchange(mocker, default_conf, id=exchange_name)
    timeframes = ('1m', '5m', '1h')
    expected = exchange._ft_has['ohlcv_candle_limit']
    for timeframe in timeframes:
        if 'ohlcv_candle_limit_per_timeframe' in exchange._ft_has:
            expected = exchange._ft_has['ohlcv_candle_limit_per_timeframe'][timeframe]
            # This should only run for bittrex
            assert exchange_name == 'bittrex'
        assert exchange.ohlcv_candle_limit(timeframe) == expected


def test_timeframe_to_minutes():
    assert timeframe_to_minutes("5m") == 5
    assert timeframe_to_minutes("10m") == 10
    assert timeframe_to_minutes("1h") == 60
    assert timeframe_to_minutes("1d") == 1440


def test_timeframe_to_seconds():
    assert timeframe_to_seconds("5m") == 300
    assert timeframe_to_seconds("10m") == 600
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("1d") == 86400


def test_timeframe_to_msecs():
    assert timeframe_to_msecs("5m") == 300000
    assert timeframe_to_msecs("10m") == 600000
    assert timeframe_to_msecs("1h") == 3600000
    assert timeframe_to_msecs("1d") == 86400000


def test_timeframe_to_prev_date():
    # 2019-08-12 13:22:08
    date = datetime.fromtimestamp(1565616128, tz=timezone.utc)

    tf_list = [
        # 5m -> 2019-08-12 13:20:00
        ("5m", datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)),
        # 10m -> 2019-08-12 13:20:00
        ("10m", datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)),
        # 1h -> 2019-08-12 13:00:00
        ("1h", datetime(2019, 8, 12, 13, 00, 0, tzinfo=timezone.utc)),
        # 2h -> 2019-08-12 12:00:00
        ("2h", datetime(2019, 8, 12, 12, 00, 0, tzinfo=timezone.utc)),
        # 4h -> 2019-08-12 12:00:00
        ("4h", datetime(2019, 8, 12, 12, 00, 0, tzinfo=timezone.utc)),
        # 1d -> 2019-08-12 00:00:00
        ("1d", datetime(2019, 8, 12, 00, 00, 0, tzinfo=timezone.utc)),
    ]
    for interval, result in tf_list:
        assert timeframe_to_prev_date(interval, date) == result

    date = datetime.now(tz=timezone.utc)
    assert timeframe_to_prev_date("5m") < date
    # Does not round
    time = datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)
    assert timeframe_to_prev_date('5m', time) == time


def test_timeframe_to_next_date():
    # 2019-08-12 13:22:08
    date = datetime.fromtimestamp(1565616128, tz=timezone.utc)
    tf_list = [
        # 5m -> 2019-08-12 13:25:00
        ("5m", datetime(2019, 8, 12, 13, 25, 0, tzinfo=timezone.utc)),
        # 10m -> 2019-08-12 13:30:00
        ("10m", datetime(2019, 8, 12, 13, 30, 0, tzinfo=timezone.utc)),
        # 1h -> 2019-08-12 14:00:00
        ("1h", datetime(2019, 8, 12, 14, 00, 0, tzinfo=timezone.utc)),
        # 2h -> 2019-08-12 14:00:00
        ("2h", datetime(2019, 8, 12, 14, 00, 0, tzinfo=timezone.utc)),
        # 4h -> 2019-08-12 14:00:00
        ("4h", datetime(2019, 8, 12, 16, 00, 0, tzinfo=timezone.utc)),
        # 1d -> 2019-08-13 00:00:00
        ("1d", datetime(2019, 8, 13, 0, 0, 0, tzinfo=timezone.utc)),
    ]

    for interval, result in tf_list:
        assert timeframe_to_next_date(interval, date) == result

    date = datetime.now(tz=timezone.utc)
    assert timeframe_to_next_date("5m") > date

    date = datetime(2019, 8, 12, 13, 30, 0, tzinfo=timezone.utc)
    assert timeframe_to_next_date("5m", date) == date + timedelta(minutes=5)


@pytest.mark.parametrize("market_symbol,base,quote,exchange,add_dict,expected_result", [
    ("BTC/USDT", 'BTC', 'USDT', "binance", {}, True),
    ("USDT/BTC", 'USDT', 'BTC', "binance", {}, True),
    ("USDT/BTC", 'BTC', 'USDT', "binance", {}, False),  # Reversed currencies
    ("BTCUSDT", 'BTC', 'USDT', "binance", {}, False),  # No seperating /
    ("BTCUSDT", None, "USDT", "binance", {}, False),  #
    ("USDT/BTC", "BTC", None, "binance", {}, False),
    ("BTCUSDT", "BTC", None, "binance", {}, False),
    ("BTC/USDT", "BTC", "USDT", "binance", {}, True),
    ("BTC/USDT", "USDT", "BTC", "binance", {}, False),  # reversed currencies
    ("BTC/USDT", "BTC", "USD", "binance", {}, False),  # Wrong quote currency
    ("BTC/", "BTC", 'UNK', "binance", {}, False),
    ("/USDT", 'UNK', 'USDT', "binance", {}, False),
    ("BTC/EUR", 'BTC', 'EUR', "kraken", {"darkpool": False}, True),
    ("EUR/BTC", 'EUR', 'BTC', "kraken", {"darkpool": False}, True),
    ("EUR/BTC", 'BTC', 'EUR', "kraken", {"darkpool": False}, False),  # Reversed currencies
    ("BTC/EUR", 'BTC', 'USD', "kraken", {"darkpool": False}, False),  # wrong quote currency
    ("BTC/EUR", 'BTC', 'EUR', "kraken", {"darkpool": True}, False),  # no darkpools
    ("BTC/EUR.d", 'BTC', 'EUR', "kraken", {"darkpool": True}, False),  # no darkpools
    ("BTC/USD", 'BTC', 'USD', "ftx", {'spot': True}, True),
    ("USD/BTC", 'USD', 'BTC', "ftx", {'spot': True}, True),
    ("BTC/USD", 'BTC', 'USDT', "ftx", {'spot': True}, False),  # Wrong quote currency
    ("BTC/USD", 'USD', 'BTC', "ftx", {'spot': True}, False),  # Reversed currencies
    ("BTC/USD", 'BTC', 'USD', "ftx", {'spot': False}, False),  # Can only trade spot markets
    ("BTC-PERP", 'BTC', 'USD', "ftx", {'spot': False}, False),  # Can only trade spot markets
])
def test_market_is_tradable(mocker, default_conf, market_symbol, base,
                            quote, add_dict, exchange, expected_result) -> None:
    ex = get_patched_exchange(mocker, default_conf, id=exchange)
    market = {
        'symbol': market_symbol,
        'base': base,
        'quote': quote,
        **(add_dict),
    }
    assert ex.market_is_tradable(market) == expected_result


@pytest.mark.parametrize("market,expected_result", [
    ({'symbol': 'ETH/BTC', 'active': True}, True),
    ({'symbol': 'ETH/BTC', 'active': False}, False),
    ({'symbol': 'ETH/BTC', }, True),
])
def test_market_is_active(market, expected_result) -> None:
    assert market_is_active(market) == expected_result


@pytest.mark.parametrize("order,expected", [
    ([{'fee'}], False),
    ({'fee': None}, False),
    ({'fee': {'currency': 'ETH/BTC'}}, False),
    ({'fee': {'currency': 'ETH/BTC', 'cost': None}}, False),
    ({'fee': {'currency': 'ETH/BTC', 'cost': 0.01}}, True),
])
def test_order_has_fee(order, expected) -> None:
    assert Exchange.order_has_fee(order) == expected


@pytest.mark.parametrize("order,expected", [
    ({'symbol': 'ETH/BTC', 'fee': {'currency': 'ETH', 'cost': 0.43}},
        (0.43, 'ETH', 0.01)),
    ({'symbol': 'ETH/USDT', 'fee': {'currency': 'USDT', 'cost': 0.01}},
        (0.01, 'USDT', 0.01)),
    ({'symbol': 'BTC/USDT', 'fee': {'currency': 'USDT', 'cost': 0.34, 'rate': 0.01}},
        (0.34, 'USDT', 0.01)),
])
def test_extract_cost_curr_rate(mocker, default_conf, order, expected) -> None:
    mocker.patch('freqtrade.exchange.Exchange.calculate_fee_rate', MagicMock(return_value=0.01))
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.extract_cost_curr_rate(order) == expected


@pytest.mark.parametrize("order,expected", [
    # Using base-currency
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05,
        'fee': {'currency': 'ETH', 'cost': 0.004, 'rate': None}}, 0.1),
    ({'symbol': 'ETH/BTC', 'amount': 0.05, 'cost': 0.05,
        'fee': {'currency': 'ETH', 'cost': 0.004, 'rate': None}}, 0.08),
    # Using quote currency
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05,
        'fee': {'currency': 'BTC', 'cost': 0.005}}, 0.1),
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05,
        'fee': {'currency': 'BTC', 'cost': 0.002, 'rate': None}}, 0.04),
    # Using foreign currency
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05,
        'fee': {'currency': 'NEO', 'cost': 0.0012}}, 0.001944),
    ({'symbol': 'ETH/BTC', 'amount': 2.21, 'cost': 0.02992561,
        'fee': {'currency': 'NEO', 'cost': 0.00027452}}, 0.00074305),
    # Rate included in return - return as is
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05,
        'fee': {'currency': 'USDT', 'cost': 0.34, 'rate': 0.01}}, 0.01),
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05,
        'fee': {'currency': 'USDT', 'cost': 0.34, 'rate': 0.005}}, 0.005),
    # 0.1% filled - no costs (kraken - #3431)
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.0,
      'fee': {'currency': 'BTC', 'cost': 0.0, 'rate': None}}, None),
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.0,
      'fee': {'currency': 'ETH', 'cost': 0.0, 'rate': None}}, 0.0),
    ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.0,
      'fee': {'currency': 'NEO', 'cost': 0.0, 'rate': None}}, None),
])
def test_calculate_fee_rate(mocker, default_conf, order, expected) -> None:
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', return_value={'last': 0.081})

    ex = get_patched_exchange(mocker, default_conf)
    assert ex.calculate_fee_rate(order) == expected


@pytest.mark.parametrize('retrycount,max_retries,expected', [
    (0, 3, 10),
    (1, 3, 5),
    (2, 3, 2),
    (3, 3, 1),
    (0, 1, 2),
    (1, 1, 1),
    (0, 4, 17),
    (1, 4, 10),
    (2, 4, 5),
    (3, 4, 2),
    (4, 4, 1),
    (0, 5, 26),
    (1, 5, 17),
    (2, 5, 10),
    (3, 5, 5),
    (4, 5, 2),
    (5, 5, 1),

])
def test_calculate_backoff(retrycount, max_retries, expected):
    assert calculate_backoff(retrycount, max_retries) == expected
