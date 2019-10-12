import copy
import logging
from unittest.mock import MagicMock, PropertyMock

import ccxt
import pytest

from freqtrade import OperationalException
from freqtrade.exchange import BaseExchange
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from tests.conftest import get_patched_baseexchange, log_has

# Make sure to always keep one exchange here which is NOT subclassed!!
EXCHANGES = ['bittrex', 'binance', 'kraken', ]


def test_init(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    get_patched_baseexchange(mocker, default_conf)
    assert log_has('Using Exchange "Bittrex"', caplog)


def test_init_ccxt_kwargs(default_conf, mocker, caplog):
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    caplog.set_level(logging.INFO)
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_async_config'] = {'aiohttp_trust_env': True}
    ex = BaseExchange(conf)
    assert log_has("Applying additional ccxt config: {'aiohttp_trust_env': True}", caplog)
    assert ex._api_async.aiohttp_trust_env
    assert not ex._api.aiohttp_trust_env

    # Reset logging and config
    caplog.clear()
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_config'] = {'TestKWARG': 11}
    ex = BaseExchange(conf)
    assert not log_has("Applying additional ccxt config: {'aiohttp_trust_env': True}", caplog)
    assert not ex._api_async.aiohttp_trust_env
    assert hasattr(ex._api, 'TestKWARG')
    assert ex._api.TestKWARG == 11
    assert not hasattr(ex._api_async, 'TestKWARG')
    assert log_has("Applying additional ccxt config: {'TestKWARG': 11}", caplog)


def test_destroy(default_conf, mocker, caplog):
    caplog.set_level(logging.DEBUG)
    get_patched_baseexchange(mocker, default_conf)
    assert log_has('Exchange object destroyed, closing async loop', caplog)


def test_init_exception(default_conf, mocker):
    default_conf['exchange']['name'] = 'wrong_exchange_name'

    with pytest.raises(OperationalException,
                       match=f"Exchange {default_conf['exchange']['name']} is not supported"):
        BaseExchange(default_conf)

    default_conf['exchange']['name'] = 'binance'
    with pytest.raises(OperationalException,
                       match=f"Exchange {default_conf['exchange']['name']} is not supported"):
        mocker.patch("ccxt.binance", MagicMock(side_effect=AttributeError))
        BaseExchange(default_conf)

    with pytest.raises(OperationalException,
                       match=r"Initialization of ccxt failed. Reason: DeadBeef"):
        mocker.patch("ccxt.binance", MagicMock(side_effect=ccxt.BaseError("DeadBeef")))
        BaseExchange(default_conf)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_exchange_resolver(default_conf, mocker, exchange_name):
    mocker.patch.multiple('freqtrade.exchange.BaseExchange',
                          _init_ccxt=MagicMock(return_value=MagicMock()),
                          validate_timeframes=MagicMock())
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          _load_async_markets=MagicMock(),
                          validate_timeframe=MagicMock(),
                          validate_pairs=MagicMock())
    exchange = ExchangeResolver(exchange_name, default_conf, base=True).base_exchange
    assert isinstance(exchange, BaseExchange)


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
    exchange = get_patched_baseexchange(mocker, default_conf, api_mock)
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
        exchange = get_patched_baseexchange(mocker, default_conf, api_mock)
        default_conf['exchange']['sandbox'] = True
        exchange.set_sandbox(exchange._api, default_conf['exchange'], 'Logname')


def test_validate_timeframes(default_conf, mocker):
    default_conf["ticker_interval"] = "5m"
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={'1m': '1m',
                                            '5m': '5m',
                                            '15m': '15m',
                                            '1h': '1h'})
    type(api_mock).timeframes = timeframes

    mocker.patch.multiple('freqtrade.exchange.BaseExchange',
                          _init_ccxt=MagicMock(return_value=api_mock))
    BaseExchange(default_conf)


def test_validate_timeframes_failed(default_conf, mocker):
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock

    # delete timeframes so magicmock does not autocreate it
    del api_mock.timeframes

    mocker.patch.multiple('freqtrade.exchange.BaseExchange',
                          _init_ccxt=MagicMock(return_value=api_mock))
    with pytest.raises(
        OperationalException,
        match=r"The ccxt library does not provide the list of timeframes for the exchange.*"
    ):
        BaseExchange(default_conf)


def test_exchange_has(default_conf, mocker):
    exchange = get_patched_baseexchange(mocker, default_conf)
    assert not exchange.exchange_has('ASDFASDF')
    api_mock = MagicMock()

    type(api_mock).has = PropertyMock(return_value={'deadbeef': True})
    exchange = get_patched_baseexchange(mocker, default_conf, api_mock)
    assert exchange.exchange_has("deadbeef")

    type(api_mock).has = PropertyMock(return_value={'deadbeef': False})
    exchange = get_patched_baseexchange(mocker, default_conf, api_mock)
    assert not exchange.exchange_has("deadbeef")


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_name(default_conf, mocker, exchange_name):
    exchange = get_patched_baseexchange(mocker, default_conf, id=exchange_name)

    assert exchange.name == exchange_name.title()
    assert exchange.id == exchange_name
