# pragma pylint: disable=missing-docstring, C0103, bad-continuation, global-statement
# pragma pylint: disable=protected-access
import copy
import logging
from datetime import datetime
from random import randint
from unittest.mock import Mock, MagicMock, PropertyMock

import arrow
import ccxt
import pytest

from freqtrade import DependencyException, OperationalException, TemporaryError
from freqtrade.exchange import API_RETRY_COUNT, Exchange
from freqtrade.tests.conftest import get_patched_exchange, log_has


# Source: https://stackoverflow.com/questions/29881236/how-to-mock-asyncio-coroutines
def get_mock_coro(return_value):
    async def mock_coro(*args, **kwargs):
        return return_value

    return Mock(wraps=mock_coro)


def ccxt_exceptionhandlers(mocker, default_conf, api_mock, fun, mock_ccxt_fun, **kwargs):
    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == API_RETRY_COUNT + 1

    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1


async def async_ccxt_exception(mocker, default_conf, api_mock, fun, mock_ccxt_fun, **kwargs):
    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == API_RETRY_COUNT + 1

    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1


def test_init(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Instance is running with dry_run enabled', caplog.record_tuples)


def test_init_ccxt_kwargs(default_conf, mocker, caplog):
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    caplog.set_level(logging.INFO)
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_async_config'] = {'aiohttp_trust_env': True}
    ex = Exchange(conf)
    assert log_has("Applying additional ccxt config: {'aiohttp_trust_env': True}",
                   caplog.record_tuples)
    assert ex._api_async.aiohttp_trust_env
    assert not ex._api.aiohttp_trust_env

    # Reset logging and config
    caplog.clear()
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_config'] = {'TestKWARG': 11}
    ex = Exchange(conf)
    assert not log_has("Applying additional ccxt config: {'aiohttp_trust_env': True}",
                       caplog.record_tuples)
    assert not ex._api_async.aiohttp_trust_env
    assert hasattr(ex._api, 'TestKWARG')
    assert ex._api.TestKWARG == 11
    assert not hasattr(ex._api_async, 'TestKWARG')
    assert log_has("Applying additional ccxt config: {'TestKWARG': 11}",
                   caplog.record_tuples)


def test_destroy(default_conf, mocker, caplog):
    caplog.set_level(logging.DEBUG)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Exchange object destroyed, closing async loop', caplog.record_tuples)


def test_init_exception(default_conf, mocker):
    default_conf['exchange']['name'] = 'wrong_exchange_name'

    with pytest.raises(
            OperationalException,
            match='Exchange {} is not supported'.format(default_conf['exchange']['name'])):
        Exchange(default_conf)

    default_conf['exchange']['name'] = 'binance'
    with pytest.raises(
            OperationalException,
            match='Exchange {} is not supported'.format(default_conf['exchange']['name'])):
        mocker.patch("ccxt.binance", MagicMock(side_effect=AttributeError))
        Exchange(default_conf)


def test_symbol_amount_prec(default_conf, mocker):
    '''
    Test rounds down to 4 Decimal places
    '''
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'LTC/BTC': '', 'XRP/BTC': '', 'NEO/BTC': ''
    })
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value='binance'))

    markets = PropertyMock(return_value={'ETH/BTC': {'precision': {'amount': 4}}})
    type(api_mock).markets = markets

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())
    exchange = Exchange(default_conf)

    amount = 2.34559
    pair = 'ETH/BTC'
    amount = exchange.symbol_amount_prec(pair, amount)
    assert amount == 2.3455


def test_symbol_price_prec(default_conf, mocker):
    '''
    Test rounds up to 4 decimal places
    '''
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'LTC/BTC': '', 'XRP/BTC': '', 'NEO/BTC': ''
    })
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value='binance'))

    markets = PropertyMock(return_value={'ETH/BTC': {'precision': {'price': 4}}})
    type(api_mock).markets = markets

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())
    exchange = Exchange(default_conf)

    price = 2.34559
    pair = 'ETH/BTC'
    price = exchange.symbol_price_prec(pair, price)
    assert price == 2.3456


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
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())

    exchange = Exchange(default_conf)
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

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())

    with pytest.raises(OperationalException, match=r'does not provide a sandbox api'):
        exchange = Exchange(default_conf)
        default_conf['exchange']['sandbox'] = True
        exchange.set_sandbox(exchange._api, default_conf['exchange'], 'Logname')


def test__load_async_markets(default_conf, mocker, caplog):
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._api_async.load_markets = get_mock_coro(None)
    exchange._load_async_markets()
    assert exchange._api_async.load_markets.call_count == 1
    caplog.set_level(logging.DEBUG)

    exchange._api_async.load_markets = Mock(side_effect=ccxt.BaseError("deadbeef"))
    exchange._load_async_markets()

    assert log_has('Could not load async markets. Reason: deadbeef',
                   caplog.record_tuples)


def test__load_markets(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value='Binance'))

    api_mock.load_markets = MagicMock(return_value={})
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', api_mock)
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())

    expected_return = {'ETH/BTC': 'available'}
    api_mock.load_markets = MagicMock(return_value=expected_return)
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    default_conf['exchange']['pair_whitelist'] = ['ETH/BTC']
    ex = Exchange(default_conf)
    assert ex.markets == expected_return

    api_mock.load_markets = MagicMock(side_effect=ccxt.BaseError())
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    Exchange(default_conf)
    assert log_has('Unable to initialize markets. Reason: ', caplog.record_tuples)


def test_validate_pairs(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'LTC/BTC': '', 'XRP/BTC': '', 'NEO/BTC': ''
    })
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())
    Exchange(default_conf)


def test_validate_pairs_not_available(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={'XRP/BTC': 'inactive'})
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())

    with pytest.raises(OperationalException, match=r'not available'):
        Exchange(default_conf)


def test_validate_pairs_not_compatible(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'TKN/BTC': '', 'TRST/BTC': '', 'SWT/BTC': '', 'BCC/BTC': ''
    })
    default_conf['stake_currency'] = 'ETH'
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())
    with pytest.raises(OperationalException, match=r'not compatible'):
        Exchange(default_conf)


def test_validate_pairs_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value='Binance'))

    api_mock.load_markets = MagicMock(return_value={})
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', api_mock)
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())

    with pytest.raises(OperationalException, match=r'Pair ETH/BTC is not available at Binance'):
        Exchange(default_conf)

    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    Exchange(default_conf)
    assert log_has('Unable to validate pairs (assuming they are correct).',
                   caplog.record_tuples)


def test_validate_pairs_stake_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    default_conf['stake_currency'] = 'ETH'
    api_mock = MagicMock()
    api_mock.name = MagicMock(return_value='binance')
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', api_mock)
    mocker.patch('freqtrade.exchange.Exchange.validate_timeframes', MagicMock())
    mocker.patch('freqtrade.exchange.Exchange._load_async_markets', MagicMock())

    with pytest.raises(
        OperationalException,
        match=r'Pair ETH/BTC not compatible with stake_currency: ETH'
    ):
        Exchange(default_conf)


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

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    Exchange(default_conf)


def test_validate_timeframes_failed(default_conf, mocker):
    default_conf["ticker_interval"] = "3m"
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
    with pytest.raises(OperationalException, match=r'Invalid ticker 3m, this Exchange supports.*'):
        Exchange(default_conf)


def test_validate_timeframes_not_in_config(default_conf, mocker):
    del default_conf["ticker_interval"]
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


def test_buy_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf)

    order = exchange.buy(pair='ETH/BTC', ordertype='limit', amount=1, rate=200)
    assert 'id' in order
    assert 'dry_run_buy_' in order['id']


def test_buy_prod(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))
    order_type = 'market'
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    order = exchange.buy(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][1] == order_type
    order_type = 'limit'
    api_mock.create_order.reset_mock()
    order = exchange.buy(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)
    assert api_mock.create_order.call_args[0][1] == order_type

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.buy(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.buy(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.buy(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.buy(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)


def test_sell_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf)

    order = exchange.sell(pair='ETH/BTC', ordertype='limit', amount=1, rate=200)
    assert 'id' in order
    assert 'dry_run_sell_' in order['id']


def test_sell_prod(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_sell_{}'.format(randint(0, 10 ** 6))
    order_type = 'market'
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    order = exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][1] == order_type

    order_type = 'limit'
    api_mock.create_order.reset_mock()
    order = exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)
    assert api_mock.create_order.call_args[0][1] == order_type

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)


def test_get_balance_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True

    exchange = get_patched_exchange(mocker, default_conf)
    assert exchange.get_balance(currency='BTC') == 999.9


def test_get_balance_prod(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={'BTC': {'free': 123.4}})
    default_conf['dry_run'] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    assert exchange.get_balance(currency='BTC') == 123.4

    with pytest.raises(OperationalException):
        api_mock.fetch_balance = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)

        exchange.get_balance(currency='BTC')

    with pytest.raises(TemporaryError, match=r'.*balance due to malformed exchange response:.*'):
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        mocker.patch('freqtrade.exchange.Exchange.get_balances', MagicMock(return_value={}))
        exchange.get_balance(currency='BTC')


def test_get_balances_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf)
    assert exchange.get_balances() == {}


def test_get_balances_prod(default_conf, mocker):
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
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert len(exchange.get_balances()) == 3
    assert exchange.get_balances()['1ST']['free'] == 10.0
    assert exchange.get_balances()['1ST']['total'] == 10.0
    assert exchange.get_balances()['1ST']['used'] == 0.0

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           "get_balances", "fetch_balance")


def test_get_tickers(default_conf, mocker):
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
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    # retrieve original ticker
    tickers = exchange.get_tickers()

    assert 'ETH/BTC' in tickers
    assert 'BCH/BTC' in tickers
    assert tickers['ETH/BTC']['bid'] == 0.5
    assert tickers['ETH/BTC']['ask'] == 1
    assert tickers['BCH/BTC']['bid'] == 0.6
    assert tickers['BCH/BTC']['ask'] == 0.5

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           "get_tickers", "fetch_tickers")

    with pytest.raises(OperationalException):
        api_mock.fetch_tickers = MagicMock(side_effect=ccxt.NotSupported)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_tickers()

    api_mock.fetch_tickers = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange.get_tickers()


def test_get_ticker(default_conf, mocker):
    api_mock = MagicMock()
    tick = {
        'symbol': 'ETH/BTC',
        'bid': 0.00001098,
        'ask': 0.00001099,
        'last': 0.0001,
    }
    api_mock.fetch_ticker = MagicMock(return_value=tick)
    api_mock.markets = {'ETH/BTC': {}}
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    # retrieve original ticker
    ticker = exchange.get_ticker(pair='ETH/BTC')

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
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    # if not caching the result we should get the same ticker
    # if not fetching a new result we should get the cached ticker
    ticker = exchange.get_ticker(pair='ETH/BTC')

    assert api_mock.fetch_ticker.call_count == 1
    assert ticker['bid'] == 0.5
    assert ticker['ask'] == 1

    assert 'ETH/BTC' in exchange._cached_ticker
    assert exchange._cached_ticker['ETH/BTC']['bid'] == 0.5
    assert exchange._cached_ticker['ETH/BTC']['ask'] == 1

    # Test caching
    api_mock.fetch_ticker = MagicMock()
    exchange.get_ticker(pair='ETH/BTC', refresh=False)
    assert api_mock.fetch_ticker.call_count == 0

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           "get_ticker", "fetch_ticker",
                           pair='ETH/BTC', refresh=True)

    api_mock.fetch_ticker = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange.get_ticker(pair='ETH/BTC', refresh=True)

    with pytest.raises(DependencyException, match=r'Pair XRP/ETH not available'):
        exchange.get_ticker(pair='XRP/ETH', refresh=True)


def test_get_history(default_conf, mocker, caplog):
    exchange = get_patched_exchange(mocker, default_conf)
    tick = [
        [
            arrow.utcnow().timestamp * 1000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]
    pair = 'ETH/BTC'

    async def mock_candle_hist(pair, tick_interval, since_ms):
        return pair, tick

    exchange._async_get_candle_history = Mock(wraps=mock_candle_hist)
    # one_call calculation * 1.8 should do 2 calls
    since = 5 * 60 * 500 * 1.8
    print(f"since = {since}")
    ret = exchange.get_history(pair, "5m", int((arrow.utcnow().timestamp - since) * 1000))

    assert exchange._async_get_candle_history.call_count == 2
    # Returns twice the above tick
    assert len(ret) == 2


def test_refresh_tickers(mocker, default_conf, caplog) -> None:
    tick = [
        [
            1511686200000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._api_async.fetch_ohlcv = get_mock_coro(tick)

    pairs = ['IOTA/ETH', 'XRP/ETH']
    # empty dicts
    assert not exchange.klines
    exchange.refresh_tickers(['IOTA/ETH', 'XRP/ETH'], '5m')

    assert log_has(f'Refreshing klines for {len(pairs)} pairs', caplog.record_tuples)
    assert exchange.klines
    for pair in pairs:
        assert exchange.klines[pair]


@pytest.mark.asyncio
async def test__async_get_candle_history(default_conf, mocker, caplog):
    tick = [
        [
            arrow.utcnow().timestamp * 1000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    # Monkey-patch async function
    exchange._api_async.fetch_ohlcv = get_mock_coro(tick)

    exchange = Exchange(default_conf)
    pair = 'ETH/BTC'
    res = await exchange._async_get_candle_history(pair, "5m")
    assert type(res) is tuple
    assert len(res) == 2
    assert res[0] == pair
    assert res[1] == tick
    assert exchange._api_async.fetch_ohlcv.call_count == 1
    assert not log_has(f"Using cached klines data for {pair} ...", caplog.record_tuples)
    # test caching
    res = await exchange._async_get_candle_history(pair, "5m")
    assert exchange._api_async.fetch_ohlcv.call_count == 1
    assert log_has(f"Using cached klines data for {pair} ...", caplog.record_tuples)

    # exchange = Exchange(default_conf)
    await async_ccxt_exception(mocker, default_conf, MagicMock(),
                               "_async_get_candle_history", "fetch_ohlcv",
                               pair='ABCD/BTC', tick_interval=default_conf['ticker_interval'])

    api_mock = MagicMock()
    with pytest.raises(OperationalException, match=r'Could not fetch ticker data*'):
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await exchange._async_get_candle_history(pair, "5m",
                                                 (arrow.utcnow().timestamp - 2000) * 1000)


@pytest.mark.asyncio
async def test__async_get_candle_history_empty(default_conf, mocker, caplog):
    """ Test empty exchange result """
    tick = []

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    # Monkey-patch async function
    exchange._api_async.fetch_ohlcv = get_mock_coro([])

    exchange = Exchange(default_conf)
    pair = 'ETH/BTC'
    res = await exchange._async_get_candle_history(pair, "5m")
    assert type(res) is tuple
    assert len(res) == 2
    assert res[0] == pair
    assert res[1] == tick
    assert exchange._api_async.fetch_ohlcv.call_count == 1


@pytest.mark.asyncio
async def test_async_get_candles_history(default_conf, mocker):
    tick = [
        [
            1511686200000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]

    async def mock_get_candle_hist(pair, tick_interval, since_ms=None):
        return (pair, tick)

    exchange = get_patched_exchange(mocker, default_conf)
    # Monkey-patch async function
    exchange._api_async.fetch_ohlcv = get_mock_coro(tick)

    exchange._async_get_candle_history = Mock(wraps=mock_get_candle_hist)

    pairs = ['ETH/BTC', 'XRP/BTC']
    res = await exchange.async_get_candles_history(pairs, "5m")
    assert type(res) is list
    assert len(res) == 2
    assert type(res[0]) is tuple
    assert res[0][0] == pairs[0]
    assert res[0][1] == tick
    assert res[1][0] == pairs[1]
    assert res[1][1] == tick
    assert exchange._async_get_candle_history.call_count == 2


def test_get_order_book(default_conf, mocker, order_book_l2):
    default_conf['exchange']['name'] = 'binance'
    api_mock = MagicMock()

    api_mock.fetch_l2_order_book = order_book_l2
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    order_book = exchange.get_order_book(pair='ETH/BTC', limit=10)
    assert 'bids' in order_book
    assert 'asks' in order_book
    assert len(order_book['bids']) == 10
    assert len(order_book['asks']) == 10


def test_get_order_book_exception(default_conf, mocker):
    api_mock = MagicMock()
    with pytest.raises(OperationalException):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NotSupported)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_order_book(pair='ETH/BTC', limit=50)
    with pytest.raises(TemporaryError):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_order_book(pair='ETH/BTC', limit=50)
    with pytest.raises(OperationalException):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_order_book(pair='ETH/BTC', limit=50)


def make_fetch_ohlcv_mock(data):
    def fetch_ohlcv_mock(pair, timeframe, since):
        if since:
            assert since > data[-1][0]
            return []
        return data
    return fetch_ohlcv_mock


def test_get_candle_history(default_conf, mocker):
    api_mock = MagicMock()
    tick = [
        [
            1511686200000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]
    type(api_mock).has = PropertyMock(return_value={'fetchOHLCV': True})
    api_mock.fetch_ohlcv = MagicMock(side_effect=make_fetch_ohlcv_mock(tick))
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    # retrieve original ticker
    ticks = exchange.get_candle_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1511686200000
    assert ticks[0][1] == 1
    assert ticks[0][2] == 2
    assert ticks[0][3] == 3
    assert ticks[0][4] == 4
    assert ticks[0][5] == 5

    # change ticker and ensure tick changes
    new_tick = [
        [
            1511686210000,  # unix timestamp ms
            6,  # open
            7,  # high
            8,  # low
            9,  # close
            10,  # volume (in quote currency)
        ]
    ]
    api_mock.fetch_ohlcv = MagicMock(side_effect=make_fetch_ohlcv_mock(new_tick))
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    ticks = exchange.get_candle_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1511686210000
    assert ticks[0][1] == 6
    assert ticks[0][2] == 7
    assert ticks[0][3] == 8
    assert ticks[0][4] == 9
    assert ticks[0][5] == 10

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           "get_candle_history", "fetch_ohlcv",
                           pair='ABCD/BTC', tick_interval=default_conf['ticker_interval'])

    with pytest.raises(OperationalException, match=r'Exchange .* does not support.*'):
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.NotSupported)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_candle_history(pair='ABCD/BTC', tick_interval=default_conf['ticker_interval'])


def test_get_candle_history_sort(default_conf, mocker):
    api_mock = MagicMock()

    # GDAX use-case (real data from GDAX)
    # This ticker history is ordered DESC (newest first, oldest last)
    tick = [
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
    type(api_mock).has = PropertyMock(return_value={'fetchOHLCV': True})
    api_mock.fetch_ohlcv = MagicMock(side_effect=make_fetch_ohlcv_mock(tick))

    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    # Test the ticker history sort
    ticks = exchange.get_candle_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1527830400000
    assert ticks[0][1] == 0.07649
    assert ticks[0][2] == 0.07651
    assert ticks[0][3] == 0.07649
    assert ticks[0][4] == 0.07651
    assert ticks[0][5] == 2.5734867

    assert ticks[9][0] == 1527833100000
    assert ticks[9][1] == 0.07666
    assert ticks[9][2] == 0.07671
    assert ticks[9][3] == 0.07666
    assert ticks[9][4] == 0.07668
    assert ticks[9][5] == 16.65244264

    # Bittrex use-case (real data from Bittrex)
    # This ticker history is ordered ASC (oldest first, newest last)
    tick = [
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
    type(api_mock).has = PropertyMock(return_value={'fetchOHLCV': True})
    api_mock.fetch_ohlcv = MagicMock(side_effect=make_fetch_ohlcv_mock(tick))
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    # Test the ticker history sort
    ticks = exchange.get_candle_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1527827700000
    assert ticks[0][1] == 0.07659999
    assert ticks[0][2] == 0.0766
    assert ticks[0][3] == 0.07627
    assert ticks[0][4] == 0.07657998
    assert ticks[0][5] == 1.85216924

    assert ticks[9][0] == 1527830400000
    assert ticks[9][1] == 0.07671
    assert ticks[9][2] == 0.07674399
    assert ticks[9][3] == 0.07629216
    assert ticks[9][4] == 0.07655213
    assert ticks[9][5] == 2.31452783


def test_cancel_order_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf)
    assert exchange.cancel_order(order_id='123', pair='TKN/BTC') is None


# Ensure that if not dry_run, we should call API
def test_cancel_order(default_conf, mocker):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value=123)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert exchange.cancel_order(order_id='_', pair='TKN/BTC') == 123

    with pytest.raises(DependencyException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == API_RETRY_COUNT + 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           "cancel_order", "cancel_order",
                           order_id='_', pair='TKN/BTC')


def test_get_order(default_conf, mocker):
    default_conf['dry_run'] = True
    order = MagicMock()
    order.myid = 123
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._dry_run_open_orders['X'] = order
    print(exchange.get_order('X', 'TKN/BTC'))
    assert exchange.get_order('X', 'TKN/BTC').myid == 123

    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value=456)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert exchange.get_order('X', 'TKN/BTC') == 456

    with pytest.raises(DependencyException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == API_RETRY_COUNT + 1

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           'get_order', 'fetch_order',
                           order_id='_', pair='TKN/BTC')


def test_name(default_conf, mocker):
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    default_conf['exchange']['name'] = 'binance'
    exchange = Exchange(default_conf)

    assert exchange.name == 'Binance'


def test_id(default_conf, mocker):
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    default_conf['exchange']['name'] = 'binance'
    exchange = Exchange(default_conf)
    assert exchange.id == 'binance'


def test_get_pair_detail_url(default_conf, mocker, caplog):
    mocker.patch('freqtrade.exchange.Exchange._load_markets', MagicMock(return_value={}))
    default_conf['exchange']['name'] = 'binance'
    exchange = Exchange(default_conf)

    url = exchange.get_pair_detail_url('TKN/ETH')
    assert 'TKN' in url
    assert 'ETH' in url

    url = exchange.get_pair_detail_url('LOOONG/BTC')
    assert 'LOOONG' in url
    assert 'BTC' in url

    default_conf['exchange']['name'] = 'bittrex'
    exchange = Exchange(default_conf)

    url = exchange.get_pair_detail_url('TKN/ETH')
    assert 'TKN' in url
    assert 'ETH' in url

    url = exchange.get_pair_detail_url('LOOONG/BTC')
    assert 'LOOONG' in url
    assert 'BTC' in url

    default_conf['exchange']['name'] = 'poloniex'
    exchange = Exchange(default_conf)
    url = exchange.get_pair_detail_url('LOOONG/BTC')
    assert '' == url
    assert log_has('Could not get exchange url for Poloniex', caplog.record_tuples)


def test_get_trades_for_order(default_conf, mocker):
    order_id = 'ABCD-ABCD'
    since = datetime(2018, 5, 5)
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
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    orders = exchange.get_trades_for_order(order_id, 'LTC/BTC', since)
    assert len(orders) == 1
    assert orders[0]['price'] == 165

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           'get_trades_for_order', 'fetch_my_trades',
                           order_id=order_id, pair='LTC/BTC', since=since)

    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=False))
    assert exchange.get_trades_for_order(order_id, 'LTC/BTC', since) == []


def test_get_markets(default_conf, mocker, markets):
    api_mock = MagicMock()
    api_mock.fetch_markets = markets
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    ret = exchange.get_markets()
    assert isinstance(ret, list)
    assert len(ret) == 6

    assert ret[0]["id"] == "ethbtc"
    assert ret[0]["symbol"] == "ETH/BTC"

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           'get_markets', 'fetch_markets')


def test_get_fee(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.calculate_fee = MagicMock(return_value={
        'type': 'taker',
        'currency': 'BTC',
        'rate': 0.025,
        'cost': 0.05
    })
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    assert exchange.get_fee() == 0.025

    ccxt_exceptionhandlers(mocker, default_conf, api_mock,
                           'get_fee', 'calculate_fee')
