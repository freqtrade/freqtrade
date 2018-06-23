# pragma pylint: disable=missing-docstring, C0103, bad-continuation, global-statement
# pragma pylint: disable=protected-access
import logging
from copy import deepcopy
from random import randint
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock

import ccxt
import pytest

from freqtrade import OperationalException, DependencyException, TemporaryError
from freqtrade.exchange import Exchange, API_RETRY_COUNT
from freqtrade.tests.conftest import log_has, get_patched_exchange


def test_init(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Instance is running with dry_run enabled', caplog.record_tuples)


def test_init_exception(default_conf):
    default_conf['exchange']['name'] = 'wrong_exchange_name'

    with pytest.raises(
            OperationalException,
            match='Exchange {} is not supported'.format(default_conf['exchange']['name'])):
        Exchange(default_conf)


def test_validate_pairs(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'LTC/BTC': '', 'XRP/BTC': '', 'NEO/BTC': ''
    })
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    Exchange(default_conf)


def test_validate_pairs_not_available(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={})
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))

    with pytest.raises(OperationalException, match=r'not available'):
        Exchange(default_conf)


def test_validate_pairs_not_compatible(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'TKN/BTC': '', 'TRST/BTC': '', 'SWT/BTC': '', 'BCC/BTC': ''
    })
    conf = deepcopy(default_conf)
    conf['stake_currency'] = 'ETH'
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))

    with pytest.raises(OperationalException, match=r'not compatible'):
        Exchange(conf)


def test_validate_pairs_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.name', PropertyMock(return_value='Binance'))

    api_mock.load_markets = MagicMock(return_value={})
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', api_mock)

    with pytest.raises(OperationalException, match=r'Pair ETH/BTC is not available at Binance'):
        Exchange(default_conf)

    api_mock.load_markets = MagicMock(side_effect=ccxt.BaseError())

    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', MagicMock(return_value=api_mock))
    Exchange(default_conf)
    assert log_has('Unable to validate pairs (assuming they are correct). Reason: ',
                   caplog.record_tuples)


def test_validate_pairs_stake_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    conf = deepcopy(default_conf)
    conf['stake_currency'] = 'ETH'
    api_mock = MagicMock()
    api_mock.name = MagicMock(return_value='binance')
    mocker.patch('freqtrade.exchange.Exchange._init_ccxt', api_mock)

    with pytest.raises(
        OperationalException,
        match=r'Pair ETH/BTC not compatible with stake_currency: ETH'
    ):
        Exchange(conf)


def test_buy_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf)

    order = exchange.buy(pair='ETH/BTC', rate=200, amount=1)
    assert 'id' in order
    assert 'dry_run_buy_' in order['id']


def test_buy_prod(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))
    api_mock.create_limit_buy_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    order = exchange.buy(pair='ETH/BTC', rate=200, amount=1)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_limit_buy_order = MagicMock(side_effect=ccxt.InsufficientFunds)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.buy(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(DependencyException):
        api_mock.create_limit_buy_order = MagicMock(side_effect=ccxt.InvalidOrder)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.buy(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(TemporaryError):
        api_mock.create_limit_buy_order = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.buy(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(OperationalException):
        api_mock.create_limit_buy_order = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.buy(pair='ETH/BTC', rate=200, amount=1)


def test_sell_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf)

    order = exchange.sell(pair='ETH/BTC', rate=200, amount=1)
    assert 'id' in order
    assert 'dry_run_sell_' in order['id']


def test_sell_prod(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_sell_{}'.format(randint(0, 10 ** 6))
    api_mock.create_limit_sell_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    order = exchange.sell(pair='ETH/BTC', rate=200, amount=1)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_limit_sell_order = MagicMock(side_effect=ccxt.InsufficientFunds)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.sell(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(DependencyException):
        api_mock.create_limit_sell_order = MagicMock(side_effect=ccxt.InvalidOrder)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.sell(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(TemporaryError):
        api_mock.create_limit_sell_order = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.sell(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(OperationalException):
        api_mock.create_limit_sell_order = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.sell(pair='ETH/BTC', rate=200, amount=1)


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

    with pytest.raises(TemporaryError):
        api_mock.fetch_balance = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_balances()
    assert api_mock.fetch_balance.call_count == API_RETRY_COUNT + 1

    with pytest.raises(OperationalException):
        api_mock.fetch_balance = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_balances()
    assert api_mock.fetch_balance.call_count == 1


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

    with pytest.raises(TemporaryError):  # test retrier
        api_mock.fetch_tickers = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_tickers()

    with pytest.raises(OperationalException):
        api_mock.fetch_tickers = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_tickers()

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

    with pytest.raises(TemporaryError):  # test retrier
        api_mock.fetch_ticker = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_ticker(pair='ETH/BTC', refresh=True)

    with pytest.raises(OperationalException):
        api_mock.fetch_ticker = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_ticker(pair='ETH/BTC', refresh=True)

    api_mock.fetch_ticker = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange.get_ticker(pair='ETH/BTC', refresh=True)


def make_fetch_ohlcv_mock(data):
    def fetch_ohlcv_mock(pair, timeframe, since):
        if since:
            assert since > data[-1][0]
            return []
        return data
    return fetch_ohlcv_mock


def test_get_ticker_history(default_conf, mocker):
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
    ticks = exchange.get_ticker_history('ETH/BTC', default_conf['ticker_interval'])
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

    ticks = exchange.get_ticker_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1511686210000
    assert ticks[0][1] == 6
    assert ticks[0][2] == 7
    assert ticks[0][3] == 8
    assert ticks[0][4] == 9
    assert ticks[0][5] == 10

    with pytest.raises(TemporaryError):  # test retrier
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        # new symbol to get around cache
        exchange.get_ticker_history('ABCD/BTC', default_conf['ticker_interval'])

    with pytest.raises(OperationalException):
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        # new symbol to get around cache
        exchange.get_ticker_history('EFGH/BTC', default_conf['ticker_interval'])


def test_get_ticker_history_sort(default_conf, mocker):
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
    ticks = exchange.get_ticker_history('ETH/BTC', default_conf['ticker_interval'])
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
    ticks = exchange.get_ticker_history('ETH/BTC', default_conf['ticker_interval'])
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

    with pytest.raises(TemporaryError):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.NetworkError)

        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == API_RETRY_COUNT + 1

    with pytest.raises(DependencyException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == API_RETRY_COUNT + 1

    with pytest.raises(OperationalException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == 1


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

    with pytest.raises(TemporaryError):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == API_RETRY_COUNT + 1

    with pytest.raises(DependencyException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == API_RETRY_COUNT + 1

    with pytest.raises(OperationalException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == 1


def test_name(default_conf, mocker):
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs',
                 side_effect=lambda s: True)
    default_conf['exchange']['name'] = 'binance'
    exchange = Exchange(default_conf)

    assert exchange.name == 'Binance'


def test_id(default_conf, mocker):
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs',
                 side_effect=lambda s: True)
    default_conf['exchange']['name'] = 'binance'
    exchange = Exchange(default_conf)
    assert exchange.id == 'binance'


def test_get_pair_detail_url(default_conf, mocker, caplog):
    mocker.patch('freqtrade.exchange.Exchange.validate_pairs',
                 side_effect=lambda s: True)
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

    # test Exceptions
    with pytest.raises(OperationalException):
        api_mock = MagicMock()
        api_mock.fetch_my_trades = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_trades_for_order(order_id, 'LTC/BTC', since)

    with pytest.raises(TemporaryError):
        api_mock = MagicMock()
        api_mock.fetch_my_trades = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_trades_for_order(order_id, 'LTC/BTC', since)
    assert api_mock.fetch_my_trades.call_count == API_RETRY_COUNT + 1


def test_get_markets(default_conf, mocker, markets):
    api_mock = MagicMock()
    api_mock.fetch_markets = markets
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    ret = exchange.get_markets()
    assert isinstance(ret, list)
    assert len(ret) == 3

    assert ret[0]["id"] == "ethbtc"
    assert ret[0]["symbol"] == "ETH/BTC"

    # test Exceptions
    with pytest.raises(OperationalException):
        api_mock = MagicMock()
        api_mock.fetch_markets = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_markets()

    with pytest.raises(TemporaryError):
        api_mock = MagicMock()
        api_mock.fetch_markets = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_markets()
    assert api_mock.fetch_markets.call_count == API_RETRY_COUNT + 1


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

    # test Exceptions
    with pytest.raises(OperationalException):
        api_mock = MagicMock()
        api_mock.calculate_fee = MagicMock(side_effect=ccxt.BaseError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_fee()

    with pytest.raises(TemporaryError):
        api_mock = MagicMock()
        api_mock.calculate_fee = MagicMock(side_effect=ccxt.NetworkError)
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        exchange.get_fee()
    assert api_mock.calculate_fee.call_count == API_RETRY_COUNT + 1


def test_get_amount_lots(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.amount_to_lots = MagicMock(return_value=1.0)
    api_mock.markets = None
    marketmock = MagicMock()
    api_mock.load_markets = marketmock
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    assert exchange.get_amount_lots('LTC/BTC', 1.54) == 1
    assert marketmock.call_count == 1
