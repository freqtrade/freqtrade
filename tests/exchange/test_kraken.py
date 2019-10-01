# pragma pylint: disable=missing-docstring, C0103, bad-continuation, global-statement
# pragma pylint: disable=protected-access
from random import randint
from unittest.mock import MagicMock

from tests.conftest import get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


def test_buy_kraken_trading_agreement(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))
    order_type = 'limit'
    time_in_force = 'ioc'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False

    mocker.patch('freqtrade.exchange.Exchange.symbol_amount_prec', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.symbol_price_prec', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="kraken")

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
    assert api_mock.create_order.call_args[0][5] == {'timeInForce': 'ioc',
                                                     'trading_agreement': 'agree'}


def test_sell_kraken_trading_agreement(default_conf, mocker):
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

    mocker.patch('freqtrade.exchange.Exchange.symbol_amount_prec', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.symbol_price_prec', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="kraken")

    order = exchange.sell(pair='ETH/BTC', ordertype=order_type, amount=1, rate=200)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'sell'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] is None
    assert api_mock.create_order.call_args[0][5] == {'trading_agreement': 'agree'}


def test_get_balances_prod(default_conf, mocker):
    balance_item = {
        'free': None,
        'total': 10.0,
        'used': 0.0
    }

    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={
        '1ST': balance_item.copy(),
        '2ST': balance_item.copy(),
        '3ST': balance_item.copy(),
        '4ST': balance_item.copy(),
    })
    kraken_open_orders = [{'symbol': '1ST/EUR',
                           'type': 'limit',
                           'side': 'sell',
                           'price': 20,
                           'cost': 0.0,
                           'amount': 1.0,
                           'filled': 0.0,
                           'average': 0.0,
                           'remaining': 1.0,
                           },
                          {'status': 'open',
                           'symbol': '2ST/EUR',
                           'type': 'limit',
                           'side': 'sell',
                           'price': 20.0,
                           'cost': 0.0,
                           'amount': 2.0,
                           'filled': 0.0,
                           'average': 0.0,
                           'remaining': 2.0,
                           },
                          {'status': 'open',
                           'symbol': '2ST/USD',
                           'type': 'limit',
                           'side': 'sell',
                           'price': 20.0,
                           'cost': 0.0,
                           'amount': 2.0,
                           'filled': 0.0,
                           'average': 0.0,
                           'remaining': 2.0,
                           },
                          {'status': 'open',
                           'symbol': 'BTC/3ST',
                           'type': 'limit',
                           'side': 'buy',
                           'price': 20,
                           'cost': 0.0,
                           'amount': 3.0,
                           'filled': 0.0,
                           'average': 0.0,
                           'remaining': 3.0,
                           }]
    api_mock.fetch_open_orders = MagicMock(return_value=kraken_open_orders)
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="kraken")
    balances = exchange.get_balances()
    assert len(balances) == 4
    assert balances['1ST']['free'] == 9.0
    assert balances['1ST']['total'] == 10.0
    assert balances['1ST']['used'] == 1.0

    assert balances['2ST']['free'] == 6.0
    assert balances['2ST']['total'] == 10.0
    assert balances['2ST']['used'] == 4.0

    assert balances['3ST']['free'] == 7.0
    assert balances['3ST']['total'] == 10.0
    assert balances['3ST']['used'] == 3.0

    assert balances['4ST']['free'] == 10.0
    assert balances['4ST']['total'] == 10.0
    assert balances['4ST']['used'] == 0.0
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, "kraken",
                           "get_balances", "fetch_balance")
