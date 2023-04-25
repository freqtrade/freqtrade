from random import randint
from unittest.mock import MagicMock

import ccxt
import pytest

from freqtrade.exceptions import DependencyException, InvalidOrderException
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


STOPLOSS_ORDERTYPE = 'stop-loss'
STOPLOSS_LIMIT_ORDERTYPE = 'stop-loss-limit'


def test_buy_kraken_trading_agreement(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))
    order_type = 'limit'
    time_in_force = 'ioc'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'symbol': 'ETH/BTC',
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False

    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="kraken")

    order = exchange.create_order(
        pair='ETH/BTC',
        ordertype=order_type,
        side="buy",
        amount=1,
        rate=200,
        leverage=1.0,
        time_in_force=time_in_force
    )

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200
    assert api_mock.create_order.call_args[0][5] == {'timeInForce': 'IOC',
                                                     'trading_agreement': 'agree'}


def test_sell_kraken_trading_agreement(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_sell_{}'.format(randint(0, 10 ** 6))
    order_type = 'market'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'symbol': 'ETH/BTC',
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False

    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="kraken")

    order = exchange.create_order(pair='ETH/BTC', ordertype=order_type,
                                  side="sell", amount=1, rate=200, leverage=1.0)

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
        'EUR': balance_item.copy(),
        'timestamp': 123123
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
                           'symbol': '3ST/EUR',
                           'type': 'limit',
                           'side': 'buy',
                           'price': 0.02,
                           'cost': 0.0,
                           'amount': 100.0,
                           'filled': 0.0,
                           'average': 0.0,
                           'remaining': 100.0,
                           }]
    api_mock.fetch_open_orders = MagicMock(return_value=kraken_open_orders)
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="kraken")
    balances = exchange.get_balances()
    assert len(balances) == 6

    assert balances['1ST']['free'] == 9.0
    assert balances['1ST']['total'] == 10.0
    assert balances['1ST']['used'] == 1.0

    assert balances['2ST']['free'] == 6.0
    assert balances['2ST']['total'] == 10.0
    assert balances['2ST']['used'] == 4.0

    assert balances['3ST']['free'] == 10.0
    assert balances['3ST']['total'] == 10.0
    assert balances['3ST']['used'] == 0.0

    assert balances['4ST']['free'] == 10.0
    assert balances['4ST']['total'] == 10.0
    assert balances['4ST']['used'] == 0.0

    assert balances['EUR']['free'] == 8.0
    assert balances['EUR']['total'] == 10.0
    assert balances['EUR']['used'] == 2.0
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, "kraken",
                           "get_balances", "fetch_balance")


@pytest.mark.parametrize('ordertype', ['market', 'limit'])
@pytest.mark.parametrize('side,adjustedprice', [
    ("sell", 217.8),
    ("buy", 222.2),
])
def test_create_stoploss_order_kraken(default_conf, mocker, ordertype, side, adjustedprice):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))

    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })

    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'kraken')

    order = exchange.create_stoploss(
        pair='ETH/BTC',
        amount=1,
        stop_price=220,
        side=side,
        order_types={
            'stoploss': ordertype,
            'stoploss_on_exchange_limit_ratio': 0.99
        },
        leverage=1.0
    )

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args_list[0][1]['symbol'] == 'ETH/BTC'
    if ordertype == 'limit':
        assert api_mock.create_order.call_args_list[0][1]['type'] == STOPLOSS_LIMIT_ORDERTYPE
        assert api_mock.create_order.call_args_list[0][1]['params'] == {
            'trading_agreement': 'agree',
            'price2': adjustedprice
        }
    else:
        assert api_mock.create_order.call_args_list[0][1]['type'] == STOPLOSS_ORDERTYPE
        assert api_mock.create_order.call_args_list[0][1]['params'] == {
            'trading_agreement': 'agree'}
    assert api_mock.create_order.call_args_list[0][1]['side'] == side
    assert api_mock.create_order.call_args_list[0][1]['amount'] == 1
    assert api_mock.create_order.call_args_list[0][1]['price'] == 220

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("0 balance"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'kraken')
        exchange.create_stoploss(
            pair='ETH/BTC',
            amount=1,
            stop_price=220,
            order_types={},
            side=side,
            leverage=1.0
        )

    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(
            side_effect=ccxt.InvalidOrder("kraken Order would trigger immediately."))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'kraken')
        exchange.create_stoploss(
            pair='ETH/BTC',
            amount=1,
            stop_price=220,
            order_types={},
            side=side,
            leverage=1.0
        )

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, "kraken",
                           "create_stoploss", "create_order", retries=1,
                           pair='ETH/BTC', amount=1, stop_price=220, order_types={},
                           side=side, leverage=1.0)


@pytest.mark.parametrize('side', ['buy', 'sell'])
def test_create_stoploss_order_dry_run_kraken(default_conf, mocker, side):
    api_mock = MagicMock()
    default_conf['dry_run'] = True
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'kraken')

    api_mock.create_order.reset_mock()

    order = exchange.create_stoploss(
        pair='ETH/BTC',
        amount=1,
        stop_price=220,
        order_types={},
        side=side,
        leverage=1.0
    )

    assert 'id' in order
    assert 'info' in order
    assert 'type' in order

    assert order['type'] == STOPLOSS_ORDERTYPE
    assert order['price'] == 220
    assert order['amount'] == 1


@pytest.mark.parametrize('sl1,sl2,sl3,side', [
    (1501, 1499, 1501, "sell"),
    (1499, 1501, 1499, "buy")
])
def test_stoploss_adjust_kraken(mocker, default_conf, sl1, sl2, sl3, side):
    exchange = get_patched_exchange(mocker, default_conf, id='kraken')
    order = {
        'type': STOPLOSS_ORDERTYPE,
        'price': 1500,
    }
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)
    # Test with invalid order case ...
    order['type'] = 'stop_loss_limit'
    assert not exchange.stoploss_adjust(sl3, order, side=side)
