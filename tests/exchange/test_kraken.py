from random import randint
from unittest.mock import MagicMock

import ccxt
import pytest

from freqtrade.exceptions import DependencyException, InvalidOrderException
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


STOPLOSS_ORDERTYPE = 'stop-loss'
STOPLOSS_LIMIT_ORDERTYPE = 'stop-loss-limit'


@pytest.mark.parametrize("order_type,time_in_force,expected_params", [
    ('limit', 'ioc', {'timeInForce': 'IOC', 'trading_agreement': 'agree'}),
    ('limit', 'PO', {'postOnly': True, 'trading_agreement': 'agree'}),
    ('market', None, {'trading_agreement': 'agree'})
])
def test_kraken_trading_agreement(default_conf, mocker, order_type, time_in_force, expected_params):
    api_mock = MagicMock()
    order_id = f'test_prod_{order_type}_{randint(0, 10 ** 6)}'
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
    assert api_mock.create_order.call_args[0][4] == (200 if order_type == 'limit' else None)

    assert api_mock.create_order.call_args[0][5] == expected_params


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
    order_id = f'test_prod_buy_{randint(0, 10 ** 6)}'

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
    assert api_mock.create_order.call_args_list[0][1]['type'] == ordertype
    assert api_mock.create_order.call_args_list[0][1]['params'] == {
        'trading_agreement': 'agree',
        'stopLossPrice': 220
    }
    assert api_mock.create_order.call_args_list[0][1]['side'] == side
    assert api_mock.create_order.call_args_list[0][1]['amount'] == 1
    if ordertype == 'limit':
        assert api_mock.create_order.call_args_list[0][1]['price'] == adjustedprice
    else:
        assert api_mock.create_order.call_args_list[0][1]['price'] is None

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

    assert order['type'] == 'market'
    assert order['price'] == 220
    assert order['amount'] == 1


@pytest.mark.parametrize('sl1,sl2,sl3,side', [
    (1501, 1499, 1501, "sell"),
    (1499, 1501, 1499, "buy")
])
def test_stoploss_adjust_kraken(mocker, default_conf, sl1, sl2, sl3, side):
    exchange = get_patched_exchange(mocker, default_conf, id='kraken')
    order = {
        'type': 'market',
        'stopLossPrice': 1500,
    }
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)
    # diff. order type ...
    order['type'] = 'limit'
    assert exchange.stoploss_adjust(sl3, order, side=side)


@pytest.mark.parametrize('trade_id, expected', [
    ('1234', False),
    ('170544369512007228', False),
    ('1705443695120072285', True),
    ('170544369512007228555', True),
])
def test__valid_trade_pagination_id_kraken(mocker, default_conf_usdt, trade_id, expected):
    exchange = get_patched_exchange(mocker, default_conf_usdt, id='kraken')
    assert exchange._valid_trade_pagination_id('XRP/USDT', trade_id) == expected
