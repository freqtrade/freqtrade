# pragma pylint: disable=missing-docstring, C0103, bad-continuation, global-statement
# pragma pylint: disable=protected-access
from random import randint
from unittest.mock import MagicMock

from tests.conftest import get_patched_exchange


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
