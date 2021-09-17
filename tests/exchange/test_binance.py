from datetime import datetime, timezone
from random import randint
from unittest.mock import MagicMock

import ccxt
import pytest

from freqtrade.exceptions import DependencyException, InvalidOrderException, OperationalException
from tests.conftest import get_mock_coro, get_patched_exchange, log_has_re
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.parametrize('limitratio,expected', [
    (None, 220 * 0.99),
    (0.99, 220 * 0.99),
    (0.98, 220 * 0.98),
])
def test_stoploss_order_binance(default_conf, mocker, limitratio, expected):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))
    order_type = 'stop_loss_limit'

    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False
    mocker.patch('freqtrade.exchange.Exchange.amount_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.price_to_precision', lambda s, x, y: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')

    with pytest.raises(OperationalException):
        order = exchange.stoploss(pair='ETH/BTC', amount=1, stop_price=190,
                                  order_types={'stoploss_on_exchange_limit_ratio': 1.05})

    api_mock.create_order.reset_mock()
    order_types = {} if limitratio is None else {'stoploss_on_exchange_limit_ratio': limitratio}
    order = exchange.stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types=order_types)

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args_list[0][1]['symbol'] == 'ETH/BTC'
    assert api_mock.create_order.call_args_list[0][1]['type'] == order_type
    assert api_mock.create_order.call_args_list[0][1]['side'] == 'sell'
    assert api_mock.create_order.call_args_list[0][1]['amount'] == 1
    # Price should be 1% below stopprice
    assert api_mock.create_order.call_args_list[0][1]['price'] == expected
    assert api_mock.create_order.call_args_list[0][1]['params'] == {'stopPrice': 220}

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("0 balance"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')
        exchange.stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={})

    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(
            side_effect=ccxt.InvalidOrder("binance Order would trigger immediately."))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')
        exchange.stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={})

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, "binance",
                           "stoploss", "create_order", retries=1,
                           pair='ETH/BTC', amount=1, stop_price=220, order_types={})


def test_stoploss_order_dry_run_binance(default_conf, mocker):
    api_mock = MagicMock()
    order_type = 'stop_loss_limit'
    default_conf['dry_run'] = True
    mocker.patch('freqtrade.exchange.Exchange.amount_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.price_to_precision', lambda s, x, y: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')

    with pytest.raises(OperationalException):
        order = exchange.stoploss(pair='ETH/BTC', amount=1, stop_price=190,
                                  order_types={'stoploss_on_exchange_limit_ratio': 1.05})

    api_mock.create_order.reset_mock()

    order = exchange.stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={})

    assert 'id' in order
    assert 'info' in order
    assert 'type' in order

    assert order['type'] == order_type
    assert order['price'] == 220
    assert order['amount'] == 1


def test_stoploss_adjust_binance(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf, id='binance')
    order = {
        'type': 'stop_loss_limit',
        'price': 1500,
        'info': {'stopPrice': 1500},
    }
    assert exchange.stoploss_adjust(1501, order)
    assert not exchange.stoploss_adjust(1499, order)
    # Test with invalid order case
    order['type'] = 'stop_loss'
    assert not exchange.stoploss_adjust(1501, order)


@pytest.mark.asyncio
async def test__async_get_historic_ohlcv_binance(default_conf, mocker, caplog):
    ohlcv = [
        [
            int((datetime.now(timezone.utc).timestamp() - 1000) * 1000),
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]

    exchange = get_patched_exchange(mocker, default_conf, id='binance')
    # Monkey-patch async function
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)

    pair = 'ETH/BTC'
    res = await exchange._async_get_historic_ohlcv(pair, "5m",
                                                   1500000000000, is_new_pair=False)
    # Call with very old timestamp - causes tons of requests
    assert exchange._api_async.fetch_ohlcv.call_count > 400
    # assert res == ohlcv
    exchange._api_async.fetch_ohlcv.reset_mock()
    res = await exchange._async_get_historic_ohlcv(pair, "5m", 1500000000000, is_new_pair=True)

    # Called twice - one "init" call - and one to get the actual data.
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert res == ohlcv
    assert log_has_re(r"Candle-data for ETH/BTC available starting with .*", caplog)
