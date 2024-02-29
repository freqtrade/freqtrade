from datetime import datetime, timezone
from random import randint
from unittest.mock import MagicMock, PropertyMock

import ccxt
import pytest

from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import DependencyException, InvalidOrderException, OperationalException
from tests.conftest import EXMS, get_mock_coro, get_patched_exchange, log_has_re
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.parametrize('side,type,time_in_force,expected', [
    ('buy', 'limit', 'gtc', {'timeInForce': 'GTC'}),
    ('buy', 'limit', 'IOC', {'timeInForce': 'IOC'}),
    ('buy', 'market', 'IOC', {}),
    ('buy', 'limit', 'PO', {'timeInForce': 'PO'}),
    ('sell', 'limit', 'PO', {'timeInForce': 'PO'}),
    ('sell', 'market', 'PO', {}),
    ])
def test__get_params_binance(default_conf, mocker, side, type, time_in_force, expected):
    exchange = get_patched_exchange(mocker, default_conf, id='binance')
    assert exchange._get_params(side, type, 1, False, time_in_force) == expected


@pytest.mark.parametrize('trademode', [TradingMode.FUTURES, TradingMode.SPOT])
@pytest.mark.parametrize('limitratio,expected,side', [
    (None, 220 * 0.99, "sell"),
    (0.99, 220 * 0.99, "sell"),
    (0.98, 220 * 0.98, "sell"),
    (None, 220 * 1.01, "buy"),
    (0.99, 220 * 1.01, "buy"),
    (0.98, 220 * 1.02, "buy"),
])
def test_create_stoploss_order_binance(default_conf, mocker, limitratio, expected, side, trademode):
    api_mock = MagicMock()
    order_id = f'test_prod_buy_{randint(0, 10 ** 6)}'
    order_type = 'stop_loss_limit' if trademode == TradingMode.SPOT else 'stop'

    api_mock.create_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    default_conf['dry_run'] = False
    default_conf['margin_mode'] = MarginMode.ISOLATED
    default_conf['trading_mode'] = trademode
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')

    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(
            pair='ETH/BTC',
            amount=1,
            stop_price=190,
            side=side,
            order_types={'stoploss': 'limit', 'stoploss_on_exchange_limit_ratio': 1.05},
            leverage=1.0
        )

    api_mock.create_order.reset_mock()
    order_types = {'stoploss': 'limit', 'stoploss_price_type': 'mark'}
    if limitratio is not None:
        order_types.update({'stoploss_on_exchange_limit_ratio': limitratio})

    order = exchange.create_stoploss(
        pair='ETH/BTC',
        amount=1,
        stop_price=220,
        order_types=order_types,
        side=side,
        leverage=1.0
    )

    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args_list[0][1]['symbol'] == 'ETH/BTC'
    assert api_mock.create_order.call_args_list[0][1]['type'] == order_type
    assert api_mock.create_order.call_args_list[0][1]['side'] == side
    assert api_mock.create_order.call_args_list[0][1]['amount'] == 1
    # Price should be 1% below stopprice
    assert api_mock.create_order.call_args_list[0][1]['price'] == expected
    if trademode == TradingMode.SPOT:
        params_dict = {'stopPrice': 220}
    else:
        params_dict = {'stopPrice': 220, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
    assert api_mock.create_order.call_args_list[0][1]['params'] == params_dict

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("0 balance"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')
        exchange.create_stoploss(
            pair='ETH/BTC',
            amount=1,
            stop_price=220,
            order_types={},
            side=side,
            leverage=1.0)

    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(
            side_effect=ccxt.InvalidOrder("binance Order would trigger immediately."))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')
        exchange.create_stoploss(
            pair='ETH/BTC',
            amount=1,
            stop_price=220,
            order_types={},
            side=side,
            leverage=1.0
        )

    ccxt_exceptionhandlers(mocker, default_conf, api_mock, "binance",
                           "create_stoploss", "create_order", retries=1,
                           pair='ETH/BTC', amount=1, stop_price=220, order_types={},
                           side=side, leverage=1.0)


def test_create_stoploss_order_dry_run_binance(default_conf, mocker):
    api_mock = MagicMock()
    order_type = 'stop_loss_limit'
    default_conf['dry_run'] = True
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')

    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(
            pair='ETH/BTC',
            amount=1,
            stop_price=190,
            side="sell",
            order_types={'stoploss_on_exchange_limit_ratio': 1.05},
            leverage=1.0
        )

    api_mock.create_order.reset_mock()

    order = exchange.create_stoploss(
        pair='ETH/BTC',
        amount=1,
        stop_price=220,
        order_types={},
        side="sell",
        leverage=1.0
    )

    assert 'id' in order
    assert 'info' in order
    assert 'type' in order

    assert order['type'] == order_type
    assert order['price'] == 220
    assert order['amount'] == 1


@pytest.mark.parametrize('sl1,sl2,sl3,side', [
    (1501, 1499, 1501, "sell"),
    (1499, 1501, 1499, "buy")
])
def test_stoploss_adjust_binance(mocker, default_conf, sl1, sl2, sl3, side):
    exchange = get_patched_exchange(mocker, default_conf, id='binance')
    order = {
        'type': 'stop_loss_limit',
        'price': 1500,
        'stopPrice': 1500,
        'info': {'stopPrice': 1500},
    }
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)


def test_fill_leverage_tiers_binance(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_leverage_tiers = MagicMock(return_value={
        'ADA/BUSD': [
            {
                "tier": 1,
                "minNotional": 0,
                "maxNotional": 100000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "info": {
                    "bracket": "1",
                    "initialLeverage": "20",
                    "maxNotional": "100000",
                    "minNotional": "0",
                    "maintMarginRatio": "0.025",
                    "cum": "0.0"
                }
            },
            {
                "tier": 2,
                "minNotional": 100000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "info": {
                    "bracket": "2",
                    "initialLeverage": "10",
                    "maxNotional": "500000",
                    "minNotional": "100000",
                    "maintMarginRatio": "0.05",
                    "cum": "2500.0"
                }
            },
            {
                "tier": 3,
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "info": {
                    "bracket": "3",
                    "initialLeverage": "5",
                    "maxNotional": "1000000",
                    "minNotional": "500000",
                    "maintMarginRatio": "0.1",
                    "cum": "27500.0"
                }
            },
            {
                "tier": 4,
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "info": {
                    "bracket": "4",
                    "initialLeverage": "3",
                    "maxNotional": "2000000",
                    "minNotional": "1000000",
                    "maintMarginRatio": "0.15",
                    "cum": "77500.0"
                }
            },
            {
                "tier": 5,
                "minNotional": 2000000,
                "maxNotional": 5000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "info": {
                    "bracket": "5",
                    "initialLeverage": "2",
                    "maxNotional": "5000000",
                    "minNotional": "2000000",
                    "maintMarginRatio": "0.25",
                    "cum": "277500.0"
                }
            },
            {
                "tier": 6,
                "minNotional": 5000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "info": {
                    "bracket": "6",
                    "initialLeverage": "1",
                    "maxNotional": "30000000",
                    "minNotional": "5000000",
                    "maintMarginRatio": "0.5",
                    "cum": "1527500.0"
                }
            }
        ],
        "ZEC/USDT": [
            {
                "tier": 1,
                "minNotional": 0,
                "maxNotional": 50000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "info": {
                    "bracket": "1",
                    "initialLeverage": "50",
                    "maxNotional": "50000",
                    "minNotional": "0",
                    "maintMarginRatio": "0.01",
                    "cum": "0.0"
                }
            },
            {
                "tier": 2,
                "minNotional": 50000,
                "maxNotional": 150000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "info": {
                    "bracket": "2",
                    "initialLeverage": "20",
                    "maxNotional": "150000",
                    "minNotional": "50000",
                    "maintMarginRatio": "0.025",
                    "cum": "750.0"
                }
            },
            {
                "tier": 3,
                "minNotional": 150000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "info": {
                    "bracket": "3",
                    "initialLeverage": "10",
                    "maxNotional": "250000",
                    "minNotional": "150000",
                    "maintMarginRatio": "0.05",
                    "cum": "4500.0"
                }
            },
            {
                "tier": 4,
                "minNotional": 250000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "info": {
                    "bracket": "4",
                    "initialLeverage": "5",
                    "maxNotional": "500000",
                    "minNotional": "250000",
                    "maintMarginRatio": "0.1",
                    "cum": "17000.0"
                }
            },
            {
                "tier": 5,
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "info": {
                    "bracket": "5",
                    "initialLeverage": "4",
                    "maxNotional": "1000000",
                    "minNotional": "500000",
                    "maintMarginRatio": "0.125",
                    "cum": "29500.0"
                }
            },
            {
                "tier": 6,
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "info": {
                    "bracket": "6",
                    "initialLeverage": "2",
                    "maxNotional": "2000000",
                    "minNotional": "1000000",
                    "maintMarginRatio": "0.25",
                    "cum": "154500.0"
                }
            },
            {
                "tier": 7,
                "minNotional": 2000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "info": {
                    "bracket": "7",
                    "initialLeverage": "1",
                    "maxNotional": "30000000",
                    "minNotional": "2000000",
                    "maintMarginRatio": "0.5",
                    "cum": "654500.0"
                }
            }
        ],
    })
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="binance")
    exchange.fill_leverage_tiers()

    assert exchange._leverage_tiers == {
        'ADA/BUSD': [
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
            }
        ],
        "ZEC/USDT": [
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

    api_mock = MagicMock()
    api_mock.load_leverage_tiers = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'fetchLeverageTiers': True})

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "binance",
        "fill_leverage_tiers",
        "fetch_leverage_tiers",
    )


def test_fill_leverage_tiers_binance_dryrun(default_conf, mocker, leverage_tiers):
    api_mock = MagicMock()
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="binance")
    exchange.fill_leverage_tiers()
    assert len(exchange._leverage_tiers.keys()) > 100
    for key, value in leverage_tiers.items():
        v = exchange._leverage_tiers[key]
        assert isinstance(v, list)
        # Assert if conftest leverage tiers have less or equal tiers than the exchange
        assert len(v) >= len(value)


def test_additional_exchange_init_binance(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fapiPrivateGetPositionSideDual = MagicMock(return_value={"dualSidePosition": True})
    api_mock.fapiPrivateGetMultiAssetsMargin = MagicMock(return_value={"multiAssetsMargin": True})
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    with pytest.raises(OperationalException,
                       match=r"Hedge Mode is not supported.*\nMulti-Asset Mode is not supported.*"):
        get_patched_exchange(mocker, default_conf, id="binance", api_mock=api_mock)
    api_mock.fapiPrivateGetPositionSideDual = MagicMock(return_value={"dualSidePosition": False})
    api_mock.fapiPrivateGetMultiAssetsMargin = MagicMock(return_value={"multiAssetsMargin": False})
    exchange = get_patched_exchange(mocker, default_conf, id="binance", api_mock=api_mock)
    assert exchange
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'binance',
                           "additional_exchange_init", "fapiPrivateGetPositionSideDual")


def test__set_leverage_binance(mocker, default_conf):

    api_mock = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'setLeverage': True})
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED

    exchange = get_patched_exchange(mocker, default_conf, api_mock, id="binance")
    exchange._set_leverage(3.2, 'BTC/USDT:USDT')
    assert api_mock.set_leverage.call_count == 1
    # Leverage is rounded to 3.
    assert api_mock.set_leverage.call_args_list[0][1]['leverage'] == 3
    assert api_mock.set_leverage.call_args_list[0][1]['symbol'] == 'BTC/USDT:USDT'

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "binance",
        "_set_leverage",
        "set_leverage",
        pair="XRP/USDT",
        leverage=5.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('candle_type', [CandleType.MARK, ''])
async def test__async_get_historic_ohlcv_binance(default_conf, mocker, caplog, candle_type):
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
    respair, restf, restype, res, _ = await exchange._async_get_historic_ohlcv(
        pair, "5m", 1500000000000, is_new_pair=False, candle_type=candle_type)
    assert respair == pair
    assert restf == '5m'
    assert restype == candle_type
    # Call with very old timestamp - causes tons of requests
    assert exchange._api_async.fetch_ohlcv.call_count > 400
    # assert res == ohlcv
    exchange._api_async.fetch_ohlcv.reset_mock()
    _, _, _, res, _ = await exchange._async_get_historic_ohlcv(
        pair, "5m", 1500000000000, is_new_pair=True, candle_type=candle_type)

    # Called twice - one "init" call - and one to get the actual data.
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert res == ohlcv
    assert log_has_re(r"Candle-data for ETH/BTC available starting with .*", caplog)


@pytest.mark.parametrize('pair,nominal_value,mm_ratio,amt', [
    ("XRP/USDT:USDT", 0.0, 0.025, 0),
    ("BNB/USDT:USDT", 100.0, 0.0065, 0),
    ("BTC/USDT:USDT", 170.30, 0.004, 0),
    ("XRP/USDT:USDT", 999999.9, 0.1, 27500.0),
    ("BNB/USDT:USDT", 5000000.0, 0.15, 233035.0),
    ("BTC/USDT:USDT", 600000000, 0.5, 1.997038E8),
])
def test_get_maintenance_ratio_and_amt_binance(
    default_conf,
    mocker,
    leverage_tiers,
    pair,
    nominal_value,
    mm_ratio,
    amt,
):
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, id="binance")
    exchange._leverage_tiers = leverage_tiers
    (result_ratio, result_amt) = exchange.get_maintenance_ratio_and_amt(pair, nominal_value)
    assert (round(result_ratio, 8), round(result_amt, 8)) == (mm_ratio, amt)
