from copy import deepcopy
from datetime import datetime, timedelta
from random import randint
from unittest.mock import MagicMock, PropertyMock

import ccxt
import pandas as pd
import pytest

from freqtrade.data.converter.trade_converter import trades_dict_to_list
from freqtrade.enums import CandleType, MarginMode, RunMode, TradingMode
from freqtrade.exceptions import DependencyException, InvalidOrderException, OperationalException
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds
from freqtrade.persistence import Trade
from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts, dt_utc
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.parametrize(
    "side,order_type,time_in_force,expected",
    [
        ("buy", "limit", "gtc", {"timeInForce": "GTC"}),
        ("buy", "limit", "IOC", {"timeInForce": "IOC"}),
        ("buy", "market", "IOC", {}),
        ("buy", "limit", "PO", {"timeInForce": "PO"}),
        ("sell", "limit", "PO", {"timeInForce": "PO"}),
        ("sell", "market", "PO", {}),
    ],
)
def test__get_params_binance(default_conf, mocker, side, order_type, time_in_force, expected):
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    assert exchange._get_params(side, order_type, 1, False, time_in_force) == expected


@pytest.mark.parametrize("trademode", [TradingMode.FUTURES, TradingMode.SPOT])
@pytest.mark.parametrize(
    "limitratio,expected,side",
    [
        (None, 220 * 0.99, "sell"),
        (0.99, 220 * 0.99, "sell"),
        (0.98, 220 * 0.98, "sell"),
        (None, 220 * 1.01, "buy"),
        (0.99, 220 * 1.01, "buy"),
        (0.98, 220 * 1.02, "buy"),
    ],
)
def test_create_stoploss_order_binance(default_conf, mocker, limitratio, expected, side, trademode):
    api_mock = MagicMock()
    order_id = f"test_prod_buy_{randint(0, 10**6)}"
    order_type = "stop_loss_limit" if trademode == TradingMode.SPOT else "stop"

    api_mock.create_order = MagicMock(return_value={"id": order_id, "info": {"foo": "bar"}})
    default_conf["dry_run"] = False
    default_conf["margin_mode"] = MarginMode.ISOLATED
    default_conf["trading_mode"] = trademode
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, "binance")

    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(
            pair="ETH/BTC",
            amount=1,
            stop_price=190,
            side=side,
            order_types={"stoploss": "limit", "stoploss_on_exchange_limit_ratio": 1.05},
            leverage=1.0,
        )

    api_mock.create_order.reset_mock()
    order_types = {"stoploss": "limit", "stoploss_price_type": "mark"}
    if limitratio is not None:
        order_types.update({"stoploss_on_exchange_limit_ratio": limitratio})

    order = exchange.create_stoploss(
        pair="ETH/BTC", amount=1, stop_price=220, order_types=order_types, side=side, leverage=1.0
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert api_mock.create_order.call_args_list[0][1]["symbol"] == "ETH/BTC"
    assert api_mock.create_order.call_args_list[0][1]["type"] == order_type
    assert api_mock.create_order.call_args_list[0][1]["side"] == side
    assert api_mock.create_order.call_args_list[0][1]["amount"] == 1
    # Price should be 1% below stopprice
    assert api_mock.create_order.call_args_list[0][1]["price"] == expected
    if trademode == TradingMode.SPOT:
        params_dict = {"stopPrice": 220}
    else:
        params_dict = {"stopPrice": 220, "reduceOnly": True, "workingType": "MARK_PRICE"}
    assert api_mock.create_order.call_args_list[0][1]["params"] == params_dict

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("0 balance"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, "binance")
        exchange.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side=side, leverage=1.0
        )

    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(
            side_effect=ccxt.InvalidOrder("binance Order would trigger immediately.")
        )
        exchange = get_patched_exchange(mocker, default_conf, api_mock, "binance")
        exchange.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side=side, leverage=1.0
        )

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "binance",
        "create_stoploss",
        "create_order",
        retries=1,
        pair="ETH/BTC",
        amount=1,
        stop_price=220,
        order_types={},
        side=side,
        leverage=1.0,
    )


def test_create_stoploss_order_dry_run_binance(default_conf, mocker):
    api_mock = MagicMock()
    order_type = "stop_loss_limit"
    default_conf["dry_run"] = True
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, "binance")

    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(
            pair="ETH/BTC",
            amount=1,
            stop_price=190,
            side="sell",
            order_types={"stoploss_on_exchange_limit_ratio": 1.05},
            leverage=1.0,
        )

    api_mock.create_order.reset_mock()

    order = exchange.create_stoploss(
        pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side="sell", leverage=1.0
    )

    assert "id" in order
    assert "info" in order
    assert "type" in order

    assert order["type"] == order_type
    assert order["price"] == 217.8
    assert order["stopPrice"] == 220
    assert order["amount"] == 1


@pytest.mark.parametrize(
    "sl1,sl2,sl3,side", [(1501, 1499, 1501, "sell"), (1499, 1501, 1499, "buy")]
)
def test_stoploss_adjust_binance(mocker, default_conf, sl1, sl2, sl3, side):
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    order = {
        "type": "stop_loss_limit",
        "price": 1500,
        "stopPrice": 1500,
        "info": {"stopPrice": 1500},
    }
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)


@pytest.mark.parametrize(
    "pair, is_short, trading_mode, margin_mode, wallet_balance, "
    "maintenance_amt, amount, open_rate, open_trades,"
    "mm_ratio, expected",
    [
        (
            "ETH/USDT:USDT",
            False,
            "futures",
            "isolated",
            1535443.01,
            135365.00,
            3683.979,
            1456.84,
            [],
            0.10,
            1114.78,
        ),
        (
            "ETH/USDT:USDT",
            False,
            "futures",
            "isolated",
            1535443.01,
            16300.000,
            109.488,
            32481.980,
            [],
            0.025,
            18778.73,
        ),
        (
            "ETH/USDT:USDT",
            False,
            "futures",
            "cross",
            1535443.01,
            135365.00,
            3683.979,  # amount
            1456.84,  # open_rate
            [
                {
                    # From calc example
                    "pair": "BTC/USDT:USDT",
                    "open_rate": 32481.98,
                    "amount": 109.488,
                    "stake_amount": 3556387.02624,  # open_rate * amount
                    "mark_price": 31967.27,
                    "mm_ratio": 0.025,
                    "maintenance_amt": 16300.0,
                },
                {
                    # From calc example
                    "pair": "ETH/USDT:USDT",
                    "open_rate": 1456.84,
                    "amount": 3683.979,
                    "stake_amount": 5366967.96,
                    "mark_price": 1335.18,
                    "mm_ratio": 0.10,
                    "maintenance_amt": 135365.00,
                },
            ],
            0.10,
            1153.26,
        ),
        (
            "BTC/USDT:USDT",
            False,
            "futures",
            "cross",
            1535443.01,
            16300.0,
            109.488,  # amount
            32481.980,  # open_rate
            [
                {
                    # From calc example
                    "pair": "BTC/USDT:USDT",
                    "open_rate": 32481.98,
                    "amount": 109.488,
                    "stake_amount": 3556387.02624,  # open_rate * amount
                    "mark_price": 31967.27,
                    "mm_ratio": 0.025,
                    "maintenance_amt": 16300.0,
                },
                {
                    # From calc example
                    "pair": "ETH/USDT:USDT",
                    "open_rate": 1456.84,
                    "amount": 3683.979,
                    "stake_amount": 5366967.96,
                    "mark_price": 1335.18,
                    "mm_ratio": 0.10,
                    "maintenance_amt": 135365.00,
                },
            ],
            0.025,
            26316.89,
        ),
    ],
)
def test_liquidation_price_binance(
    mocker,
    default_conf,
    pair,
    is_short,
    trading_mode,
    margin_mode,
    wallet_balance,
    maintenance_amt,
    amount,
    open_rate,
    open_trades,
    mm_ratio,
    expected,
):
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = margin_mode
    default_conf["liquidation_buffer"] = 0.0
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")

    def get_maint_ratio(pair_, stake_amount):
        if pair_ != pair:
            oc = next(c for c in open_trades if c["pair"] == pair_)
            return oc["mm_ratio"], oc["maintenance_amt"]
        return mm_ratio, maintenance_amt

    def fetch_funding_rates(*args, **kwargs):
        return {
            t["pair"]: {
                "symbol": t["pair"],
                "markPrice": t["mark_price"],
            }
            for t in open_trades
        }

    exchange.get_maintenance_ratio_and_amt = get_maint_ratio
    exchange.fetch_funding_rates = fetch_funding_rates

    open_trade_objects = [
        Trade(
            pair=t["pair"],
            open_rate=t["open_rate"],
            amount=t["amount"],
            stake_amount=t["stake_amount"],
            fee_open=0,
        )
        for t in open_trades
    ]

    assert (
        pytest.approx(
            round(
                exchange.get_liquidation_price(
                    pair=pair,
                    open_rate=open_rate,
                    is_short=is_short,
                    wallet_balance=wallet_balance,
                    amount=amount,
                    stake_amount=open_rate * amount,
                    leverage=5,
                    open_trades=open_trade_objects,
                ),
                2,
            )
        )
        == expected
    )


def test_fill_leverage_tiers_binance(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_leverage_tiers = MagicMock(
        return_value={
            "ADA/BUSD": [
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
                        "cum": "0.0",
                    },
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
                        "cum": "2500.0",
                    },
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
                        "cum": "27500.0",
                    },
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
                        "cum": "77500.0",
                    },
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
                        "cum": "277500.0",
                    },
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
                        "cum": "1527500.0",
                    },
                },
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
                        "cum": "0.0",
                    },
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
                        "cum": "750.0",
                    },
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
                        "cum": "4500.0",
                    },
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
                        "cum": "17000.0",
                    },
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
                        "cum": "29500.0",
                    },
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
                        "cum": "154500.0",
                    },
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
                        "cum": "654500.0",
                    },
                },
            ],
        }
    )
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="binance")
    exchange.fill_leverage_tiers()

    assert exchange._leverage_tiers == {
        "ADA/BUSD": [
            {
                "minNotional": 0,
                "maxNotional": 100000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 100000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 2500.0,
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 27500.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 77500.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 5000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 277500.0,
            },
            {
                "minNotional": 5000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 1527500.0,
            },
        ],
        "ZEC/USDT": [
            {
                "minNotional": 0,
                "maxNotional": 50000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 50000,
                "maxNotional": 150000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 750.0,
            },
            {
                "minNotional": 150000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 4500.0,
            },
            {
                "minNotional": 250000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 17000.0,
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "maintAmt": 29500.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 154500.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 654500.0,
            },
        ],
    }

    api_mock = MagicMock()
    api_mock.load_leverage_tiers = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"fetchLeverageTiers": True})

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
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="binance")
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
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    with pytest.raises(
        OperationalException,
        match=r"Hedge Mode is not supported.*\nMulti-Asset Mode is not supported.*",
    ):
        get_patched_exchange(mocker, default_conf, exchange="binance", api_mock=api_mock)
    api_mock.fapiPrivateGetPositionSideDual = MagicMock(return_value={"dualSidePosition": False})
    api_mock.fapiPrivateGetMultiAssetsMargin = MagicMock(return_value={"multiAssetsMargin": False})
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance", api_mock=api_mock)
    assert exchange
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "binance",
        "additional_exchange_init",
        "fapiPrivateGetPositionSideDual",
    )


def test__set_leverage_binance(mocker, default_conf):
    api_mock = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setLeverage": True})
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="binance")
    exchange._set_leverage(3.2, "BTC/USDT:USDT")
    assert api_mock.set_leverage.call_count == 1
    # Leverage is rounded to 3.
    assert api_mock.set_leverage.call_args_list[0][1]["leverage"] == 3
    assert api_mock.set_leverage.call_args_list[0][1]["symbol"] == "BTC/USDT:USDT"

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


def patch_binance_vision_ohlcv(mocker, start, archive_end, api_end, timeframe):
    def make_storage(start: datetime, end: datetime, timeframe: str):
        date = pd.date_range(start, end, freq=timeframe.replace("m", "min"))
        df = pd.DataFrame(
            data=dict(date=date, open=1.0, high=1.0, low=1.0, close=1.0),
        )
        return df

    archive_storage = make_storage(start, archive_end, timeframe)
    api_storage = make_storage(start, api_end, timeframe)

    ohlcv = [[dt_ts(start), 1, 1, 1, 1]]
    # (pair, timeframe, candle_type, ohlcv, True)
    candle_history = [None, None, None, ohlcv, None]

    def get_historic_ohlcv(
        # self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: CandleType,
        is_new_pair: bool = False,
        until_ms: int | None = None,
    ):
        since = dt_from_ts(since_ms)
        until = dt_from_ts(until_ms) if until_ms else api_end + timedelta(seconds=1)
        return api_storage.loc[(api_storage["date"] >= since) & (api_storage["date"] < until)]

    async def download_archive_ohlcv(
        candle_type,
        pair,
        timeframe,
        since_ms,
        until_ms,
        markets=None,
        stop_on_404=False,
    ):
        since = dt_from_ts(since_ms)
        until = dt_from_ts(until_ms) if until_ms else archive_end + timedelta(seconds=1)
        if since < start:
            pass
        return archive_storage.loc[
            (archive_storage["date"] >= since) & (archive_storage["date"] < until)
        ]

    candle_mock = mocker.patch(f"{EXMS}._async_get_candle_history", return_value=candle_history)
    api_mock = mocker.patch(f"{EXMS}.get_historic_ohlcv", side_effect=get_historic_ohlcv)
    archive_mock = mocker.patch(
        "freqtrade.exchange.binance.download_archive_ohlcv", side_effect=download_archive_ohlcv
    )
    return candle_mock, api_mock, archive_mock


@pytest.mark.parametrize(
    "timeframe,is_new_pair,since,until,first_date,last_date,candle_called,archive_called,"
    "api_called",
    [
        (
            "1m",
            True,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59),
            True,
            True,
            False,
        ),
        (
            "1m",
            True,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 3),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2, 23, 59),
            True,
            True,
            True,
        ),
        (
            "1m",
            True,
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 2, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 2, 0, 59),
            True,
            False,
            True,
        ),
        (
            "1m",
            False,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59),
            False,
            True,
            False,
        ),
        (
            "1m",
            True,
            dt_utc(2019, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59),
            True,
            True,
            False,
        ),
        (
            "1m",
            False,
            dt_utc(2019, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59),
            False,
            True,
            False,
        ),
        (
            "1m",
            False,
            dt_utc(2019, 1, 1),
            dt_utc(2019, 1, 2),
            None,
            None,
            False,
            True,
            True,
        ),
        (
            "1m",
            True,
            dt_utc(2019, 1, 1),
            dt_utc(2019, 1, 2),
            None,
            None,
            True,
            False,
            False,
        ),
        (
            "1m",
            False,
            dt_utc(2021, 1, 1),
            dt_utc(2021, 1, 2),
            None,
            None,
            False,
            False,
            False,
        ),
        (
            "1m",
            True,
            dt_utc(2021, 1, 1),
            dt_utc(2021, 1, 2),
            None,
            None,
            True,
            False,
            False,
        ),
        (
            "1h",
            False,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23),
            False,
            False,
            True,
        ),
        (
            "1m",
            False,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 3, 50, 30),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 3, 50),
            False,
            True,
            False,
        ),
    ],
)
def test_get_historic_ohlcv_binance(
    mocker,
    default_conf,
    timeframe,
    is_new_pair,
    since,
    until,
    first_date,
    last_date,
    candle_called,
    archive_called,
    api_called,
):
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")

    start = dt_utc(2020, 1, 1)
    archive_end = dt_utc(2020, 1, 2)
    api_end = dt_utc(2020, 1, 3)
    candle_mock, api_mock, archive_mock = patch_binance_vision_ohlcv(
        mocker, start=start, archive_end=archive_end, api_end=api_end, timeframe=timeframe
    )

    candle_type = CandleType.SPOT
    pair = "BTC/USDT"

    since_ms = dt_ts(since)
    until_ms = dt_ts(until)

    df = exchange.get_historic_ohlcv(pair, timeframe, since_ms, candle_type, is_new_pair, until_ms)

    if df.empty:
        assert first_date is None
        assert last_date is None
    else:
        assert df["date"].iloc[0] == first_date
        assert df["date"].iloc[-1] == last_date
        assert (
            df["date"].diff().iloc[1:] == timedelta(seconds=timeframe_to_seconds(timeframe))
        ).all()

    if candle_called:
        candle_mock.assert_called_once()
    if archive_called:
        archive_mock.assert_called_once()
    if api_called:
        api_mock.assert_called_once()
    candle_mock.reset_mock()
    api_mock.reset_mock()
    archive_mock.reset_mock()

    # binanceus does not use archive mode!
    exchange._can_use_data_download_fast = False
    df = exchange.get_historic_ohlcv(pair, timeframe, since_ms, candle_type, is_new_pair, until_ms)
    # Never uses archive
    assert archive_mock.call_count == 0
    assert candle_mock.call_count == (0 if not candle_called else 1)
    if api_called:
        assert api_mock.call_count == 1


@pytest.mark.parametrize(
    "pair,notional_value,mm_ratio,amt",
    [
        ("XRP/USDT:USDT", 0.0, 0.025, 0),
        ("BNB/USDT:USDT", 100.0, 0.0065, 0),
        ("BTC/USDT:USDT", 170.30, 0.004, 0),
        ("XRP/USDT:USDT", 999999.9, 0.1, 27500.0),
        ("BNB/USDT:USDT", 5000000.0, 0.15, 233035.0),
        ("BTC/USDT:USDT", 600000000, 0.5, 1.997038e8),
    ],
)
def test_get_maintenance_ratio_and_amt_binance(
    default_conf,
    mocker,
    leverage_tiers,
    pair,
    notional_value,
    mm_ratio,
    amt,
):
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    exchange._leverage_tiers = leverage_tiers
    (result_ratio, result_amt) = exchange.get_maintenance_ratio_and_amt(pair, notional_value)
    assert (round(result_ratio, 8), round(result_amt, 8)) == (mm_ratio, amt)


async def test__async_get_trade_history_id_binance(default_conf_usdt, mocker, fetch_trades_result):
    default_conf_usdt["exchange"]["only_from_ccxt"] = True
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="binance")

    async def mock_get_trade_hist(pair, *args, **kwargs):
        if "since" in kwargs:
            # older than initial call
            if kwargs["since"] < 1565798399752:
                return []
            else:
                # Don't expect to get here
                raise ValueError("Unexpected call")
                # return fetch_trades_result[:-2]
        elif kwargs.get("params", {}).get(exchange._ft_has["trades_pagination_arg"]) == "0":
            # Return first 3
            return fetch_trades_result[:-2]
        elif kwargs.get("params", {}).get(exchange._ft_has["trades_pagination_arg"]) in (
            fetch_trades_result[-3]["id"],
            1565798399752,
        ):
            # Return 2
            return fetch_trades_result[-3:-1]
        else:
            # Return last 2
            return fetch_trades_result[-2:]

    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)

    pair = "ETH/BTC"
    ret = await exchange._async_get_trade_history_id(
        pair,
        since=fetch_trades_result[0]["timestamp"],
        until=fetch_trades_result[-1]["timestamp"] - 1,
    )
    assert ret[0] == pair
    assert isinstance(ret[1], list)
    assert exchange._api_async.fetch_trades.call_count == 4

    fetch_trades_cal = exchange._api_async.fetch_trades.call_args_list
    # first call (using since, not fromId)
    assert fetch_trades_cal[0][0][0] == pair
    assert fetch_trades_cal[0][1]["since"] == fetch_trades_result[0]["timestamp"]

    # 2nd call
    assert fetch_trades_cal[1][0][0] == pair
    assert "params" in fetch_trades_cal[1][1]
    pagination_arg = exchange._ft_has["trades_pagination_arg"]
    assert pagination_arg in fetch_trades_cal[1][1]["params"]
    # Initial call was with from_id = "0"
    assert fetch_trades_cal[1][1]["params"][pagination_arg] == "0"

    assert fetch_trades_cal[2][1]["params"][pagination_arg] != "0"
    assert fetch_trades_cal[3][1]["params"][pagination_arg] != "0"

    # Clean up event loop to avoid warnings
    exchange.close()


async def test__async_get_trade_history_id_binance_fast(
    default_conf_usdt, mocker, fetch_trades_result
):
    default_conf_usdt["exchange"]["only_from_ccxt"] = False
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="binance")

    async def mock_get_trade_hist(pair, *args, **kwargs):
        if "since" in kwargs:
            pass
            # older than initial call
            # if kwargs["since"] < 1565798399752:
            #     return []
            # else:
            #     # Don't expect to get here
            #     raise ValueError("Unexpected call")
            #     # return fetch_trades_result[:-2]
        elif kwargs.get("params", {}).get(exchange._ft_has["trades_pagination_arg"]) == "0":
            # Return first 3
            return fetch_trades_result[:-2]
        # elif kwargs.get("params", {}).get(exchange._ft_has['trades_pagination_arg']) in (
        #     fetch_trades_result[-3]["id"],
        #     1565798399752,
        # ):
        #     # Return 2
        #     return fetch_trades_result[-3:-1]
        # else:
        #     # Return last 2
        #     return fetch_trades_result[-2:]

    pair = "ETH/BTC"
    mocker.patch(
        "freqtrade.exchange.binance.download_archive_trades",
        return_value=(pair, trades_dict_to_list(fetch_trades_result[-2:])),
    )

    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)

    ret = await exchange._async_get_trade_history(
        pair,
        since=fetch_trades_result[0]["timestamp"],
        until=fetch_trades_result[-1]["timestamp"] - 1,
    )

    assert ret[0] == pair
    assert isinstance(ret[1], list)

    # Clean up event loop to avoid warnings
    exchange.close()


def test_check_delisting_time_binance(default_conf_usdt, mocker):
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="binance")
    exchange._config["runmode"] = RunMode.BACKTEST
    delist_mock = MagicMock(return_value=None)
    delist_fut_mock = MagicMock(return_value=None)
    mocker.patch.object(exchange, "_get_spot_pair_delist_time", delist_mock)
    mocker.patch.object(exchange, "_check_delisting_futures", delist_fut_mock)

    # Invalid run mode
    resp = exchange.check_delisting_time("BTC/USDT")
    assert resp is None
    assert delist_mock.call_count == 0
    assert delist_fut_mock.call_count == 0

    # Delist spot called
    exchange._config["runmode"] = RunMode.DRY_RUN
    resp1 = exchange.check_delisting_time("BTC/USDT")
    assert resp1 is None
    assert delist_mock.call_count == 1
    assert delist_fut_mock.call_count == 0
    delist_mock.reset_mock()

    # Delist futures called
    exchange.trading_mode = TradingMode.FUTURES
    resp1 = exchange.check_delisting_time("BTC/USDT:USDT")
    assert resp1 is None
    assert delist_mock.call_count == 0
    assert delist_fut_mock.call_count == 1


def test__check_delisting_futures_binance(default_conf_usdt, mocker, markets):
    markets["BTC/USDT:USDT"] = deepcopy(markets["SOL/BUSD:BUSD"])
    markets["BTC/USDT:USDT"]["info"]["deliveryDate"] = 4133404800000
    markets["SOL/BUSD:BUSD"]["info"]["deliveryDate"] = 4133404800000
    markets["ADA/USDT:USDT"]["info"]["deliveryDate"] = 1760745600000  # 2025-10-18
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="binance")
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))

    resp_sol = exchange._check_delisting_futures("SOL/BUSD:BUSD")
    # Delisting is equal to BTC
    assert resp_sol is None
    # Actually has a delisting date
    resp_ada = exchange._check_delisting_futures("ADA/USDT:USDT")
    assert resp_ada == dt_utc(2025, 10, 18)


def test__get_spot_delist_schedule_binance(default_conf_usdt, mocker):
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="binance")
    ret_value = [{"delistTime": 1759114800000, "symbols": ["ETCBTC"]}]
    schedule_mock = mocker.patch.object(exchange, "_get_spot_delist_schedule", return_value=None)

    # None - mode is DRY
    assert exchange._get_spot_pair_delist_time("ETC/BTC") is None
    # Switch to live
    exchange._config["runmode"] = RunMode.LIVE
    assert exchange._get_spot_pair_delist_time("ETC/BTC") is None

    mocker.patch.object(exchange, "_get_spot_delist_schedule", return_value=ret_value)
    resp = exchange._get_spot_pair_delist_time("ETC/BTC")
    assert resp == dt_utc(2025, 9, 29, 3, 0)
    assert schedule_mock.call_count == 1
    schedule_mock.reset_mock()

    # Caching - don't refresh.
    assert exchange._get_spot_pair_delist_time("ETC/BTC", refresh=False) == dt_utc(
        2025, 9, 29, 3, 0
    )
    assert schedule_mock.call_count == 0

    api_mock = MagicMock()
    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "binance",
        "_get_spot_delist_schedule",
        "sapi_get_spot_delist_schedule",
        retries=1,
    )
