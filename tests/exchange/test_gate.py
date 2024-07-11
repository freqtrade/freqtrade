from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from freqtrade.enums import MarginMode, TradingMode
from tests.conftest import EXMS, get_patched_exchange


@pytest.mark.usefixtures("init_persistence")
def test_fetch_stoploss_order_gate(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")

    fetch_order_mock = MagicMock()
    exchange.fetch_order = fetch_order_mock

    exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert fetch_order_mock.call_count == 1
    assert fetch_order_mock.call_args_list[0][1]["order_id"] == "1234"
    assert fetch_order_mock.call_args_list[0][1]["pair"] == "ETH/BTC"
    assert fetch_order_mock.call_args_list[0][1]["params"] == {"stop": True}

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"

    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")

    exchange.fetch_order = MagicMock(
        return_value={
            "status": "closed",
            "id": "1234",
            "stopPrice": 5.62,
            "info": {"trade_id": "222555"},
        }
    )

    exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert exchange.fetch_order.call_count == 2
    assert exchange.fetch_order.call_args_list[0][1]["order_id"] == "1234"
    assert exchange.fetch_order.call_args_list[1][1]["order_id"] == "222555"


def test_cancel_stoploss_order_gate(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")

    cancel_order_mock = MagicMock()
    exchange.cancel_order = cancel_order_mock

    exchange.cancel_stoploss_order("1234", "ETH/BTC")
    assert cancel_order_mock.call_count == 1
    assert cancel_order_mock.call_args_list[0][1]["order_id"] == "1234"
    assert cancel_order_mock.call_args_list[0][1]["pair"] == "ETH/BTC"
    assert cancel_order_mock.call_args_list[0][1]["params"] == {"stop": True}


@pytest.mark.parametrize(
    "sl1,sl2,sl3,side", [(1501, 1499, 1501, "sell"), (1499, 1501, 1499, "buy")]
)
def test_stoploss_adjust_gate(mocker, default_conf, sl1, sl2, sl3, side):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")
    order = {
        "price": 1500,
        "stopPrice": 1500,
    }
    assert exchange.stoploss_adjust(sl1, order, side)
    assert not exchange.stoploss_adjust(sl2, order, side)


@pytest.mark.parametrize(
    "takerormaker,rate,cost",
    [
        ("taker", 0.0005, 0.0001554325),
        ("maker", 0.0, 0.0),
    ],
)
def test_fetch_my_trades_gate(mocker, default_conf, takerormaker, rate, cost):
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    tick = {
        "ETH/USDT:USDT": {
            "info": {
                "user_id": "",
                "taker_fee": "0.0018",
                "maker_fee": "0.0018",
                "gt_discount": False,
                "gt_taker_fee": "0",
                "gt_maker_fee": "0",
                "loan_fee": "0.18",
                "point_type": "1",
                "futures_taker_fee": "0.0005",
                "futures_maker_fee": "0",
            },
            "symbol": "ETH/USDT:USDT",
            "maker": 0.0,
            "taker": 0.0005,
        }
    }
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED

    api_mock = MagicMock()
    api_mock.fetch_my_trades = MagicMock(
        return_value=[
            {
                "fee": {"cost": None},
                "price": 3108.65,
                "cost": 0.310865,
                "order": "22255",
                "takerOrMaker": takerormaker,
                "amount": 1,  # 1 contract
            }
        ]
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock=api_mock, exchange="gate")
    exchange._trading_fees = tick
    trades = exchange.get_trades_for_order("22255", "ETH/USDT:USDT", datetime.now(timezone.utc))
    trade = trades[0]
    assert trade["fee"]
    assert trade["fee"]["rate"] == rate
    assert trade["fee"]["currency"] == "USDT"
    assert trade["fee"]["cost"] == cost
