from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.enums import MarginMode, RunMode, TradingMode
from freqtrade.util import dt_utc
from tests.conftest import EXMS, get_mock_coro, get_patched_exchange, log_has
from tests.exchange.test_exchange import ccxt_exceptionhandlers


def test_additional_exchange_init_bybit(default_conf, mocker, caplog):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.set_position_mode = MagicMock(return_value={"dualSidePosition": False})
    api_mock.is_unified_enabled = MagicMock(return_value=[False, False])

    exchange = get_patched_exchange(mocker, default_conf, exchange="bybit", api_mock=api_mock)
    assert api_mock.set_position_mode.call_count == 1
    assert api_mock.is_unified_enabled.call_count == 1
    assert exchange.unified_account is False

    assert log_has("Bybit: Standard account.", caplog)

    api_mock.set_position_mode.reset_mock()
    api_mock.is_unified_enabled = MagicMock(return_value=[False, True])
    exchange = get_patched_exchange(mocker, default_conf, exchange="bybit", api_mock=api_mock)
    assert log_has("Bybit: Unified account. Assuming dedicated subaccount for this bot.", caplog)
    assert api_mock.set_position_mode.call_count == 1
    assert api_mock.is_unified_enabled.call_count == 1
    assert exchange.unified_account is True

    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, "bybit", "additional_exchange_init", "set_position_mode"
    )


async def test_bybit_fetch_funding_rate(default_conf, mocker):
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    api_mock = MagicMock()
    api_mock.fetch_funding_rate_history = get_mock_coro(return_value=[])
    exchange = get_patched_exchange(mocker, default_conf, exchange="bybit", api_mock=api_mock)
    limit = 200
    # Test fetch_funding_rate_history (current data)
    await exchange._fetch_funding_rate_history(
        pair="BTC/USDT:USDT",
        timeframe="4h",
        limit=limit,
    )

    assert api_mock.fetch_funding_rate_history.call_count == 1
    assert api_mock.fetch_funding_rate_history.call_args_list[0][0][0] == "BTC/USDT:USDT"
    kwargs = api_mock.fetch_funding_rate_history.call_args_list[0][1]
    assert kwargs["since"] is None

    api_mock.fetch_funding_rate_history.reset_mock()
    since_ms = 1610000000000
    # Test fetch_funding_rate_history (current data)
    await exchange._fetch_funding_rate_history(
        pair="BTC/USDT:USDT",
        timeframe="4h",
        limit=limit,
        since_ms=since_ms,
    )

    assert api_mock.fetch_funding_rate_history.call_count == 1
    assert api_mock.fetch_funding_rate_history.call_args_list[0][0][0] == "BTC/USDT:USDT"
    kwargs = api_mock.fetch_funding_rate_history.call_args_list[0][1]
    assert kwargs["since"] == since_ms


def test_bybit_get_funding_fees(default_conf, mocker):
    now = datetime.now(UTC)
    exchange = get_patched_exchange(mocker, default_conf, exchange="bybit")
    exchange._fetch_and_calculate_funding_fees = MagicMock()
    exchange.get_funding_fees("BTC/USDT:USDT", 1, False, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 0

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="bybit")
    exchange._fetch_and_calculate_funding_fees = MagicMock()
    exchange.get_funding_fees("BTC/USDT:USDT", 1, False, now)

    assert exchange._fetch_and_calculate_funding_fees.call_count == 1


def test_bybit_fetch_orders(default_conf, mocker, limit_order):
    api_mock = MagicMock()
    api_mock.fetch_orders = MagicMock(
        return_value=[
            limit_order["buy"],
            limit_order["sell"],
        ]
    )
    api_mock.fetch_open_orders = MagicMock(
        side_effect=[
            [{**limit_order["buy"], "id": 1}],
            [{**limit_order["buy"], "id": 2}],
            [{**limit_order["buy"], "id": 3}],
        ]
    )
    api_mock.fetch_closed_orders = MagicMock(
        side_effect=[
            [{**limit_order["buy"], "id": 5}],
            [{**limit_order["buy"], "id": 6}],
            [{**limit_order["buy"], "id": 7}],
        ]
    )

    def exchange_has(value):
        if value == "fetchOrders":
            return False
        return True

    mocker.patch(f"{EXMS}.exchange_has", side_effect=exchange_has)
    start_time = datetime.now(UTC) - timedelta(days=20)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bybit")
    # Not available in dry-run
    assert exchange.fetch_orders("mocked", start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    default_conf["dry_run"] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bybit")
    res = exchange.fetch_orders("mocked", start_time)
    # Bybit will call the endpoint 3 times, as it has a limit of 7 days per call
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 3
    assert api_mock.fetch_closed_orders.call_count == 3
    assert len(res) == 2 * 3


def test_bybit_fetch_order_canceled_empty(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False

    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(
        return_value={
            "id": "123",
            "symbol": "BTC/USDT",
            "status": "canceled",
            "filled": 0.0,
            "remaining": 0.0,
            "amount": 20.0,
        }
    )

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange="bybit")

    res = exchange.fetch_order("123", "BTC/USDT")
    assert res["remaining"] is None
    assert res["filled"] == 0.0
    assert res["amount"] == 20.0
    assert res["status"] == "canceled"

    api_mock.fetch_order = MagicMock(
        return_value={
            "id": "123",
            "symbol": "BTC/USDT",
            "status": "canceled",
            "filled": 0.0,
            "remaining": 20.0,
            "amount": 20.0,
        }
    )
    # Don't touch orders which return correctly.
    res1 = exchange.fetch_order("123", "BTC/USDT")
    assert res1["remaining"] == 20.0
    assert res1["filled"] == 0.0
    assert res1["amount"] == 20.0
    assert res1["status"] == "canceled"

    # Reverse test - remaining is not touched
    api_mock.fetch_order = MagicMock(
        return_value={
            "id": "124",
            "symbol": "BTC/USDT",
            "status": "open",
            "filled": 0.0,
            "remaining": 20.0,
            "amount": 20.0,
        }
    )
    res2 = exchange.fetch_order("123", "BTC/USDT")
    assert res2["remaining"] == 20.0
    assert res2["filled"] == 0.0
    assert res2["amount"] == 20.0
    assert res2["status"] == "open"


@pytest.mark.parametrize(
    "side,order_type,uta,tradingmode,expected",
    [
        ("buy", "limit", False, "spot", True),
        ("buy", "limit", False, "futures", True),
        ("sell", "limit", False, "spot", True),
        ("sell", "limit", False, "futures", True),
        ("buy", "market", False, "spot", True),
        ("buy", "market", False, "futures", False),
        ("buy", "market", True, "spot", False),
        ("buy", "market", True, "futures", False),
    ],
)
def test_bybit__order_needs_price(
    default_conf, mocker, side, order_type, uta, tradingmode, expected
):
    exchange = get_patched_exchange(mocker, default_conf, exchange="bybit")
    exchange.trading_mode = tradingmode
    exchange.unified_account = uta

    assert exchange._order_needs_price(side, order_type) == expected


def test_check_delisting_time_bybit(default_conf_usdt, mocker):
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="bybit")
    exchange._config["runmode"] = RunMode.BACKTEST
    delist_fut_mock = MagicMock(return_value=None)
    mocker.patch.object(exchange, "_check_delisting_futures", delist_fut_mock)

    # Invalid run mode
    resp = exchange.check_delisting_time("BTC/USDT:USDT")
    assert resp is None
    assert delist_fut_mock.call_count == 0

    # Delist spot called
    exchange._config["runmode"] = RunMode.DRY_RUN
    resp1 = exchange.check_delisting_time("BTC/USDT")
    assert resp1 is None
    assert delist_fut_mock.call_count == 0

    # Delist futures called
    exchange.trading_mode = TradingMode.FUTURES
    resp1 = exchange.check_delisting_time("BTC/USDT:USDT")
    assert resp1 is None
    assert delist_fut_mock.call_count == 1


def test__check_delisting_futures_bybit(default_conf_usdt, mocker, markets):
    markets["BTC/USDT:USDT"] = deepcopy(markets["SOL/BUSD:BUSD"])
    markets["BTC/USDT:USDT"]["info"]["deliveryTime"] = "0"
    markets["SOL/BUSD:BUSD"]["info"]["deliveryTime"] = "0"
    markets["ADA/USDT:USDT"]["info"]["deliveryTime"] = "1760745600000"  # 2025-10-18
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="bybit")
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))

    resp_sol = exchange._check_delisting_futures("SOL/BUSD:BUSD")
    # SOL has no delisting date
    assert resp_sol is None
    # Actually has a delisting date
    resp_ada = exchange._check_delisting_futures("ADA/USDT:USDT")
    assert resp_ada == dt_utc(2025, 10, 18)
