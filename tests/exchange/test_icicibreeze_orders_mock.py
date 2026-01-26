import pytest

from adapters.ccxt_shim.breeze_ccxt import BreezeAsyncCCXT, BreezeCCXT
from freqtrade.exceptions import OperationalException


def test_sync_mock_execution():
    config = {"options": {"mode": "mock"}}
    exchange = BreezeCCXT(config)

    # Balance
    balance = exchange.fetch_balance()
    assert balance["free"]["INR"] == 100000.0

    # Positions
    positions = exchange.fetch_positions()
    assert positions == []

    # Order Lifecycle
    symbol = "RELIANCE/INR"
    order = exchange.create_order(symbol, "limit", "buy", 1.0, 2500.0)
    assert order["id"].startswith("ord_")
    assert order["status"] == "open"

    fetched = exchange.fetch_order(order["id"])
    assert fetched["id"] == order["id"]

    open_orders = exchange.fetch_open_orders(symbol)
    assert any(o["id"] == order["id"] for o in open_orders)

    canceled = exchange.cancel_order(order["id"])
    assert canceled["status"] == "canceled"

    fetched_canceled = exchange.fetch_order(order["id"])
    assert fetched_canceled["status"] == "canceled"


@pytest.mark.asyncio
async def test_async_mock_execution():
    config = {"options": {"mode": "mock"}}
    exchange = BreezeAsyncCCXT(config)

    # Balance
    balance = await exchange.fetch_balance()
    assert balance["free"]["INR"] == 100000.0

    # Order Lifecycle
    symbol = "NIFTY/INR"
    order = await exchange.create_order(symbol, "limit", "sell", 50.0, 20000.0)
    assert order["id"].startswith("ord_")
    assert order["status"] == "open"

    canceled = await exchange.cancel_order(order["id"])
    assert canceled["status"] == "canceled"

    await exchange.close()


def test_fetch_open_orders_only_open():
    config = {"options": {"mode": "mock"}}
    exchange = BreezeCCXT(config)
    symbol = "RELIANCE/INR"

    order1 = exchange.create_order(symbol, "limit", "buy", 1.0, 2500.0)
    order2 = exchange.create_order(symbol, "limit", "buy", 1.0, 2600.0)

    exchange.cancel_order(order1["id"])

    open_orders = exchange.fetch_open_orders(symbol)
    ids = [o["id"] for o in open_orders]
    assert order2["id"] in ids
    assert order1["id"] not in ids
    assert all(o["status"] == "open" for o in open_orders)


def test_fetch_orders_includes_closed():
    config = {"options": {"mode": "mock"}}
    exchange = BreezeCCXT(config)
    symbol = "RELIANCE/INR"

    order1 = exchange.create_order(symbol, "limit", "buy", 1.0, 2500.0)
    exchange.cancel_order(order1["id"])

    all_orders = exchange.fetch_orders(symbol)
    ids = [o["id"] for o in all_orders]
    assert order1["id"] in ids
    assert any(o["status"] == "canceled" for o in all_orders)


def test_fetch_positions_returns_list():
    config = {"options": {"mode": "mock"}}
    exchange = BreezeCCXT(config)
    positions = exchange.fetch_positions()
    assert isinstance(positions, list)


def test_fetch_balance_has_inr_free_total():
    config = {"options": {"mode": "mock"}}
    exchange = BreezeCCXT(config)
    balance = exchange.fetch_balance()
    assert "INR" in balance["free"]
    assert "INR" in balance["total"]
    assert balance["free"]["INR"] == 100000.0
    assert balance["total"]["INR"] == 100000.0


def test_cancel_unknown_order_raises_clear_error():
    config = {"options": {"mode": "mock"}}
    exchange = BreezeCCXT(config)
    with pytest.raises(OperationalException, match="Mock order unknown_id not found"):
        exchange.cancel_order("unknown_id")


def test_fetch_unknown_order_raises_clear_error():
    config = {"options": {"mode": "mock"}}
    exchange = BreezeCCXT(config)
    with pytest.raises(OperationalException, match="Mock order unknown_id not found"):
        exchange.fetch_order("unknown_id")
