import pytest
import asyncio
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT, BreezeAsyncCCXT


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
