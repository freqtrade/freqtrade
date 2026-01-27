"""Integration tests for BreezeCCXT Market Hours blocking."""

import os
from unittest import mock

import pytest
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


@pytest.fixture
def exchange_forced_closed():
    """Fixture providing BreezeCCXT instance with market forced closed."""
    with mock.patch.dict(
        os.environ,
        {"FT_FORCE_MARKET_CLOSED": "1", "BREEZE_MOCK": "1", "RISK_GUARD_ENABLED": "false"},
    ):
        # Initialize exchange after env var set to ensure guard picks it up
        config = {"dry_run": True, "risk_guard": {"enabled": False}}
        exchange = BreezeCCXT(config)
        yield exchange


@pytest.fixture
def exchange_forced_open():
    """Fixture providing BreezeCCXT instance with market forced open."""
    with mock.patch.dict(
        os.environ, {"FT_FORCE_MARKET_OPEN": "1", "BREEZE_MOCK": "1", "RISK_GUARD_ENABLED": "false"}
    ):
        config = {"dry_run": True, "risk_guard": {"enabled": False}}
        exchange = BreezeCCXT(config)
        yield exchange


def test_market_closed_blocks_buy(exchange_forced_closed):
    """Test that creating a buy order raises exception when market is closed."""
    with pytest.raises(Exception, match="market_closed: blocking entry order"):
        exchange_forced_closed.create_order("RELIANCE/INR", "limit", "buy", 1, 2500)


def test_market_closed_allows_sell(exchange_forced_closed):
    """Test that creating a sell order is allowed (exit) even when closed."""
    # Mock position to satisfy OrderRouter Buyer Only check
    exchange_forced_closed.fetch_positions = mock.MagicMock(
        return_value=[{"symbol": "RELIANCE/INR", "contracts": 10}]
    )

    # Should not raise
    exchange_forced_closed.create_order("RELIANCE/INR", "limit", "sell", 1, 2500)


def test_market_closed_blocks_cancel(exchange_forced_closed):
    """Test that cancel order raises exception when market is closed."""
    with pytest.raises(Exception, match="market_closed: blocking cancel order"):
        exchange_forced_closed.cancel_order("123", "RELIANCE/INR")


def test_market_open_allows_all(exchange_forced_open):
    """Test that all operations are allowed when market is open."""
    # Buy
    order = exchange_forced_open.create_order("RELIANCE/INR", "limit", "buy", 1, 2500)

    # Mock positions for Sell
    exchange_forced_open.fetch_positions = mock.MagicMock(
        return_value=[{"symbol": "RELIANCE/INR", "contracts": 10}]
    )

    # Sell
    exchange_forced_open.create_order("RELIANCE/INR", "limit", "sell", 1, 2500)
    # Cancel
    exchange_forced_open.cancel_order(order["id"], "RELIANCE/INR")
