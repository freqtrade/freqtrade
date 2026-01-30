import asyncio
import os
import ccxt
import pytest
from unittest.mock import patch
from freqtrade.exceptions import OperationalException
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


@pytest.fixture
def mock_exchange():
    config = {
        "risk_guard": {
            "enabled": True,
            "max_trades_per_day": 0,  # Block everything
            "allow_exits_when_blocked": True,
        },
        "options": {"key": "mock_key", "secret": "mock_secret", "session_token": "mock_token"},
        "icicibreeze": {"live_trading": {"enabled": True}},
    }
    exchange = BreezeCCXT(config)
    # exchange._set_mock_mode(True) # Not needed/doesn't exist, api_key="mock_key" triggers it
    return exchange


@pytest.mark.asyncio
async def test_risk_block_buy_entry(mock_exchange):
    # Should raise OperationalException with risk_block prefix
    # Must force market open to reach risk check
    with patch.dict(os.environ, {"FT_FORCE_MARKET_OPEN": "1", "FT_ENABLE_LIVE_ORDERS": "1"}):
        with pytest.raises(OperationalException, match="risk_block:max_trades_per_day"):
            # Try await, if fails catch type error? No, better to detect.
            if hasattr(mock_exchange, "create_order") and asyncio.iscoroutinefunction(
                mock_exchange.create_order
            ):
                await mock_exchange.create_order("RELIANCE/INR", "limit", "buy", 1, 2500.0)
            else:
                mock_exchange.create_order("RELIANCE/INR", "limit", "buy", 1, 2500.0)


@pytest.mark.asyncio
async def test_risk_allow_sell_exit(mock_exchange):
    # Selling should be allowed even if entries are blocked
    # Mock mode create_order just returns a dict
    with patch.dict(os.environ, {"FT_FORCE_MARKET_OPEN": "1"}):
        if hasattr(mock_exchange, "create_order") and asyncio.iscoroutinefunction(
            mock_exchange.create_order
        ):
            order = await mock_exchange.create_order("RELIANCE/INR", "limit", "sell", 1, 2500.0)
        else:
            order = mock_exchange.create_order("RELIANCE/INR", "limit", "sell", 1, 2500.0)

    assert order["id"] is not None
    assert order["side"] == "sell"
