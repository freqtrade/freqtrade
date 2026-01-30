import sys
import os
import time
from unittest.mock import MagicMock

# Ensure project root is in path
sys.path.append(os.getcwd())

from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
from freqtrade.exceptions import OperationalException


def verify_p30_guard():
    # 1. Test Blocked (Default state assuming no ENV)
    print(">>> Testing Live Guard BLOCKING...")

    # Unset ENV just in case
    if "FT_ENABLE_LIVE_ORDERS" in os.environ:
        del os.environ["FT_ENABLE_LIVE_ORDERS"]

    config = {
        "icicibreeze": {"live_trading": {"enabled": True}},  # Config Enabled
        "options": {"key": "test", "secret": "test", "session_token": "test"},
        "pair_whitelist": ["RELIANCE/INR"],
    }

    exchange = BreezeCCXT(config)
    # Mock ticker to pass risk check
    exchange.fetch_ticker = lambda symbol, params=None: {
        "symbol": symbol,
        "last": 2500.0,
        "bid": 2499.0,
        "ask": 2501.0,
    }

    try:
        exchange.create_order("RELIANCE/INR", "limit", "buy", 1, 2500.0)
        print("ERROR: Guard FAILED! Order was allowed without ENV.")
        sys.exit(1)
    except OperationalException as e:
        if "Live Trading Guard: Blocked" in str(e):
            print("[OK] Guard correctly blocked order.")
        else:
            print(f"ERROR: Unexpected exception: {e}")
            sys.exit(1)

    # 2. Test Allowed (With Mocked SDK)
    print(">>> Testing Live Guard ALLOW (Mocked SDK)...")
    os.environ["FT_ENABLE_LIVE_ORDERS"] = "1"

    # Mock Breeze SDK on the exchange instance
    mock_breeze = MagicMock()
    mock_breeze.place_order.return_value = {
        "status": 200,
        "Success": {"order_id": "live_123", "message": "Order Placed"},
    }
    exchange.breeze = mock_breeze

    try:
        order = exchange.create_order("RELIANCE/INR", "limit", "buy", 1, 2500.0)
        if order["id"] == "live_123":
            print("[OK] Order passed guard and hit Mock SDK.")
        else:
            print(f"ERROR: Order ID mismatch: {order}")
            sys.exit(1)

        # Verify call args
        mock_breeze.place_order.assert_called_once()
        print("[OK] SDK place_order called.")

    except Exception as e:
        print(f"ERROR: Logic Flow Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    verify_p30_guard()
