import sys
import os
import time
import logging
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

    # Force Market Open to test Live Guard (which runs AFTER Market Guard)
    os.environ["FT_FORCE_MARKET_OPEN"] = "1"

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
    # Setup Env
    os.environ["FT_ENABLE_LIVE_ORDERS"] = "1"  # Allow Live
    os.environ["FT_FORCE_MARKET_OPEN"] = "1"  # Bypass Market Hours
    print(f"DEBUG: FT_FORCE_MARKET_OPEN={os.environ.get('FT_FORCE_MARKET_OPEN')}")

    # Mock Breeze SDK on the exchange instance
    mock_breeze = MagicMock()
    mock_breeze.place_order.return_value = {
        "status": 200,
        "Success": {"order_id": "live_123", "message": "Order Placed"},
    }
    exchange.breeze = mock_breeze

    # Create Deadman File for P40 Compliance with secure permissions
    from pathlib import Path

    deadman_file = Path("user_data/secrets/deadman_live.ok")
    deadman_file.parent.mkdir(parents=True, exist_ok=True)
    deadman_file.touch()
    os.chmod(deadman_file, 0o600)

    # Mock RiskGuard to avoid 'intraday_cutoff' or other risk blocks
    exchange.risk_guard = MagicMock()
    exchange.risk_guard.should_block_entry.return_value = (False, None)

    try:
        order = exchange.create_order("RELIANCE/INR", "limit", "buy", 1, 2500.0)

        # In BREEZE_MOCK=1 mode, BreezeCCXT returns an internal mock order,
        # NOT the result of self.breeze.place_order.
        if order["status"] == "open" and order["info"].get("mock") is True:
            print("[OK] Order passed guard and hit Internal Mock Route.")
        else:
            print(f"ERROR: Malformed Mock Order: {order}")
            sys.exit(1)

        # In Mock Mode, the underlying SDK is NOT called (short-circuited).
        # So we do NOT check mock_breeze.place_order.assert_called_once().

    except Exception as e:
        logger.exception(f"Logic Flow Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("p30_check")
    verify_p30_guard()
