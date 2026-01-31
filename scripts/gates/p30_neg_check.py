import sys
import os
from unittest.mock import MagicMock
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
from freqtrade.exceptions import OperationalException


def check_p30_neg():
    # Setup Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("p30_neg")

    print(">>> P30 Neg: Testing Guard Layering (Double Lock Open + Market Closed)...")

    # 1. Open Double Lock
    os.environ["FT_ENABLE_LIVE_ORDERS"] = "1"

    # 2. Force Market Closed
    os.environ["FT_FORCE_MARKET_CLOSED"] = "1"

    config = {
        "icicibreeze": {"live_trading": {"enabled": True}},
        "options": {"key": "test", "secret": "test", "session_token": "test"},
        "pair_whitelist": ["RELIANCE/INR"],
    }

    exchange = BreezeCCXT(config)
    # Mock ticker
    exchange.fetch_ticker = lambda symbol, params=None: {
        "symbol": symbol,
        "last": 2500.0,
        "bid": 2499.0,
        "ask": 2501.0,
    }
    # Mock SDK just in case (should not be reached)
    exchange.breeze = MagicMock()

    try:
        exchange.create_order("RELIANCE/INR", "limit", "buy", 1, 2500.0)
        print("ERROR: Order was Allowed! (Should be blocked by Market Hours)")
        sys.exit(1)
    except Exception as e:
        msg = str(e)

        # Check for expected block first logic
        if "market_closed" in msg and "blocking entry" in msg:
            # P19/P30 Requirement: Log expected block cleanly without traceback
            logger.info(f"EXPECTED_BLOCK: {msg}")
            print("P30_NEG_EXPECTED_BLOCK")
            print("[OK] Blocked by Market Hours despite Live Enablement.")
            sys.exit(0)

        # Unexpected error: Log with traceback
        logger.error(f"Caught unexpected exception: {msg}", exc_info=True)
        print(f"ERROR: Unexpected exception: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    check_p30_neg()
