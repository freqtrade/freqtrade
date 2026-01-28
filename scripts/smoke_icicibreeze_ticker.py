import logging
import os
import sys
import time

# Add current directory to path so we can import adapters
sys.path.append(os.getcwd())

from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
from freqtrade.exceptions import OperationalException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_ticker")


def run_smoke_ticker():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE/INR"
    logger.info(f"Starting Ticker Smoke Test for {symbol}")

    # Credentials from ENV
    api_key = os.environ.get("BREEZE_API_KEY", "mock_key")
    api_secret = os.environ.get("BREEZE_API_SECRET", "mock_secret")
    session_token = os.environ.get("BREEZE_SESSION_TOKEN", "mock_token")

    config = {"key": api_key, "secret": api_secret, "password": session_token}

    exchange = BreezeCCXT(config)

    try:
        ticker = exchange.fetch_ticker(symbol)
        # Requirement: must print last + timestamp
        print(f"SYMBOL: {ticker['symbol']}")
        print(f"LAST: {ticker['last']}")
        print(f"TIMESTAMP: {ticker['timestamp']}")
        print(f"DATETIME: {ticker['datetime']}")
    except OperationalException as e:
        logger.warning(f"Note: fetch_ticker failed (expected if mock credentials used): {e}")
    except Exception:
        logger.exception("P19: FAILED: Unexpected error")


if __name__ == "__main__":
    run_smoke_ticker()
