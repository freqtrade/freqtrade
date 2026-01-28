import logging
import os
import sys
import time

# Add current directory to path so we can import adapters
sys.path.append(os.getcwd())

from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
from freqtrade.exceptions import OperationalException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_markets")


def run_smoke_markets():
    logger.info("Starting Market Data Smoke Test")

    # Check if credentials are present (mocked or real)
    # We use mock keys for structure validation if SDK allows,
    # or expects real ones for network calls.
    # In real mode testing, User should provide valid ENV.
    api_key = os.environ.get("BREEZE_API_KEY", "mock_key")
    api_secret = os.environ.get("BREEZE_API_SECRET", "mock_secret")
    session_token = os.environ.get("BREEZE_SESSION_TOKEN", "mock_token")

    config = {
        "key": api_key,
        "secret": api_secret,
        "password": session_token,
        "pair_whitelist": ["RELIANCE/INR"],  # For SecurityMaster if fetch_markets is called
    }

    exchange = BreezeCCXT(config)

    # 1. Test fetch_ticker
    logger.info("Testing fetch_ticker('RELIANCE/INR')")
    try:
        ticker = exchange.fetch_ticker("RELIANCE/INR")
        logger.info(f"PASSED: Ticker received: {ticker['symbol']} Last: {ticker['last']}")
        # Validate format
        required_keys = ["symbol", "timestamp", "datetime", "last", "close"]
        for k in required_keys:
            if k not in ticker:
                logger.error(f"FAILED: Ticker missing key: {k}")
                return
    except OperationalException as e:
        logger.warning(f"Note: fetch_ticker failed (expected if mock credentials used): {e}")
    except Exception:
        logger.exception("P19: Failed: Unexpected error in fetch_ticker")
        return

    # 2. Test fetch_ohlcv
    logger.info("Testing fetch_ohlcv('RELIANCE/INR', '5m')")
    try:
        ohlcv = exchange.fetch_ohlcv("RELIANCE/INR", "5m", limit=10)
        logger.info(f"PASSED: OHLCV received: {len(ohlcv)} candles")
        if ohlcv:
            # Validate format: [ms, o, h, l, c, v]
            candle = ohlcv[0]
            if not isinstance(candle, list) or len(candle) != 6:
                logger.error(f"FAILED: Invalid candle format: {candle}")
                return
            logger.info(f"First candle: {candle}")
    except OperationalException as e:
        logger.warning(f"Note: fetch_ohlcv failed (expected if mock credentials used): {e}")
    except Exception:
        logger.exception("P19: Failed: Unexpected error in fetch_ohlcv")
        return

    logger.info("SMOKE TEST FINISHED")


if __name__ == "__main__":
    run_smoke_markets()
