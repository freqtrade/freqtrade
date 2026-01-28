import ccxt
import logging
import sys
import pprint
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import freqtrade.exchange.icicibreeze
try:
    import freqtrade.exchange.icicibreeze
except ImportError as e:
    logger.error(f"Could not import freqtrade.exchange.icicibreeze: {e}")
    sys.exit(1)

try:
    logger.info("Initializing Exchange...")
    ex = ccxt.icicibreeze({"enableRateLimit": True})
    ex.load_markets()

    logger.info("\nFetching OHLCV for BTC/USDT (limit=5, timeframe='1m')...")
    # Using '1m' to verify it works, although we also support '5m'
    ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=5)

    logger.info("OHLCV Result:")
    logger.info(pprint.pformat(ohlcv))

    assert isinstance(ohlcv, list), "OHLCV must be a list"
    assert len(ohlcv) == 5, f"Expected 5 candles, got {len(ohlcv)}"

    first_candle = ohlcv[0]
    assert len(first_candle) == 6, "Candle must have 6 elements [ms, o, h, l, c, v]"

    # Verify timestamps are sequential
    ts0 = ohlcv[0][0]
    ts1 = ohlcv[1][0]
    diff = ts1 - ts0
    logger.info(f"\nTime difference between candles: {diff}ms")
    assert diff == 60000, f"Expected 60000ms (1m) difference, got {diff}"

    logger.info("\nSUCCESS: fetchOHLCV verified.")

except Exception:
    logger.exception("P19: Failed to verify fetchOHLCV")
    sys.exit(1)
