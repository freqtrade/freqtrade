import ccxt
import logging
import sys
import time
import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import freqtrade.exchange.icicibreeze
# This import is needed to register the shim
try:
    import freqtrade.exchange.icicibreeze
except ImportError as e:
    logger.error(f"Could not import freqtrade.exchange.icicibreeze: {e}")
    sys.exit(1)

try:
    logger.info("Initializing Exchange...")
    ex = ccxt.icicibreeze({"enableRateLimit": True})

    logger.info("Loading markets...")
    ex.load_markets()

    logger.info("Fetching Ticker for BTC/USDT...")
    t = ex.fetchTicker("BTC/USDT")

    logger.info("\nTicker Result:")
    logger.info(pprint.pformat(t))

    assert isinstance(t, dict), "Ticker must be a dictionary"
    assert t["symbol"] == "BTC/USDT", f"Symbol mismatch: {t.get('symbol')}"
    assert "last" in t, "Missing 'last' price"
    assert "timestamp" in t, "Missing 'timestamp'"
    assert "datetime" in t, "Missing 'datetime'"
    assert "bid" in t, "Missing 'bid'"
    assert "ask" in t, "Missing 'ask'"

    logger.info("\nSUCCESS: fetchTicker verified.")

except Exception:
    logger.exception("P19: Failed to verify fetchTicker")
    sys.exit(1)
