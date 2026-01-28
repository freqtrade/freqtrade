import ccxt
import logging
import sys
import pprint

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

    logger.info("Loading markets...")
    mk = ex.load_markets()

    logger.info(f"\nMarkets loaded: {len(mk)}")
    for sym in ["RELIANCE/INR", "TCS/INR", "BTC/USDT"]:
        logger.info(f"Market {sym} present: {sym in mk}")
        if sym in mk:
            logger.info(f"  Quote: {mk[sym]['quote']}")
            logger.info(f"  Active: {mk[sym].get('active')}")

    logger.info("\nCurrencies:")
    logger.info(ex.currencies.keys())

    assert "INR" in ex.currencies, "INR missing from currencies"
    assert "RELIANCE/INR" in mk, "RELIANCE/INR missing from markets"

    logger.info("\nSUCCESS: INR support verification passed.")

except Exception:
    logger.exception("P19: Verification failed")
    sys.exit(1)
