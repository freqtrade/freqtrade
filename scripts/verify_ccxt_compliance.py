import ccxt
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import freqtrade.exchange.icicibreeze
# This import is needed to register the shim
try:
    import freqtrade.exchange.icicibreeze
except ImportError:
    logger.error("Could not import freqtrade.exchange.icicibreeze")
    sys.exit(1)

logger.info(f"ccxt.icicibreeze present: {hasattr(ccxt, 'icicibreeze')}")

try:
    ex = ccxt.icicibreeze({"enableRateLimit": True})
    mk = ex.load_markets()

    logger.info(f"Markets count: {len(mk)}")
    logger.info(f"Symbols count: {len(ex.symbols) if ex.symbols else 0}")

    has_market = "BTC/USDT" in mk
    logger.info(f"has BTC/USDT in markets: {has_market}")

    has_symbol = "BTC/USDT" in ex.symbols if ex.symbols else False
    logger.info(f"symbols contains BTC/USDT: {has_symbol}")

    if has_market:
        logger.info(f"market: {ex.market('BTC/USDT')}")

        m = ex.market("BTC/USDT")
        logger.info(f"market active: {m.get('active')}")
        logger.info(f"market base: {m.get('base')}")
        logger.info(f"market quote: {m.get('quote')}")

except Exception:
    logger.exception("P19: Verification failed")
    sys.exit(1)
