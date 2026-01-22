import ccxt
import sys
import os

# Ensure we can import freqtrade.exchange.icicibreeze
# This import is needed to register the shim
try:
    import freqtrade.exchange.icicibreeze
except ImportError:
    print("Could not import freqtrade.exchange.icicibreeze")
    sys.exit(1)

print("ccxt.icicibreeze present:", hasattr(ccxt, "icicibreeze"))

try:
    ex = ccxt.icicibreeze({"enableRateLimit": True})
    mk = ex.load_markets()

    print(f"Markets count: {len(mk)}")
    print(f"Symbols count: {len(ex.symbols) if ex.symbols else 0}")

    has_market = "BTC/USDT" in mk
    print("has BTC/USDT in markets:", has_market)

    has_symbol = "BTC/USDT" in ex.symbols if ex.symbols else False
    print("symbols contains BTC/USDT:", has_symbol)

    if has_market:
        print("market:", ex.market("BTC/USDT"))

        m = ex.market("BTC/USDT")
        print("market active:", m.get("active"))
        print("market base:", m.get("base"))
        print("market quote:", m.get("quote"))

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
