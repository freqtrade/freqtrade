import logging
import json
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT

logging.basicConfig(level=logging.DEBUG)


def debug_fetch():
    with open("user_data/config_icicibreeze.json", "r") as f:
        config = json.load(f)

    exchange_config = config["exchange"]
    # Freqtrade typically passes the 'exchange' sub-dict to CCXT
    # but some fields are modified or added.

    print(f"Exchange Config Keys: {list(exchange_config.keys())}")
    print(f"Pair Whitelist: {exchange_config.get('pair_whitelist')}")

    exchange = BreezeCCXT(exchange_config)
    print(f"Exchange Name: {exchange.name}")
    print(f"Exchange Config in object: {list(exchange.config.keys())}")

    markets = exchange.fetch_markets()
    print(f"Fetched {len(markets)} markets.")
    for m in markets:
        print(f"  - {m['symbol']}")


if __name__ == "__main__":
    debug_fetch()
