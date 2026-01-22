# flake8: noqa: F401
# isort: off
# --- ccxt shim registration for icicibreeze (must run early) ---
try:
    import ccxt  # type: ignore

    if "icicibreeze" not in getattr(ccxt, "exchanges", []):
        # ccxt.exchanges is a list of ids
        ccxt.exchanges.append("icicibreeze")

    if not hasattr(ccxt, "icicibreeze"):
        # Minimal ccxt exchange stub so freqtrade/ccxt checks pass.
        class icicibreeze(ccxt.Exchange):  # noqa: N801
            def describe(self):
                base = super().describe()
                base.update(
                    {
                        "id": "icicibreeze",
                        "name": "ICICI Breeze (Shim)",
                        "countries": ["IN"],
                        "rateLimit": 1000,
                        "has": {
                            "fetchMarkets": True,
                            "loadMarkets": True,
                            "fetchTicker": True,
                            "fetchOHLCV": True,
                            "createOrder": False,
                            "cancelOrder": False,
                        },
                    }
                )
                return base

        setattr(ccxt, "icicibreeze", icicibreeze)

except Exception:
    # Do not break import if ccxt is unavailable in some environments.
    pass
# --- end shim ---

from freqtrade.exchange.common import MAP_EXCHANGE_CHILDCLASS
from freqtrade.exchange.exchange import Exchange

# isort: on
from freqtrade.exchange.binance import Binance, Binanceus, Binanceusdm
from freqtrade.exchange.bingx import Bingx
from freqtrade.exchange.bitget import Bitget
from freqtrade.exchange.bitmart import Bitmart
from freqtrade.exchange.bitpanda import Bitpanda
from freqtrade.exchange.bitvavo import Bitvavo
from freqtrade.exchange.bybit import Bybit
from freqtrade.exchange.coinex import Coinex
from freqtrade.exchange.cryptocom import Cryptocom
from freqtrade.exchange.exchange_utils import (
    ROUND_DOWN,
    ROUND_UP,
    amount_to_contract_precision,
    amount_to_contracts,
    amount_to_precision,
    available_exchanges,
    ccxt_exchanges,
    contracts_to_amount,
    date_minus_candles,
    is_exchange_known_ccxt,
    list_available_exchanges,
    market_is_active,
    price_to_precision,
    validate_exchange,
)
from freqtrade.exchange.exchange_utils_timeframe import (
    timeframe_to_minutes,
    timeframe_to_msecs,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    timeframe_to_resample_freq,
    timeframe_to_seconds,
)
from freqtrade.exchange.gate import Gate
from freqtrade.exchange.hitbtc import Hitbtc
from freqtrade.exchange.htx import Htx
from freqtrade.exchange.hyperliquid import Hyperliquid
from freqtrade.exchange.icicibreeze import Icicibreeze, IciciBreeze
from freqtrade.exchange.idex import Idex
from freqtrade.exchange.kraken import Kraken
from freqtrade.exchange.kucoin import Kucoin
from freqtrade.exchange.lbank import Lbank
from freqtrade.exchange.luno import Luno
from freqtrade.exchange.modetrade import Modetrade
from freqtrade.exchange.okx import Myokx, Okx, Okxus

MAP_EXCHANGE_CHILDCLASS["icicibreeze"] = "Icicibreeze"
MAP_EXCHANGE_CHILDCLASS["IciciBreeze"] = "IciciBreeze"
