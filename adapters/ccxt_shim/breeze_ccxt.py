import logging
from typing import Any, Dict

import ccxt
import ccxt.async_support as ccxt_async
from breeze_connect import BreezeConnect

from adapters.ccxt_shim.security_master import (
    find_latest_master_file,
    load_nfo_options_master,
    parse_pair_whitelist_for_options,
    resolve_underlying,
)
from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


class BreezeCCXT(ccxt.Exchange):
    """
    Sync CCXT Shim for Breeze Connect SDK.
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        super().__init__(config)
        self.config = config
        self.name = "IciciBreeze"
        self.breeze = BreezeConnect(api_key=config.get("key"))
        if config.get("secret"):
            # Setup session if secret is available (dummy session for now as we don't have full auth flow here)
            # The user provided skeleton typically expects generating session
            pass

    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "id": "icicibreeze",
                "name": "IciciBreeze",
                "countries": ["IN"],
                "rateLimit": 1000,
                "has": {
                    "fetchTicker": True,
                    "fetchOHLCV": True,
                    "fetchOrder": True,
                    "fetchOpenOrders": True,
                    "fetchClosedOrders": True,
                    "fetchMyTrades": True,
                    "fetchBalance": True,
                },
                "timeframes": {"1m": "1minute", "5m": "5minute"},
            },
        )

    def fetch_markets(self, params={}):
        """
        Fetch markets strictly from SecurityMaster based on whitelist.
        """
        master_file = find_latest_master_file()
        if not master_file:
            logger.warning(
                "SecurityMaster file (FONSEScripMaster.txt) not found. No markets loaded."
            )
            return []

        master = load_nfo_options_master(master_file)
        if not master["by_contract"]:
            logger.warning("SecurityMaster is empty or failed to parse. No markets loaded.")
            return []

        whitelist = self.config.get("pair_whitelist", [])
        if not whitelist:
            logger.info("Pair whitelist is empty. No markets to build.")
            return []

        specs = parse_pair_whitelist_for_options(whitelist)
        resolved, unresolved = resolve_underlying(specs, master)

        if unresolved:
            logger.warning(f"Could not resolve these pairs from SecurityMaster: {unresolved}")

        markets = []
        by_contract = master["by_contract"]
        for spec in resolved:
            key = (spec["underlying"], spec["expiry"], spec["strike"], spec["right"])
            if key in by_contract:
                info = by_contract[key]
                # Symbol convention: UNDERLYING/INR:YYYY-MM-DD:STRIKE:RIGHT
                symbol = (
                    f"{info['underlying']}/INR:{info['expiry']}:{info['strike']}:{info['right']}"
                )

                market = {
                    "id": info["token"],
                    "symbol": symbol,
                    "base": info["underlying"],
                    "quote": "INR",
                    "baseId": info["underlying"],
                    "quoteId": "INR",
                    "active": True,
                    "type": "option",
                    "option": True,
                    "expiry": info["expiry"],
                    "strike": info["strike"],
                    "right": info["right"],
                    "lot": info["lot_size"],
                    "precision": {
                        "amount": 1,
                        "price": info["tick_size"],
                    },
                    "limits": {
                        "amount": {"min": info["lot_size"], "max": None},
                        "price": {"min": info["tick_size"], "max": None},
                    },
                    "info": info,
                }
                markets.append(market)
            else:
                logger.warning(f"Contract not found in SecurityMaster: {spec['original']}")

        return markets

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        raise OperationalException("Orders not implemented in p06")

    def cancel_order(self, id, symbol=None, params={}):
        raise OperationalException("Orders not implemented in p06")

    def fetch_order(self, id, symbol=None, params={}):
        raise OperationalException("Orders not implemented in p06")


class BreezeAsyncCCXT(ccxt_async.Exchange):
    """
    Async CCXT Shim for Breeze Connect SDK.
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        super().__init__(config)
        self.config = config
        self.name = "IciciBreeze"
        self.breeze = BreezeConnect(api_key=config.get("key"))

    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "id": "icicibreeze",
                "name": "IciciBreeze",
                "countries": ["IN"],
                "rateLimit": 1000,
                "has": {
                    "fetchTicker": True,
                    "fetchOHLCV": True,
                    "fetchOrder": True,
                    "fetchOpenOrders": True,
                    "fetchClosedOrders": True,
                    "fetchMyTrades": True,
                    "fetchBalance": True,
                },
                "timeframes": {"1m": "1minute", "5m": "5minute"},
            },
        )

    async def fetch_markets(self, params={}):
        """
        Async version of fetch_markets.
        Since master file loading is local FS, we can reuse the sync logic.
        """
        # In a real async shim, we might want to offload I/O, but for this shim,
        # call the sync logic for now.
        return BreezeCCXT.fetch_markets(self, params)

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        raise OperationalException("Orders not implemented in p06")

    async def cancel_order(self, id, symbol=None, params={}):
        raise OperationalException("Orders not implemented in p06")

    async def fetch_order(self, id, symbol=None, params={}):
        raise OperationalException("Orders not implemented in p06")
