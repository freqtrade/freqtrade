"""
ICICI Breeze exchange integration.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import ccxt
import ccxt.async_support as ccxt_async
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.common import MAP_EXCHANGE_CHILDCLASS
from freqtrade.exchange.exchange import Exchange

try:
    import ccxt.pro as ccxt_pro
except ImportError:
    ccxt_pro = None

logger = logging.getLogger(__name__)

# --- CCXT Patching ---
# This ensures that even if Freqtrade falls back to the generic Exchange class,
# it can still find our shim under ccxt.icicibreeze.


def patch_ccxt():
    from adapters.ccxt_shim.breeze_ccxt import BreezeAsyncCCXT, BreezeCCXT

    if "icicibreeze" not in ccxt.exchanges:
        ccxt.exchanges.append("icicibreeze")

    # We set it globally. Freqtrade's Exchange class will use these if it can't find a subclass.
    # We use a wrapper to handle the configuration-based mode selection if possible,
    # or just default to the real Shim for this p06 gate.

    setattr(ccxt, "icicibreeze", BreezeCCXT)
    setattr(ccxt_async, "icicibreeze", BreezeAsyncCCXT)
    if ccxt_pro:
        setattr(ccxt_pro, "icicibreeze", BreezeAsyncCCXT)


patch_ccxt()

# --- Freqtrade Exchange Class ---


class Icicibreeze(Exchange):
    """
    ICICI Breeze exchange integration for Freqtrade.
    """

    _params: dict = {"name": "icicibreeze"}
    _ft_has: dict = {
        "ohlcv_has_history": True,
        "mark_ohlcv_price": "close",
        "fetch_config": True,
    }

    def _init_ccxt(
        self, exchange_config: Dict[str, Any], sync: bool, ccxt_kwargs: Dict[str, Any]
    ) -> Any:
        # Determine Mode
        mode = self._config.get("icici_mode") or exchange_config.get("icici_mode") or "stub"

        if mode == "stub":
            logger.info("Initializing Icicibreeze in Stub mode.")
            # We would normally return a stub here, but for p06 we prioritize the real shim.
            # If the user specifically wants stub, we could keep the old shim,
            # but for now let's just use the patched ccxt.
            if sync:
                return ccxt.icicibreeze(exchange_config)
            return ccxt_async.icicibreeze(exchange_config)

        logger.info(f"Initializing Icicibreeze in Real mode. Sync={sync}")
        from adapters.ccxt_shim.breeze_ccxt import BreezeAsyncCCXT, BreezeCCXT

        # Inject necessary config for the shim to see key/secret/dry_run in self.options
        exchange_config["key"] = exchange_config.get("key")
        exchange_config["secret"] = exchange_config.get("secret")
        exchange_config["dry_run"] = self._config.get("dry_run")

        if sync:
            return BreezeCCXT(exchange_config)
        return BreezeAsyncCCXT(exchange_config)


# Register the class
MAP_EXCHANGE_CHILDCLASS["icicibreeze"] = "Icicibreeze"
IciciBreeze = Icicibreeze
