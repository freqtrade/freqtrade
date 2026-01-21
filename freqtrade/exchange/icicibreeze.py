"""
ICICI Breeze exchange integration.
"""

import asyncio
import logging
from typing import Any

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange import Exchange

logger = logging.getLogger(__name__)


class _BreezeCCXTAsync:
    """
    Async wrapper for BreezeCCXT sync implementation.
    """

    def __init__(self, sync_api):
        self._sync_api = sync_api
        self.has = sync_api.has if hasattr(sync_api, "has") else {}
        self.id = sync_api.id if hasattr(sync_api, "id") else "icicibreeze"
        self.name = sync_api.name if hasattr(sync_api, "name") else "ICICI Breeze"
        self.timeframes = sync_api.timeframes if hasattr(sync_api, "timeframes") else {}
        # Mimic common ccxt async properties
        self.session = None  # No persistent session to close in this simplified wrapper

    async def close(self):
        pass

    async def fetch_ohlcv(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_ohlcv, *args, **kwargs)

    async def load_markets(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.load_markets, *args, **kwargs)

    async def fetch_ticker(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_ticker, *args, **kwargs)

    async def fetch_tickers(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_tickers, *args, **kwargs)

    async def create_order(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.create_order, *args, **kwargs)

    async def cancel_order(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.cancel_order, *args, **kwargs)

    async def fetch_order(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_order, *args, **kwargs)

    async def fetch_open_orders(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_open_orders, *args, **kwargs)

    async def fetch_closed_orders(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_closed_orders, *args, **kwargs)

    async def fetch_my_trades(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_my_trades, *args, **kwargs)

    async def fetch_balance(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_balance, *args, **kwargs)

    async def fetch_positions(self, *args, **kwargs):
        return await asyncio.to_thread(self._sync_api.fetch_positions, *args, **kwargs)

    def __getattr__(self, name):
        # Fallback for other methods not explicitly wrapped (careful with async/sync mismatch)
        return getattr(self._sync_api, name)


class Icicibreeze(Exchange):
    """
    ICICI Breeze exchange integration.
    """

    _params: dict = {"name": "icicibreeze"}
    _ft_has: dict = {
        "ohlcv_has_history": True,
        "mark_ohlcv_price": "close",
        "fetch_config": True,
    }

    def _init_ccxt(
        self, exchange_config: dict[str, Any], sync: bool, ccxt_kwargs: dict[str, Any]
    ) -> Any:
        """
        Initialize BreezeCCXT shim instead of standard CCXT.
        """
        # We only support sync implementation wrapped in async, or strict async shim.
        # Since BreezeCCXT is likely sync, we'll implement the async wrapper below.

        # 1. Attempt imports
        try:
            from trade_bot.adapters.breeze_rest_adapter import BreezeRestAdapter

            try:
                # Try new location first (if scaffolded)
                from trade_bot.adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
            except ImportError:
                # Fallback to direct import if shim is not yet moved
                from trade_bot.adapters.ccxt_shim import BreezeCCXT

            from trade_bot.services.ohlcv_service import OhlcvService
            from trade_bot.services.security_master_service import SecurityMasterService
        except ImportError as e:
            raise OperationalException(
                f"Failed to import trade_bot dependencies for Icicibreeze: {e}. "
                "Ensure trade_bot is installed/available in the environment."
            ) from e

        # 2. Validate Config
        api_key = exchange_config.get("key")
        secret_key = exchange_config.get("secret")
        session_token = exchange_config.get("password")  # Mapping password to session_token

        if not api_key or not secret_key or not session_token:
            raise OperationalException(
                "Icicibreeze require 'key', 'secret', and 'password' (session_token) in config."
            )

        # 3. Instantiate Services (Manual DI)
        try:
            # TODO: Ideally use trade_bot's composition root/container if available
            rest_adapter = BreezeRestAdapter(
                api_key=api_key, secret_key=secret_key, session_token=session_token
            )

            # Assuming these services might take the adapter or need other config
            # Adjusting instantiation based on likely signature
            sec_master = SecurityMasterService(rest_adapter)
            ohlcv_service = OhlcvService(rest_adapter, sec_master)

            # Instantiate Shim
            breeze_ccxt = BreezeCCXT(
                rest_adapter=rest_adapter, security_master=sec_master, ohlcv_service=ohlcv_service
            )

        except Exception as e:
            raise OperationalException(f"Failed to initialize BreezeCCXT components: {e}") from e

        # 4. Return correct instance based on sync/async request
        if sync:
            return breeze_ccxt

        # 5. Create Async Wrapper
        return _BreezeCCXTAsync(breeze_ccxt)
