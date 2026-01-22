"""
ICICI Breeze exchange integration.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

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

# --- Configuration Patching ---
# Ensure resolver finds our class
MAP_EXCHANGE_CHILDCLASS["icicibreeze"] = "Icicibreeze"
MAP_EXCHANGE_CHILDCLASS["IciciBreeze"] = "IciciBreeze"

# --- CCXT Patching start ---
if "icicibreeze" not in ccxt.exchanges:
    ccxt.exchanges.append("icicibreeze")


class IcicibreezeShim(ccxt.Exchange):
    """
    CCXT Shim for Icicibreeze (Sync).
    """

    @property
    def features(self):
        return {"spot": {"fetchOHLCV": {"limit": 1000, "days": 100}}}

    @features.setter
    def features(self, value):
        pass

    def __init__(self, config={}):
        # Check both top-level and ccxt_config used by Freqtrade
        mode = config.get("icici_mode") or config.get("ccxt_config", {}).get("icici_mode") or "stub"
        print(f"DEBUG: IcicibreezeShim initialized. Mode: {mode}")
        if mode == "real":
            raise ccxt.ConfigurationError("Real ICICI mode is not yet implemented. Use 'stub'.")
        super().__init__(config)

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

    def load_markets(self, reload=False, params={}):
        print("DEBUG: IcicibreezeShim.load_markets override ENTERED")
        markets = super().load_markets(reload, params)
        for _, market in markets.items():
            market["active"] = True
        self.markets = markets
        # Ensure currencies are set (CCXT usually does this, but we force it for completeness)
        if hasattr(self, "currencies") and self.currencies:
            self.currencies["INR"] = {"id": "INR", "code": "INR"}
        else:
            self.currencies = {
                "INR": {"id": "INR", "code": "INR"},
                "USDT": {"id": "USDT", "code": "USDT"},
                "BTC": {"id": "BTC", "code": "BTC"},
            }

        print(f"DEBUG: Manually set active=True on {len(markets)} markets")
        return markets

    def fetch_markets(self, params={}):
        print("DEBUG: fetch_markets called")
        return [
            {
                "symbol": "BTC/USDT",
                "id": "BTC-USDT",
                "base": "BTC",
                "quote": "USDT",
                "spot": True,
                "margin": False,
                "future": False,
                "precision": {"amount": 6, "price": 2},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1}},
                "active": True,
                "info": {},
            },
            {
                "symbol": "ETH/USDT",
                "id": "ETH-USDT",
                "base": "ETH",
                "quote": "USDT",
                "spot": True,
                "margin": False,
                "future": False,
                "precision": {"amount": 6, "price": 2},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1}},
                "active": True,
                "info": {},
            },
            {
                "symbol": "RELIANCE/INR",
                "id": "RELIANCE-INR",
                "base": "RELIANCE",
                "quote": "INR",
                "spot": True,
                "margin": False,
                "future": False,
                "precision": {"amount": 0, "price": 2},
                "limits": {"amount": {"min": 1}, "cost": {"min": 1}},
                "active": True,
                "info": {},
            },
            {
                "symbol": "TCS/INR",
                "id": "TCS-INR",
                "base": "TCS",
                "quote": "INR",
                "spot": True,
                "margin": False,
                "future": False,
                "precision": {"amount": 0, "price": 2},
                "limits": {"amount": {"min": 1}, "cost": {"min": 1}},
                "active": True,
                "info": {},
            },
        ]

    def fetch_ticker(self, symbol, params={}):
        market = self.market(symbol)
        return {
            "symbol": market["symbol"],
            "timestamp": 1672531200000,
            "datetime": "2023-01-01T00:00:00Z",
            "high": 50000.0,
            "low": 49000.0,
            "bid": 49500.0,
            "bidVolume": 1.0,
            "ask": 49501.0,
            "askVolume": 1.0,
            "vwap": 49500.0,
            "open": 49000.0,
            "close": 49500.0,
            "last": 49500.0,
            "previousClose": 49000.0,
            "change": 500.0,
            "percentage": 1.0,
            "average": 49250.0,
            "baseVolume": 100.0,
            "quoteVolume": 4950000.0,
            "info": {"stub": True},
        }

    def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=None, params={}):
        # Default limits
        if limit is None:
            limit = 100

        # Duration in milliseconds
        duration = self.parse_timeframe(timeframe) * 1000

        current_time = self.milliseconds()

        if since is None:
            # If since is not provided, generate 'limit' candles ending now
            since = current_time - (limit * duration)

        data = []

        for i in range(limit):
            timestamp = since + (i * duration)
            if timestamp > current_time:
                break

            # Deterministic stub data
            # Just some oscillation around 50000
            price_base = 50000.0 + (i % 100) * 10

            open_p = price_base
            high_p = price_base + 100
            low_p = price_base - 100
            close_p = price_base + 50
            volume = 100.0 + (i % 10)

            data.append([timestamp, open_p, high_p, low_p, close_p, volume])

        return data

    def fetch_balance(self, params={}):
        return {
            "free": {"USDT": 10000.0, "BTC": 1.0},
            "used": {"USDT": 0.0, "BTC": 0.0},
            "total": {"USDT": 10000.0, "BTC": 1.0},
            "info": {},
        }

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        return {
            "id": "123456_dry",
            "info": {},
            "symbol": symbol,
            "type": type,
            "side": side,
            "status": "open",
            "amount": amount,
            "price": price,
        }

    def cancel_order(self, id, symbol=None, params={}):
        return {"id": id, "status": "canceled"}

    def fetch_order(self, id, symbol=None, params={}):
        return {
            "id": id,
            "info": {},
            "symbol": symbol or "BTC/USDT",
            "type": "limit",
            "side": "buy",
            "status": "closed",
        }

    def fetch_open_orders(self, symbol=None, since=None, limit=None, params={}):
        return []

    def fetch_closed_orders(self, symbol=None, since=None, limit=None, params={}):
        return []

    def fetch_my_trades(self, symbol=None, since=None, limit=None, params={}):
        return []


# Force overwrite to ensure correct class is used
setattr(ccxt, "icicibreeze", IcicibreezeShim)


class IcicibreezeAsyncShim(ccxt_async.Exchange):
    """
    CCXT Shim for Icicibreeze (Async).
    """

    def __init__(self, config={}):
        mode = config.get("icici_mode") or config.get("ccxt_config", {}).get("icici_mode") or "stub"
        if mode == "real":
            raise ccxt_async.ConfigurationError(
                "Real ICICI mode is not yet implemented. Use 'stub'."
            )
        super().__init__(config)

    @property
    def features(self):
        return {"spot": {"fetchOHLCV": {"limit": 1000, "days": 100}}}

    @features.setter
    def features(self, value):
        pass

    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "id": "icicibreeze",
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

    async def load_markets(self, reload=False, params={}):
        self.markets = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "id": "BTC-USDT",
                "base": "BTC",
                "quote": "USDT",
                "spot": True,
                "margin": False,
                "future": False,
                "precision": {"amount": 6, "price": 2},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1}},
                "active": True,
                "info": {},
            },
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "id": "ETH-USDT",
                "base": "ETH",
                "quote": "USDT",
                "spot": True,
                "margin": False,
                "future": False,
                "precision": {"amount": 6, "price": 2},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1}},
                "active": True,
                "info": {},
            },
            "RELIANCE/INR": {
                "symbol": "RELIANCE/INR",
                "id": "RELIANCE-INR",
                "base": "RELIANCE",
                "quote": "INR",
                "spot": True,
                "margin": False,
                "future": False,
                "precision": {"amount": 0, "price": 2},
                "limits": {"amount": {"min": 1}, "cost": {"min": 1}},
                "active": True,
                "info": {},
            },
            "TCS/INR": {
                "symbol": "TCS/INR",
                "id": "TCS-INR",
                "base": "TCS",
                "quote": "INR",
                "spot": True,
                "margin": False,
                "future": False,
                "precision": {"amount": 0, "price": 2},
                "limits": {"amount": {"min": 1}, "cost": {"min": 1}},
                "active": True,
                "info": {},
            },
        }

        # Ensure currencies are set (CCXT usually does this, but we force it for completeness)
        if hasattr(self, "currencies") and self.currencies:
            self.currencies["INR"] = {"id": "INR", "code": "INR"}
        else:
            self.currencies = {
                "INR": {"id": "INR", "code": "INR"},
                "USDT": {"id": "USDT", "code": "USDT"},
                "BTC": {"id": "BTC", "code": "BTC"},
            }

        return self.markets

    async def fetch_ticker(self, symbol, params={}):
        market = self.market(symbol)
        return {
            "symbol": market["symbol"],
            "timestamp": 1672531200000,
            "datetime": "2023-01-01T00:00:00Z",
            "high": 50000.0,
            "low": 49000.0,
            "bid": 49500.0,
            "bidVolume": 1.0,
            "ask": 49501.0,
            "askVolume": 1.0,
            "vwap": 49500.0,
            "open": 49000.0,
            "close": 49500.0,
            "last": 49500.0,
            "previousClose": 49000.0,
            "change": 500.0,
            "percentage": 1.0,
            "average": 49250.0,
            "baseVolume": 100.0,
            "quoteVolume": 4950000.0,
            "info": {"stub": True},
        }

    async def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=None, params={}):
        # Default limits
        if limit is None:
            limit = 100

        # Duration in milliseconds
        duration = self.parse_timeframe(timeframe) * 1000

        current_time = self.milliseconds()

        if since is None:
            # If since is not provided, generate 'limit' candles ending now
            since = current_time - (limit * duration)

        data = []

        for i in range(limit):
            timestamp = since + (i * duration)
            if timestamp > current_time:
                break

            # Deterministic stub data
            # Just some oscillation around 50000
            price_base = 50000.0 + (i % 100) * 10

            open_p = price_base
            high_p = price_base + 100
            low_p = price_base - 100
            close_p = price_base + 50
            volume = 100.0 + (i % 10)

            data.append([timestamp, open_p, high_p, low_p, close_p, volume])

        return data

    async def fetch_balance(self, params={}):
        return {
            "free": {"USDT": 10000.0, "BTC": 1.0},
            "used": {"USDT": 0.0, "BTC": 0.0},
            "total": {"USDT": 10000.0, "BTC": 1.0},
            "info": {},
        }

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        return {
            "id": "123456_dry",
            "info": {},
            "symbol": symbol,
            "type": type,
            "side": side,
            "status": "open",
            "amount": amount,
            "price": price,
        }

    async def cancel_order(self, id, symbol=None, params={}):
        return {"id": id, "status": "canceled"}

    async def fetch_order(self, id, symbol=None, params={}):
        return {
            "id": id,
            "info": {},
            "symbol": symbol or "BTC/USDT",
            "type": "limit",
            "side": "buy",
            "status": "closed",
        }

    async def fetch_open_orders(self, symbol=None, since=None, limit=None, params={}):
        return []

    async def fetch_closed_orders(self, symbol=None, since=None, limit=None, params={}):
        return []

    async def fetch_my_trades(self, symbol=None, since=None, limit=None, params={}):
        return []

    async def close(self):
        pass


if not hasattr(ccxt_async, "icicibreeze"):
    setattr(ccxt_async, "icicibreeze", IcicibreezeAsyncShim)

if ccxt_pro and not hasattr(ccxt_pro, "icicibreeze"):
    setattr(ccxt_pro, "icicibreeze", IcicibreezeAsyncShim)


class _BreezeCCXTAsync:
    """
    Async wrapper for BreezeCCXT sync implementation (Live Mode).
    """

    def __init__(self, sync_api):
        self._sync_api = sync_api
        self.has = sync_api.has if hasattr(sync_api, "has") else {}
        self.id = sync_api.id if hasattr(sync_api, "id") else "icicibreeze"
        self.name = sync_api.name if hasattr(sync_api, "name") else "ICICI Breeze"
        self.timeframes = sync_api.timeframes if hasattr(sync_api, "timeframes") else {}
        self.session = None

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
        self, exchange_config: Dict[str, Any], sync: bool, ccxt_kwargs: Dict[str, Any]
    ) -> Any:
        # 1. Check Dry Run
        if self._config.get("dry_run", False):
            if sync:
                logger.info("Initializing Icicibreeze in Dry Run mode (using stub).")
                return IcicibreezeShim(exchange_config)
            return IcicibreezeAsyncShim(exchange_config)

        # 2. Live Mode - Attempt imports
        try:
            from trade_bot.adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
        except ImportError as e:
            raise OperationalException(
                "trade_bot is not installed in this venv. Run: cd ~/work/trade-bot && python -m pip install -e ."
            ) from e

        # 3. Validate Config (Live)
        api_key = exchange_config.get("key")
        secret_key = exchange_config.get("secret")

        if not api_key or not secret_key:
            raise OperationalException(
                "Icicibreeze logic require 'key' and 'secret' in config for live trading."
            )

        # 4. Instantiate Shim (Live)
        try:
            breeze_ccxt = BreezeCCXT(config=exchange_config)
        except Exception as e:
            raise OperationalException(f"Failed to initialize BreezeCCXT: {e}") from e

        # 5. Return correct instance based on sync/async request
        if sync:
            return breeze_ccxt

        return _BreezeCCXTAsync(breeze_ccxt)

    def fetch_ticker(self, pair: str):
        return super().fetch_ticker(pair)


IciciBreeze = Icicibreeze
