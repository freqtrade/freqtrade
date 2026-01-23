import asyncio
import logging
import os
import time
from collections import deque
from typing import Any, Optional

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


class BreezeRateLimiter:
    """
    Internal rate limiter for Breeze API.
    Limits: 100 requests / minute, 5000 requests / day.
    """

    def __init__(self):
        self.minute_requests = deque()
        self.day_requests = deque()
        self.limit_min = 100
        self.limit_day = 5000

    def check_and_record(self):
        now = time.time()

        # Cleanup old requests
        while self.minute_requests and self.minute_requests[0] < now - 60:
            self.minute_requests.popleft()
        while self.day_requests and self.day_requests[0] < now - 86400:
            self.day_requests.popleft()

        if len(self.minute_requests) >= self.limit_min:
            wait_time = 60 - (now - self.minute_requests[0])
            raise OperationalException(
                f"Breeze Rate Limit exceeded (100/min). Retrying in {wait_time:.1f}s"
            )

        if len(self.day_requests) >= self.limit_day:
            raise OperationalException("Breeze Rate Limit exceeded (5000/day).")

        self.minute_requests.append(now)
        self.day_requests.append(now)


class BreezeCCXT(ccxt.Exchange):
    """
    Sync CCXT Shim for Breeze Connect SDK.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        if config is None:
            config = {}
        super().__init__(config)
        self.config = config
        self.name = "IciciBreeze"
        self.rate_limiter = BreezeRateLimiter()

        # 1. Resolve Credentials (ENV > Config)
        api_key = os.environ.get("BREEZE_API_KEY") or config.get("key")
        api_secret = os.environ.get("BREEZE_API_SECRET") or config.get("secret")
        session_token = os.environ.get("BREEZE_SESSION_TOKEN") or config.get("password")

        if not api_key:
            self.breeze = None
            return

        self.breeze = BreezeConnect(api_key=api_key)

        if api_secret and session_token:
            try:
                logger.info("Initializing Breeze session with provided credentials.")
                self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
            except Exception as e:
                logger.error(f"Failed to initialize Breeze session: {e}")

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
                "timeframes": {
                    "1m": "1minute",
                    "5m": "5minute",
                    "30m": "30minute",
                    "1d": "1day",
                },
            },
        )

    def _parse_symbol(self, symbol: str) -> dict:
        """
        Map RELIANCE/INR -> stock_code="RELIANCE", exchange_code="NSE", product_type="cash"
        """
        parts = symbol.split("/")
        if len(parts) < 2:
            raise OperationalException(f"Invalid symbol format: {symbol}")

        base = parts[0]
        # For Prompt 6.5, we focus on Cash Equity (NSE)
        return {"stock_code": base, "exchange_code": "NSE", "product_type": "cash"}

    def fetch_markets(self, params: Optional[dict] = None):
        if self.rate_limiter:
            self.rate_limiter.check_and_record()
        master_file = find_latest_master_file()
        if not master_file:
            logger.warning("SecurityMaster file not found.")
            return []

        master = load_nfo_options_master(master_file)
        if not master["by_contract"]:
            return []

        whitelist = self.config.get("pair_whitelist", [])
        if not whitelist:
            return []

        specs = parse_pair_whitelist_for_options(whitelist)
        resolved, _ = resolve_underlying(specs, master)

        markets = []
        by_contract = master["by_contract"]
        for spec in resolved:
            key = (spec["underlying"], spec["expiry"], spec["strike"], spec["right"])
            if key in by_contract:
                info = by_contract[key]
                symbol = (
                    f"{info['underlying']}/INR:{info['expiry']}:{info['strike']}:{info['right']}"
                )
                markets.append(
                    {
                        "id": info["token"],
                        "symbol": symbol,
                        "base": info["underlying"],
                        "quote": "INR",
                        "active": True,
                        "type": "option",
                        "option": True,
                        "expiry": info["expiry"],
                        "strike": info["strike"],
                        "right": info["right"],
                        "lot": info["lot_size"],
                        "precision": {"amount": 1, "price": info["tick_size"]},
                        "info": info,
                    }
                )
        return markets

    def fetch_ticker(self, symbol: str, params: Optional[dict] = None):
        if not self.breeze:
            raise OperationalException("Breeze session not initialized.")

        self.rate_limiter.check_and_record()
        s_params = self._parse_symbol(symbol)

        try:
            resp = self.breeze.get_quotes(
                stock_code=s_params["stock_code"],
                exchange_code=s_params["exchange_code"],
                product_type=s_params["product_type"],
            )

            if resp is None or resp.get("status") != 200 or not resp.get("Success"):
                err_msg = resp.get("error") if resp else "Empty response from SDK"
                raise OperationalException(f"Breeze fetch_ticker failed: {err_msg}")

            data = resp["Success"][0]
            last = float(data["ltp"])
            timestamp = self.milliseconds()

            return {
                "symbol": symbol,
                "timestamp": timestamp,
                "datetime": self.iso8601(timestamp),
                "high": float(data.get("high", last)),
                "low": float(data.get("low", last)),
                "bid": float(data.get("best_bid_price", last)),
                "ask": float(data.get("best_ask_price", last)),
                "last": last,
                "close": last,
                "info": data,
            }
        except Exception as e:
            raise OperationalException(f"Error in fetch_ticker: {e}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None,
    ):
        if not self.breeze:
            raise OperationalException("Breeze session not initialized.")

        self.rate_limiter.check_and_record()
        s_params = self._parse_symbol(symbol)

        interval = self.timeframes.get(timeframe)
        if not interval:
            raise OperationalException(f"Unsupported timeframe: {timeframe}")

        # limit logic
        if limit is None:
            limit = 100
        limit = min(limit, 1000)

        # from/to date logic
        if since:
            from_date = self.iso8601(since).split(".")[0]  # Breeze wants YYYY-MM-DDTHH:mm:ss
        else:
            # Default to some time in the past based on limit and timeframe
            # Simplified for now: last 2 days
            from_date = self.iso8601(self.milliseconds() - 2 * 86400 * 1000).split(".")[0]

        to_date = self.iso8601(self.milliseconds()).split(".")[0]

        try:
            resp = self.breeze.get_historical_data_v2(
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                stock_code=s_params["stock_code"],
                exchange_code=s_params["exchange_code"],
                product_type=s_params["product_type"],
            )

            if resp is None:
                raise OperationalException("Breeze fetch_ohlcv failed: Empty response from SDK")

            if resp.get("status") != 200 or not resp.get("Success"):
                # Breeze returns empty Success if no data found
                if resp.get("status") == 200 and not resp.get("Success"):
                    return []
                raise OperationalException(f"Breeze fetch_ohlcv failed: {resp.get('error')}")

            ohlcv = []
            for d in resp["Success"]:
                ohlcv.append(
                    [
                        self.parse8601(d["datetime"]),
                        float(d["open"]),
                        float(d["high"]),
                        float(d["low"]),
                        float(d["close"]),
                        float(d["volume"]),
                    ]
                )

            # Sort and apply limit
            ohlcv.sort(key=lambda x: x[0])
            return ohlcv[-limit:] if limit else ohlcv

        except Exception as e:
            raise OperationalException(f"Error in fetch_ohlcv: {e}")

    def create_order(self, symbol, type, side, amount, price=None, params: Optional[dict] = None):
        raise OperationalException("Orders not implemented in p06")

    def cancel_order(self, id, symbol=None, params: Optional[dict] = None):
        raise OperationalException("Orders not implemented in p06")

    def fetch_order(self, id, symbol=None, params: Optional[dict] = None):
        raise OperationalException("Orders not implemented in p06")


class BreezeAsyncCCXT(ccxt_async.Exchange):
    """
    Async CCXT Shim for Breeze Connect SDK.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        if config is None:
            config = {}
        super().__init__(config)
        self.config = config
        self.name = "IciciBreeze"
        self.sync_exchange = BreezeCCXT(config)

    def describe(self):
        return self.sync_exchange.describe()

    async def fetch_markets(self, params: Optional[dict] = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_markets, params)

    async def fetch_ticker(self, symbol: str, params: Optional[dict] = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_ticker, symbol, params)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None,
    ):
        return await asyncio.to_thread(
            self.sync_exchange.fetch_ohlcv, symbol, timeframe, since, limit, params
        )

    async def create_order(
        self, symbol, order_type, side, amount, price=None, params: dict | None = None
    ):
        raise OperationalException("Orders not implemented in p06")

    async def cancel_order(self, order_id, symbol=None, params: dict | None = None):
        raise OperationalException("Orders not implemented in p06")

    async def fetch_order(self, order_id, symbol=None, params: dict | None = None):
        raise OperationalException("Orders not implemented in p06")
