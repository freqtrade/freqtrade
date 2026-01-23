import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Any

import ccxt
import ccxt.async_support as ccxt_async
from breeze_connect import BreezeConnect

from adapters.ccxt_shim.security_master import (
    find_latest_master_file,
    load_nfo_options_master,
    load_nse_cash_master,
    parse_pair_whitelist_for_options,
    resolve_underlying,
)
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class InternalRateLimiter:
    def __init__(self, rpm: int = 100, rpd: int = 5000):
        self.rpm = rpm
        self.rpd = rpd
        self.history = deque()
        self.daily_count = 0
        self.last_reset = time.time()

    def check_and_record(self):
        now = time.time()
        if now - self.last_reset > 86400:
            self.daily_count = 0
            self.last_reset = now
        if self.daily_count >= self.rpd:
            raise OperationalException("Daily rate limit exceeded (5000)")
        while self.history and now - self.history[0] > 60:
            self.history.popleft()
        if len(self.history) >= self.rpm:
            sleep_time = 60 - (now - self.history[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit hit. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                now = time.time()
        self.history.append(now)
        self.daily_count += 1


class BreezeCCXT(ccxt.Exchange):
    _mock_mode_logged = False

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            config = {}
        super().__init__(config)
        self.config = config
        self.name = "IciciBreeze"

        # Rate Limiting
        rl_config = self.options.get("rateLimit", 100)
        self.rate_limiter = InternalRateLimiter(rpm=rl_config)

        # Credentials lookup (Options > ENV)
        self.api_key = self.options.get("key") or os.environ.get("BREEZE_API_KEY")
        api_secret = self.options.get("secret") or os.environ.get("BREEZE_API_SECRET")
        session_token = self.options.get("session_token") or os.environ.get("BREEZE_SESSION_TOKEN")

        if self._is_mock_mode():
            if not BreezeCCXT._mock_mode_logged:
                logger.info("Mock mode enabled: bypassing Breeze session.")
                BreezeCCXT._mock_mode_logged = True
            self.breeze = BreezeConnect(api_key=self.api_key or "mock_key")
            self._setup_mock_breeze()
            return

        if not self.api_key:
            logger.warning("Breeze API Key not found in Config or ENV.")
            self.breeze = None
            return

        self.breeze = BreezeConnect(api_key=self.api_key)

        if self.api_key == "mock_key":
            self._setup_mock_breeze()
            return

        if api_secret and session_token:
            try:
                logger.info("Initializing Breeze session with provided credentials.")
                self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
            except Exception as e:
                logger.error(f"Failed to initialize Breeze session: {e}")

    def _setup_mock_breeze(self):
        logger.info("Setting up Mock Breeze SDK mode for validation.")

        def mock_get_quotes(**kwargs):
            return {
                "status": 200,
                "Success": [
                    {
                        "stock_code": kwargs.get("stock_code"),
                        "ltp": "2500.00",
                        "high": "2550.00",
                        "low": "2480.00",
                        "best_bid_price": "2499.00",
                        "best_ask_price": "2501.00",
                        "ltt": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
                    }
                ],
            }

        def mock_get_historical_v2(**kwargs):
            interval = kwargs.get("interval", "5minute")
            count = 100
            now_ms = int(time.time() * 1000)
            step_ms = 5 * 60 * 1000 if "5" in interval else 60 * 60 * 1000
            success = []
            for i in range(count):
                ts = now_ms - (count - i) * step_ms
                dt_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
                success.append(
                    {
                        "datetime": dt_str,
                        "open": "2500",
                        "high": "2510",
                        "low": "2490",
                        "close": f"{2500 + i % 10}",
                        "volume": "1000",
                    }
                )
            return {"status": 200, "Success": success}

        self.breeze.get_quotes = mock_get_quotes
        self.breeze.get_historical_data_v2 = mock_get_historical_v2

    def _is_mock_mode(self) -> bool:
        """
        Deterministic mock mode check.
        """
        if self.options.get("dry_run") is True:
            return True
        if self.options.get("mode") in {"mock", "dry_run"}:
            return True
        if self.options.get("icici_mode") == "mock":
            return True
        if self.options.get("key") == "mock_key" or getattr(self, "api_key", None) == "mock_key":
            return True
        if os.getenv("BREEZE_MOCK") == "1":
            return True
        return False

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
                "features": {
                    "spot": {
                        "fetchOHLCV": {
                            "limit": 1000,
                            "days": 30,
                            "timeframes": ["1m", "5m", "30m", "1d"],
                        }
                    }
                },
            },
        )

    def _parse_symbol(self, symbol: str) -> dict:
        parts = symbol.split("/")
        if len(parts) < 2:
            raise OperationalException(f"Invalid symbol format: {symbol}")
        base = parts[0]
        return {"stock_code": base, "exchange_code": "NSE", "product_type": "cash"}

    def fetch_markets(self, params: dict | None = None):
        if self.rate_limiter:
            self.rate_limiter.check_and_record()
        nfo_file = find_latest_master_file("FONSEScripMaster.txt")
        nse_file = find_latest_master_file("NSEScripMaster.txt")
        nfo_master = (
            load_nfo_options_master(nfo_file)
            if nfo_file
            else {"by_contract": {}, "company_search": {}}
        )
        nse_master = (
            load_nse_cash_master(nse_file) if nse_file else {"by_symbol": {}, "company_search": {}}
        )
        if not nfo_master["by_contract"] and not nse_master["by_symbol"]:
            logger.warning("No security master data loaded.")
            return []
        whitelist = self.config.get("pair_whitelist", [])
        if not whitelist:
            return []
        specs = parse_pair_whitelist_for_options(whitelist)
        resolved, _ = resolve_underlying(specs, nfo_master, nse_master)
        markets = []
        nfo_contracts = nfo_master["by_contract"]
        nse_symbols = nse_master["by_symbol"]
        for spec in resolved:
            if spec["type"] == "option":
                key = (spec["underlying"], spec["expiry"], spec["strike"], spec["right"])
                if key in nfo_contracts:
                    info = nfo_contracts[key]
                    symbol = (
                        f"{info['underlying']}/INR:{info['expiry']}:"
                        f"{info['strike']}:{info['right']}"
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
            elif spec["type"] == "cash":
                und = spec["underlying"]
                if und in nse_symbols:
                    info = nse_symbols[und]
                    markets.append(
                        {
                            "id": info["token"],
                            "symbol": spec["original"],
                            "base": und,
                            "quote": "INR",
                            "active": True,
                            "type": "spot",
                            "spot": True,
                            "lot": info["lot_size"],
                            "precision": {"amount": 1, "price": info["tick_size"]},
                            "info": info,
                        }
                    )
        return markets

    def fetch_ticker(self, symbol: str, params: dict | None = None):
        if self._is_mock_mode():
            # Deterministic ticker
            import hashlib

            h = hashlib.md5(symbol.encode()).hexdigest()
            last = 2500.0 + (int(h[:3], 16) % 100)
            ts = int(time.time() * 1000)
            return {
                "symbol": symbol,
                "timestamp": ts,
                "datetime": self.iso8601(ts),
                "high": last + 10,
                "low": last - 10,
                "bid": last - 0.05,
                "ask": last + 0.05,
                "last": last,
                "close": last,
                "info": {"mock": True},
            }

        if not self.breeze:
            raise OperationalException("Breeze session not initialized.")
        self.rate_limiter.check_and_record()
        s_params = self._parse_symbol(symbol)
        try:
            res = self.breeze.get_quotes(**s_params)
            if not res or res.get("status") != 200 or not res.get("Success"):
                raise OperationalException(
                    f"Breeze fetch_ticker failed: "
                    f"{res.get('Error') if res else 'Empty response from SDK'}"
                )
            data = res["Success"][0]
            ts = int(time.time() * 1000)
            if data.get("ltt"):
                try:
                    ts = int(datetime.strptime(data["ltt"], "%d-%b-%Y %H:%M:%S").timestamp() * 1000)
                except Exception:
                    logger.warning("Could not parse LTT timestamp: %s", data.get("ltt"))
            return {
                "symbol": symbol,
                "timestamp": ts,
                "datetime": self.iso8601(ts),
                "high": float(data.get("high", 0)),
                "low": float(data.get("low", 0)),
                "bid": float(data.get("best_bid_price", 0)),
                "ask": float(data.get("best_ask_price", 0)),
                "last": float(data.get("ltp", 0)),
                "close": float(data.get("ltp", 0)),
                "info": data,
            }
        except Exception as e:
            raise OperationalException(f"Error in fetch_ticker: {e}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        if self._is_mock_mode():
            import hashlib

            # 1) compute step_ms from timeframe (5m=300000)
            multiplier = 1
            if timeframe.endswith("m"):
                multiplier = 60
            elif timeframe.endswith("h"):
                multiplier = 3600
            elif timeframe.endswith("d"):
                multiplier = 86400

            try:
                num = int(timeframe[:-1])
            except ValueError:
                num = 5  # Default to 5m if parsing fails

            step_ms = num * multiplier * 1000

            # 2) choose limit_eff = limit or 500
            limit_eff = limit if limit is not None else 500
            limit_eff = min(limit_eff, 1000)

            now_ms = int(time.time() * 1000)

            # 3) choose since_eff = since or (now_ms - limit_eff*step_ms)
            since_eff = since if since is not None else (now_ms - limit_eff * step_ms)
            # Align since_eff to step_ms
            since_eff = since_eff - (since_eff % step_ms)

            h = hashlib.md5((symbol + timeframe).encode()).hexdigest()
            base_price = 2500.0 + (int(h[:3], 16) % 100)

            ohlcv = []
            # 4) generate candles starting at since_eff stepping by step_ms until either:
            # - limit_eff candles produced, OR
            # - time reaches now_ms
            curr_ts = since_eff
            while len(ohlcv) < limit_eff and curr_ts < now_ms:
                # Deterministic "price" offset
                offset = (curr_ts // step_ms) % 50
                price = base_price + offset
                ohlcv.append(
                    [
                        curr_ts,
                        price,  # open
                        price + 2,  # high
                        price - 2,  # low
                        price + 1,  # close
                        1000.0,  # volume
                    ]
                )
                curr_ts += step_ms

            return ohlcv

        if not self.breeze:
            raise OperationalException("Breeze session not initialized.")
        self.rate_limiter.check_and_record()
        s_params = self._parse_symbol(symbol)
        interval = self.timeframes.get(timeframe)
        if not interval:
            raise OperationalException(f"Unsupported timeframe: {timeframe}")
        if limit is None:
            limit = 100
        limit = min(limit, 1000)
        end_dt = datetime.now()
        start_dt = (
            datetime.fromtimestamp(since / 1000)
            if since
            else datetime.fromtimestamp(time.time() - 86400 * 2)
        )
        try:
            res = self.breeze.get_historical_data_v2(
                stock_code=s_params["stock_code"],
                exchange_code=s_params["exchange_code"],
                product_type=s_params["product_type"],
                from_date=start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                to_date=end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                interval=interval,
            )
            if not res or res.get("status") != 200 or not res.get("Success"):
                return []
            ohlcv = []
            for row in res["Success"]:
                ts = int(datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
                ohlcv.append(
                    [
                        ts,
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row.get("volume", 0)),
                    ]
                )
            ohlcv.sort(key=lambda x: x[0])
            return ohlcv[:limit]
        except Exception as e:
            logger.error(f"Error in fetch_ohlcv: {e}")
            return []

    def fetch_balance(self, params: dict | None = None):
        raise OperationalException("fetch_balance not implemented in p06")

    def create_order(
        self, symbol, order_type, side, amount, price=None, params: dict | None = None
    ):
        raise OperationalException("Orders not implemented in p06")

    def cancel_order(self, order_id, symbol=None, params: dict | None = None):
        raise OperationalException("Orders not implemented in p06")

    def fetch_order(self, order_id, symbol=None, params: dict | None = None):
        raise OperationalException("Orders not implemented in p06")


class BreezeAsyncCCXT(ccxt_async.Exchange):
    _mock_mode_logged = False

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            config = {}
        self.sync_exchange = BreezeCCXT(config)
        super().__init__(config)
        self.config = config
        self.name = "IciciBreeze"

        if self._is_mock_mode() and not BreezeAsyncCCXT._mock_mode_logged:
            logger.info("Mock mode enabled: bypassing Breeze session.")
            BreezeAsyncCCXT._mock_mode_logged = True

    def _is_mock_mode(self) -> bool:
        return self.sync_exchange._is_mock_mode()

    def describe(self):
        res = {
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
            "timeframes": {"1m": "1minute", "5m": "5minute", "30m": "30minute", "1d": "1day"},
        }
        if hasattr(self, "sync_exchange") and self.sync_exchange:
            return self.deep_extend(res, self.sync_exchange.describe())
        return res

    async def load_markets(self, reload: bool = False, params: dict | None = None):
        markets = await asyncio.to_thread(self.sync_exchange.load_markets, reload, params)
        self.markets = markets
        self.symbols = list(markets.keys())
        return markets

    async def fetch_markets(self, params: dict | None = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_markets, params)

    async def fetch_ticker(self, symbol: str, params: dict | None = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_ticker, symbol, params)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        return await asyncio.to_thread(
            self.sync_exchange.fetch_ohlcv, symbol, timeframe, since, limit, params
        )
