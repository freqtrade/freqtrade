import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import ccxt
import ccxt.async_support as ccxt_async
from breeze_connect import BreezeConnect

from adapters.ccxt_shim.icicibreeze.mock_ohlcv import synth_ohlcv, timeframe_to_ms
from adapters.ccxt_shim.instrument import (
    InstrumentSpec,
    InstrumentType,
    format_pair,
    parse_pair,
)
from adapters.ccxt_shim.security_master import (
    find_latest_master_file,
    load_nfo_options_master,
    load_nse_cash_master,
)
from adapters.ccxt_shim.market_hours import MarketHoursGuard
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
    _mock_ohlcv_lock = threading.Lock()
    _MOCK_BASE_PRICES = {
        "RELIANCE/INR": 2500.0,
        "NIFTY/INR": 20000.0,
        "BANKNIFTY/INR": 45000.0,
    }
    _MOCK_MAX_PCT_MOVE = 0.003

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            config = {}
        super().__init__(config)
        self.config = config
        self.name = "IciciBreeze"

        # Rate Limiting
        rl_config = self.options.get("rateLimit", 100)
        self.rate_limiter = InternalRateLimiter(rpm=rl_config)
        self._security_master_cache: dict[str, Any] | None = None
        self.market_hours = MarketHoursGuard()

        # Mock Order Storage
        self._mock_orders: dict[str, dict] = {}

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
            except Exception:
                logger.error("Failed to initialize Breeze session. Please check your credentials.")

    def _setup_mock_breeze(self):
        logger.info("Setting up Mock Breeze SDK mode for validation.")

        def mock_get_quotes(**kwargs):
            stock_code = kwargs.get("stock_code", "UNKNOWN")
            base_price = self._mock_base_price(f"{stock_code}/INR")
            return {
                "status": 200,
                "Success": [
                    {
                        "stock_code": stock_code,
                        "ltp": f"{base_price:.2f}",
                        "high": f"{base_price * 1.01:.2f}",
                        "low": f"{base_price * 0.99:.2f}",
                        "best_bid_price": f"{base_price - 0.05:.2f}",
                        "best_ask_price": f"{base_price + 0.05:.2f}",
                        "ltt": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
                    }
                ],
            }

        def mock_get_historical_v2(**kwargs):
            symbol = f"{kwargs.get('stock_code')}/INR"
            # If it's an option, we should ideally parse it back, but for mock quotes
            # we just use the raw symbol which _generate_mock_ohlcv handles.
            timeframe = "5m"
            if kwargs.get("interval") == "1minute":
                timeframe = "1m"
            elif kwargs.get("interval") == "30minute":
                timeframe = "30m"
            elif kwargs.get("interval") == "1day":
                timeframe = "1d"

            ohlcv_list = self._generate_mock_ohlcv(symbol, timeframe, None, 5)
            success = [
                {
                    "datetime": datetime.fromtimestamp(o[0] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "open": str(o[1]),
                    "high": str(o[2]),
                    "low": str(o[3]),
                    "close": str(o[4]),
                    "volume": str(o[5]),
                }
                for o in ohlcv_list
            ]
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
                    "createOrder": True,
                    "cancelOrder": True,
                    "fetchOpenOrders": True,
                    "fetchClosedOrders": True,
                    "fetchOrders": True,
                    "fetchMyTrades": True,
                    "fetchBalance": True,
                    "fetchOrderBook": True,
                    "fetchL2OrderBook": True,
                    "fetchPositions": True,
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

    def _load_security_master(self) -> dict[str, Any]:
        if self._security_master_cache is not None:
            return self._security_master_cache
        nfo_file = find_latest_master_file("FONSEScripMaster.txt")
        nse_file = find_latest_master_file("NSEScripMaster.txt")
        nfo_master = (
            load_nfo_options_master(nfo_file)
            if nfo_file
            else {"by_contract": {}, "by_future": {}, "company_search": {}}
        )
        nse_master = (
            load_nse_cash_master(nse_file) if nse_file else {"by_symbol": {}, "company_search": {}}
        )
        self._security_master_cache = {"nfo": nfo_master, "nse": nse_master}
        return self._security_master_cache

    def _build_breeze_params(self, spec: InstrumentSpec, info: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "stock_code": spec.underlying,
            "exchange_code": "NSE" if spec.type == InstrumentType.CASH else "NFO",
            "product_type": "cash",
        }
        if spec.type == InstrumentType.OPT:
            params.update(
                {
                    "product_type": "options",
                    "expiry_date": info["expiry_iso"],
                    "strike_price": spec.strike,
                    "right": spec.right,
                }
            )
        elif spec.type == InstrumentType.FUT:
            params.update(
                {
                    "product_type": "futures",
                    "expiry_date": info["expiry_iso"],
                }
            )
        return params

    def _parse_symbol(self, symbol: str) -> dict[str, Any]:
        try:
            spec = parse_pair(symbol)
        except ValueError as exc:
            raise OperationalException(f"Invalid symbol format: {symbol}") from exc
        master = self._load_security_master()
        nfo_master = master["nfo"]
        nse_master = master["nse"]
        if spec.type == InstrumentType.CASH:
            info = nse_master.get("by_symbol", {}).get(spec.underlying)
            if not info:
                raise OperationalException(f"Cash symbol not found in SecurityMaster: {symbol}")
            return self._build_breeze_params(spec, info)
        if spec.type == InstrumentType.OPT:
            key = (spec.underlying, spec.expiry_yyyymmdd, float(spec.strike), spec.right)
            info = nfo_master.get("by_contract", {}).get(key)
            if not info:
                raise OperationalException(f"Option contract not found in SecurityMaster: {symbol}")
            return self._build_breeze_params(spec, info)
        if spec.type == InstrumentType.FUT:
            key = (spec.underlying, spec.expiry_yyyymmdd)
            info = nfo_master.get("by_future", {}).get(key)
            if not info:
                raise OperationalException(f"Future contract not found in SecurityMaster: {symbol}")
            return self._build_breeze_params(spec, info)
        raise OperationalException(f"Unsupported symbol type: {symbol}")

    def fetch_markets(self, params: dict | None = None):
        if self.rate_limiter:
            self.rate_limiter.check_and_record()
        master = self._load_security_master()
        whitelist = self.config.get("pair_whitelist", [])
        if not whitelist:
            return []
        markets = []
        for pair in whitelist:
            try:
                spec = parse_pair(pair)
            except ValueError:
                logger.warning("Skipping non-canonical pair in whitelist: %s", pair)
                continue
            market_info = self._fetch_specific_market(spec, master)
            if market_info:
                markets.append(market_info)
        return markets

    def _fetch_specific_market(self, spec: InstrumentSpec, master: dict[str, Any]) -> dict | None:
        if spec.type == InstrumentType.OPT:
            return self._fetch_option_market(spec, master["nfo"]["by_contract"])
        if spec.type == InstrumentType.FUT:
            return self._fetch_future_market(spec, master["nfo"].get("by_future", {}))
        if spec.type == InstrumentType.CASH:
            return self._fetch_cash_market(spec, master["nse"]["by_symbol"])
        return None

    def _fetch_option_market(self, spec: InstrumentSpec, contracts: dict) -> dict | None:
        key = (spec.underlying, spec.expiry_yyyymmdd, float(spec.strike), spec.right)
        info = contracts.get(key)
        if not info:
            logger.warning("Option contract not found for whitelist entry: %s", format_pair(spec))
            return None
        return {
            "id": info["token"],
            "symbol": format_pair(spec),
            "base": info["underlying"],
            "quote": "INR",
            "active": True,
            "type": "option",
            "spot": True,
            "option": True,
            "future": False,
            "margin": False,
            "swap": False,
            "expiry": info["expiry_yyyymmdd"],
            "strike": info["strike"],
            "right": info["right"],
            "lot": info["lot_size"],
            "precision": {"amount": 1, "price": info["tick_size"]},
            "info": info,
        }

    def _fetch_future_market(self, spec: InstrumentSpec, futures: dict) -> dict | None:
        key = (spec.underlying, spec.expiry_yyyymmdd)
        info = futures.get(key)
        if not info:
            logger.warning("Future contract not found for whitelist entry: %s", format_pair(spec))
            return None
        return {
            "id": info["token"],
            "symbol": format_pair(spec),
            "base": info["underlying"],
            "quote": "INR",
            "active": True,
            "type": "future",
            "spot": True,
            "option": False,
            "future": True,
            "margin": False,
            "swap": False,
            "expiry": info["expiry_yyyymmdd"],
            "lot": info["lot_size"],
            "precision": {"amount": 1, "price": info["tick_size"]},
            "info": info,
        }

    def _fetch_cash_market(self, spec: InstrumentSpec, cash_symbols: dict) -> dict | None:
        info = cash_symbols.get(spec.underlying)
        if not info and self._is_mock_mode():
            # Synthetic Index Cash or Mock Pairs support (BTC/USDT, NIFTY/INR etc)
            is_index = spec.underlying in self._MOCK_BASE_PRICES or spec.underlying in {
                "NIFTY",
                "BANKNIFTY",
            }
            is_mock_pair = spec.underlying == "BTC" and spec.quote == "USDT"
            if is_index or is_mock_pair:
                info = {
                    "token": f"mock_{spec.underlying}",
                    "symbol": spec.underlying,
                    "lot_size": 1,
                    "tick_size": 0.05,
                    "company_name": spec.underlying,
                }

        if not info:
            logger.warning("Cash symbol not found for whitelist entry: %s", format_pair(spec))
            return None

        symbol = format_pair(spec) if spec.quote == "INR" else f"{spec.underlying}/{spec.quote}"
        return {
            "id": info["token"],
            "symbol": symbol,
            "base": spec.underlying,
            "quote": spec.quote,
            "active": True,
            "type": "spot",
            "spot": True,
            "option": False,
            "future": False,
            "margin": False,
            "swap": False,
            "lot": info.get("lot_size", 1),
            "precision": {"amount": 1, "price": info.get("tick_size", 0.05)},
            "info": info,
        }

    def fetch_ticker(self, symbol: str, params: dict | None = None):
        if self._is_mock_mode():
            return self._generate_mock_ticker(symbol)

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
            logger.debug("fetch_ticker error: %s", e)
            raise OperationalException("Error in fetch_ticker. See debug logs for details.") from e

    def fetch_order_book(self, symbol: str, limit: int | None = None, params: dict | None = None):
        """Fetch order book. In mock mode, returns a synthetic one."""
        if self._is_mock_mode():
            ticker = self.fetch_ticker(symbol)
            price = ticker["last"]
            return {
                "symbol": symbol,
                "bids": [[price - 0.05, 1000.0], [price - 0.10, 2000.0]],
                "asks": [[price + 0.05, 1000.0], [price + 0.10, 2000.0]],
                "timestamp": ticker["timestamp"],
                "datetime": ticker["datetime"],
                "nonce": None,
            }
        raise OperationalException("fetchOrderBook not supported yet in real mode.")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        if self._is_mock_mode():
            return self._generate_mock_ohlcv(symbol, timeframe, since, limit)

        if not self.breeze:
            raise OperationalException("Breeze session not initialized.")
        self.rate_limiter.check_and_record()
        s_params = self._parse_symbol(symbol)
        interval = self.timeframes.get(timeframe)
        if not interval:
            raise OperationalException(f"Unsupported timeframe: {timeframe}")
        if limit is None:
            limit = 1000
        limit = min(limit, 1000)
        end_dt = datetime.now()
        start_dt = (
            datetime.fromtimestamp(since / 1000)
            if since
            else datetime.fromtimestamp(time.time() - 86400 * 2)
        )
        try:
            request_params = {
                "stock_code": s_params["stock_code"],
                "exchange_code": s_params["exchange_code"],
                "product_type": s_params["product_type"],
                "from_date": start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "to_date": end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "interval": interval,
            }
            if s_params.get("expiry_date"):
                request_params["expiry_date"] = s_params["expiry_date"]
            if s_params.get("strike_price") is not None:
                request_params["strike_price"] = s_params["strike_price"]
            if s_params.get("right"):
                request_params["right"] = s_params["right"]
            res = self.breeze.get_historical_data_v2(**request_params)
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
            logger.debug("fetch_ohlcv error: %s", e)
            return []

    def fetch_balance(self, params: dict | None = None):
        if self._is_mock_mode():
            return {
                "free": {"INR": 100000.0},
                "used": {"INR": 0.0},
                "total": {"INR": 100000.0},
                "info": {"mock": True},
            }
        raise OperationalException("fetch_balance not supported in real mode yet.")

    def create_order(
        self, symbol, order_type, side, amount, price=None, params: dict | None = None
    ):
        self.market_hours.assert_can_create_order(side, symbol)

        if self._is_mock_mode():
            import hashlib  # nosec
            import random  # nosec

            ts = int(time.time() * 1000)
            rand = random.randint(0, 1000000)
            seed = f"{symbol}-{side}-{amount}-{ts}-{rand}"
            order_id = f"ord_{hashlib.md5(seed.encode()).hexdigest()[:12]}"  # nosec
            order = {
                "id": order_id,
                "clientOrderId": order_id,
                "timestamp": ts,
                "datetime": self.iso8601(ts),
                "lastTradeTimestamp": None,
                "status": "open",
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "filled": 0.0,
                "remaining": amount,
                "cost": 0.0,
                "trades": [],
                "info": {"mock": True},
            }
            self._mock_orders[order_id] = order
            return order
        raise OperationalException("create_order not supported in real mode yet.")

    def cancel_order(self, order_id, symbol=None, params: dict | None = None):
        self.market_hours.assert_can_cancel_order(order_id, str(symbol))

        if self._is_mock_mode():
            if order_id in self._mock_orders:
                self._mock_orders[order_id]["status"] = "canceled"
                return self._mock_orders[order_id]
            raise OperationalException(f"Mock order {order_id} not found.")
        raise OperationalException("cancel_order not supported in real mode yet.")

    def fetch_order(self, order_id, symbol=None, params: dict | None = None):
        if self._is_mock_mode():
            if order_id in self._mock_orders:
                return self._mock_orders[order_id]
            raise OperationalException(f"Mock order {order_id} not found.")
        raise OperationalException("fetch_order not supported in real mode yet.")

    def fetch_open_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        if self._is_mock_mode():
            orders = [o for o in self._mock_orders.values() if o["status"] == "open"]
            if symbol:
                orders = [o for o in orders if o["symbol"] == symbol]
            return orders
        raise OperationalException("fetch_open_orders not supported in real mode yet.")

    def fetch_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        if self._is_mock_mode():
            orders = list(self._mock_orders.values())
            if symbol:
                orders = [o for o in orders if o["symbol"] == symbol]
            return orders
        raise OperationalException("fetch_orders not supported in real mode yet.")

    def fetch_positions(self, symbols: list[str] | None = None, params: dict | None = None):
        if self._is_mock_mode():
            return []
        raise OperationalException("fetch_positions not supported in real mode yet.")

    def _generate_mock_ticker(self, symbol: str) -> dict[str, Any]:
        import hashlib

        h = hashlib.sha256(symbol.encode()).hexdigest()
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

    def _get_mock_data_path(self, symbol: str, timeframe: str) -> Path:
        """Get path for mock data persistence."""
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        target_dir = Path("user_data") / "data" / "icicibreeze" / "mock_cache"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{safe_symbol}-{timeframe}.json"

    def _generate_mock_ohlcv(
        self, symbol: str, timeframe: str, since: int | None, limit: int | None
    ) -> list[list[Any]]:
        path = self._get_mock_data_path(symbol, timeframe)

        with self._mock_ohlcv_lock:
            stored_ohlcv = []
            if path.exists():
                try:
                    with path.open("r") as f:
                        stored_ohlcv = json.load(f)
                except Exception as e:
                    logger.warning("Failed to load mock data from %s: %s", path, e)

            # Check if stored data covers the requested range
            if stored_ohlcv and since is not None:
                first_ts = int(stored_ohlcv[0][0])
                last_ts = int(stored_ohlcv[-1][0])
                now_ms = int(time.time() * 1000)
                tf_ms = timeframe_to_ms(timeframe)
                # If since is within stored range and last_ts is close to now
                # Or if we have enough candles to satisfy limit
                if first_ts <= since <= last_ts:
                    requested_slice = [cand for cand in stored_ohlcv if int(cand[0]) >= since]
                    # If we have at least 2 more candles or we are very close to now
                    if (limit is not None and len(requested_slice) >= limit) or (
                        last_ts > now_ms - (tf_ms * 2)
                    ):
                        logger.debug(
                            "Returning %d cached candles from since=%d", len(requested_slice), since
                        )
                        return requested_slice[:limit] if limit else requested_slice

            # Use seed from environment if provided
            seed = int(os.environ.get("MOCK_OHLCV_SEED", 42))

            # Synthesize data
            # Ensure we use a large enough limit if not provided
            synth_limit = limit if limit is not None else 15000
            new_ohlcv = synth_ohlcv(symbol, timeframe, since, synth_limit, seed)
            logger.debug("Synthesized %d candles for %s", len(new_ohlcv), symbol)

            # Merge and dedupe
            full_map = {int(c[0]): c for c in stored_ohlcv}
            for cand in new_ohlcv:
                cand[0] = int(cand[0])
                full_map[cand[0]] = cand

            merged = [full_map[ts] for ts in sorted(full_map.keys())]
            logger.debug("Merged total of %d candles for %s", len(merged), symbol)

            # Persist atomically
            temp_fd, temp_path_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            temp_path = Path(temp_path_str)
            try:
                with os.fdopen(temp_fd, "w") as f:
                    json.dump(merged, f)
                temp_path.replace(path)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                logger.error("Failed to persist mock data to %s: %s", path, e)

        # Return requested slice (outside lock if we want, but inside is safer for consistent view)
        if since is not None:
            res = [cand for cand in merged if cand[0] >= since][:limit]
            logger.debug("Returning %d candles from since=%d", len(res), since)
            return res
        res = merged[-limit:] if limit is not None else merged
        logger.debug("Returning %d candles (no since)", len(res))
        return res

    def _mock_base_price(self, symbol: str) -> float:
        """Return a deterministic base price per symbol for mock data."""
        if symbol in self._MOCK_BASE_PRICES:
            return self._MOCK_BASE_PRICES[symbol]
        try:
            spec = parse_pair(symbol)
        except ValueError:
            spec = None
        if spec is not None and spec.type == InstrumentType.OPT:
            base_symbol = f"{spec.underlying}/INR"
            if base_symbol in self._MOCK_BASE_PRICES:
                return self._MOCK_BASE_PRICES[base_symbol]
        import hashlib

        h = hashlib.sha256(symbol.encode()).hexdigest()
        return 1000.0 + (int(h[:6], 16) % 4000)


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
                "fetchOrderBook": True,
                "fetchL2OrderBook": True,
            },
            "timeframes": {"1m": "1minute", "5m": "5minute", "30m": "30minute", "1d": "1day"},
        }
        if hasattr(self, "sync_exchange") and self.sync_exchange:
            return self.deep_extend(res, self.sync_exchange.describe())
        return res

    async def load_markets(self, reload: bool = False, params: dict | None = None):
        if not reload and self.markets:
            return self.markets
        markets = await asyncio.to_thread(self.sync_exchange.load_markets, reload, params)
        self.set_markets(markets)
        if "INR" not in self.currencies:
            self.currencies["INR"] = {
                "id": "INR",
                "code": "INR",
                "precision": 2,
            }
        return self.markets

    async def fetch_markets(self, params: dict | None = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_markets, params)

    async def fetch_ticker(self, symbol: str, params: dict | None = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_ticker, symbol, params)

    async def fetch_order_book(
        self, symbol: str, limit: int | None = None, params: dict | None = None
    ):
        return await asyncio.to_thread(self.sync_exchange.fetch_order_book, symbol, limit, params)

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

    async def fetch_balance(self, params: dict | None = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_balance, params)

    async def create_order(
        self, symbol, order_type, side, amount, price=None, params: dict | None = None
    ):
        return await asyncio.to_thread(
            self.sync_exchange.create_order, symbol, order_type, side, amount, price, params
        )

    async def cancel_order(self, order_id, symbol=None, params: dict | None = None):
        return await asyncio.to_thread(self.sync_exchange.cancel_order, order_id, symbol, params)

    async def fetch_order(self, order_id, symbol=None, params: dict | None = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_order, order_id, symbol, params)

    async def fetch_open_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        return await asyncio.to_thread(
            self.sync_exchange.fetch_open_orders, symbol, since, limit, params
        )

    async def fetch_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        return await asyncio.to_thread(
            self.sync_exchange.fetch_orders, symbol, since, limit, params
        )

    async def fetch_positions(self, symbols: list[str] | None = None, params: dict | None = None):
        return await asyncio.to_thread(self.sync_exchange.fetch_positions, symbols, params)
