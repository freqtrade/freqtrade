"""ccxt-like XCoin facade backed by the internal REST connector."""

import asyncio
from copy import deepcopy
from typing import Any

import ccxt
from ccxt import DECIMAL_PLACES

from freqtrade.exchange.xcoin_connector import (
    XCOIN_DEFAULT_BASE_URL,
    XCOIN_TIMEFRAMES,
    XCoinClient,
    ccxt_symbol_to_xcoin,
    xcoin_symbol_to_ccxt,
)


__all__ = [
    "XCOIN_DEFAULT_BASE_URL",
    "XCOIN_HAS",
    "XCOIN_TIMEFRAMES",
    "XCoinAsync",
    "XCoinSync",
    "ccxt_symbol_to_xcoin",
    "xcoin_symbol_to_ccxt",
]


XCOIN_HAS = {
    "fetchOrder": True,
    "fetchOpenOrder": False,
    "fetchClosedOrder": False,
    "fetchOpenOrders": True,
    "fetchOrders": False,
    "fetchBalance": True,
    "fetchOHLCV": True,
    "fetchTicker": True,
    "fetchTickers": True,
    "fetchL2OrderBook": True,
    "fetchOrderBook": True,
    "fetchMyTrades": False,
    "fetchTrades": False,
    "cancelOrder": True,
    "createOrder": True,
    "createLimitOrder": True,
    "createMarketOrder": True,
    "watchOHLCV": False,
}

XCOIN_STATUS_MAP = {
    "new": "open",
    "partially_filled": "open",
    "filled": "closed",
    "canceled": "canceled",
    "partially_canceled": "canceled",
}


def _clean_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _num(value: Any, fallback: float = 0.0) -> float:
    result = _clean_float(value)
    return fallback if result is None else result


def _str_num(value: float | int | str | None) -> str | None:
    if value is None:
        return None
    return str(value)


class XCoinSync:
    """Small ccxt-compatible subset used by Freqtrade's Exchange base class."""

    id = "xcoin"
    name = "XCoin"
    precisionMode = DECIMAL_PLACES
    timeframes = XCOIN_TIMEFRAMES
    features = {"spot": {"fetchOHLCV": {"limit": 1000}}}
    has = XCOIN_HAS
    session = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.client = XCoinClient(config)
        self.apiKey = self.client.api_key
        self.secret = self.client.api_secret
        self.password = self.client.password
        self.uid = self.client.uid
        self.account_name = self.client.account_name
        self.base_url = self.client.base_url
        self.timeout = self.client.timeout
        self.markets: dict[str, dict[str, Any]] = {}
        self.markets_by_id: dict[str, dict[str, Any]] = {}
        self.options = {
            "createMarketBuyOrderRequiresPrice": False,
            "timeframes": {"spot": XCOIN_TIMEFRAMES},
        }

    def milliseconds(self) -> int:
        return self.client.milliseconds()

    def iso8601(self, timestamp: int | None) -> str | None:
        return ccxt.Exchange.iso8601(timestamp) if timestamp is not None else None

    def set_markets_from_exchange(self, exchange: Any) -> None:
        self.markets = deepcopy(exchange.markets)
        self.markets_by_id = {
            market["id"]: market for market in self.markets.values() if market.get("id")
        }

    def market_id(self, symbol: str) -> str:
        if symbol in self.markets:
            return self.markets[symbol]["id"]
        return ccxt_symbol_to_xcoin(symbol)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        private: bool = False,
    ) -> dict[str, Any]:
        return self.client.request(method, path, params=params, data=data, private=private)

    def _private_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.client.private_params(params)

    def _parse_market(self, raw: dict[str, Any]) -> dict[str, Any]:
        symbol = xcoin_symbol_to_ccxt(raw["symbol"])
        order_params = raw.get("orderParameters") or {}
        return {
            "id": raw["symbol"],
            "symbol": symbol,
            "base": raw.get("baseCurrency"),
            "quote": raw.get("quoteCurrency"),
            "settle": raw.get("settleCurrency"),
            "type": "spot",
            "spot": raw.get("businessType") == "spot",
            "margin": False,
            "swap": False,
            "future": False,
            "active": raw.get("status") == "trading",
            "precision": {
                "amount": int(raw.get("quantityPrecision") or 8),
                "price": int(raw.get("pricePrecision") or 8),
            },
            "limits": {
                "amount": {
                    "min": _clean_float(order_params.get("minOrderQty")),
                    "max": _clean_float(order_params.get("maxLmtOrderQty")),
                },
                "price": {"min": _clean_float(raw.get("tickSize")), "max": None},
                "cost": {
                    "min": _clean_float(order_params.get("minOrderAmt")),
                    "max": _clean_float(order_params.get("maxLmtOrderAmt")),
                },
            },
            "maker": 0.001,
            "taker": 0.001,
            "info": raw,
        }

    def load_markets(
        self, reload: bool = False, params: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        if self.markets and not reload:
            return self.markets
        payload = self.client.public_symbols(params)
        markets = {
            market["symbol"]: market
            for market in (self._parse_market(item) for item in payload.get("data", []))
        }
        self.markets = markets
        self.markets_by_id = {market["id"]: market for market in markets.values()}
        return markets

    def _parse_ticker(
        self,
        raw: dict[str, Any],
        *,
        bid: float | None = None,
        ask: float | None = None,
        timestamp: int | None = None,
    ) -> dict[str, Any]:
        last = _clean_float(raw.get("lastPrice"))
        symbol = xcoin_symbol_to_ccxt(raw["symbol"])
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "datetime": self.iso8601(timestamp),
            "high": None,
            "low": None,
            "bid": bid if bid is not None else last,
            "bidVolume": None,
            "ask": ask if ask is not None else last,
            "askVolume": None,
            "vwap": None,
            "open": None,
            "close": last,
            "last": last,
            "previousClose": None,
            "change": _clean_float(raw.get("priceChange")),
            "percentage": _clean_float(raw.get("priceChangePercent")),
            "average": None,
            "baseVolume": _clean_float(raw.get("fillQty")),
            "quoteVolume": _clean_float(raw.get("fillAmount")),
            "info": raw,
        }

    def fetch_ticker(self, symbol: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        market_id = self.market_id(symbol)
        payload = self.client.ticker_mini(market_id, params)
        data = payload.get("data") or []
        if not data:
            raise ccxt.BadSymbol(f"XCoin ticker not found for {symbol}")
        orderbook = self.fetch_l2_order_book(symbol, 1)
        bid = orderbook["bids"][0][0] if orderbook["bids"] else None
        ask = orderbook["asks"][0][0] if orderbook["asks"] else None
        return self._parse_ticker(data[0], bid=bid, ask=ask, timestamp=_num(payload.get("ts"), 0))

    def fetch_tickers(
        self, symbols: list[str] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        market_id = self.market_id(symbols[0]) if symbols and len(symbols) == 1 else None
        payload = self.client.ticker_mini(market_id, params)
        tickers = {
            ticker["symbol"]: ticker
            for ticker in (
                self._parse_ticker(item, timestamp=_num(payload.get("ts"), 0))
                for item in payload.get("data", [])
            )
        }
        if symbols:
            symbols = [xcoin_symbol_to_ccxt(s) for s in symbols]
            tickers = {symbol: ticker for symbol, ticker in tickers.items() if symbol in symbols}
        return tickers

    def _parse_depth_side(self, values: list[Any]) -> list[list[float]]:
        if (
            values
            and isinstance(values[0], list)
            and len(values) == 1
            and isinstance(values[0][0], list)
        ):
            values = values[0]
        return [[float(price), float(amount)] for price, amount in values]

    def fetch_l2_order_book(
        self, symbol: str, limit: int | None = 100, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = self.client.depth(self.market_id(symbol), limit, params)
        data = payload.get("data") or {}
        timestamp = _num(payload.get("ts"), 0)
        return {
            "symbol": xcoin_symbol_to_ccxt(symbol),
            "bids": self._parse_depth_side(data.get("bids") or []),
            "asks": self._parse_depth_side(data.get("asks") or []),
            "timestamp": timestamp,
            "datetime": self.iso8601(timestamp),
            "nonce": _clean_float(data.get("lastUpdateId")),
        }

    def fetch_order_book(
        self, symbol: str, limit: int | None = 100, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.fetch_l2_order_book(symbol, limit, params)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        if timeframe not in self.timeframes:
            raise ccxt.BadRequest(f"Unsupported XCoin timeframe: {timeframe}")
        payload = self.client.klines(
            self.market_id(symbol),
            self.timeframes[timeframe],
            since=since,
            limit=limit,
            params=params,
        )
        candles = [
            [
                int(row[1]),
                float(row[3]),
                float(row[5]),
                float(row[6]),
                float(row[4]),
                float(row[7]),
            ]
            for row in payload.get("data", [])
        ]
        return sorted(candles, key=lambda candle: candle[0])

    def fetch_balance(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self.client.balance(params)
        balances: dict[str, Any] = {"info": payload, "free": {}, "used": {}, "total": {}}
        for item in (payload.get("data") or {}).get("details", []):
            currency = item["currency"]
            free = _num(item.get("cashBalance"), 0.0)
            used = _num(item.get("frozen"), 0.0)
            total = _num(item.get("totalBalance"), free + used)
            balances[currency] = {"free": free, "used": used, "total": total}
            balances["free"][currency] = free
            balances["used"][currency] = used
            balances["total"][currency] = total
        return balances

    def _parse_order(
        self,
        raw: dict[str, Any],
        *,
        symbol: str | None = None,
        amount: float | None = None,
        price: float | None = None,
        order_type: str | None = None,
        side: str | None = None,
        status: str | None = None,
        timestamp: int | None = None,
    ) -> dict[str, Any]:
        symbol = xcoin_symbol_to_ccxt(raw.get("symbol") or symbol or "")
        timestamp = int(raw.get("createTime") or timestamp or raw.get("ts") or self.milliseconds())
        amount = _clean_float(raw.get("qty")) if raw.get("qty") is not None else amount
        price = _clean_float(raw.get("price")) if raw.get("price") is not None else price
        filled = _num(raw.get("totalFillQty"), 0.0)
        average = _clean_float(raw.get("avgPrice"))
        cost = _clean_float(raw.get("quoteQty"))
        if cost is None and filled and average:
            cost = filled * average
        fee = self._parse_order_fee(raw, symbol)
        return {
            "id": raw.get("orderId"),
            "clientOrderId": raw.get("clientOrderId") or None,
            "timestamp": timestamp,
            "datetime": self.iso8601(timestamp),
            "lastTradeTimestamp": int(raw["updateTime"]) if raw.get("updateTime") else None,
            "symbol": symbol,
            "type": (raw.get("orderType") or order_type or "limit").lower(),
            "timeInForce": (raw.get("timeInForce") or "").upper() or None,
            "side": raw.get("side") or side,
            "price": price,
            "average": average,
            "amount": amount,
            "filled": filled,
            "remaining": max((amount or 0.0) - filled, 0.0) if amount is not None else None,
            "cost": cost,
            "status": status or XCOIN_STATUS_MAP.get(raw.get("status"), raw.get("status", "open")),
            "fee": fee,
            "trades": [],
            "info": raw,
        }

    def _parse_order_fee(self, raw: dict[str, Any], symbol: str) -> dict[str, Any] | None:
        market = self.markets.get(symbol, {})
        base = market.get("base")
        quote = market.get("quote")
        quote_fee = _clean_float(raw.get("quoteFee"))
        if quote_fee:
            return {"currency": quote, "cost": abs(quote_fee), "rate": None}
        base_fee = _clean_float(raw.get("baseFee"))
        if base_fee:
            return {"currency": base, "cost": abs(base_fee), "rate": None}
        return None

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        order_type = order_type.lower()
        if order_type not in {"limit", "market", "post_only"}:
            raise ccxt.InvalidOrder(
                "XCoin adapter only supports limit/market/post_only spot orders"
            )
        if order_type in {"limit", "post_only"} and price is None:
            raise ccxt.InvalidOrder("XCoin limit orders require a price")
        params = params or {}
        if order_type == "market":
            time_in_force = "ioc"
        else:
            time_in_force = (
                params.get("timeInForce") or params.get("time_in_force") or "gtc"
            ).lower()
        body = {
            "symbol": self.market_id(symbol),
            "side": side.lower(),
            "orderType": order_type,
            "qty": _str_num(amount),
            "marketUnit": params.get("marketUnit") or params.get("market_unit") or "baseCoin",
            "timeInForce": time_in_force,
            "clientOrderId": params.get("clientOrderId"),
        }
        if order_type in {"limit", "post_only"}:
            body["price"] = _str_num(price)
        payload = self.client.place_order(body)
        raw = {
            **(payload.get("data") or {}),
            "symbol": self.market_id(symbol),
            "side": side.lower(),
            "orderType": order_type,
            "qty": _str_num(amount),
            "price": _str_num(price),
            "status": "new",
            "ts": payload.get("ts"),
        }
        return self._parse_order(
            raw,
            symbol=symbol,
            amount=amount,
            price=price,
            order_type=order_type,
            side=side.lower(),
            status="open",
            timestamp=int(payload.get("ts") or self.milliseconds()),
        )

    def cancel_order(
        self, order_id: str, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = self.client.cancel_order(
            {
                "symbol": self.market_id(symbol or ""),
                "orderFilter": "order",
                "orderId": order_id,
                **(params or {}),
            }
        )
        raw = {
            **(payload.get("data") or {}),
            "symbol": self.market_id(symbol or ""),
            "status": "canceled",
            "ts": payload.get("ts"),
        }
        return self._parse_order(
            raw,
            symbol=symbol,
            status="canceled",
            timestamp=int(payload["ts"]),
        )

    def fetch_order(
        self, order_id: str, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = self.client.order_info(
            {"orderId": order_id, "orderFilter": "order", **(params or {})}
        )
        return self._parse_order(payload.get("data") or {}, symbol=symbol)

    def fetch_open_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        payload = self.client.open_orders(
            {
                "businessType": "spot",
                "symbol": self.market_id(symbol) if symbol else None,
                "orderFilter": "order",
                **(params or {}),
            }
        )
        orders = [self._parse_order(item) for item in payload.get("data", [])]
        return orders[:limit] if limit else orders

    def calculate_fee(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        takerOrMaker: str = "maker",
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        market = self.markets.get(symbol, {})
        rate = _num(market.get(takerOrMaker), 0.001)
        return {
            "type": takerOrMaker,
            "currency": market.get("quote"),
            "rate": rate,
            "cost": amount * price * rate,
        }

    def close(self) -> None:
        self.client.close()


class XCoinAsync(XCoinSync):
    """Async facade used by Freqtrade candle and market refresh paths."""

    async def load_markets(
        self, reload: bool = False, params: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(super().load_markets, reload, params)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        return await asyncio.to_thread(super().fetch_ohlcv, symbol, timeframe, since, limit, params)

    async def close(self) -> None:
        super().close()
