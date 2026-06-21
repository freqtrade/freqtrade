"""Low-level XCoin REST connector.

This module intentionally stays free of Freqtrade exchange semantics. It owns
HTTP transport, authentication, response validation, endpoint paths, and small
request-shaping helpers. The ccxt-like response parsing lives in xcoin_api.py.
"""

import hmac
import json
import time
from hashlib import sha256
from typing import Any
from urllib.parse import urlencode

import ccxt
import requests

from freqtrade.exchange.xcoin_connector.constants import XCOIN_DEFAULT_BASE_URL


class XCoinClient:
    """Thin XCoin REST client used by the Freqtrade exchange facade."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.api_key = config.get("apiKey") or ""
        self.api_secret = config.get("secret") or ""
        self.password = config.get("password") or ""
        self.uid = config.get("uid") or ""
        self.account_name = config.get("accountName") or config.get("account_name") or ""
        self.base_url = (config.get("base_url") or XCOIN_DEFAULT_BASE_URL).rstrip("/")
        self.timeout = config.get("timeout", 10)
        self._http = requests.Session()

    def milliseconds(self) -> int:
        return int(time.time() * 1000)

    def close(self) -> None:
        self._http.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        private: bool = False,
    ) -> dict[str, Any]:
        query = self._query(params)
        body = self._body(data)
        headers = {"Content-Type": "application/json"}
        if private:
            if not self.api_key or not self.api_secret:
                raise ccxt.AuthenticationError("XCoin API credentials are required")
            timestamp = str(self.milliseconds())
            headers.update(
                {
                    "X-ACCESS-APIKEY": self.api_key,
                    "X-ACCESS-TIMESTAMP": timestamp,
                    "X-ACCESS-SIGN": self._sign(timestamp, method, path, query, body),
                }
            )

        url = f"{self.base_url}{path}{query}"
        try:
            response = self._http.request(
                method.upper(),
                url,
                data=body or None,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise ccxt.NetworkError(str(e)) from e

        if response.status_code >= 500:
            raise ccxt.ExchangeNotAvailable(f"XCoin HTTP {response.status_code}")
        if response.status_code != 200:
            raise ccxt.ExchangeError(f"XCoin HTTP {response.status_code}: {response.text}")
        try:
            payload = response.json()
        except ValueError as e:
            raise ccxt.ExchangeError("XCoin returned invalid JSON") from e
        return self._handle_response(payload)

    def public_symbols(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request(
            "GET",
            "/v2/public/symbols",
            params={"businessType": "spot", **(params or {})},
        )

    def ticker_mini(
        self, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        request_params = {"businessType": "spot", **(params or {})}
        if symbol:
            request_params["symbol"] = symbol
        return self.request("GET", "/v1/market/ticker/mini", params=request_params)

    def depth(
        self, symbol: str, limit: int | None = 100, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            "/v1/market/depth",
            params={
                "businessType": "spot",
                "symbol": symbol,
                "limit": limit or 100,
                **(params or {}),
            },
        )

    def klines(
        self,
        symbol: str,
        period: str,
        *,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_params = {
            "businessType": "spot",
            "symbol": symbol,
            "period": period,
            "limit": min(limit or 1000, 1000),
            **(params or {}),
        }
        if since:
            request_params["startTime"] = since
        return self.request("GET", "/v1/market/kline", params=request_params)

    def balance(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request(
            "GET",
            "/v1/account/balance",
            params=self.private_params(params),
            private=True,
        )

    def place_order(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.request(
            "POST",
            "/v2/trade/order",
            data=self.private_params(data),
            private=True,
        )

    def cancel_order(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.request(
            "POST",
            "/v1/trade/cancelOrder",
            data=self.private_params(data),
            private=True,
        )

    def order_info(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.request(
            "GET",
            "/v2/trade/order/info",
            params=self.private_params(params),
            private=True,
        )

    def open_orders(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.request(
            "GET",
            "/v2/trade/openOrders",
            params=self.private_params(params),
            private=True,
        )

    def private_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        result = dict(params or {})
        if self.account_name and "accountName" not in result:
            result["accountName"] = self.account_name
        return result

    def _sign(self, timestamp: str, method: str, path: str, query: str, body: str) -> str:
        message = f"{timestamp}{method.upper()}{path}{query}{body}"
        return hmac.new(self.api_secret.encode(), message.encode(), sha256).hexdigest()

    def _query(self, params: dict[str, Any] | None) -> str:
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        return f"?{urlencode(sorted(clean.items()))}" if clean else ""

    def _body(self, data: dict[str, Any] | None) -> str:
        if not data:
            return ""
        clean = {k: v for k, v in data.items() if v is not None}
        return json.dumps(clean, separators=(",", ":"), ensure_ascii=False)

    def _handle_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        code = str(payload.get("code", "0"))
        if code == "0":
            return payload
        msg = payload.get("msg") or payload.get("message") or "XCoin API error"
        if code in {"11004", "14001"}:
            raise ccxt.DDoSProtection(msg)
        if code in {"40013", "20010"}:
            raise ccxt.OrderNotFound(msg)
        if code in {"60103", "60104", "60106"}:
            raise ccxt.InsufficientFunds(msg)
        if code.startswith("5") or code.startswith("4"):
            raise ccxt.InvalidOrder(msg)
        raise ccxt.ExchangeError(f"{code}: {msg}")
