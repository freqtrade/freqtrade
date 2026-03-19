"""
Reddit sentiment PairList provider

Provides a dynamic crypto pairlist based on remote Reddit sentiment data.
"""

import logging
from typing import Any

import requests

from freqtrade import __version__
from freqtrade.constants import PairPrefixes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import FtTTLCache


logger = logging.getLogger(__name__)

VALID_TRENDS = {"rising", "stable", "falling"}


class RedditSentimentPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.BIASED

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._mode = self._pairlistconfig.get("mode", "whitelist")
        if (self._mode == "whitelist") and ("number_assets" not in self._pairlistconfig):
            raise OperationalException(
                "`number_assets` not specified. Please check your configuration "
                'for "pairlist.config.number_assets"'
            )

        self._api_url = self._pairlistconfig.get("api_url", "")
        if not self._api_url:
            raise OperationalException(
                "`api_url` not specified. Please check your configuration "
                'for "pairlist.config.api_url"'
            )

        self._api_key = self._pairlistconfig.get("api_key", "")
        if not self._api_key:
            raise OperationalException(
                "`api_key` not specified. Please check your configuration "
                'for "pairlist.config.api_key"'
            )

        self._stake_currency = self._config["stake_currency"]
        self._number_assets = self._pairlistconfig.get("number_assets", 30)
        self._max_tokens = self._pairlistconfig.get("max_tokens", 50)
        self._days = self._pairlistconfig.get("days", 1)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 21600)
        self._read_timeout = self._pairlistconfig.get("read_timeout", 30)
        self._keep_pairlist_on_failure = self._pairlistconfig.get("keep_pairlist_on_failure", True)
        self._min_buzz_score = self._pairlistconfig.get("min_buzz_score", 0.0)
        self._min_mentions = self._pairlistconfig.get("min_mentions", 0)
        self._min_bullish_pct = self._pairlistconfig.get("min_bullish_pct", None)
        self._allowed_trends = self._pairlistconfig.get("allowed_trends", [])
        self._sentiment_cache: FtTTLCache = FtTTLCache(maxsize=1, ttl=self._refresh_period)
        self._last_sentiment_rows: list[dict[str, Any]] = []
        self._init_done = False

        if self._mode not in ["whitelist", "blacklist"]:
            raise OperationalException(
                '`mode` not configured correctly. Supported Modes are "whitelist","blacklist"'
            )

        if self._max_tokens > 100:
            self.logger.warning(
                f"The max_tokens you have set ({self._max_tokens}) exceeds the API limit of 100. "
                "It will be capped to 100."
            )
            self._max_tokens = 100

        if self._allowed_trends and not set(self._allowed_trends).issubset(VALID_TRENDS):
            raise OperationalException(
                "`allowed_trends` not configured correctly. "
                'Supported values are "rising", "stable", "falling".'
            )

    def short_desc(self) -> str:
        num = self._number_assets if self._mode == "whitelist" else "blacklisting"
        return f"{self.name} - {num} pairs filtered by Reddit sentiment."

    @staticmethod
    def description() -> str:
        return "Provides pair list based on a remote Reddit crypto sentiment API."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "api_url": {
                "type": "string",
                "default": "",
                "description": "Remote API URL",
                "help": "API endpoint returning Reddit crypto sentiment rows.",
            },
            "api_key": {
                "type": "string",
                "default": "",
                "description": "API key",
                "help": "API key for the remote sentiment service.",
            },
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "Number of assets",
                "help": "Number of assets to use from the sentiment-ranked pairlist.",
            },
            "max_tokens": {
                "type": "number",
                "default": 50,
                "description": "Maximum remote tokens to evaluate",
                "help": "Maximum number of remote tokens fetched from the sentiment endpoint.",
            },
            "days": {
                "type": "number",
                "default": 1,
                "description": "Lookback window in days",
                "help": "Lookback window in days for the remote sentiment request.",
            },
            "min_buzz_score": {
                "type": "number",
                "default": 0,
                "description": "Minimum buzz score",
                "help": "Minimum buzz score required for a token to pass the filter.",
            },
            "min_mentions": {
                "type": "number",
                "default": 0,
                "description": "Minimum mentions",
                "help": "Minimum Reddit mentions required for a token to pass the filter.",
            },
            "min_bullish_pct": {
                "type": "number",
                "default": None,
                "description": "Minimum bullish percentage",
                "help": "Optional minimum bullish percentage required for a token to pass.",
            },
            "allowed_trends": {
                "type": "list",
                "default": [],
                "description": "Allowed trends",
                "help": 'Optional list of allowed trends. Supported values: ["rising", "stable", "falling"].',
            },
            "mode": {
                "type": "option",
                "default": "whitelist",
                "options": ["whitelist", "blacklist"],
                "description": "Mode of operation",
                "help": "Mode of operation (whitelist/blacklist)",
            },
            "keep_pairlist_on_failure": {
                "type": "boolean",
                "default": True,
                "description": "Keep last sentiment set on failure",
                "help": "Keep the previous sentiment data if the remote API request fails.",
            },
            "read_timeout": {
                "type": "number",
                "default": 30,
                "description": "Read timeout",
                "help": "Request timeout in seconds for the remote sentiment service.",
            },
            **IPairList.refresh_period_parameter(),
        }

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _process_rows(self, jsonparse: Any) -> list[dict[str, Any]]:
        if not isinstance(jsonparse, list):
            raise OperationalException("Remote sentiment API response is not a JSON list.")

        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        for row in jsonparse:
            if not isinstance(row, dict):
                continue

            symbol = str(row.get("symbol") or "").upper().strip()
            if not symbol or symbol in seen:
                continue

            buzz_score = self._as_float(row.get("buzz_score"))
            mentions = self._as_int(row.get("mentions"))
            bullish_pct_raw = row.get("bullish_pct")
            bullish_pct = None if bullish_pct_raw is None else self._as_int(bullish_pct_raw)
            trend = str(row.get("trend") or "").lower().strip() or None

            if buzz_score < self._min_buzz_score:
                continue
            if mentions < self._min_mentions:
                continue
            if self._min_bullish_pct is not None and (
                bullish_pct is None or bullish_pct < self._min_bullish_pct
            ):
                continue
            if self._allowed_trends and trend not in self._allowed_trends:
                continue

            seen.add(symbol)
            results.append(
                {
                    "symbol": symbol,
                    "buzz_score": buzz_score,
                    "mentions": mentions,
                    "bullish_pct": bullish_pct,
                    "trend": trend,
                }
            )

        results.sort(key=lambda row: (row["buzz_score"], row["mentions"]), reverse=True)
        self._init_done = True
        return results

    def _handle_error(self, error: str) -> list[dict[str, Any]]:
        if self._init_done and self._keep_pairlist_on_failure:
            self.log_once("Error: " + error, logger.info)
            self.log_once("Keeping last fetched sentiment universe", logger.info)
            return self._last_sentiment_rows.copy()
        raise OperationalException(error)

    def fetch_sentiment_rows(self) -> list[dict[str, Any]]:
        cached_rows = self._sentiment_cache.get("rows")
        if cached_rows is not None:
            return cached_rows.copy()

        headers = {
            "User-Agent": "Freqtrade/" + __version__ + " RedditSentimentPairList",
            "X-API-Key": self._api_key,
        }
        params = {"days": self._days, "limit": self._max_tokens}

        try:
            response = requests.get(
                self._api_url,
                headers=headers,
                params=params,
                timeout=self._read_timeout,
            )
        except requests.exceptions.RequestException:
            rows = self._handle_error(
                f"Was not able to fetch sentiment data from: {self._api_url}"
            )
            self._sentiment_cache["rows"] = rows.copy()
            return rows

        content_type = response.headers.get("content-type", "")
        if response.status_code != 200:
            rows = self._handle_error(
                f"Sentiment API returned status {response.status_code}: {self._api_url}"
            )
            self._sentiment_cache["rows"] = rows.copy()
            return rows

        if "application/json" not in str(content_type):
            rows = self._handle_error(f"Sentiment API is not of type JSON. {self._api_url}")
            self._sentiment_cache["rows"] = rows.copy()
            return rows

        try:
            rows = self._process_rows(response.json())
        except Exception as exc:
            rows = self._handle_error(f"Failed processing sentiment JSON data: {type(exc)}")

        self._last_sentiment_rows = rows.copy()
        self._sentiment_cache["rows"] = rows.copy()
        return rows

    def get_markets_exchange(self) -> list[str]:
        return [
            market
            for market in self._exchange.get_markets(
                quote_currencies=[self._stake_currency], tradable_only=True, active_only=True
            ).keys()
        ]

    def resolve_sentiment_pair(
        self,
        pair: str,
        pairlist: list[str],
        markets: list[str],
        filtered_pairlist: list[str],
    ) -> str | None:
        if pair in filtered_pairlist:
            return None

        if pair in pairlist:
            return pair

        if pair not in markets:
            for prefix in PairPrefixes:
                test_prefix = f"{prefix}{pair}"
                if test_prefix in pairlist:
                    return test_prefix

        return None

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        pairlist = self.get_markets_exchange()
        pairlist = self.verify_blacklist(pairlist, logger.info)
        return self.filter_pairlist(pairlist, tickers)

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        is_whitelist_mode = self._mode == "whitelist"
        filtered_pairlist: list[str] = []
        sentiment_rows = self.fetch_sentiment_rows()

        if not sentiment_rows:
            return [] if is_whitelist_mode else pairlist

        market = self._exchange._config["trading_mode"]
        pair_format = f"{self._stake_currency.upper()}" + (
            f":{self._stake_currency.upper()}" if market == "futures" else ""
        )
        markets = self.get_markets_exchange()

        for row in sentiment_rows[: self._max_tokens]:
            pair = f"{row['symbol'].upper()}/{pair_format}"
            resolved = self.resolve_sentiment_pair(pair, pairlist, markets, filtered_pairlist)

            if not resolved:
                continue

            if not is_whitelist_mode:
                pairlist.remove(resolved)
                continue

            filtered_pairlist.append(resolved)
            if len(filtered_pairlist) == self._number_assets:
                break

        return filtered_pairlist if is_whitelist_mode else pairlist
