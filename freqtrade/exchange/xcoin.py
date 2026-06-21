"""XCoin exchange subclass using the native XCoin REST API."""

import logging
import os
from typing import Any

from freqtrade.constants import ExchangeConfig
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas
from freqtrade.exchange.xcoin_api import XCoinAsync, XCoinSync
from freqtrade.exchange.xcoin_connector import XCOIN_DEFAULT_BASE_URL


logger = logging.getLogger(__name__)


class Xcoin(Exchange):
    """XCoin exchange class.

    XCoin is not provided by ccxt, so this subclass injects a small ccxt-like
    REST wrapper while keeping Freqtrade's regular Exchange call path intact.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,
        "l2_limit_range_required": False,
        "l2_limit_upper": 5000,
        "order_time_in_force": ["GTC", "IOC", "FOK"],
        "stoploss_on_exchange": False,
        "tickers_have_bid_ask": True,
        "tickers_have_price": True,
        "exchange_has_overrides": {
            "createMarketOrder": True,
            "fetchMyTrades": False,
            "fetchTrades": False,
        },
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
    ]

    def _init_ccxt(
        self, exchange_config: ExchangeConfig, sync: bool, ccxt_kwargs: dict[str, Any]
    ) -> XCoinSync:
        if self.trading_mode != TradingMode.SPOT:
            raise OperationalException("XCoin adapter currently supports spot trading only.")

        live_enabled = exchange_config.get("xcoin_live_trading_enabled", False)
        if not self._config.get("dry_run", True) and not live_enabled:
            raise OperationalException(
                "XCoin live trading is disabled. Set "
                "`exchange.xcoin_live_trading_enabled=true` explicitly after dry-run testing."
            )

        api_key = os.environ.get("FREQTRADE__EXCHANGE__KEY") or os.environ.get("XCOIN_API_KEY")
        api_secret = os.environ.get("FREQTRADE__EXCHANGE__SECRET") or os.environ.get(
            "XCOIN_API_SECRET"
        )
        if not self._config.get("dry_run", True) and (not api_key or not api_secret):
            raise OperationalException(
                "XCoin live trading requires API credentials from environment variables "
                "`FREQTRADE__EXCHANGE__KEY` and `FREQTRADE__EXCHANGE__SECRET`."
            )

        sanitized_ccxt_kwargs = {
            key: value
            for key, value in ccxt_kwargs.items()
            if key
            not in {
                "apiKey",
                "api_key",
                "key",
                "secret",
                "password",
                "privateKey",
                "private_key",
            }
        }
        wrapper_config = {
            "apiKey": api_key or "",
            "secret": api_secret or "",
            "accountName": exchange_config.get("account_name")
            or exchange_config.get("accountName")
            or os.environ.get("XCOIN_ACCOUNT_NAME", ""),
            "base_url": exchange_config.get("xcoin_base_url")
            or exchange_config.get("base_url")
            or XCOIN_DEFAULT_BASE_URL,
            "timeout": exchange_config.get("xcoin_timeout", 10),
        }
        wrapper_config.update(sanitized_ccxt_kwargs)

        return XCoinSync(wrapper_config) if sync else XCoinAsync(wrapper_config)

    def close(self) -> None:
        if api := getattr(self, "_api", None):
            api.close()
        super().close()
