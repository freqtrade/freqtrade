"""PortfolioBench exchange subclass.

Extends Binance to support non-crypto assets (US stocks, global indices)
for offline backtesting. This avoids patching the vendored exchange.py.

Quote-currency convention:
  - Cryptocurrency pairs use USDT  (e.g. BTC/USDT, ETH/USDT)
  - US stocks and global indices use USD (e.g. AAPL/USD, DJI/USD)
"""

import asyncio
import logging
from copy import deepcopy
from typing import Any

import ccxt

from freqtrade.enums import TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange.binance import Binance
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import FtHas
from freqtrade.util.datetime_helpers import dt_ts


logger = logging.getLogger(__name__)

# Recognised cryptocurrency base tickers.  Everything else is treated as a
# stock or index and quoted in USD rather than USDT.
CRYPTO_BASES: set[str] = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "BNB", "BCH", "DOGE", "STETH", "TRX",
}


def is_crypto_pair(pair: str) -> bool:
    """Return True if *pair* is a cryptocurrency pair (quoted in USDT)."""
    base = pair.split("/")[0]
    return base in CRYPTO_BASES


def default_quote(pair: str) -> str:
    """Return the appropriate quote currency for *pair*."""
    if "/" in pair:
        return pair.split("/")[1]
    return "USDT" if is_crypto_pair(pair) else "USD"


def _synthetic_market(pair: str) -> dict:
    """Build a synthetic market entry so freqtrade treats any pair as tradeable."""
    return {
        "symbol": pair,
        "base": pair.split("/")[0],
        "quote": default_quote(pair),
        "spot": True,
        "swap": True,
        "future": True,
        "linear": True,
        "type": "swap",
        "contract": True,
        "active": True,
        "precision": {"amount": 8, "price": 8},
        "limits": {
            "amount": {"min": 1e-8, "max": 1e8},
            "price": {"min": 1e-8, "max": 1e8},
            "cost": {"min": 1e-8, "max": 1e8},
        },
        "info": {},
    }


class Portfoliobench(Binance):
    """Exchange adapter for PortfolioBench multi-asset backtesting.

    Overrides Binance behaviour to:
    - Tolerate offline / unreachable exchange (timeout + zero retries)
    - Auto-inject synthetic market entries for non-crypto pairs
    - Return zero fees for assets without exchange fee data
    - Provide a default 1x leverage tier for non-crypto assets
    """

    _ft_has: FtHas = {
        "needs_trading_fees": False,
    }

    # ------------------------------------------------------------------
    # ccxt init — use binance as the underlying ccxt exchange
    # ------------------------------------------------------------------

    def _init_ccxt(
        self, exchange_config: dict[str, Any], sync: bool, ccxt_kwargs: dict[str, Any]
    ) -> ccxt.Exchange:
        """Map 'portfoliobench' back to 'binance' for ccxt initialisation."""
        patched = deepcopy(exchange_config)
        patched["name"] = "binance"
        return super()._init_ccxt(patched, sync, ccxt_kwargs)

    # ------------------------------------------------------------------
    # Market loading — offline-tolerant
    # ------------------------------------------------------------------

    async def _api_reload_markets(self, reload: bool = False) -> None:
        try:
            await asyncio.wait_for(
                self._api_async.load_markets(reload=reload, params={}),
                timeout=5.0,
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise TemporaryError(f"Market loading timed out: {e}") from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Error in reload_markets due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise TemporaryError(e) from e

    def reload_markets(self, force: bool = False, *, load_leverage_tiers: bool = True) -> None:
        is_initial = self._last_markets_refresh == 0
        if (
            not force
            and self._last_markets_refresh > 0
            and (self._last_markets_refresh + self.markets_refresh_interval > dt_ts())
        ):
            return None
        logger.debug("Performing scheduled market reload..")

        exchange_loaded = False
        try:
            retrier(self._load_async_markets, retries=0)(reload=True)
            self._markets = self._api_async.markets
            exchange_loaded = True
        except (ccxt.BaseError, TemporaryError):
            logger.warning(
                "Could not load markets from exchange, will use locally injected pairs."
            )
            if not self._markets:
                self._markets = {}

        try:
            self._inject_synthetic_markets()

            if exchange_loaded:
                self._api.set_markets_from_exchange(self._api_async)
                self._api.options = self._api_async.options
                if self._exchange_ws:
                    self._ws_async.set_markets_from_exchange(self._api_async)
                    self._ws_async.options = self._api.options

            # Sync injected markets to the sync api
            self._api.markets = self._markets

            self._last_markets_refresh = dt_ts()

            if exchange_loaded and is_initial and self._ft_has["needs_trading_fees"]:
                self._trading_fees = self.fetch_trading_fees()

            if exchange_loaded and load_leverage_tiers and self.trading_mode == TradingMode.FUTURES:
                self.fill_leverage_tiers()
        except (ccxt.BaseError, TemporaryError):
            logger.exception("Could not load markets.")

    def _inject_synthetic_markets(self) -> None:
        """Inject synthetic market entries for any configured pair not on the exchange."""
        whitelist = self._config.get("exchange", {}).get("pair_whitelist", [])
        cli_pairs = self._config.get("pairs", [])
        for pair in set(whitelist + cli_pairs):
            if pair not in self._markets:
                logger.info("Auto-injecting pair: %s", pair)
                self._markets[pair] = _synthetic_market(pair)

    # ------------------------------------------------------------------
    # Validation — accept both USD and USDT as stake-compatible quotes
    # ------------------------------------------------------------------

    def validate_stakecurrency(self, stake_currency: str) -> None:
        """Skip the strict quote-currency check.

        PortfolioBench intentionally mixes USD (stocks/indices) and USDT
        (crypto) quote currencies.  Both are treated as equivalent for
        backtesting purposes.
        """
        if not self._markets:
            # Still need *some* markets to exist.
            from freqtrade.exceptions import OperationalException

            raise OperationalException(
                "Could not load markets, therefore cannot start. "
                "Please investigate the above error for more details."
            )
        # Accept USD or USDT — no further validation needed.

    def get_pair_quote_currency(self, pair: str) -> str:
        """Normalise USD and USDT to the configured stake currency.

        This lets freqtrade's pairlist filtering treat ``AAPL/USD`` and
        ``BTC/USDT`` as compatible with a single ``stake_currency`` setting.
        """
        raw = self.markets.get(pair, {}).get("quote", "")
        if raw in ("USD", "USDT"):
            return self._config.get("stake_currency", raw)
        return raw

    # ------------------------------------------------------------------
    # Fees — graceful fallback for non-crypto assets
    # ------------------------------------------------------------------

    @retrier
    def get_fee(
        self,
        symbol: str,
        order_type: str = "",
        side: str = "",
        amount: float = 1,
        price: float = 1,
        taker_or_maker: str = "maker",
    ) -> float:
        if order_type and order_type == "market":
            taker_or_maker = "taker"
        try:
            if self._config["dry_run"] and self._config.get("fee", None) is not None:
                return self._config["fee"]
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets(params={})
            try:
                return self._api.calculate_fee(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                    takerOrMaker=taker_or_maker,
                )["rate"]
            except KeyError:
                return 0.0
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Could not get fee info due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    # ------------------------------------------------------------------
    # Leverage tiers — default 1x for non-crypto
    # ------------------------------------------------------------------

    async def get_market_leverage_tiers(self, symbol: str) -> tuple[str, list[dict]]:
        try:
            tier = await self._api_async.fetch_market_leverage_tiers(symbol)
            return symbol, tier
        except (ccxt.OperationFailed, ccxt.ExchangeError, ccxt.BaseError):
            return symbol, [
                {
                    "minNotional": 0,
                    "maxNotional": 100000000,
                    "maintenanceMarginRate": 0.0,
                    "maxLeverage": 1.0,
                    "maintAmt": 0.0,
                    "info": {},
                }
            ]
