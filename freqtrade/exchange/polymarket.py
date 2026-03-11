"""Polymarket exchange subclass for event contract trading.

Extends the Portfoliobench exchange to support Polymarket binary outcome
contracts (YES/NO shares priced $0–$1) for backtesting prediction market
strategies.

Key differences from crypto/stock trading:
  - Prices are bounded [0, 1] representing probability of an outcome
  - Each event has two complementary contracts: YES + NO (prices sum to ~$1)
  - Contracts settle at exactly $0 or $1 at resolution
  - There is no "volume" in the traditional sense; liquidity comes from the CLOB
  - Profit = (settlement_price - entry_price) * shares

Pair convention:  {EVENT_SLUG}-{YES|NO}/USDT  (Polymarket contracts use USDT)
  e.g. "TRUMP-WIN-YES/USDT", "ETH-10K-NO/USDT"
"""

import logging
from copy import deepcopy
from typing import Any

import ccxt

from freqtrade.exchange.portfoliobench import Portfoliobench

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract type constants
# ---------------------------------------------------------------------------
CONTRACT_YES = "YES"
CONTRACT_NO = "NO"
PRICE_FLOOR = 0.001  # Minimum contract price ($0.001)
PRICE_CEIL = 0.999   # Maximum contract price ($0.999)


def is_polymarket_pair(pair: str) -> bool:
    """Return True if the pair follows Polymarket naming: *-YES/USDT or *-NO/USDT."""
    base = pair.split("/")[0] if "/" in pair else pair
    return base.endswith(f"-{CONTRACT_YES}") or base.endswith(f"-{CONTRACT_NO}")


def get_complement_pair(pair: str) -> str:
    """Return the complementary contract pair (YES <-> NO)."""
    if f"-{CONTRACT_YES}/" in pair:
        return pair.replace(f"-{CONTRACT_YES}/", f"-{CONTRACT_NO}/")
    elif f"-{CONTRACT_NO}/" in pair:
        return pair.replace(f"-{CONTRACT_NO}/", f"-{CONTRACT_YES}/")
    return pair


def get_event_slug(pair: str) -> str:
    """Extract the event slug from a Polymarket pair.

    'TRUMP-WIN-YES/USDT' -> 'TRUMP-WIN'
    """
    base = pair.split("/")[0]
    for suffix in (f"-{CONTRACT_YES}", f"-{CONTRACT_NO}"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _polymarket_synthetic_market(pair: str) -> dict:
    """Build a synthetic market entry for a Polymarket event contract.

    Polymarket contracts differ from regular assets:
    - Prices are bounded [0, 1] (probability)
    - Amount precision is whole shares (integer)
    - Minimum order is 1 share
    """
    return {
        "symbol": pair,
        "base": pair.split("/")[0],
        "quote": pair.split("/")[1] if "/" in pair else "USDT",
        "spot": True,
        "swap": False,
        "future": False,
        "linear": True,
        "type": "spot",
        "contract": False,
        "active": True,
        "precision": {"amount": 0, "price": 4},  # whole shares, 4-decimal prices
        "limits": {
            "amount": {"min": 1, "max": 1e8},
            "price": {"min": PRICE_FLOOR, "max": PRICE_CEIL},
            "cost": {"min": 0.01, "max": 1e8},
        },
        "info": {
            "polymarket": True,
            "contract_type": CONTRACT_YES if f"-{CONTRACT_YES}/" in pair else CONTRACT_NO,
            "event_slug": get_event_slug(pair),
        },
    }


class Polymarket(Portfoliobench):
    """Exchange adapter for Polymarket event contract backtesting.

    Extends Portfoliobench to handle:
    - Binary outcome contract market entries (YES/NO shares)
    - Price bounds enforcement [0, 1]
    - Zero trading fees (Polymarket's maker model)
    - Contract-specific synthetic market injection
    - Complement pair awareness (YES + NO = $1)
    """

    _ft_has = {
        "needs_trading_fees": False,
    }

    # ------------------------------------------------------------------
    # ccxt init — reuse binance underneath
    # ------------------------------------------------------------------

    def _init_ccxt(
        self, exchange_config: dict[str, Any], sync: bool, ccxt_kwargs: dict[str, Any]
    ) -> ccxt.Exchange:
        """Map 'polymarket' back to 'binance' for ccxt initialisation."""
        patched = deepcopy(exchange_config)
        patched["name"] = "binance"
        return super()._init_ccxt(patched, sync, ccxt_kwargs)

    # ------------------------------------------------------------------
    # Market injection — Polymarket-aware
    # ------------------------------------------------------------------

    def _inject_synthetic_markets(self) -> None:
        """Inject synthetic market entries for event contracts and regular pairs."""
        whitelist = self._config.get("exchange", {}).get("pair_whitelist", [])
        cli_pairs = self._config.get("pairs", [])
        for pair in set(whitelist + cli_pairs):
            if pair not in self._markets:
                if is_polymarket_pair(pair):
                    logger.info("Auto-injecting Polymarket contract: %s", pair)
                    self._markets[pair] = _polymarket_synthetic_market(pair)
                else:
                    # Fall back to parent's generic synthetic market
                    from freqtrade.exchange.portfoliobench import _synthetic_market

                    logger.info("Auto-injecting pair: %s", pair)
                    self._markets[pair] = _synthetic_market(pair)

    # ------------------------------------------------------------------
    # Fees — Polymarket uses 0 maker fees
    # ------------------------------------------------------------------

    def get_fee(
        self,
        symbol: str,
        order_type: str = "",
        side: str = "",
        amount: float = 1,
        price: float = 1,
        taker_or_maker: str = "maker",
    ) -> float:
        """Polymarket charges 0% maker fees; small taker fee on some markets."""
        if is_polymarket_pair(symbol):
            return 0.0
        return super().get_fee(symbol, order_type, side, amount, price, taker_or_maker)
