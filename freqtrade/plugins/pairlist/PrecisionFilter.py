"""
Precision pair list filter
"""

import logging
from typing import Optional

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import ROUND_UP
from freqtrade.exchange.exchange_types import Ticker
from freqtrade.plugins.pairlist.IPairList import IPairList, SupportsBacktesting


logger = logging.getLogger(__name__)


class PrecisionFilter(IPairList):
    supports_backtesting = SupportsBacktesting.BIASED

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "stoploss" not in self._config:
            raise OperationalException(
                "PrecisionFilter can only work with stoploss defined. Please add the "
                "stoploss key to your configuration (overwrites eventual strategy settings)."
            )
        self._stoploss = self._config["stoploss"]
        self._enabled = self._stoploss != 0

        # Precalculate sanitized stoploss value to avoid recalculation for every pair
        self._stoploss = 1 - abs(self._stoploss)

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return True

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - Filtering untradable pairs."

    @staticmethod
    def description() -> str:
        return "Filters low-value coins which would not allow setting stoplosses."

    def _validate_pair(self, pair: str, ticker: Optional[Ticker]) -> bool:
        """
        Check if pair has enough room to add a stoploss to avoid "unsellable" buys of very
        low value pairs.
        :param pair: Pair that's currently validated
        :param ticker: ticker dict as returned from ccxt.fetch_ticker
        :return: True if the pair can stay, false if it should be removed
        """
        if not ticker or ticker.get("last", None) is None:
            self.log_once(
                f"Removed {pair} from whitelist, because "
                "ticker['last'] is empty (Usually no trade in the last 24h).",
                logger.info,
            )
            return False
        stop_price = ticker["last"] * self._stoploss

        # Adjust stop-prices to precision
        sp = self._exchange.price_to_precision(pair, stop_price, rounding_mode=ROUND_UP)

        stop_gap_price = self._exchange.price_to_precision(
            pair, stop_price * 0.99, rounding_mode=ROUND_UP
        )
        logger.debug(f"{pair} - {sp} : {stop_gap_price}")

        if sp <= stop_gap_price:
            self.log_once(
                f"Removed {pair} from whitelist, because "
                f"stop price {sp} would be <= stop limit {stop_gap_price}",
                logger.info,
            )
            return False

        return True
