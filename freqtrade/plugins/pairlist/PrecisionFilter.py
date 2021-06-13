"""
Precision pair list filter
"""
import logging
from typing import Any, Dict

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class PrecisionFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'stoploss' not in self._config:
            raise OperationalException(
                'PrecisionFilter can only work with stoploss defined. Please add the '
                'stoploss key to your configuration (overwrites eventual strategy settings).')
        self._stoploss = self._config['stoploss']
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

    def _validate_pair(self, pair: str, ticker: Dict[str, Any]) -> bool:
        """
        Check if pair has enough room to add a stoploss to avoid "unsellable" buys of very
        low value pairs.
        :param pair: Pair that's currently validated
        :param ticker: ticker dict as returned from ccxt.fetch_tickers()
        :return: True if the pair can stay, false if it should be removed
        """
        stop_price = ticker['ask'] * self._stoploss

        # Adjust stop-prices to precision
        sp = self._exchange.price_to_precision(pair, stop_price)

        stop_gap_price = self._exchange.price_to_precision(pair, stop_price * 0.99)
        logger.debug(f"{ticker['symbol']} - {sp} : {stop_gap_price}")

        if sp <= stop_gap_price:
            self.log_once(f"Removed {ticker['symbol']} from whitelist, because "
                          f"stop price {sp} would be <= stop limit {stop_gap_price}", logger.info)
            return False

        return True
