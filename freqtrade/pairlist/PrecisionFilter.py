import logging
from copy import deepcopy
from typing import Dict, List

from freqtrade.pairlist.IPairList import IPairList

logger = logging.getLogger(__name__)


class PrecisionFilter(IPairList):

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requries tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """
        return True

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - Filtering untradable pairs."

    def _validate_precision_filter(self, ticker: dict, stoploss: float) -> bool:
        """
        Check if pair has enough room to add a stoploss to avoid "unsellable" buys of very
        low value pairs.
        :param ticker: ticker dict as returned from ccxt.load_markets()
        :param stoploss: stoploss value as set in the configuration
                        (already cleaned to be 1 - stoploss)
        :return: True if the pair can stay, false if it should be removed
        """
        stop_price = ticker['ask'] * stoploss
        # Adjust stop-prices to precision
        sp = self._exchange.price_to_precision(ticker["symbol"], stop_price)
        stop_gap_price = self._exchange.price_to_precision(ticker["symbol"], stop_price * 0.99)
        logger.debug(f"{ticker['symbol']} - {sp} : {stop_gap_price}")
        if sp <= stop_gap_price:
            logger.info(f"Removed {ticker['symbol']} from whitelist, "
                        f"because stop price {sp} would be <= stop limit {stop_gap_price}")
            return False
        return True

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlists and assigns and returns them again.
        """
        stoploss = None
        if self._config.get('stoploss') is not None:
            # Precalculate sanitized stoploss value to avoid recalculation for every pair
            stoploss = 1 - abs(self._config.get('stoploss'))
        # Copy list since we're modifying this list
        for p in deepcopy(pairlist):
            ticker = tickers.get(p)
            # Filter out assets which would not allow setting a stoploss
            if not ticker or (stoploss and not self._validate_precision_filter(ticker, stoploss)):
                pairlist.remove(p)
                continue

        return pairlist
