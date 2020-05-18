"""
Price pair list filter
"""
import logging
from copy import deepcopy
from typing import Any, Dict, List

from freqtrade.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class PriceFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._low_price_ratio = pairlistconfig.get('low_price_ratio', 0)

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
        return f"{self.name} - Filtering pairs priced below {self._low_price_ratio * 100}%."

    def _validate_ticker_lowprice(self, ticker) -> bool:
        """
        Check if if one price-step (pip) is > than a certain barrier.
        :param ticker: ticker dict as returned from ccxt.load_markets()
        :return: True if the pair can stay, false if it should be removed
        """
        if ticker['last'] is None:
            self.log_on_refresh(logger.info,
                                f"Removed {ticker['symbol']} from whitelist, because "
                                "ticker['last'] is empty (Usually no trade in the last 24h).")
            return False
        compare = self._exchange.price_get_one_pip(ticker['symbol'], ticker['last'])
        changeperc = compare / ticker['last']
        if changeperc > self._low_price_ratio:
            self.log_on_refresh(logger.info, f"Removed {ticker['symbol']} from whitelist, "
                                             f"because 1 unit is {changeperc * 100:.3f}%")
            return False
        return True

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        if self._low_price_ratio:
            # Copy list since we're modifying this list
            for p in deepcopy(pairlist):
                # Filter out assets which would not allow setting a stoploss
                if not self._validate_ticker_lowprice(tickers[p]):
                    pairlist.remove(p)

        return pairlist
