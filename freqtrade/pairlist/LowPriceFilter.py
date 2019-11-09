import logging
from copy import deepcopy
from typing import Dict, List

from freqtrade.pairlist.IPairList import IPairList

logger = logging.getLogger(__name__)


class LowPriceFilter(IPairList):

    def __init__(self, exchange, pairlistmanager, config, pairlistconfig: dict) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig)

        self._low_price_percent = pairlistconfig.get('low_price_percent', 0)

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
        return f"{self.name} - Filtering pairs priced below {self._low_price_percent * 100}%."

    def _validate_ticker_lowprice(self, ticker) -> bool:
        """
        Check if if one price-step is > than a certain barrier.
        :param ticker: ticker dict as returned from ccxt.load_markets()
        :param precision: Precision
        :return: True if the pair can stay, false if it should be removed
        """
        precision = self._exchange.markets[ticker['symbol']]['precision']['price']

        compare = ticker['last'] + 1 / pow(10, precision)
        changeperc = (compare - ticker['last']) / ticker['last']
        if changeperc > self._low_price_percent:
            logger.info(f"Removed {ticker['symbol']} from whitelist, "
                        f"because 1 unit is {changeperc * 100:.3f}%")
            return False
        return True

    def filter_pairlist(self, pairlist: List[str], tickers: List[Dict]) -> List[str]:

        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Copy list since we're modifying this list
        for p in deepcopy(pairlist):
            ticker = [t for t in tickers if t['symbol'] == p][0]

            # Filter out assets which would not allow setting a stoploss
            if self._low_price_percent and not self._validate_ticker_lowprice(ticker):
                pairlist.remove(p)

        return pairlist
