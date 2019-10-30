import logging
from copy import deepcopy
from typing import Dict, List

from freqtrade.pairlist.IPairListFilter import IPairListFilter

logger = logging.getLogger(__name__)


class LowPriceFilter(IPairListFilter):

    def __init__(self, freqtrade, config: dict) -> None:
        super().__init__(freqtrade, config)

        self._low_price_percent = config['pairlist']['filters']['LowPriceFilter'].get(
            'low_price_percent', 0)

    def _validate_ticker_lowprice(self, ticker) -> bool:
        """
        Check if if one price-step is > than a certain barrier.
        :param ticker: ticker dict as returned from ccxt.load_markets()
        :param precision: Precision
        :return: True if the pair can stay, false if it should be removed
        """
        precision = self._freqtrade.exchange.markets[ticker['symbol']]['precision']['price']

        compare = ticker['last'] + 1 / pow(10, precision)
        changeperc = (compare - ticker['last']) / ticker['last']
        if changeperc > self._low_price_percent:
            logger.info(f"Removed {ticker['symbol']} from whitelist, "
                        f"because 1 unit is {changeperc * 100:.3f}%")
            return False
        return True

    def filter_pairlist(self, pairlist: List[str], tickers: List[Dict]) -> List[str]:
        """
        Method doing the filtering
        """

        # Copy list since we're modifying this list
        for p in deepcopy(pairlist):
            ticker = [t for t in tickers if t['symbol'] == p][0]

            # Filter out assets which would not allow setting a stoploss
            if self._low_price_percent and not self._validate_ticker_lowprice(ticker):
                pairlist.remove(p)

        return pairlist
