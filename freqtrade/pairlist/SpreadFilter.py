import logging
from copy import deepcopy
from typing import Dict, List

from freqtrade.pairlist.IPairList import IPairList

logger = logging.getLogger(__name__)


class SpreadFilter(IPairList):

    def __init__(self, exchange, pairlistmanager, config, pairlistconfig: dict,
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._max_spread_ratio = pairlistconfig.get('max_spread_ratio', 0.005)

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
        return (f"{self.name} - Filtering pairs with ask/bid diff above "
                f"{self._max_spread_ratio * 100}%.")

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:

        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Copy list since we're modifying this list

        spread = None
        for p in deepcopy(pairlist):
            ticker = tickers.get(p)
            assert ticker is not None
            if 'bid' in ticker and 'ask' in ticker:
                spread = 1 - ticker['bid'] / ticker['ask']
                if not ticker or spread > self._max_spread_ratio:
                    logger.info(f"Removed {ticker['symbol']} from whitelist, "
                                f"because spread {spread * 100:.3f}% >"
                                f"{self._max_spread_ratio * 100}%")
                    pairlist.remove(p)
            else:
                pairlist.remove(p)

        return pairlist
