"""
Shuffle pair list filter
"""
import logging
import random
from typing import Any, Dict, List

from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class ShuffleFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._seed = pairlistconfig.get('seed')
        self._random = random.Random(self._seed)

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (f"{self.name} - Shuffling pairs" +
                (f", seed = {self._seed}." if self._seed is not None else "."))

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Shuffle is done inplace
        self._random.shuffle(pairlist)

        return pairlist
