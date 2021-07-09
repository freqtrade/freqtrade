"""
Offset pair list filter
"""
import logging
from typing import Any, Dict, List

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class OffsetFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._offset = pairlistconfig.get('offset', 0)

        if self._offset < 0:
            raise OperationalException("OffsetFilter requires offset to be >= 0")

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
        return f"{self.name} - Offseting pairs by {self._offset}."

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        if self._offset > len(pairlist):
            self.log_once(f"Offset of {self._offset} is larger than " +
                          f"pair count of {len(pairlist)}", logger.warning)
        pairs = pairlist[self._offset:]
        self.log_once(f"Searching {len(pairs)} pairs: {pairs}", logger.info)
        return pairs
