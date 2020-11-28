"""
Performance pair list filter
"""
import logging
from typing import Any, Dict, List

import pandas as pd

from freqtrade.pairlist.IPairList import IPairList

from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class PerformanceFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requries tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - Sorting pairs by performance."

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Get the trading performance for pairs from database
        perf = pd.DataFrame(Trade.get_overall_performance())
        # get pairlist from performance dataframe values
        list_df = pd.DataFrame({'pair': pairlist})
        # set initial value for pairs with no trades to 0
        # and sort the list using performance and count
        sorted_df = list_df.join(perf.set_index('pair'), on='pair')\
            .fillna(0).sort_values(by=['profit', 'count'], ascending=False)
        pairlist = sorted_df['pair'].tolist()

        return pairlist
