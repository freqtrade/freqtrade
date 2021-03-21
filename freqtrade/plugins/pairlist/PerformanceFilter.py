"""
Performance pair list filter
"""
import logging
from typing import Dict, List

import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class PerformanceFilter(IPairList):

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
        Short allowlist method description - used for startup-messages
        """
        return f"{self.name} - Sorting pairs by performance."

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the allowlist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new allowlist
        """
        # Get the trading performance for pairs from database
        performance = pd.DataFrame(Trade.get_overall_performance())

        # Skip performance-based sorting if no performance data is available
        if len(performance) == 0:
            return pairlist

        # Get pairlist from performance dataframe values
        list_df = pd.DataFrame({'pair': pairlist})

        # Set initial value for pairs with no trades to 0
        # Sort the list using:
        #  - primarily performance (high to low)
        #  - then count (low to high, so as to favor same performance with fewer trades)
        #  - then pair name alphametically
        sorted_df = list_df.merge(performance, on='pair', how='left')\
            .fillna(0).sort_values(by=['count', 'pair'], ascending=True)\
            .sort_values(by=['profit'], ascending=False)
        pairlist = sorted_df['pair'].tolist()

        return pairlist
