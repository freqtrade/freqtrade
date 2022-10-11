"""
Performance pair list filter
"""
import logging
from typing import Any, Dict, List

import pandas as pd

from freqtrade.constants import Config
from freqtrade.exchange.types import Tickers
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class PerformanceFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._minutes = pairlistconfig.get('minutes', 0)
        self._min_profit = pairlistconfig.get('min_profit')

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short allowlist method description - used for startup-messages
        """
        return f"{self.name} - Sorting pairs by performance."

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        """
        Filters and sorts pairlist and returns the allowlist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new allowlist
        """
        # Get the trading performance for pairs from database
        try:
            performance = pd.DataFrame(Trade.get_overall_performance(self._minutes))
        except AttributeError:
            # Performancefilter does not work in backtesting.
            self.log_once("PerformanceFilter is not available in this mode.", logger.warning)
            return pairlist

        # Skip performance-based sorting if no performance data is available
        if len(performance) == 0:
            return pairlist

        # Get pairlist from performance dataframe values
        list_df = pd.DataFrame({'pair': pairlist})
        list_df['prior_idx'] = list_df.index

        # Set initial value for pairs with no trades to 0
        # Sort the list using:
        #  - primarily performance (high to low)
        #  - then count (low to high, so as to favor same performance with fewer trades)
        #  - then pair name alphametically
        sorted_df = list_df.merge(performance, on='pair', how='left')\
            .fillna(0).sort_values(by=['count', 'prior_idx'], ascending=True)\
            .sort_values(by=['profit_ratio'], ascending=False)
        if self._min_profit is not None:
            removed = sorted_df[sorted_df['profit_ratio'] < self._min_profit]
            for _, row in removed.iterrows():
                self.log_once(
                    f"Removing pair {row['pair']} since {row['profit_ratio']} is "
                    f"below {self._min_profit}", logger.info)
            sorted_df = sorted_df[sorted_df['profit_ratio'] >= self._min_profit]

        pairlist = sorted_df['pair'].tolist()

        return pairlist
