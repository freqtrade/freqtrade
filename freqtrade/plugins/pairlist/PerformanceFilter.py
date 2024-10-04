"""
Performance pair list filter
"""

import logging

import pandas as pd

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class PerformanceFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO_ACTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._minutes = self._pairlistconfig.get("minutes", 0)
        self._min_profit = self._pairlistconfig.get("min_profit")

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

    @staticmethod
    def description() -> str:
        return "Filter pairs by performance."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "minutes": {
                "type": "number",
                "default": 0,
                "description": "Minutes",
                "help": "Consider trades from the last X minutes. 0 means all trades.",
            },
            "min_profit": {
                "type": "number",
                "default": None,
                "description": "Minimum profit",
                "help": "Minimum profit in percent. Pairs with less profit are removed.",
            },
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
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
        list_df = pd.DataFrame({"pair": pairlist})
        list_df["prior_idx"] = list_df.index

        # Set initial value for pairs with no trades to 0
        # Sort the list using:
        #  - primarily performance (high to low)
        #  - then count (low to high, so as to favor same performance with fewer trades)
        #  - then by prior index, keeping original sorting order
        sorted_df = (
            list_df.merge(performance, on="pair", how="left")
            .fillna(0)
            .sort_values(by=["profit_ratio", "count", "prior_idx"], ascending=[False, True, True])
        )
        if self._min_profit is not None:
            removed = sorted_df[sorted_df["profit_ratio"] < self._min_profit]
            for _, row in removed.iterrows():
                self.log_once(
                    f"Removing pair {row['pair']} since {row['profit_ratio']} is "
                    f"below {self._min_profit}",
                    logger.info,
                )
            sorted_df = sorted_df[sorted_df["profit_ratio"] >= self._min_profit]

        pairlist = sorted_df["pair"].tolist()

        return pairlist
