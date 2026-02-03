"""
Full trade slots pair list filter
"""

import logging

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList, SupportsBacktesting


logger = logging.getLogger(__name__)


class FullTradesFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO_ACTION

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
        return f"{self.name} - Shrink whitelist when trade slots are full."

    @staticmethod
    def description() -> str:
        return "Shrink whitelist when trade slots are full."

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Filters and sorts pairlist and returns the allowlist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new allowlist
        """
        # Get the number of open trades and max open trades config
        num_open = Trade.get_open_trade_count()
        max_trades = self._config["max_open_trades"]

        if (num_open >= max_trades) and (max_trades > 0):
            return []

        return pairlist
