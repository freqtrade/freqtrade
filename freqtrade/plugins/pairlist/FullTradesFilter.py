"""
Performance pair list filter
"""
import logging
from typing import Any, Dict, List

import pandas as pd

from freqtrade.constants import Config
from freqtrade.exchange.types import Tickers
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter


logger = logging.getLogger(__name__)


class FullTradesFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        # self._minutes = pairlistconfig.get('minutes', 0)
        # self._min_profit = pairlistconfig.get('min_profit')

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

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        return {
            
        }

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
            trades = Trade.get_trades(Trade.is_open.is_(True)).all()
        except AttributeError:
            # Performancefilter does not work in backtesting.
            self.log_once("FullTradesFilter is not available in this mode.", logger.warning)
            return pairlist

        num_open = len(trades)
        max_trades = self._config['max_open_trades']

        # self.log_once(f"Max open trades: {max_trades}, current open trades: {num_open}", logger.info)

        if (num_open >= max_trades):
            # logger.info('Slots full. Emptying pairlist!!')
            return [];

        return pairlist
