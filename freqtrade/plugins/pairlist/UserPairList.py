"""
User Pair List provider

Reads pairs from the user_pairs table. RPC can add and remove them at runtime.
"""

import logging

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.persistence.user_pair import UserPair
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class UserPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._allow_inactive = self._pairlistconfig.get("allow_inactive", False)

    def short_desc(self) -> str:
        """Short whitelist method description - used for startup-messages."""
        return f"{self.name} - {len(UserPair.list_pairs())} pairs."

    @staticmethod
    def description() -> str:
        return "Use pairs stored in the database. RPC can add and remove them at runtime."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "allow_inactive": {
                "type": "boolean",
                "default": False,
                "description": "Allow inactive pairs",
                "help": "Allow inactive pairs to be in the whitelist.",
            },
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist from the user_pairs table.
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        stored = UserPair.list_pairs()
        if not stored:
            return []

        pairlist = self.verify_whitelist(stored, logger.info, keep_invalid=True)
        if self._allow_inactive:
            return pairlist
        return self._whitelist_for_active_markets(pairlist)

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Append stored pairs that are not already in the incoming list.
        Called on each bot iteration.
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        pairlist_ = pairlist.copy()
        for pair in self.gen_pairlist(tickers):
            if pair not in pairlist_:
                pairlist_.append(pair)
        return pairlist_
