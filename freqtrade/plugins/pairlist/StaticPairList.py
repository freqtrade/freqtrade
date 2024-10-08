"""
Static Pair List provider

Provides pair white list as it configured in config
"""

import logging
from copy import deepcopy

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class StaticPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.YES

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._allow_inactive = self._pairlistconfig.get("allow_inactive", False)

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
        -> Please overwrite in subclasses
        """
        return f"{self.name}"

    @staticmethod
    def description() -> str:
        return "Use pairlist as configured in config."

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
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        wl = self.verify_whitelist(
            self._config["exchange"]["pair_whitelist"], logger.info, keep_invalid=True
        )
        if self._allow_inactive:
            return wl
        else:
            # Avoid implicit filtering of "verify_whitelist" to keep
            # proper warnings in the log
            return self._whitelist_for_active_markets(wl)

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        pairlist_ = deepcopy(pairlist)
        for pair in self._config["exchange"]["pair_whitelist"]:
            if pair not in pairlist_:
                pairlist_.append(pair)
        return pairlist_
