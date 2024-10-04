"""
Shuffle pair list filter
"""

import logging
import random
from typing import Literal

from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util.periodic_cache import PeriodicCache


logger = logging.getLogger(__name__)

ShuffleValues = Literal["candle", "iteration"]


class ShuffleFilter(IPairList):
    supports_backtesting = SupportsBacktesting.YES

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Apply seed in backtesting mode to get comparable results,
        # but not in live modes to get a non-repeating order of pairs during live modes.
        if self._config.get("runmode") in (RunMode.LIVE, RunMode.DRY_RUN):
            self._seed = None
            logger.info("Live mode detected, not applying seed.")
        else:
            self._seed = self._pairlistconfig.get("seed")
            logger.info(f"Backtesting mode detected, applying seed value: {self._seed}")

        self._random = random.Random(self._seed)  # noqa: S311
        self._shuffle_freq: ShuffleValues = self._pairlistconfig.get("shuffle_frequency", "candle")
        self.__pairlist_cache = PeriodicCache(
            maxsize=1000, ttl=timeframe_to_seconds(self._config["timeframe"])
        )

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
        return f"{self.name} - Shuffling pairs every {self._shuffle_freq}" + (
            f", seed = {self._seed}." if self._seed is not None else "."
        )

    @staticmethod
    def description() -> str:
        return "Randomize pairlist order."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "shuffle_frequency": {
                "type": "option",
                "default": "candle",
                "options": ["candle", "iteration"],
                "description": "Shuffle frequency",
                "help": "Shuffle frequency. Can be either 'candle' or 'iteration'.",
            },
            "seed": {
                "type": "number",
                "default": None,
                "description": "Random Seed",
                "help": "Seed for random number generator. Not used in live mode.",
            },
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        pairlist_bef = tuple(pairlist)
        pairlist_new = self.__pairlist_cache.get(pairlist_bef)
        if pairlist_new and self._shuffle_freq == "candle":
            # Use cached pairlist.
            return pairlist_new
        # Shuffle is done inplace
        self._random.shuffle(pairlist)
        self.__pairlist_cache[pairlist_bef] = pairlist

        return pairlist
