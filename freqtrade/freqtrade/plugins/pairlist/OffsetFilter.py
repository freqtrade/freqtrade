"""
Offset pair list filter
"""

import logging

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class OffsetFilter(IPairList):
    supports_backtesting = SupportsBacktesting.YES

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._offset = self._pairlistconfig.get("offset", 0)
        self._number_pairs = self._pairlistconfig.get("number_assets", 0)

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
        if self._number_pairs:
            return f"{self.name} - Taking {self._number_pairs} Pairs, starting from {self._offset}."
        return f"{self.name} - Offsetting pairs by {self._offset}."

    @staticmethod
    def description() -> str:
        return "Offset pair list filter."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "offset": {
                "type": "number",
                "default": 0,
                "description": "Offset",
                "help": "Offset of the pairlist.",
            },
            "number_assets": {
                "type": "number",
                "default": 0,
                "description": "Number of assets",
                "help": "Number of assets to use from the pairlist, starting from offset.",
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
        if self._offset > len(pairlist):
            self.log_once(
                f"Offset of {self._offset} is larger than " + f"pair count of {len(pairlist)}",
                logger.warning,
            )
        pairs = pairlist[self._offset :]
        if self._number_pairs:
            pairs = pairs[: self._number_pairs]

        return pairs
