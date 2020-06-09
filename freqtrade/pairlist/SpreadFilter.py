"""
Spread pair list filter
"""
import logging
from typing import Any, Dict

from freqtrade.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class SpreadFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._max_spread_ratio = pairlistconfig.get('max_spread_ratio', 0.005)
        self._enabled = self._max_spread_ratio != 0

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requries tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """
        return True

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (f"{self.name} - Filtering pairs with ask/bid diff above "
                f"{self._max_spread_ratio * 100}%.")

    def _validate_pair(self, ticker: dict) -> bool:
        """
        Validate spread for the ticker
        :param ticker: ticker dict as returned from ccxt.load_markets()
        :return: True if the pair can stay, False if it should be removed
        """
        if 'bid' in ticker and 'ask' in ticker:
            spread = 1 - ticker['bid'] / ticker['ask']
            if spread > self._max_spread_ratio:
                self.log_on_refresh(logger.info, f"Removed {ticker['symbol']} from whitelist, "
                                                 f"because spread {spread * 100:.3f}% >"
                                                 f"{self._max_spread_ratio * 100}%")
                return False
            else:
                return True
        return False
