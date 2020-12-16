"""
Minimum age (days listed) pair list filter
"""
import logging
from typing import Any, Dict, List

import arrow

from freqtrade.exceptions import OperationalException
from freqtrade.misc import plural
from freqtrade.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class AgeFilter(IPairList):

    # Checked symbols cache (dictionary of ticker symbol => timestamp)
    _symbolsChecked: Dict[str, int] = {}

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._min_days_listed = pairlistconfig.get('min_days_listed', 10)

        if self._min_days_listed < 1:
            raise OperationalException("AgeFilter requires min_days_listed to be >= 1")
        if self._min_days_listed > exchange.ohlcv_candle_limit:
            raise OperationalException("AgeFilter requires min_days_listed to not exceed "
                                       "exchange max request size "
                                       f"({exchange.ohlcv_candle_limit})")

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return True

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (f"{self.name} - Filtering pairs with age less than "
                f"{self._min_days_listed} {plural(self._min_days_listed, 'day')}.")

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new allowlist
        """
        needed_pairs = [(p, '1d') for p in pairlist if p not in self._symbolsChecked]
        if not needed_pairs:
            return pairlist

        since_ms = int(arrow.utcnow()
                       .floor('day')
                       .shift(days=-self._min_days_listed - 1)
                       .float_timestamp) * 1000
        candles = self._exchange.refresh_latest_ohlcv(needed_pairs, since_ms=since_ms, cache=False)
        pairlist_new = []
        if self._enabled:
            for p, _ in needed_pairs:

                age = len(candles[(p, '1d')]) if (p, '1d') in candles else 0
                if age > self._min_days_listed:
                    pairlist_new.append(p)
                    self._symbolsChecked[p] = int(arrow.utcnow().float_timestamp) * 1000
                else:
                    self.log_once(f"Removed {p} from whitelist, because age "
                                  f"{age} is less than {self._min_days_listed} "
                                  f"{plural(self._min_days_listed, 'day')}", logger.info)
        logger.info(f"Validated {len(pairlist_new)} pairs.")
        return pairlist_new
