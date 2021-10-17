"""
Volatility pairlist filter
"""
import logging
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

import arrow
import numpy as np
from cachetools.ttl import TTLCache
from pandas import DataFrame

from freqtrade.exceptions import OperationalException
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class VolatilityFilter(IPairList):
    """
    Filters pairs by volatility
    """

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._days = pairlistconfig.get('lookback_days', 10)
        self._min_volatility = pairlistconfig.get('min_volatility', 0)
        self._max_volatility = pairlistconfig.get('max_volatility', sys.maxsize)
        self._refresh_period = pairlistconfig.get('refresh_period', 1440)

        self._pair_cache: TTLCache = TTLCache(maxsize=1000, ttl=self._refresh_period)

        if self._days < 1:
            raise OperationalException("VolatilityFilter requires lookback_days to be >= 1")
        if self._days > exchange.ohlcv_candle_limit('1d'):
            raise OperationalException("VolatilityFilter requires lookback_days to not "
                                       "exceed exchange max request size "
                                       f"({exchange.ohlcv_candle_limit('1d')})")

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
        Short whitelist method description - used for startup-messages
        """
        return (f"{self.name} - Filtering pairs with volatility range "
                f"{self._min_volatility}-{self._max_volatility} "
                f" the last {self._days} {plural(self._days, 'day')}.")

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Validate trading range
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new allowlist
        """
        needed_pairs = [(p, '1d') for p in pairlist if p not in self._pair_cache]

        since_ms = (arrow.utcnow()
                         .floor('day')
                         .shift(days=-self._days - 1)
                         .int_timestamp) * 1000
        # Get all candles
        candles = {}
        if needed_pairs:
            candles = self._exchange.refresh_latest_ohlcv(needed_pairs, since_ms=since_ms,
                                                          cache=False)

        if self._enabled:
            for p in deepcopy(pairlist):
                daily_candles = candles[(p, '1d')] if (p, '1d') in candles else None
                if not self._validate_pair_loc(p, daily_candles):
                    pairlist.remove(p)
        return pairlist

    def _validate_pair_loc(self, pair: str, daily_candles: Optional[DataFrame]) -> bool:
        """
        Validate trading range
        :param pair: Pair that's currently validated
        :param ticker: ticker dict as returned from ccxt.fetch_tickers()
        :return: True if the pair can stay, false if it should be removed
        """
        # Check symbol in cache
        cached_res = self._pair_cache.get(pair, None)
        if cached_res is not None:
            return cached_res

        result = False
        if daily_candles is not None and not daily_candles.empty:
            returns = (np.log(daily_candles.close / daily_candles.close.shift(-1)))
            returns.fillna(0, inplace=True)

            volatility_series = returns.rolling(window=self._days).std()*np.sqrt(self._days)
            volatility_avg = volatility_series.mean()

            if self._min_volatility <= volatility_avg <= self._max_volatility:
                result = True
            else:
                self.log_once(f"Removed {pair} from whitelist, because volatility "
                              f"over {self._days} {plural(self._days, 'day')} "
                              f"is: {volatility_avg:.3f} "
                              f"which is not in the configured range of "
                              f"{self._min_volatility}-{self._max_volatility}.",
                              logger.info)
                result = False
            self._pair_cache[pair] = result

        return result
