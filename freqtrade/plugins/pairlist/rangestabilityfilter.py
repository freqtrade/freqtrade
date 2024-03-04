"""
Rate of change pairlist filter
"""
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from cachetools import TTLCache
from pandas import DataFrame

from freqtrade.constants import Config, ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.types import Tickers
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter
from freqtrade.util import dt_floor_day, dt_now, dt_ts


logger = logging.getLogger(__name__)


class RangeStabilityFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._days = pairlistconfig.get('lookback_days', 10)
        self._min_rate_of_change = pairlistconfig.get('min_rate_of_change', 0.01)
        self._max_rate_of_change = pairlistconfig.get('max_rate_of_change')
        self._refresh_period = pairlistconfig.get('refresh_period', 86400)
        self._def_candletype = self._config['candle_type_def']
        self._sort_direction: Optional[str] = pairlistconfig.get('sort_direction', None)

        self._pair_cache: TTLCache = TTLCache(maxsize=1000, ttl=self._refresh_period)

        candle_limit = exchange.ohlcv_candle_limit('1d', self._config['candle_type_def'])
        if self._days < 1:
            raise OperationalException("RangeStabilityFilter requires lookback_days to be >= 1")
        if self._days > candle_limit:
            raise OperationalException("RangeStabilityFilter requires lookback_days to not "
                                       f"exceed exchange max request size ({candle_limit})")
        if self._sort_direction not in [None, 'asc', 'desc']:
            raise OperationalException("RangeStabilityFilter requires sort_direction to be "
                                       "either None (undefined), 'asc' or 'desc'")

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
        max_rate_desc = ""
        if self._max_rate_of_change:
            max_rate_desc = (f" and above {self._max_rate_of_change}")
        return (f"{self.name} - Filtering pairs with rate of change below "
                f"{self._min_rate_of_change}{max_rate_desc} over the "
                f"last {plural(self._days, 'day')}.")

    @staticmethod
    def description() -> str:
        return "Filters pairs by their rate of change."

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        return {
            "lookback_days": {
                "type": "number",
                "default": 10,
                "description": "Lookback Days",
                "help": "Number of days to look back at.",
            },
            "min_rate_of_change": {
                "type": "number",
                "default": 0.01,
                "description": "Minimum Rate of Change",
                "help": "Minimum rate of change to filter pairs.",
            },
            "max_rate_of_change": {
                "type": "number",
                "default": None,
                "description": "Maximum Rate of Change",
                "help": "Maximum rate of change to filter pairs.",
            },
            "sort_direction": {
                "type": "option",
                "default": None,
                "options": ["", "asc", "desc"],
                "description": "Sort pairlist",
                "help": "Sort Pairlist ascending or descending by rate of change.",
            },
            **IPairList.refresh_period_parameter()
        }

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        """
        Validate trading range
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new allowlist
        """
        needed_pairs: ListPairsWithTimeframes = [
            (p, '1d', self._def_candletype) for p in pairlist if p not in self._pair_cache]

        since_ms = dt_ts(dt_floor_day(dt_now()) - timedelta(days=self._days + 1))
        candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms=since_ms)

        resulting_pairlist: List[str] = []
        pct_changes: Dict[str, float] = {}

        for p in pairlist:
            daily_candles = candles.get((p, '1d', self._def_candletype), None)

            pct_change = self._calculate_rate_of_change(p, daily_candles)

            if pct_change is not None:
                if self._validate_pair_loc(p, pct_change):
                    resulting_pairlist.append(p)
                    pct_changes[p] = pct_change
            else:
                self.log_once(f"Removed {p} from whitelist, no candles found.", logger.info)

        if self._sort_direction:
            resulting_pairlist = sorted(resulting_pairlist,
                                        key=lambda p: pct_changes[p],
                                        reverse=self._sort_direction == 'desc')
        return resulting_pairlist

    def _calculate_rate_of_change(self, pair: str, daily_candles: DataFrame) -> Optional[float]:
        # Check symbol in cache
        if (pct_change := self._pair_cache.get(pair, None)) is not None:
            return pct_change
        if daily_candles is not None and not daily_candles.empty:

            highest_high = daily_candles['high'].max()
            lowest_low = daily_candles['low'].min()
            pct_change = ((highest_high - lowest_low) / lowest_low) if lowest_low > 0 else 0
            self._pair_cache[pair] = pct_change
            return pct_change
        else:
            return None

    def _validate_pair_loc(self, pair: str, pct_change: float) -> bool:
        """
        Validate trading range
        :param pair: Pair that's currently validated
        :param pct_change: Rate of change
        :return: True if the pair can stay, false if it should be removed
        """

        result = True
        if pct_change < self._min_rate_of_change:
            self.log_once(f"Removed {pair} from whitelist, because rate of change "
                          f"over {self._days} {plural(self._days, 'day')} is {pct_change:.3f}, "
                          f"which is below the threshold of {self._min_rate_of_change}.",
                          logger.info)
            result = False
        if self._max_rate_of_change:
            if pct_change > self._max_rate_of_change:
                self.log_once(
                    f"Removed {pair} from whitelist, because rate of change "
                    f"over {self._days} {plural(self._days, 'day')} is {pct_change:.3f}, "
                    f"which is above the threshold of {self._max_rate_of_change}.",
                    logger.info)
                result = False
        return result
