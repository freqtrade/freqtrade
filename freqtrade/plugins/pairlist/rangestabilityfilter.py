"""
Rate of change pairlist filter
"""
import logging
from copy import deepcopy
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
        self._refresh_period = pairlistconfig.get('refresh_period', 1440)
        self._def_candletype = self._config['candle_type_def']

        self._pair_cache: TTLCache = TTLCache(maxsize=1000, ttl=self._refresh_period)

        candle_limit = exchange.ohlcv_candle_limit('1d', self._config['candle_type_def'])
        if self._days < 1:
            raise OperationalException("RangeStabilityFilter requires lookback_days to be >= 1")
        if self._days > candle_limit:
            raise OperationalException("RangeStabilityFilter requires lookback_days to not "
                                       f"exceed exchange max request size ({candle_limit})")

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

        since_ms = dt_ts(dt_floor_day(dt_now()) - timedelta(days=self._days - 1))
        # Get all candles
        candles = {}
        if needed_pairs:
            candles = self._exchange.refresh_latest_ohlcv(needed_pairs, since_ms=since_ms,
                                                          cache=False)

        if self._enabled:
            for p in deepcopy(pairlist):
                daily_candles = candles[(p, '1d', self._def_candletype)] if (
                    p, '1d', self._def_candletype) in candles else None
                if not self._validate_pair_loc(p, daily_candles):
                    pairlist.remove(p)
        return pairlist

    def _validate_pair_loc(self, pair: str, daily_candles: Optional[DataFrame]) -> bool:
        """
        Validate trading range
        :param pair: Pair that's currently validated
        :param daily_candles: Downloaded daily candles
        :return: True if the pair can stay, false if it should be removed
        """
        # Check symbol in cache
        cached_res = self._pair_cache.get(pair, None)
        if cached_res is not None:
            return cached_res

        result = True
        if daily_candles is not None and not daily_candles.empty:
            highest_high = daily_candles['high'].max()
            lowest_low = daily_candles['low'].min()
            pct_change = ((highest_high - lowest_low) / lowest_low) if lowest_low > 0 else 0
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
            self._pair_cache[pair] = result
        else:
            self.log_once(f"Removed {pair} from whitelist, no candles found.", logger.info)
        return result
