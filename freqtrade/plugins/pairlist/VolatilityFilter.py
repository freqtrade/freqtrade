"""
Volatility pairlist filter
"""

import logging
import sys
from datetime import timedelta
from typing import Optional

import numpy as np
from cachetools import TTLCache
from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_floor_day, dt_now, dt_ts


logger = logging.getLogger(__name__)


class VolatilityFilter(IPairList):
    """
    Filters pairs by volatility
    """

    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._days = self._pairlistconfig.get("lookback_days", 10)
        self._min_volatility = self._pairlistconfig.get("min_volatility", 0)
        self._max_volatility = self._pairlistconfig.get("max_volatility", sys.maxsize)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1440)
        self._def_candletype = self._config["candle_type_def"]
        self._sort_direction: Optional[str] = self._pairlistconfig.get("sort_direction", None)

        self._pair_cache: TTLCache = TTLCache(maxsize=1000, ttl=self._refresh_period)

        candle_limit = self._exchange.ohlcv_candle_limit("1d", self._config["candle_type_def"])
        if self._days < 1:
            raise OperationalException("VolatilityFilter requires lookback_days to be >= 1")
        if self._days > candle_limit:
            raise OperationalException(
                "VolatilityFilter requires lookback_days to not "
                f"exceed exchange max request size ({candle_limit})"
            )
        if self._sort_direction not in [None, "asc", "desc"]:
            raise OperationalException(
                "VolatilityFilter requires sort_direction to be "
                "either None (undefined), 'asc' or 'desc'"
            )

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
        return (
            f"{self.name} - Filtering pairs with volatility range "
            f"{self._min_volatility}-{self._max_volatility} "
            f" the last {self._days} {plural(self._days, 'day')}."
        )

    @staticmethod
    def description() -> str:
        return "Filter pairs by their recent volatility."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "lookback_days": {
                "type": "number",
                "default": 10,
                "description": "Lookback Days",
                "help": "Number of days to look back at.",
            },
            "min_volatility": {
                "type": "number",
                "default": 0,
                "description": "Minimum Volatility",
                "help": "Minimum volatility a pair must have to be considered.",
            },
            "max_volatility": {
                "type": "number",
                "default": None,
                "description": "Maximum Volatility",
                "help": "Maximum volatility a pair must have to be considered.",
            },
            "sort_direction": {
                "type": "option",
                "default": None,
                "options": ["", "asc", "desc"],
                "description": "Sort pairlist",
                "help": "Sort Pairlist ascending or descending by volatility.",
            },
            **IPairList.refresh_period_parameter(),
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Validate trading range
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new allowlist
        """
        needed_pairs: ListPairsWithTimeframes = [
            (p, "1d", self._def_candletype) for p in pairlist if p not in self._pair_cache
        ]

        since_ms = dt_ts(dt_floor_day(dt_now()) - timedelta(days=self._days))
        candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms=since_ms)

        resulting_pairlist: list[str] = []
        volatilitys: dict[str, float] = {}
        for p in pairlist:
            daily_candles = candles.get((p, "1d", self._def_candletype), None)

            volatility_avg = self._calculate_volatility(p, daily_candles)

            if volatility_avg is not None:
                if self._validate_pair_loc(p, volatility_avg):
                    resulting_pairlist.append(p)
                    volatilitys[p] = (
                        volatility_avg if volatility_avg and not np.isnan(volatility_avg) else 0
                    )
            else:
                self.log_once(f"Removed {p} from whitelist, no candles found.", logger.info)

        if self._sort_direction:
            resulting_pairlist = sorted(
                resulting_pairlist,
                key=lambda p: volatilitys[p],
                reverse=self._sort_direction == "desc",
            )
        return resulting_pairlist

    def _calculate_volatility(self, pair: str, daily_candles: DataFrame) -> Optional[float]:
        # Check symbol in cache
        if (volatility_avg := self._pair_cache.get(pair, None)) is not None:
            return volatility_avg

        if daily_candles is not None and not daily_candles.empty:
            returns = np.log(daily_candles["close"].shift(1) / daily_candles["close"])
            returns.fillna(0, inplace=True)

            volatility_series = returns.rolling(window=self._days).std() * np.sqrt(self._days)
            volatility_avg = volatility_series.mean()
            self._pair_cache[pair] = volatility_avg

            return volatility_avg
        else:
            return None

    def _validate_pair_loc(self, pair: str, volatility_avg: float) -> bool:
        """
        Validate trading range
        :param pair: Pair that's currently validated
        :param volatility_avg: Average volatility
        :return: True if the pair can stay, false if it should be removed
        """

        if self._min_volatility <= volatility_avg <= self._max_volatility:
            result = True
        else:
            self.log_once(
                f"Removed {pair} from whitelist, because volatility "
                f"over {self._days} {plural(self._days, 'day')} "
                f"is: {volatility_avg:.3f} "
                f"which is not in the configured range of "
                f"{self._min_volatility}-{self._max_volatility}.",
                logger.info,
            )
            result = False
        return result
