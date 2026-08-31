"""
Volatility pairlist filter
"""

import logging
import sys

import numpy as np
from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import FtTTLCache


logger = logging.getLogger(__name__)


class VolatilityFilter(IPairList):
    """
    Filters pairs by volatility
    """

    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._min_volatility = self._pairlistconfig.get("min_volatility", 0)
        self._max_volatility = self._pairlistconfig.get("max_volatility", sys.maxsize)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1440)
        self._def_candletype = self._config["candle_type_def"]
        self._sort_direction: str | None = self._pairlistconfig.get("sort_direction", None)

        self._pair_cache: FtTTLCache = FtTTLCache(maxsize=1000, ttl=self._refresh_period)

        self._init_lookback_config(required=True, deprecated_fallback=10)

        if self._sort_direction not in [None, "asc", "desc"]:
            raise OperationalException(
                f"{self.name} requires sort_direction to be "
                "either None (undefined), 'asc' or 'desc'"
            )

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (
            f"{self.name} - Filtering pairs with volatility range "
            f"{self._min_volatility}-{self._max_volatility} over the "
            f"last {self._lookback_period} x {self._lookback_timeframe} "
            f"{plural(self._lookback_period, 'candle')}."
        )

    @staticmethod
    def description() -> str:
        return "Filter pairs by their recent volatility."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
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
            **IPairList.lookback_parameters(default_period=10),
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Validate trading range
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new allowlist
        """
        needed_pairs: ListPairsWithTimeframes = [
            (p, self._lookback_timeframe, self._def_candletype)
            for p in pairlist
            if p not in self._pair_cache
        ]

        candles = self._exchange.refresh_ohlcv_with_cache(
            needed_pairs, lookback_period=self._lookback_period
        )

        resulting_pairlist: list[str] = []
        volatilitys: dict[str, float] = {}
        for p in pairlist:
            pair_candles = candles.get((p, self._lookback_timeframe, self._def_candletype), None)

            volatility_avg = self._calculate_volatility(p, pair_candles)

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

    def _calculate_volatility(self, pair: str, pair_candles: DataFrame) -> float | None:
        # Check symbol in cache
        if (volatility_avg := self._pair_cache.get(pair, None)) is not None:
            return volatility_avg

        if pair_candles is not None and not pair_candles.empty:
            returns = np.log(pair_candles["close"].shift(1) / pair_candles["close"])
            returns.fillna(0, inplace=True)

            volatility_series = returns.rolling(window=self._lookback_period).std() * np.sqrt(
                self._lookback_period
            )
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
                f"over {self._lookback_period} x {self._lookback_timeframe} "
                f"{plural(self._lookback_period, 'candle')} "
                f"is: {volatility_avg:.3f} "
                f"which is not in the configured range of "
                f"{self._min_volatility}-{self._max_volatility}.",
                logger.info,
            )
            result = False
        return result
