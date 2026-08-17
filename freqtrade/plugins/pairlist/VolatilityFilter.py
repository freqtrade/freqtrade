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

        _lookback_days = self._pairlistconfig.get("lookback_days", 0)
        self._lookback_timeframe = self._pairlistconfig.get("lookback_timeframe", "1d")
        _lookback_period: int | None = self._pairlistconfig.get("lookback_period", None)
        self._min_volatility = self._pairlistconfig.get("min_volatility", 0)
        self._max_volatility = self._pairlistconfig.get("max_volatility", sys.maxsize)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1440)
        self._def_candletype = self._config["candle_type_def"]
        self._sort_direction: str | None = self._pairlistconfig.get("sort_direction", None)

        self._pair_cache: FtTTLCache = FtTTLCache(maxsize=1000, ttl=self._refresh_period)

        if (_lookback_days > 0) and _lookback_period and (_lookback_period > 0):
            raise OperationalException(
                "Ambiguous configuration: lookback_days and lookback_period both set in pairlist "
                "config. Please set lookback_days only or lookback_period and lookback_timeframe "
                "and restart the bot."
            )

        # overwrite lookback timeframe and period when lookback_days is set
        if "lookback_days" in self._pairlistconfig:
            self._lookback_timeframe = "1d"
            _lookback_period = _lookback_days
        if _lookback_period is None:
            if "lookback_timeframe" in self._pairlistconfig:
                raise OperationalException(
                    f"{self.name} requires lookback_period to be set when using lookback_timeframe."
                )
            logger.warning(
                f"DEPRECATED: Using {self.name} without lookback_days or lookback_period is "
                "deprecated and will result in an error in a future version. "
                "Please set either lookback_days or lookback_period and lookback_timeframe. "
                "Falling back to lookback_days: 10."
            )
            _lookback_period = 10
        self._lookback_period: int = _lookback_period

        candle_limit = self._exchange.ohlcv_candle_limit(
            self._lookback_timeframe, self._def_candletype
        )
        if self._lookback_period < 1:
            raise OperationalException(f"{self.name} requires lookback_period to be >= 1")
        if self._lookback_period > candle_limit:
            raise OperationalException(
                f"{self.name} requires lookback_period to not "
                f"exceed exchange max request size ({candle_limit})"
            )
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
            f"last {self._lookback_period} {plural(self._lookback_period, 'candle')} of "
            f"{self._lookback_timeframe}."
        )

    @staticmethod
    def description() -> str:
        return "Filter pairs by their recent volatility."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "lookback_days": {
                "type": "number",
                "default": None,
                "description": "Lookback Days",
                "help": "Number of days to look back at. Implies a lookback_timeframe of 1d.",
            },
            "lookback_timeframe": {
                "type": "string",
                "default": "1d",
                "description": "Lookback Timeframe",
                "help": "Timeframe to use for lookback.",
            },
            "lookback_period": {
                "type": "number",
                "default": 10,
                "description": "Lookback Period",
                "help": "Number of periods to look back at.",
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
                f"over {self._lookback_period} {plural(self._lookback_period, 'candle')} of "
                f"{self._lookback_timeframe} "
                f"is: {volatility_avg:.3f} "
                f"which is not in the configured range of "
                f"{self._min_volatility}-{self._max_volatility}.",
                logger.info,
            )
            result = False
        return result
