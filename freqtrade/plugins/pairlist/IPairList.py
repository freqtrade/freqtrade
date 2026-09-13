"""
PairList Handler base class
"""

import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal, TypedDict

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange, market_is_active
from freqtrade.exchange.exchange_types import Ticker, Tickers
from freqtrade.mixins import LoggingMixin


logger = logging.getLogger(__name__)


class __PairlistParameterBase(TypedDict):
    description: str
    help: str


class __NumberPairlistParameter(__PairlistParameterBase):
    type: Literal["number"]
    default: int | float | None


class __StringPairlistParameter(__PairlistParameterBase):
    type: Literal["string"]
    default: str | None


class __OptionPairlistParameter(__PairlistParameterBase):
    type: Literal["option"]
    default: str | None
    options: list[str]


class __ListPairListParamenter(__PairlistParameterBase):
    type: Literal["list"]
    default: list[str] | None


class __BoolPairlistParameter(__PairlistParameterBase):
    type: Literal["boolean"]
    default: bool | None


PairlistParameter = (
    __NumberPairlistParameter
    | __StringPairlistParameter
    | __OptionPairlistParameter
    | __BoolPairlistParameter
    | __ListPairListParamenter
)


class SupportsBacktesting(StrEnum):
    """
    Enum to indicate if a Pairlist Handler supports backtesting.
    """

    YES = "yes"
    NO = "no"
    NO_ACTION = "no_action"
    BIASED = "biased"


class IPairList(LoggingMixin, ABC):
    is_pairlist_generator = False
    supports_backtesting: SupportsBacktesting = SupportsBacktesting.NO

    def __init__(
        self,
        exchange: Exchange,
        pairlistmanager,
        config: Config,
        pairlistconfig: dict[str, Any],
        pairlist_pos: int,
    ) -> None:
        """
        :param exchange: Exchange instance
        :param pairlistmanager: Instantiated Pairlist manager
        :param config: Global bot configuration
        :param pairlistconfig: Configuration for this Pairlist Handler - can be empty.
        :param pairlist_pos: Position of the Pairlist Handler in the chain
        """
        self._enabled = True

        self._exchange: Exchange = exchange
        self._pairlistmanager = pairlistmanager
        self._config = config
        self._pairlistconfig = pairlistconfig
        self._pairlist_pos = pairlist_pos
        self.refresh_period = self._pairlistconfig.get("refresh_period", 1800)
        LoggingMixin.__init__(self, logger, self.refresh_period)

    @property
    def name(self) -> str:
        """
        Gets name of the class
        -> no need to overwrite in subclasses
        """
        return self.__class__.__name__

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    @staticmethod
    @abstractmethod
    def description() -> str:
        """
        Return description of this Pairlist Handler
        -> Please overwrite in subclasses
        """
        return ""

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        """
        Return parameters used by this Pairlist Handler, and their type
        contains a dictionary with the parameter name as key, and a dictionary
        with the type and default value.
        -> Please overwrite in subclasses
        """
        return {}

    @staticmethod
    def refresh_period_parameter() -> dict[str, PairlistParameter]:
        return {
            "refresh_period": {
                "type": "number",
                "default": 1800,
                "description": "Refresh period",
                "help": "Refresh period in seconds",
            }
        }

    @staticmethod
    def lookback_parameters(default_period: int = 0) -> dict[str, PairlistParameter]:
        """
        Lookback parameters, for Pairlist Handlers supporting a lookback range.
        :param default_period: Default lookback_period - 0 when the lookback range is optional.
        """
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
                "default": default_period,
                "description": "Lookback Period",
                "help": "Number of periods to look back at.",
            },
        }

    def _init_lookback_config(
        self, *, required: bool = False, deprecated_fallback: int = 0
    ) -> None:
        """
        Resolve the lookback configuration (`lookback_days`, `lookback_timeframe` and
        `lookback_period`) into `self._lookback_timeframe` and `self._lookback_period`,
        and validate it against the exchange's max request size.
        `lookback_days` is a convenience alias for `lookback_period` on daily candles.
        :param required: Whether this handler needs a lookback range. Handlers with an optional
                         lookback end up with a period of 0, disabling the lookback range.
        :param deprecated_fallback: Number of days to fall back to when `required` is set, but
                                    neither `lookback_days` nor `lookback_period` is configured.
                                    Deprecated - will be removed in a future version.
        """
        lookback_days: int = self._pairlistconfig.get("lookback_days", 0) or 0
        lookback_period: int | None = self._pairlistconfig.get("lookback_period", None)
        self._lookback_timeframe: str = self._pairlistconfig.get("lookback_timeframe", "1d")

        has_period = (lookback_period or 0) > 0

        if lookback_days > 0 and has_period:
            raise OperationalException(
                "Ambiguous configuration: lookback_days and lookback_period both set in pairlist "
                "config. Please set lookback_days only or lookback_period and lookback_timeframe "
                "and restart the bot."
            )
        # 0 means "no lookback" - only acceptable if the lookback is optional,
        # or if lookback_period provides the range instead.
        min_days = 1 if required and not has_period else 0
        if "lookback_days" in self._pairlistconfig and lookback_days < min_days:
            raise OperationalException(f"{self.name} requires lookback_days to be >= {min_days}")

        # lookback_days implies daily candles
        if lookback_days > 0:
            if self._lookback_timeframe != "1d":
                raise OperationalException(
                    "Ambiguous configuration: lookback_days implies a lookback_timeframe of 1d, "
                    f"but lookback_timeframe is set to {self._lookback_timeframe}. Please set "
                    "lookback_period instead of lookback_days and restart the bot."
                )
            lookback_period = lookback_days

        if lookback_period is None and required:
            if "lookback_timeframe" in self._pairlistconfig:
                raise OperationalException(
                    f"{self.name} requires lookback_period to be set when using lookback_timeframe."
                )
            if not deprecated_fallback:
                raise OperationalException(
                    f"{self.name} requires either lookback_days or lookback_period to be set."
                )
            logger.warning(
                f"DEPRECATED: Using {self.name} without lookback_days or lookback_period is "
                "deprecated and will result in an error in a future version. "
                "Please set either lookback_days or lookback_period and lookback_timeframe. "
                f"Falling back to lookback_days: {deprecated_fallback}."
            )
            lookback_period = deprecated_fallback

        self._lookback_period: int = lookback_period or 0

        min_period = 1 if required else 0
        if self._lookback_period < min_period:
            raise OperationalException(
                f"{self.name} requires lookback_period to be >= {min_period}"
            )

        if self._lookback_period == 0 and "lookback_timeframe" in self._pairlistconfig:
            # Required handlers raise above - for optional ones, the timeframe has no effect.
            logger.warning(
                f"{self.name} is configured with lookback_timeframe "
                f"{self._lookback_timeframe}, but without lookback_period - "
                "the lookback range is disabled and the timeframe has no effect. "
                "Please set lookback_period to enable it."
            )

        candle_limit = self._exchange.ohlcv_candle_limit(
            self._lookback_timeframe, self._config["candle_type_def"]
        )
        if self._lookback_period > candle_limit:
            raise OperationalException(
                f"{self.name} requires lookback_period to not "
                f"exceed exchange max request size ({candle_limit})"
            )

    @abstractmethod
    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """

    def _validate_pair(self, pair: str, ticker: Ticker | None) -> bool:
        """
        Check one pair against Pairlist Handler's specific conditions.

        Either implement it in the Pairlist Handler or override the generic
        filter_pairlist() method.

        :param pair: Pair that's currently validated
        :param ticker: ticker dict as returned from ccxt.fetch_ticker
        :return: True if the pair can stay, false if it should be removed
        """
        raise NotImplementedError()

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist.

        This method is called once by the pairlistmanager in the refresh_pairlist()
        method to supply the starting pairlist for the chain of the Pairlist Handlers.
        Pairlist Filters (those Pairlist Handlers that cannot be used at the first
        position in the chain) shall not override this base implementation --
        it will raise the exception if a Pairlist Handler is used at the first
        position in the chain.

        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        raise OperationalException(
            "This Pairlist Handler should not be used "
            "at the first position in the list of Pairlist Handlers."
        )

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.

        Called on each bot iteration - please use internal caching if necessary
        This generic implementation calls self._validate_pair() for each pair
        in the pairlist.

        Some Pairlist Handlers override this generic implementation and employ
        own filtration.

        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        if self._enabled:
            # Copy list since we're modifying this list
            for p in pairlist.copy():
                # Filter out assets
                if not self._validate_pair(p, tickers.get(p, None)):
                    pairlist.remove(p)

        return pairlist

    def verify_blacklist(self, pairlist: list[str], logmethod) -> list[str]:
        """
        Proxy method to verify_blacklist for easy access for child classes.
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`.
        :return: pairlist - blacklisted pairs
        """
        return self._pairlistmanager.verify_blacklist(pairlist, logmethod)

    def verify_whitelist(
        self, pairlist: list[str], logmethod, keep_invalid: bool = False
    ) -> list[str]:
        """
        Proxy method to verify_whitelist for easy access for child classes.
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`
        :param keep_invalid: If sets to True, drops invalid pairs silently while expanding regexes.
        :return: pairlist - whitelisted pairs
        """
        return self._pairlistmanager.verify_whitelist(pairlist, logmethod, keep_invalid)

    def _whitelist_for_active_markets(self, pairlist: list[str]) -> list[str]:
        """
        Check available markets and remove pair from whitelist if necessary
        :param pairlist: the sorted list of pairs the user might want to trade
        :return: the list of pairs the user wants to trade without those unavailable or
        black_listed
        """
        markets = self._exchange.markets
        if not markets:
            raise OperationalException(
                "Markets not loaded. Make sure that exchange is initialized correctly."
            )

        sanitized_whitelist: list[str] = []
        for pair in pairlist:
            # pair is not in the generated dynamic market or has the wrong stake currency
            if pair not in markets:
                self.log_once(
                    f"Pair {pair} is not compatible with exchange "
                    f"{self._exchange.name}. Removing it from whitelist..",
                    logger.warning,
                    True,
                )
                continue

            if not self._exchange.market_is_tradable(markets[pair]):
                self.log_once(
                    f"Pair {pair} is not tradable with Freqtrade. Removing it from whitelist..",
                    logger.warning,
                    True,
                )
                continue

            if self._exchange.get_pair_quote_currency(pair) != self._config["stake_currency"]:
                self.log_once(
                    f"Pair {pair} is not compatible with your stake currency "
                    f"{self._config['stake_currency']}. Removing it from whitelist..",
                    logger.warning,
                    True,
                )
                continue

            # Check if market is active
            market = markets[pair]
            if not market_is_active(market):
                self.log_once(
                    f"Ignoring {pair} from whitelist. Market is not active.",
                    logger.info,
                    True,
                )
                continue
            if pair not in sanitized_whitelist:
                sanitized_whitelist.append(pair)

        # We need to remove pairs that are unknown
        return sanitized_whitelist
