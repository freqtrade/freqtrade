"""
PairList Handler base class
"""
import logging
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange, market_is_active
from freqtrade.exchange.types import Ticker, Tickers
from freqtrade.mixins import LoggingMixin


logger = logging.getLogger(__name__)


class __PairlistParameterBase(TypedDict):
    description: str
    help: str


class __NumberPairlistParameter(__PairlistParameterBase):
    type: Literal["number"]
    default: Union[int, float, None]


class __StringPairlistParameter(__PairlistParameterBase):
    type: Literal["string"]
    default: Union[str, None]


class __OptionPairlistParameter(__PairlistParameterBase):
    type: Literal["option"]
    default: Union[str, None]
    options: List[str]


class __BoolPairlistParameter(__PairlistParameterBase):
    type: Literal["boolean"]
    default: Union[bool, None]


PairlistParameter = Union[
    __NumberPairlistParameter,
    __StringPairlistParameter,
    __OptionPairlistParameter,
    __BoolPairlistParameter
    ]


class IPairList(LoggingMixin, ABC):

    is_pairlist_generator = False

    def __init__(self, exchange: Exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
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
        self.refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        LoggingMixin.__init__(self, logger, self.refresh_period)

    @property
    def name(self) -> str:
        """
        Gets name of the class
        -> no need to overwrite in subclasses
        """
        return self.__class__.__name__

    @abstractproperty
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
    def available_parameters() -> Dict[str, PairlistParameter]:
        """
        Return parameters used by this Pairlist Handler, and their type
        contains a dictionary with the parameter name as key, and a dictionary
        with the type and default value.
        -> Please overwrite in subclasses
        """
        return {}

    @staticmethod
    def refresh_period_parameter() -> Dict[str, PairlistParameter]:
        return {
            "refresh_period": {
                "type": "number",
                "default": 1800,
                "description": "Refresh period",
                "help": "Refresh period in seconds",
            }
        }

    @abstractmethod
    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """

    def _validate_pair(self, pair: str, ticker: Optional[Ticker]) -> bool:
        """
        Check one pair against Pairlist Handler's specific conditions.

        Either implement it in the Pairlist Handler or override the generic
        filter_pairlist() method.

        :param pair: Pair that's currently validated
        :param ticker: ticker dict as returned from ccxt.fetch_ticker
        :return: True if the pair can stay, false if it should be removed
        """
        raise NotImplementedError()

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
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
        raise OperationalException("This Pairlist Handler should not be used "
                                   "at the first position in the list of Pairlist Handlers.")

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
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
            for p in deepcopy(pairlist):
                # Filter out assets
                if not self._validate_pair(p, tickers[p] if p in tickers else None):
                    pairlist.remove(p)

        return pairlist

    def verify_blacklist(self, pairlist: List[str], logmethod) -> List[str]:
        """
        Proxy method to verify_blacklist for easy access for child classes.
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`.
        :return: pairlist - blacklisted pairs
        """
        return self._pairlistmanager.verify_blacklist(pairlist, logmethod)

    def verify_whitelist(self, pairlist: List[str], logmethod,
                         keep_invalid: bool = False) -> List[str]:
        """
        Proxy method to verify_whitelist for easy access for child classes.
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`
        :param keep_invalid: If sets to True, drops invalid pairs silently while expanding regexes.
        :return: pairlist - whitelisted pairs
        """
        return self._pairlistmanager.verify_whitelist(pairlist, logmethod, keep_invalid)

    def _whitelist_for_active_markets(self, pairlist: List[str]) -> List[str]:
        """
        Check available markets and remove pair from whitelist if necessary
        :param pairlist: the sorted list of pairs the user might want to trade
        :return: the list of pairs the user wants to trade without those unavailable or
        black_listed
        """
        markets = self._exchange.markets
        if not markets:
            raise OperationalException(
                'Markets not loaded. Make sure that exchange is initialized correctly.')

        sanitized_whitelist: List[str] = []
        for pair in pairlist:
            # pair is not in the generated dynamic market or has the wrong stake currency
            if pair not in markets:
                self.log_once(f"Pair {pair} is not compatible with exchange "
                              f"{self._exchange.name}. Removing it from whitelist..",
                              logger.warning)
                continue

            if not self._exchange.market_is_tradable(markets[pair]):
                self.log_once(f"Pair {pair} is not tradable with Freqtrade."
                              "Removing it from whitelist..", logger.warning)
                continue

            if self._exchange.get_pair_quote_currency(pair) != self._config['stake_currency']:
                self.log_once(f"Pair {pair} is not compatible with your stake currency "
                              f"{self._config['stake_currency']}. Removing it from whitelist..",
                              logger.warning)
                continue

            # Check if market is active
            market = markets[pair]
            if not market_is_active(market):
                self.log_once(f"Ignoring {pair} from whitelist. Market is not active.", logger.info)
                continue
            if pair not in sanitized_whitelist:
                sanitized_whitelist.append(pair)

        # We need to remove pairs that are unknown
        return sanitized_whitelist
