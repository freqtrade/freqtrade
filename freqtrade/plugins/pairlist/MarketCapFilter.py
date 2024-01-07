"""
Market Cap PairList provider

Provides dynamic pair list based on Market Cap
"""
import logging
from datetime import timedelta
from typing import Any, Dict, List, Literal

from cachetools import TTLCache

from freqtrade.constants import Config, ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter
from freqtrade.util import dt_now, format_ms_time

from pycoingecko import CoinGeckoAPI

logger = logging.getLogger(__name__)


SORT_VALUES = ['quoteVolume']


class MarketCapFilter(IPairList):

    is_pairlist_generator = True

    def __init__(self, exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'max_rank' not in self._pairlistconfig:
            raise OperationalException(
                '`max_rank` not specified. Please check your configuration '
                'for "pairlist.config.max_rank"')

        self._stake_currency = config['stake_currency']
        self._max_rank = self._pairlistconfig['max_rank']
        self._refresh_period = self._pairlistconfig.get('refresh_period', 86400)
        self._marketcap_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._def_candletype = self._config['candle_type_def']
        self._coingekko: CoinGeckoAPI = CoinGeckoAPI()

        if self._max_rank > 250:
            raise OperationalException(
                "This filter only support up to rank 250."
            )


    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    def _validate_keys(self, key):
        return key in SORT_VALUES

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - Only use top {self._pairlistconfig['max_rank']} market cap pairs."

    @staticmethod
    def description() -> str:
        return "Filter pair list based on market cap."

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        return {
            "max_rank": {
                "type": "number",
                "default": 30,
                "description": "Max market cap rank",
                "help": "Only use assets that ranked within top max_rank market cap",
            },
            "refresh_period": {
                "type": "number",
                "default": 86400,
                "description": "Refresh period",
                "help": "Refresh period in seconds",
            }
        }

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        marketcap_list = self._marketcap_cache.get('marketcap')
        can_filter = False

        if marketcap_list:
            can_filter = True
        else:
            data = self._coingekko.get_coins_markets(vs_currency='usd', order='market_cap_desc',
                                                     per_page='250', page='1', sparkline='false',
                                                     locale='en')
            if data:
                marketcap_list = []
                for row in data:
                    marketcap_list.append(row['symbol'])

                if len(marketcap_list) > 0:
                    self._marketcap_cache['marketcap'] = marketcap_list
                    can_filter = True


        if can_filter:
            filtered_pairlist = []
            top_marketcap = marketcap_list[:self._max_rank:]
            logger.info(top_marketcap)
            logger.info(len(top_marketcap))

            # for pair in pairlist:


        return pairlist
