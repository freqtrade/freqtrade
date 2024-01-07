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

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        # Generate dynamic whitelist
        # Must always run if this pairlist is not the first in the list.
        pairlist = self._marketcap_cache.get('pairlist_mc')
        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:
            # Use fresh pairlist
            # Check if pair quote currency equals to the stake currency.
            _pairlist = [k for k in self._exchange.get_markets(
                quote_currencies=[self._stake_currency],
                tradable_only=True, active_only=True).keys()]
            # No point in testing for blacklisted pairs...
            _pairlist = self.verify_blacklist(_pairlist, logger.info)
            # if not self._use_range:
            #     filtered_tickers = [
            #         v for k, v in tickers.items()
            #         if (self._exchange.get_pair_quote_currency(k) == self._stake_currency
            #             and (self._use_range or v.get(self._sort_key) is not None)
            #             and v['symbol'] in _pairlist)]
            #     pairlist = [s['symbol'] for s in filtered_tickers]
            # else:
            #     pairlist = _pairlist

            pairlist = self.filter_pairlist(_pairlist, tickers)
            self._marketcap_cache['pairlist_mc'] = pairlist.copy()

        return pairlist

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

            for pair in pairlist:
                base = pair.split('/')[0]
                if base.lower() in top_marketcap:
                    filtered_pairlist.append(pair)
                else:
                    logger.info(f"Remove {pair} from whitelist because it's not in "
                                f"top {self._max_rank} market cap")

            if len(filtered_pairlist) > 0:
                return filtered_pairlist


        return pairlist
