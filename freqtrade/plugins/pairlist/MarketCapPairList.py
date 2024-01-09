"""
Market Cap PairList provider

Provides dynamic pair list based on Market Cap
"""
import logging
from typing import Any, Dict, List

from cachetools import TTLCache
from pycoingecko import CoinGeckoAPI

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter


logger = logging.getLogger(__name__)


MODE_VALUES = ['top_rank', 'total_assets']


class MarketCapPairList(IPairList):

    is_pairlist_generator = True

    def __init__(self, exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                '`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')

        self._stake_currency = config['stake_currency']
        self._mode = self._pairlistconfig.get('mode', 'top_rank')
        self._number_assets = self._pairlistconfig['number_assets']
        self._refresh_period = self._pairlistconfig.get('refresh_period', 86400)
        self._marketcap_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._def_candletype = self._config['candle_type_def']
        self._coingekko: CoinGeckoAPI = CoinGeckoAPI()

        if self._number_assets > 250:
            raise OperationalException(
                "This filter only support number_assets value up to 250."
            )

        if not self._validate_keys(self._mode):
            raise OperationalException(
                f'key {self._mode} not in {MODE_VALUES}')

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    def _validate_keys(self, key):
        return key in MODE_VALUES

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        num = self._pairlistconfig['number_assets']
        msg = f"{self.name} - Only include pairs ranked within top {num} market cap."
        if self._mode == "total_assets":
            msg = f"{self.name} - top {num} pairs sorted by market cap."
        return msg

    @staticmethod
    def description() -> str:
        return "Provides pair list based on CoinGecko's market cap rank."

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "Max market cap rank",
                "help": "Only use assets with high market cap rank",
            },
            "mode": {
                "type": "option",
                "default": "top_rank",
                "options": MODE_VALUES,
                "description": "Mode of number",
                "help": "How to interpret the number",
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
                marketcap_list = [row['symbol'] for row in data]
                self._marketcap_cache['marketcap'] = marketcap_list
                can_filter = True

        if can_filter:
            filtered_pairlist = []

            if self._mode == 'top_rank':
                top_marketcap = marketcap_list[:self._number_assets:]

                for pair in pairlist:
                    base = pair.split('/')[0]
                    if base.lower() in top_marketcap:
                        filtered_pairlist.append(pair)
                    else:
                        logger.info(f"Remove {pair} from whitelist because it's not ranked "
                                    f"within top {self._number_assets} market cap")

            else:
                market = self._config['trading_mode']
                pair_format = f"{self._stake_currency.upper()}"
                if (market == 'futures'):
                    pair_format += f":{self._stake_currency.upper()}"
                for mc_pair in marketcap_list:
                    test_pair = f"{mc_pair.upper()}/{pair_format}"
                    if test_pair in pairlist:
                        filtered_pairlist.append(test_pair)
                        if len(filtered_pairlist) == self._number_assets:
                            break

            if len(filtered_pairlist) > 0:
                return filtered_pairlist

        return pairlist
