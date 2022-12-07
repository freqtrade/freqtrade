"""
Remote PairList provider

Provides pair list fetched from a remote source
"""
import json
import logging
from typing import Any, Dict, List

import requests
from cachetools import TTLCache

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class RemotePairList(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                '`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')

        if 'pairlist_url' not in self._pairlistconfig:
            raise OperationalException(
                '`pairlist_url` not specified. Please check your configuration '
                'for "pairlist.config.pairlist_url"')

        self._number_pairs = self._pairlistconfig['number_assets']
        self._refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        self._keep_pairlist_on_failure = self._pairlistconfig.get('keep_pairlist_on_failure', True)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._pairlist_url = self._pairlistconfig.get('pairlist_url', '')
        self._read_timeout = self._pairlistconfig.get('read_timeout', 60)
        self._last_pairlist: List[Any] = list()

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - {self._pairlistconfig['number_assets']} pairs from RemotePairlist."

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        pairlist = self._pair_cache.get('pairlist')
        info = ""

        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:
            # Fetch Pairlist from Remote
            headers = {
                'User-Agent': 'Freqtrade - Remotepairlist',
            }

            try:
                response = requests.get(self._pairlist_url, headers=headers,
                                        timeout=self._read_timeout)
                content_type = response.headers.get('content-type')
                time_elapsed = response.elapsed.total_seconds()

                rsplit = response.text.split("#")

                if "text/html" in str(content_type):
                    if len(rsplit) > 1:
                        plist = rsplit[0].strip()
                        plist = json.loads(plist)
                        info = rsplit[1].strip()
                    else:
                        plist = json.loads(rsplit[0])
                elif "application/json" in str(content_type):
                    jsonp = json.loads(' '.join(rsplit))
                    plist = jsonp['pairs']
                    info = jsonp['info']

            except requests.exceptions.RequestException:
                self.log_once(f'Was not able to fetch pairlist from:'
                              f' {self._pairlist_url}', logger.info)

                if self._keep_pairlist_on_failure:
                    plist = str(self._last_pairlist)
                    self.log_once('Keeping last fetched pairlist', logger.info)
                else:
                    plist = ""

                time_elapsed = 0

            pairlist = []

            for i in plist:
                if i not in pairlist:
                    pairlist.append(i)
                else:
                    continue

        pairlist = self.filter_pairlist(pairlist, tickers)
        self._pair_cache['pairlist'] = pairlist.copy()

        if(time_elapsed):
            self.log_once(info + " | " + " Fetched in " + str(time_elapsed) +
                          " seconds.", logger.info)

        self._last_pairlist = list(pairlist)
        return pairlist

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """

        # Validate whitelist to only have active market pairs
        pairlist = self._whitelist_for_active_markets(pairlist)
        pairlist = self.verify_blacklist(pairlist, logger.info)
        # Limit pairlist to the requested number of pairs
        pairlist = pairlist[:self._number_pairs]
        self.log_once(f"Searching {self._number_pairs} pairs: {pairlist}", logger.info)

        return pairlist
