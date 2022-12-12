"""
Remote PairList provider

Provides pair list fetched from a remote source
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from cachetools import TTLCache

from freqtrade import __version__
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
        self._bearer_token = self._pairlistconfig.get('bearer_token', '')
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

    def fetch_pairlist(self) -> Tuple[List[str], float, str]:

        headers = {
            'User-Agent': 'Freqtrade/' + __version__ + ' Remotepairlist'
        }

        if self._bearer_token:
            headers['Authorization'] = f'Bearer {self._bearer_token}'

        info = "Pairlist"

        try:
            with requests.get(self._pairlist_url, headers=headers,
                              timeout=self._read_timeout) as response:
                content_type = response.headers.get('content-type')
                time_elapsed = response.elapsed.total_seconds()

                if "application/json" in str(content_type):
                    jsonparse = response.json()
                    pairlist = jsonparse['pairs']
                    info = jsonparse.get('info', '')[:1000]
                else:
                    raise OperationalException(
                        'Remotepairlist is not of type JSON abort')

                self._refresh_period = jsonparse.get('refresh_period', self._refresh_period)
                self._pair_cache = TTLCache(maxsize=1, ttl=self._refresh_period)

        except requests.exceptions.RequestException:
            self.log_once(f'Was not able to fetch pairlist from:'
                          f' {self._pairlist_url}', logger.info)

            if self._keep_pairlist_on_failure:
                pairlist = self._last_pairlist
                self.log_once('Keeping last fetched pairlist', logger.info)
            else:
                pairlist = []

            time_elapsed = 0

        return pairlist, time_elapsed, info

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """

        pairlist = self._pair_cache.get('pairlist')
        time_elapsed = 0.0

        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:
            if self._pairlist_url.startswith("file:///"):
                filename = self._pairlist_url.split("file:///", 1)[1]
                file_path = Path(filename)

                if file_path.exists():
                    with open(filename) as json_file:
                        # Load the JSON data into a dictionary
                        jsonparse = json.load(json_file)
                        pairlist = jsonparse['pairs']
                        info = jsonparse.get('info', '')[:1000]
                        self._refresh_period = jsonparse.get('refresh_period', self._refresh_period)
                        self._pair_cache = TTLCache(maxsize=1, ttl=self._refresh_period)

                else:
                    raise ValueError(f"{self._pairlist_url} does not exist.")
            else:
                # Fetch Pairlist from Remote URL
                pairlist, time_elapsed, info = self.fetch_pairlist()

        pairlist = self.filter_pairlist(pairlist, tickers)
        self._pair_cache['pairlist'] = pairlist.copy()

        if time_elapsed != 0.0:
            self.log_once(f'{info} Fetched in {time_elapsed} seconds.', logger.info)
        else:
            self.log_once(f'{info} Fetched Pairlist.', logger.info)

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
