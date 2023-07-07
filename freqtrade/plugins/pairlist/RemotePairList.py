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
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter


logger = logging.getLogger(__name__)


class RemotePairList(IPairList):

    is_pairlist_generator = True

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
        self._refresh_period: int = self._pairlistconfig.get('refresh_period', 1800)
        self._keep_pairlist_on_failure = self._pairlistconfig.get('keep_pairlist_on_failure', True)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._pairlist_url = self._pairlistconfig.get('pairlist_url', '')
        self._read_timeout = self._pairlistconfig.get('read_timeout', 60)
        self._bearer_token = self._pairlistconfig.get('bearer_token', '')
        self._init_done = False
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

    @staticmethod
    def description() -> str:
        return "Retrieve pairs from a remote API."

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 0,
                "description": "Number of assets",
                "help": "Number of assets to use from the pairlist.",
            },
            "pairlist_url": {
                "type": "string",
                "default": "",
                "description": "URL to fetch pairlist from",
                "help": "URL to fetch pairlist from",
            },
            **IPairList.refresh_period_parameter(),
            "keep_pairlist_on_failure": {
                "type": "boolean",
                "default": True,
                "description": "Keep last pairlist on failure",
                "help": "Keep last pairlist on failure",
            },
            "read_timeout": {
                "type": "number",
                "default": 60,
                "description": "Read timeout",
                "help": "Request timeout for remote pairlist",
            },
            "bearer_token": {
                "type": "string",
                "default": "",
                "description": "Bearer token",
                "help": "Bearer token - used for auth against the upstream service.",
            },
        }

    def process_json(self, jsonparse) -> List[str]:

        pairlist = jsonparse.get('pairs', [])
        remote_refresh_period = int(jsonparse.get('refresh_period', self._refresh_period))

        if self._refresh_period < remote_refresh_period:
            self.log_once(f'Refresh Period has been increased from {self._refresh_period}'
                          f' to minimum allowed: {remote_refresh_period} from Remote.', logger.info)

            self._refresh_period = remote_refresh_period
            self._pair_cache = TTLCache(maxsize=1, ttl=remote_refresh_period)

        self._init_done = True

        return pairlist

    def return_last_pairlist(self) -> List[str]:
        if self._keep_pairlist_on_failure:
            pairlist = self._last_pairlist
            self.log_once('Keeping last fetched pairlist', logger.info)
        else:
            pairlist = []

        return pairlist

    def fetch_pairlist(self) -> Tuple[List[str], float]:

        headers = {
            'User-Agent': 'Freqtrade/' + __version__ + ' Remotepairlist'
        }

        if self._bearer_token:
            headers['Authorization'] = f'Bearer {self._bearer_token}'

        try:
            response = requests.get(self._pairlist_url, headers=headers,
                                    timeout=self._read_timeout)
            content_type = response.headers.get('content-type')
            time_elapsed = response.elapsed.total_seconds()

            if "application/json" in str(content_type):
                jsonparse = response.json()

                try:
                    pairlist = self.process_json(jsonparse)
                except Exception as e:

                    if self._init_done:
                        pairlist = self.return_last_pairlist()
                        logger.warning(f'Error while processing JSON data: {type(e)}')
                    else:
                        raise OperationalException(f'Error while processing JSON data: {type(e)}')

            else:
                if self._init_done:
                    self.log_once(f'Error: RemotePairList is not of type JSON: '
                                  f' {self._pairlist_url}', logger.info)
                    pairlist = self.return_last_pairlist()
                else:
                    raise OperationalException('RemotePairList is not of type JSON, abort.')

        except requests.exceptions.RequestException:
            self.log_once(f'Was not able to fetch pairlist from:'
                          f' {self._pairlist_url}', logger.info)

            pairlist = self.return_last_pairlist()

            time_elapsed = 0

        return pairlist, time_elapsed

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """

        if self._init_done:
            pairlist = self._pair_cache.get('pairlist')
            if pairlist == [None]:
                # Valid but empty pairlist.
                return []
        else:
            pairlist = []

        time_elapsed = 0.0

        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:
            if self._pairlist_url.startswith("file:///"):
                filename = self._pairlist_url.split("file:///", 1)[1]
                file_path = Path(filename)

                if file_path.exists():
                    with file_path.open() as json_file:
                        # Load the JSON data into a dictionary
                        jsonparse = json.load(json_file)

                        try:
                            pairlist = self.process_json(jsonparse)
                        except Exception as e:
                            if self._init_done:
                                pairlist = self.return_last_pairlist()
                                logger.warning(f'Error while processing JSON data: {type(e)}')
                            else:
                                raise OperationalException('Error while processing'
                                                           f'JSON data: {type(e)}')
                else:
                    raise ValueError(f"{self._pairlist_url} does not exist.")
            else:
                # Fetch Pairlist from Remote URL
                pairlist, time_elapsed = self.fetch_pairlist()

        self.log_once(f"Fetched pairs: {pairlist}", logger.debug)

        pairlist = self._whitelist_for_active_markets(pairlist)
        pairlist = pairlist[:self._number_pairs]

        if pairlist:
            self._pair_cache['pairlist'] = pairlist.copy()
        else:
            # If pairlist is empty, set a dummy value to avoid fetching again
            self._pair_cache['pairlist'] = [None]

        if time_elapsed != 0.0:
            self.log_once(f'Pairlist Fetched in {time_elapsed} seconds.', logger.info)
        else:
            self.log_once('Fetched Pairlist.', logger.info)

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
        rpl_pairlist = self.gen_pairlist(tickers)
        merged_list = pairlist + rpl_pairlist
        merged_list = sorted(set(merged_list), key=merged_list.index)
        return merged_list
