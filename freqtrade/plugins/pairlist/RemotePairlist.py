"""
Remote PairList provider

Provides dynamic pair list based on trade volumes
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


class RemotePairlist(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                '`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')

        self._number_pairs = self._pairlistconfig['number_assets']
        self._refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        self._keep_pairlist_on_failure = self._pairlistconfig.get('keep_pairlist_on_failure', True)        
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._pairlist_url = self._pairlistconfig.get('pairlist_url',
                                                      'http://pairlist.robot.co.network')
        self._stake_currency = config['stake_currency']

        if (self._refresh_period < 850):
            raise OperationalException(
                'Please set a Refresh Period higher than 850 for the Remotepairlist.'
            )

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
        return f"{self.name} - {self._pairlistconfig['number_assets']} pairs from Remote."

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        hick = "'"
        double = '"'
        # Generate dynamic whitelist
        # Must always run if this pairlist is not the first in the list.
        pairlist = self._pair_cache.get('pairlist')

        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:

            headers = {
                'User-Agent': 'Freqtrade Pairlist Fetcher',
            }

            if "limit" not in self._pairlist_url:
                url = self._pairlist_url + "&limit=" + str(self._number_pairs)
            else:
                url = self._pairlist_url

            if "stake" not in self._pairlist_url:
                url = self._pairlist_url + "&stake=" + str(self._config['stake_currency'])
            else:
                url = self._pairlist_url

            if "exchange" not in self._pairlist_url:
                url = self._pairlist_url + "&exchange=" + str(self._config['exchange'])
            else:
                url = self._pairlist_url

            try:
                response = requests.get(url, headers=headers, timeout=60)
                responser = response.text.replace(hick, double)
                time_elapsed = response.elapsed.total_seconds()
                rsplit = responser.split("#")
                plist = rsplit[0].strip()
                plist = plist.replace("<br>", "")
                plist = json.loads(plist)
                info = rsplit[1].strip()

            except Exception as e:
                print(e)
                self.log_once(f'Was not able to receive pairlist from'
                              f' {self._pairlist_url}', logger.info)

                if self._keep_pairlist_on_failure:                
                    plist = pairlist
                else:
                    plist = ""


            pairlist = []

            for i in plist:
                if i not in pairlist:
                    if "/" in i:
                        if self._stake_currency in i:
                            pairlist.append(i)
                        else:
                            continue
                    else:
                        pairlist.append(i + "/" + self._config['stake_currency'])

        pairlist = self.filter_pairlist(pairlist, tickers)
        self._pair_cache['pairlist'] = pairlist.copy()
        self.log_once(info + " | " + "Fetched in " + str(time_elapsed) + " seconds.", logger.info)
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
