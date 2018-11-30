"""
Static List provider

Provides lists as configured in config.json

 """
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class StaticList(object):

    def __init__(self, freqtrade, config: dict) -> None:
        self._freqtrade = freqtrade
        self._config = config
        self._whitelist = self._config['exchange']['pair_whitelist']
        self._blacklist = self._config['exchange'].get('pair_blacklist', [])
        # self.refresh_whitelist()

    @property
    def whitelist(self) -> List[str]:
        """ Contains the current whitelist """
        return self._whitelist

    @property
    def blacklist(self) -> List[str]:
        return self._blacklist

    def refresh_whitelist(self) -> bool:
        """
        Refreshes whitelist.
        """
        return self.validate_whitelist(self._config['exchange']['pair_whitelist'])

    def validate_whitelist(self, whitelist: List[str]) -> bool:
        """
        Check available markets and remove pair from whitelist if necessary
        :param whitelist: the sorted list (based on BaseVolume) of pairs the user might want to
        trade
        :return: the list of pairs the user wants to trade without the one unavailable or
        black_listed
        """
        sanitized_whitelist = whitelist
        markets = self._freqtrade.exchange.get_markets()

        # Filter to markets in stake currency
        markets = [m for m in markets if m['quote'] == self._config['stake_currency']]
        known_pairs = set()

        for market in markets:
            pair = market['symbol']
            # pair is not int the generated dynamic market, or in the blacklist ... ignore it
            if pair not in whitelist or pair in self.blacklist:
                continue
            # else the pair is valid
            known_pairs.add(pair)
            # Market is not active
            if not market['active']:
                sanitized_whitelist.remove(pair)
                logger.info(
                    'Ignoring %s from whitelist. Market is not active.',
                    pair
                )

        # We need to remove pairs that are unknown
        self._whitelist = [x for x in sanitized_whitelist if x in known_pairs]

        return True
