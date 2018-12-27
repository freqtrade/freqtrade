"""
Static List provider

Provides lists as configured in config.json

 """
import logging

from freqtrade.pairlist.IPairList import IPairList

logger = logging.getLogger(__name__)


class StaticPairList(IPairList):

    def __init__(self, freqtrade, config: dict) -> None:
        super().__init__(freqtrade, config)

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """
        return f"{self.name}: {self.whitelist}"

    def refresh_pairlist(self) -> None:
        """
        Refreshes pairlists and assigns them to self._whitelist and self._blacklist respectively
        """
        self._whitelist = self._validate_whitelist(self._config['exchange']['pair_whitelist'])
