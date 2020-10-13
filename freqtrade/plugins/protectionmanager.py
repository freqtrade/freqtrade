"""
Protection manager class
"""
import logging
from typing import Dict, List

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.protections import IProtection
from freqtrade.resolvers import ProtectionResolver


logger = logging.getLogger(__name__)


class ProtectionManager():

    def __init__(self, exchange, config: dict) -> None:
        self._exchange = exchange
        self._config = config

        self._protection_handlers: List[IProtection] = []
        self._tickers_needed = False
        for protection_handler_config in self._config.get('protections', None):
            if 'method' not in protection_handler_config:
                logger.warning(f"No method found in {protection_handler_config}, ignoring.")
                continue
            protection_handler = ProtectionResolver.load_protection(
                protection_handler_config['method'],
                exchange=exchange,
                protectionmanager=self,
                config=config,
                protection_config=protection_handler_config,
            )
            self._tickers_needed |= protection_handler.needstickers
            self._protection_handlers.append(protection_handler)

        if not self._protection_handlers:
            raise OperationalException("No protection Handlers defined")

    @property
    def name_list(self) -> List[str]:
        """
        Get list of loaded Protection Handler names
        """
        return [p.name for p in self._protection_handlers]

    def short_desc(self) -> List[Dict]:
        """
        List of short_desc for each Pairlist Handler
        """
        return [{p.name: p.short_desc()} for p in self._pairlist_handlers]
