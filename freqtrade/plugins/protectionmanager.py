"""
Protection manager class
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from freqtrade.constants import Config, LongShort
from freqtrade.persistence import PairLocks
from freqtrade.persistence.models import PairLock
from freqtrade.plugins.protections import IProtection
from freqtrade.resolvers import ProtectionResolver


logger = logging.getLogger(__name__)


class ProtectionManager:

    def __init__(self, config: Config, protections: List) -> None:
        self._config = config

        self._protection_handlers: List[IProtection] = []
        for protection_handler_config in protections:
            protection_handler = ProtectionResolver.load_protection(
                protection_handler_config['method'],
                config=config,
                protection_config=protection_handler_config,
            )
            self._protection_handlers.append(protection_handler)

        if not self._protection_handlers:
            logger.info("No protection Handlers defined.")

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
        return [{p.name: p.short_desc()} for p in self._protection_handlers]

    def global_stop(self, now: Optional[datetime] = None,
                    side: LongShort = 'long') -> Optional[PairLock]:
        if not now:
            now = datetime.now(timezone.utc)
        result = None
        for protection_handler in self._protection_handlers:
            if protection_handler.has_global_stop:
                lock = protection_handler.global_stop(date_now=now, side=side)
                if lock and lock.until:
                    if not PairLocks.is_global_lock(lock.until, side=lock.lock_side):
                        result = PairLocks.lock_pair(
                            '*', lock.until, lock.reason, now=now, side=lock.lock_side)
        return result

    def stop_per_pair(self, pair, now: Optional[datetime] = None,
                      side: LongShort = 'long') -> Optional[PairLock]:
        if not now:
            now = datetime.now(timezone.utc)
        result = None
        for protection_handler in self._protection_handlers:
            if protection_handler.has_local_stop:
                lock = protection_handler.stop_per_pair(
                    pair=pair, date_now=now, side=side)
                if lock and lock.until:
                    if not PairLocks.is_pair_locked(pair, lock.until, lock.lock_side):
                        result = PairLocks.lock_pair(
                            pair, lock.until, lock.reason, now=now, side=lock.lock_side)
        return result
