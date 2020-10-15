
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from freqtrade.mixins import LoggingMixin


logger = logging.getLogger(__name__)

ProtectionReturn = Tuple[bool, Optional[datetime], Optional[str]]


class IProtection(LoggingMixin, ABC):

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        self._config = config
        self._protection_config = protection_config
        LoggingMixin.__init__(self, logger)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        -> Please overwrite in subclasses
        """

    @abstractmethod
    def global_stop(self, date_now: datetime) -> ProtectionReturn:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        """
