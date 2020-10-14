
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict


logger = logging.getLogger(__name__)


class IProtection(ABC):

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        self._config = config
        self._protection_config = protection_config

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
    def stop_trade_enters_global(self, date_now: datetime) -> bool:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        """
