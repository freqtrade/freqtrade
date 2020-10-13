
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


logger = logging.getLogger(__name__)


class IProtection(ABC):

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        -> Please overwrite in subclasses
        """
