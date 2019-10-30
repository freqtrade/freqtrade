import logging
from abc import ABC, abstractmethod
from typing import Dict, List

logger = logging.getLogger(__name__)


class IPairListFilter(ABC):

    def __init__(self, freqtrade, config: dict) -> None:
        self._freqtrade = freqtrade
        self._config = config

    @abstractmethod
    def filter_pairlist(self, pairlist: List[str], tickers: List[Dict]) -> List[str]:
        """
        Method doing the filtering
        """
