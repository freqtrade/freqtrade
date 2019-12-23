"""
Abstract datahandler interface.
It's subclasses handle and storing data from disk.

"""

from abc import ABC, abstractmethod, abstractclassmethod
from pathlib import Path
from typing import Dict, List, Optional

from pandas import DataFrame

from freqtrade.configuration import TimeRange


class IDataHandler(ABC):

    def __init__(self, datadir: Path, pair: str) -> None:
        self._datadir = datadir
        self._pair = pair

    @abstractclassmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        Returns a list of all pairs available in this datadir
        """

    @abstractmethod
    def ohlcv_store(self, timeframe: str, data: DataFrame):
        """
        Store data
        """

    @abstractmethod
    def ohlcv_append(self, timeframe: str, data: DataFrame):
        """
        Append data to existing files
        """

    @abstractmethod
    def ohlcv_load(self, timeframe: str, timerange: Optional[TimeRange] = None) -> DataFrame:
        """
        Load data for one pair
        :return: Dataframe
        """

    @abstractclassmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        Returns a list of all pairs available in this datadir
        """

    @abstractmethod
    def trades_store(self, data: DataFrame):
        """
        Store data
        """

    @abstractmethod
    def trades_append(self, data: DataFrame):
        """
        Append data to existing files
        """

    @abstractmethod
    def trades_load(self, timerange: Optional[TimeRange] = None):
        """
        Load data for one pair
        :return: Dataframe
        """

    @staticmethod
    def trim_tickerlist(tickerlist: List[Dict], timerange: TimeRange) -> List[Dict]:
        """
        TODO: investigate if this is needed ... we can probably cover this in a dataframe
        Trim tickerlist based on given timerange
        """
        if not tickerlist:
            return tickerlist

        start_index = 0
        stop_index = len(tickerlist)

        if timerange.starttype == 'date':
            while (start_index < len(tickerlist) and
                   tickerlist[start_index][0] < timerange.startts * 1000):
                start_index += 1

        if timerange.stoptype == 'date':
            while (stop_index > 0 and
                   tickerlist[stop_index-1][0] > timerange.stopts * 1000):
                stop_index -= 1

        if start_index > stop_index:
            raise ValueError(f'The timerange [{timerange.startts},{timerange.stopts}] is incorrect')

        return tickerlist[start_index:stop_index]
