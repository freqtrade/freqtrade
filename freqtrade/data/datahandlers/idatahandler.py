"""
Abstract datahandler interface.
It's subclasses handle and storing data from disk.

"""
import logging
from abc import ABC, abstractclassmethod, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import arrow
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.exchange import timeframe_to_seconds

logger = logging.getLogger(__name__)


class IDataHandler(ABC):

    def __init__(self, datadir: Path) -> None:
        self._datadir = datadir

    # TODO: create abstract interface

    def ohlcv_load(self, pair, timeframe: str,
                   timerange: Optional[TimeRange] = None,
                   fill_missing: bool = True,
                   drop_incomplete: bool = True,
                   startup_candles: int = 0,
                   ) -> DataFrame:
        """
        Load cached ticker history for the given pair.

        :param pair: Pair to load data for
        :param timeframe: Ticker timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange
        :param fill_up_missing: Fill missing values with "No action"-candles
        :param drop_incomplete: Drop last candle assuming it may be incomplete.
        :param startup_candles: Additional candles to load at the start of the period
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        # Fix startup period
        timerange_startup = deepcopy(timerange)
        if startup_candles > 0 and timerange_startup:
            timerange_startup.subtract_start(timeframe_to_seconds(timeframe) * startup_candles)

        pairdf = self._ohlcv_load(pair, timeframe,
                                  timerange=timerange_startup,
                                  fill_missing=fill_missing,
                                  drop_incomplete=drop_incomplete)
        if pairdf.empty:
            logger.warning(
                f'No history data for pair: "{pair}", timeframe: {timeframe}. '
                'Use `freqtrade download-data` to download the data'
            )
            return pairdf
        else:
            if timerange_startup:
                self._validate_pairdata(pair, pairdf, timerange_startup)
            return pairdf

    def _validate_pairdata(self, pair, pairdata: DataFrame, timerange: TimeRange):
        """
        Validates pairdata for missing data at start end end and logs warnings.
        :param pairdata: Dataframe to validate
        :param timerange: Timerange specified for start and end dates
        """

        if timerange.starttype == 'date' and pairdata[0][0] > timerange.startts * 1000:
            logger.warning('Missing data at start for pair %s, data starts at %s',
                           pair, arrow.get(pairdata[0][0] // 1000).strftime('%Y-%m-%d %H:%M:%S'))
        if timerange.stoptype == 'date' and pairdata[-1][0] < timerange.stopts * 1000:
            logger.warning('Missing data at end for pair %s, data ends at %s',
                           pair, arrow.get(pairdata[-1][0] // 1000).strftime('%Y-%m-%d %H:%M:%S'))

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
