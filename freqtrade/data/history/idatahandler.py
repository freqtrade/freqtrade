"""
Abstract datahandler interface.
It's subclasses handle and storing data from disk.

"""
import logging
from abc import ABC, abstractclassmethod, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Type

from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.data.converter import clean_ohlcv_dataframe, trim_dataframe
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
                   warn_no_data: bool = True
                   ) -> DataFrame:
        """
        Load cached ticker history for the given pair.

        :param pair: Pair to load data for
        :param timeframe: Ticker timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange
        :param fill_missing: Fill missing values with "No action"-candles
        :param drop_incomplete: Drop last candle assuming it may be incomplete.
        :param startup_candles: Additional candles to load at the start of the period
        :param warn_no_data: Log a warning message when no data is found
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        # Fix startup period
        timerange_startup = deepcopy(timerange)
        if startup_candles > 0 and timerange_startup:
            timerange_startup.subtract_start(timeframe_to_seconds(timeframe) * startup_candles)

        pairdf = self._ohlcv_load(pair, timeframe,
                                  timerange=timerange_startup)
        if pairdf.empty:
            if warn_no_data:
                logger.warning(
                    f'No history data for pair: "{pair}", timeframe: {timeframe}. '
                    'Use `freqtrade download-data` to download the data'
                )
            return pairdf
        else:
            enddate = pairdf.iloc[-1]['date']

            if timerange_startup:
                self._validate_pairdata(pair, pairdf, timerange_startup)
                pairdf = trim_dataframe(pairdf, timerange_startup)

            # incomplete candles should only be dropped if we didn't trim the end beforehand.
            return clean_ohlcv_dataframe(pairdf, timeframe,
                                         pair=pair,
                                         fill_missing=fill_missing,
                                         drop_incomplete=(drop_incomplete and
                                                          enddate == pairdf.iloc[-1]['date']))

    def _validate_pairdata(self, pair, pairdata: DataFrame, timerange: TimeRange):
        """
        Validates pairdata for missing data at start end end and logs warnings.
        :param pairdata: Dataframe to validate
        :param timerange: Timerange specified for start and end dates
        """

        if timerange.starttype == 'date':
            start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)
            if pairdata.iloc[0]['date'] > start:
                logger.warning(f"Missing data at start for pair {pair}, "
                               f"data starts at {pairdata.iloc[0]['date']:%Y-%m-%d %H:%M:%S}")
        if timerange.stoptype == 'date':
            stop = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)
            if pairdata.iloc[-1]['date'] < stop:
                logger.warning(f"Missing data at end for pair {pair}, "
                               f"data ends at {pairdata.iloc[-1]['date']:%Y-%m-%d %H:%M:%S}")


def get_datahandlerclass(datatype: str) -> Type[IDataHandler]:
    """
    Get datahandler class.
    Could be done using Resolvers, but since this may be called often and resolvers
    are rather expensive, doing this directly should improve performance.
    :param datatype: datatype to use.
    :return: Datahandler class
    """

    if datatype == 'json':
        from .jsondatahandler import JsonDataHandler
        return JsonDataHandler
    elif datatype == 'jsongz':
        from .jsondatahandler import JsonGzDataHandler
        return JsonGzDataHandler
    else:
        raise ValueError(f"No datahandler for datatype {datatype} available.")


def get_datahandler(datadir: Path, data_format: str = None,
                    data_handler: IDataHandler = None) -> IDataHandler:
    """
    :param datadir: Folder to save data
    :data_format: dataformat to use
    :data_handler: returns this datahandler if it exists or initializes a new one
    """

    if not data_handler:
        HandlerClass = get_datahandlerclass(data_format or 'json')
        data_handler = HandlerClass(datadir)
    return data_handler
