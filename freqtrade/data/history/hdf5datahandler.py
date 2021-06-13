import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import (DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS,
                                 ListPairsWithTimeframes, TradeList)

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class HDF5DataHandler(IDataHandler):

    _columns = DEFAULT_DATAFRAME_COLUMNS

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path) -> ListPairsWithTimeframes:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        :param datadir: Directory to search for ohlcv files
        :return: List of Tuples of (pair, timeframe)
        """
        _tmp = [re.search(r'^([a-zA-Z_]+)\-(\d+\S+)(?=.h5)', p.name)
                for p in datadir.glob("*.h5")]
        return [(match[1].replace('_', '/'), match[2]) for match in _tmp
                if match and len(match.groups()) > 1]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        for the specified timeframe
        :param datadir: Directory to search for ohlcv files
        :param timeframe: Timeframe to search pairs for
        :return: List of Pairs
        """

        _tmp = [re.search(r'^(\S+)(?=\-' + timeframe + '.h5)', p.name)
                for p in datadir.glob(f"*{timeframe}.h5")]
        # Check if regex found something and only return these results
        return [match[0].replace('_', '/') for match in _tmp if match]

    def ohlcv_store(self, pair: str, timeframe: str, data: pd.DataFrame) -> None:
        """
        Store data in hdf5 file.
        :param pair: Pair - used to generate filename
        :timeframe: Timeframe - used to generate filename
        :data: Dataframe containing OHLCV data
        :return: None
        """
        key = self._pair_ohlcv_key(pair, timeframe)
        _data = data.copy()

        filename = self._pair_data_filename(self._datadir, pair, timeframe)

        ds = pd.HDFStore(filename, mode='a', complevel=9, complib='blosc')
        ds.put(key, _data.loc[:, self._columns], format='table', data_columns=['date'])

        ds.close()

    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange] = None) -> pd.DataFrame:
        """
        Internal method used to load data for one pair from disk.
        Implements the loading and conversion to a Pandas dataframe.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        key = self._pair_ohlcv_key(pair, timeframe)
        filename = self._pair_data_filename(self._datadir, pair, timeframe)

        if not filename.exists():
            return pd.DataFrame(columns=self._columns)
        where = []
        if timerange:
            if timerange.starttype == 'date':
                where.append(f"date >= Timestamp({timerange.startts * 1e9})")
            if timerange.stoptype == 'date':
                where.append(f"date <= Timestamp({timerange.stopts * 1e9})")

        pairdata = pd.read_hdf(filename, key=key, mode="r", where=where)

        if list(pairdata.columns) != self._columns:
            raise ValueError("Wrong dataframe format")
        pairdata = pairdata.astype(dtype={'open': 'float', 'high': 'float',
                                          'low': 'float', 'close': 'float', 'volume': 'float'})
        return pairdata

    def ohlcv_purge(self, pair: str, timeframe: str) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :param timeframe: Timeframe (e.g. "5m")
        :return: True when deleted, false if file did not exist.
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def ohlcv_append(self, pair: str, timeframe: str, data: pd.DataFrame) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        """
        raise NotImplementedError()

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        Returns a list of all pairs for which trade data is available in this
        :param datadir: Directory to search for ohlcv files
        :return: List of Pairs
        """
        _tmp = [re.search(r'^(\S+)(?=\-trades.h5)', p.name)
                for p in datadir.glob("*trades.h5")]
        # Check if regex found something and only return these results to avoid exceptions.
        return [match[0].replace('_', '/') for match in _tmp if match]

    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        key = self._pair_trades_key(pair)

        ds = pd.HDFStore(self._pair_trades_filename(self._datadir, pair),
                         mode='a', complevel=9, complib='blosc')
        ds.put(key, pd.DataFrame(data, columns=DEFAULT_TRADES_COLUMNS),
               format='table', data_columns=['timestamp'])
        ds.close()

    def trades_append(self, pair: str, data: TradeList):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(self, pair: str, timerange: Optional[TimeRange] = None) -> TradeList:
        """
        Load a pair from h5 file.
        :param pair: Load trades for this pair
        :param timerange: Timerange to load trades for - currently not implemented
        :return: List of trades
        """
        key = self._pair_trades_key(pair)
        filename = self._pair_trades_filename(self._datadir, pair)

        if not filename.exists():
            return []
        where = []
        if timerange:
            if timerange.starttype == 'date':
                where.append(f"timestamp >= {timerange.startts * 1e3}")
            if timerange.stoptype == 'date':
                where.append(f"timestamp < {timerange.stopts * 1e3}")

        trades: pd.DataFrame = pd.read_hdf(filename, key=key, mode="r", where=where)
        trades[['id', 'type']] = trades[['id', 'type']].replace({np.nan: None})
        return trades.values.tolist()

    def trades_purge(self, pair: str) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :return: True when deleted, false if file did not exist.
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        if filename.exists():
            filename.unlink()
            return True
        return False

    @classmethod
    def _pair_ohlcv_key(cls, pair: str, timeframe: str) -> str:
        return f"{pair}/ohlcv/tf_{timeframe}"

    @classmethod
    def _pair_trades_key(cls, pair: str) -> str:
        return f"{pair}/trades"

    @classmethod
    def _pair_data_filename(cls, datadir: Path, pair: str, timeframe: str) -> Path:
        pair_s = misc.pair_to_filename(pair)
        filename = datadir.joinpath(f'{pair_s}-{timeframe}.h5')
        return filename

    @classmethod
    def _pair_trades_filename(cls, datadir: Path, pair: str) -> Path:
        pair_s = misc.pair_to_filename(pair)
        filename = datadir.joinpath(f'{pair_s}-trades.h5')
        return filename
