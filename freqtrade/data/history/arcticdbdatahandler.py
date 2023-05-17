import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, ListPairsWithTimeframes, TradeList
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class ArcticDBDataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    @staticmethod
    def _get_arcticdb_connection(datadir: Path):
        """
        Internal method used to connect arcticdb.
        Convert the dir path to a valid arcticdb connection string.

        :param datadir: Path object to the data dir.

        :return: ArcticDB connection
        """
        try:
            from arcticdb import Arctic
        except ImportError:
            logger.error("Please install `arcticdb` to use this data handler.")
            raise ImportError("Please install `arcticdb` to use this data handler.")

        dir_string = str(datadir)

        # Fix path string for arcticdb connection
        if dir_string.startswith("s3s:/"):
            dir_string = dir_string.replace("s3s:/", "s3s://")
        elif dir_string.startswith("lmdb:/"):
            dir_string = dir_string.replace("lmdb:/", "lmdb://")
        else:
            # Assume local filesystem, append lmdb prefix
            dir_string = "lmdb://" + dir_string

        return Arctic(dir_string)

    @classmethod
    def ohlcv_get_available_data(
            cls, datadir: Path, trading_mode: TradingMode) -> ListPairsWithTimeframes:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        :param datadir: Directory to search for ohlcv files
        :param trading_mode: trading-mode to be used
        :return: List of Tuples of (pair, timeframe, CandleType)
        """
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath('futures')

        db = cls._get_arcticdb_connection(datadir)

        if 'ohlcv' not in db.list_libraries():
            return []

        lib = db['ohlcv']
        keys = lib.list_symbols()

        splitted_keys = [x.split(':') for x in keys]

        return [
            (
                cls.rebuild_pair_from_filename(match[1]),
                cls.rebuild_timeframe_from_filename(match[2]),
                CandleType.from_string(match[0])
            ) for match in splitted_keys if match]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str, candle_type: CandleType) -> List[str]:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        for the specified timeframe
        :param datadir: Directory to search for ohlcv files
        :param timeframe: Timeframe to search pairs for
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: List of Pairs
        """

        db = cls._get_arcticdb_connection(datadir)

        if 'ohlcv' not in db.list_libraries():
            return []

        lib = db['ohlcv']
        keys = lib.list_symbols()

        splitted_keys = [x.split(':') for x in keys]

        # Filter candle type
        filtered_pairs = [cls.rebuild_pair_from_filename(x[1]) for x in splitted_keys
                          if x[0] == candle_type]

        return filtered_pairs

    def ohlcv_store(
            self, pair: str, timeframe: str, data: pd.DataFrame, candle_type: CandleType) -> None:
        """
        Store data in arcticdb.
        :param pair: Pair - used to generate filename
        :param timeframe: Timeframe - used to generate filename
        :param data: Dataframe containing OHLCV data
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: None
        """

        _data = data.copy()

        # Date is now set to be the index, makes the database happy to do time based query.
        _data.set_index('date', inplace=True)
        columns = self._columns.copy()
        columns.remove('date')

        key = self._pair_ohlcv_key(pair, timeframe, candle_type)
        db = self._get_arcticdb_connection(self._datadir)

        if 'ohlcv' not in db.list_libraries():
            db.create_library('ohlcv')

        lib = db['ohlcv']

        if lib.has_symbol(key):
            lib.delete(key)

        lib.write(key, _data.loc[:, columns])

    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange], candle_type: CandleType
                    ) -> pd.DataFrame:
        """
        Internal method used to load data for one pair from arcticdb.
        Implements the loading and conversion to a Pandas dataframe.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        key = self._pair_ohlcv_key(pair, timeframe, candle_type)

        db = self._get_arcticdb_connection(self._datadir)

        if 'ohlcv' not in db.list_libraries():
            return pd.DataFrame(columns=self._columns)

        lib = db['ohlcv']

        if not lib.has_symbol(key):
            return pd.DataFrame(columns=self._columns)

        columns = self._columns.copy()
        columns.remove('date')

        pairdata = lib.read(key,
                            date_range=(timerange.startdt, timerange.stopdt) if timerange else None,
                            columns=columns).data

        pairdata['date'] = pairdata.index

        pairdata = pairdata.astype(dtype={'open': 'float', 'high': 'float',
                                          'low': 'float', 'close': 'float', 'volume': 'float'})
        pairdata = pairdata.reset_index(drop=True)
        return pairdata

    def ohlcv_append(
            self,
            pair: str,
            timeframe: str,
            data: pd.DataFrame,
            candle_type: CandleType
    ) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        raise NotImplementedError()

    def ohlcv_purge(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:

        key = self._pair_ohlcv_key(pair, timeframe, candle_type)

        db = self._get_arcticdb_connection(self._datadir)

        if 'ohlcv' not in db.list_libraries():
            # Library does not exist, nothing to purge
            return True

        lib = db['ohlcv']

        if lib.has_symbol(key):
            # Dataframe exists, delete it
            lib.delete(key)

        return True

    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """

        raise NotImplementedError()

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

        raise NotImplementedError()

    @classmethod
    def _get_file_extension(cls):
        raise NotImplementedError()

    @classmethod
    def _pair_ohlcv_key(cls, pair: str, timeframe: str, candle_type: CandleType) -> str:
        # Escape futures pairs to avoid warnings
        pair_esc = pair.replace(':', '_')
        return f"{candle_type}:{pair_esc}:{timeframe}"
