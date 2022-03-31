import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.constants import (DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS,
                                 ListPairsWithTimeframes, TradeList)
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class HDF5DataHandler(IDataHandler):

    _columns = DEFAULT_DATAFRAME_COLUMNS

    @classmethod
    def ohlcv_get_available_data(
            cls, datadir: Path, trading_mode: TradingMode) -> ListPairsWithTimeframes:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        :param datadir: Directory to search for ohlcv files
        :param trading_mode: trading-mode to be used
        :return: List of Tuples of (pair, timeframe)
        """
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath('futures')
        _tmp = [
            re.search(
                cls._OHLCV_REGEX, p.name
            ) for p in datadir.glob("*.h5")
        ]
        return [
            (
                cls.rebuild_pair_from_filename(match[1]),
                match[2],
                CandleType.from_string(match[3])
            ) for match in _tmp if match and len(match.groups()) > 1]

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
        candle = ""
        if candle_type != CandleType.SPOT:
            datadir = datadir.joinpath('futures')
            candle = f"-{candle_type}"

        _tmp = [re.search(r'^(\S+)(?=\-' + timeframe + candle + '.h5)', p.name)
                for p in datadir.glob(f"*{timeframe}{candle}.h5")]
        # Check if regex found something and only return these results
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    def ohlcv_store(
            self, pair: str, timeframe: str, data: pd.DataFrame, candle_type: CandleType) -> None:
        """
        Store data in hdf5 file.
        :param pair: Pair - used to generate filename
        :param timeframe: Timeframe - used to generate filename
        :param data: Dataframe containing OHLCV data
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: None
        """
        key = self._pair_ohlcv_key(pair, timeframe)
        _data = data.copy()

        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)

        _data.loc[:, self._columns].to_hdf(
            filename, key, mode='a', complevel=9, complib='blosc',
            format='table', data_columns=['date']
        )

    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange], candle_type: CandleType
                    ) -> pd.DataFrame:
        """
        Internal method used to load data for one pair from disk.
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
        key = self._pair_ohlcv_key(pair, timeframe)
        filename = self._pair_data_filename(
            self._datadir,
            pair,
            timeframe,
            candle_type=candle_type
        )

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
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        key = self._pair_trades_key(pair)

        pd.DataFrame(data, columns=DEFAULT_TRADES_COLUMNS).to_hdf(
            self._pair_trades_filename(self._datadir, pair), key,
            mode='a', complevel=9, complib='blosc',
            format='table', data_columns=['timestamp']
        )

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

    @classmethod
    def _get_file_extension(cls):
        return "h5"

    @classmethod
    def _pair_ohlcv_key(cls, pair: str, timeframe: str) -> str:
        # Escape futures pairs to avoid warnings
        pair_esc = pair.replace(':', '_')
        return f"{pair_esc}/ohlcv/tf_{timeframe}"

    @classmethod
    def _pair_trades_key(cls, pair: str) -> str:
        return f"{pair}/trades"
