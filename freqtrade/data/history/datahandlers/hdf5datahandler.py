import logging
from typing import Optional

import numpy as np
import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class HDF5DataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(
        self, pair: str, timeframe: str, data: pd.DataFrame, candle_type: CandleType
    ) -> None:
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
            filename,
            key=key,
            mode="a",
            complevel=9,
            complib="blosc",
            format="table",
            data_columns=["date"],
        )

    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: Optional[TimeRange], candle_type: CandleType
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
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)

        if not filename.exists():
            # Fallback mode for 1M files
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            if not filename.exists():
                return pd.DataFrame(columns=self._columns)
        where = []
        if timerange:
            if timerange.starttype == "date":
                where.append(f"date >= Timestamp({timerange.startts * 1e9})")
            if timerange.stoptype == "date":
                where.append(f"date <= Timestamp({timerange.stopts * 1e9})")

        pairdata = pd.read_hdf(filename, key=key, mode="r", where=where)

        if list(pairdata.columns) != self._columns:
            raise ValueError("Wrong dataframe format")
        pairdata = pairdata.astype(
            dtype={
                "open": "float",
                "high": "float",
                "low": "float",
                "close": "float",
                "volume": "float",
            }
        )
        pairdata = pairdata.reset_index(drop=True)
        return pairdata

    def ohlcv_append(
        self, pair: str, timeframe: str, data: pd.DataFrame, candle_type: CandleType
    ) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: pd.DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        :param trading_mode: Trading mode to use (used to determine the filename)
        """
        key = self._pair_trades_key(pair)

        data.to_hdf(
            self._pair_trades_filename(self._datadir, pair, trading_mode),
            key=key,
            mode="a",
            complevel=9,
            complib="blosc",
            format="table",
            data_columns=["timestamp"],
        )

    def trades_append(self, pair: str, data: pd.DataFrame):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: Optional[TimeRange] = None
    ) -> pd.DataFrame:
        """
        Load a pair from h5 file.
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Timerange to load trades for - currently not implemented
        :return: Dataframe containing trades
        """
        key = self._pair_trades_key(pair)
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)

        if not filename.exists():
            return pd.DataFrame(columns=DEFAULT_TRADES_COLUMNS)
        where = []
        if timerange:
            if timerange.starttype == "date":
                where.append(f"timestamp >= {timerange.startts * 1e3}")
            if timerange.stoptype == "date":
                where.append(f"timestamp < {timerange.stopts * 1e3}")

        trades: pd.DataFrame = pd.read_hdf(filename, key=key, mode="r", where=where)
        trades[["id", "type"]] = trades[["id", "type"]].replace({np.nan: None})
        return trades

    @classmethod
    def _get_file_extension(cls):
        return "h5"

    @classmethod
    def _pair_ohlcv_key(cls, pair: str, timeframe: str) -> str:
        # Escape futures pairs to avoid warnings
        pair_esc = pair.replace(":", "_")
        return f"{pair_esc}/ohlcv/tf_{timeframe}"

    @classmethod
    def _pair_trades_key(cls, pair: str) -> str:
        return f"{pair}/trades"
