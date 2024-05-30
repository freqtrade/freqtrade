import logging
from typing import Optional

import numpy as np
from pandas import DataFrame, read_json, to_datetime

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.data.converter import trades_dict_to_list, trades_list_to_df
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class JsonDataHandler(IDataHandler):
    _use_zip = False
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Store data in json format "values".
            format looks as follows:
            [[<date>,<open>,<high>,<low>,<close>]]
        :param pair: Pair - used to generate filename
        :param timeframe: Timeframe - used to generate filename
        :param data: Dataframe containing OHLCV data
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: None
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)
        _data = data.copy()
        # Convert date to int
        _data["date"] = _data["date"].astype(np.int64) // 1000 // 1000

        # Reset index, select only appropriate columns and save as json
        _data.reset_index(drop=True).loc[:, self._columns].to_json(
            filename, orient="values", compression="gzip" if self._use_zip else None
        )

    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: Optional[TimeRange], candle_type: CandleType
    ) -> DataFrame:
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
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        if not filename.exists():
            # Fallback mode for 1M files
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            if not filename.exists():
                return DataFrame(columns=self._columns)
        try:
            pairdata = read_json(filename, orient="values")
            pairdata.columns = self._columns
        except ValueError:
            logger.error(f"Could not load data for {pair}.")
            return DataFrame(columns=self._columns)
        pairdata = pairdata.astype(
            dtype={
                "open": "float",
                "high": "float",
                "low": "float",
                "close": "float",
                "volume": "float",
            }
        )
        pairdata["date"] = to_datetime(pairdata["date"], unit="ms", utc=True)
        return pairdata

    def ohlcv_append(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        :param trading_mode: Trading mode to use (used to determine the filename)
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        trades = data.values.tolist()
        misc.file_dump_json(filename, trades, is_zip=self._use_zip)

    def trades_append(self, pair: str, data: DataFrame):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: Optional[TimeRange] = None
    ) -> DataFrame:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: respect timerange ...
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Timerange to load trades for - currently not implemented
        :return: Dataframe containing trades
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        tradesdata = misc.file_load_json(filename)

        if not tradesdata:
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        if isinstance(tradesdata[0], dict):
            # Convert trades dict to list
            logger.info("Old trades format detected - converting")
            tradesdata = trades_dict_to_list(tradesdata)
            pass
        return trades_list_to_df(tradesdata, convert=False)

    @classmethod
    def _get_file_extension(cls):
        return "json.gz" if cls._use_zip else "json"


class JsonGzDataHandler(JsonDataHandler):
    _use_zip = True
