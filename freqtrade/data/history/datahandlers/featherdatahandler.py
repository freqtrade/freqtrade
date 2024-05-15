import logging
from typing import Optional

from pandas import DataFrame, read_feather, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class FeatherDataHandler(IDataHandler):
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

        data.reset_index(drop=True).loc[:, self._columns].to_feather(
            filename, compression_level=9, compression="lz4"
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

        pairdata = read_feather(filename)
        pairdata.columns = self._columns
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
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).to_feather(filename, compression_level=9, compression="lz4")

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
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        tradesdata = read_feather(filename)

        return tradesdata

    @classmethod
    def _get_file_extension(cls):
        return "feather"
