import logging
from abc import abstractmethod
from pathlib import Path

from pandas import DataFrame
from pyarrow import dataset

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class ArrowDataHandler(IDataHandler):
    """
    Base class for columnar data handlers backed by pyarrow (feather / parquet).

    Subclasses need to implement the format-specific read/write of a single
    DataFrame (_load_dataframe / _store_dataframe) and _get_file_extension.
    """

    _columns = DEFAULT_DATAFRAME_COLUMNS

    @abstractmethod
    def _store_dataframe(self, data: DataFrame, filename: Path) -> None:
        """
        Store a DataFrame to disk in the handler's file format.
        :param data: DataFrame to store (already reset/reordered by the caller)
        :param filename: Target filename
        """

    @abstractmethod
    def _load_dataframe(self, filename: Path) -> DataFrame:
        """
        Load a DataFrame from disk in the handler's file format.
        :param filename: File to load
        :return: DataFrame as stored on disk
        """

    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Store data in the handler's columnar format.
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

        self._store_dataframe(data.reset_index(drop=True).loc[:, self._columns], filename)

    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: TimeRange | None, candle_type: CandleType
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
            pairdata = self._load_dataframe(filename)
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
            pairdata["date"] = pairdata["date"].dt.as_unit("ms")
            return pairdata
        except Exception as e:
            logger.exception(
                f"Error loading data from {filename}. Exception: {e}. Returning empty dataframe."
            )
            return DataFrame(columns=self._columns)

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
        self._store_dataframe(data.reset_index(drop=True), filename)

    def trades_append(self, pair: str, data: DataFrame):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _build_arrow_time_filter(self, timerange: TimeRange | None):
        """
        Build Arrow predicate filter for timerange filtering.
        Treats 0 as unbounded (no filter on that side).
        :param timerange: TimeRange object with start/stop timestamps
        :return: Arrow filter expression or None if fully unbounded
        """
        if not timerange:
            return None

        # Treat 0 as unbounded
        start_set = bool(timerange.startts and timerange.startts > 0)
        stop_set = bool(timerange.stopts and timerange.stopts > 0)

        if not (start_set or stop_set):
            return None

        ts_field = dataset.field("timestamp")
        exprs = []

        if start_set:
            exprs.append(ts_field >= timerange.startts)
        if stop_set:
            exprs.append(ts_field <= timerange.stopts)

        if len(exprs) == 1:
            return exprs[0]
        else:
            return exprs[0] & exprs[1]

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None
    ) -> DataFrame:
        """
        Load a pair from file
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Timerange to load trades for - filters data to this range if provided
        :return: Dataframe containing trades
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        # Use Arrow dataset with optional timerange filtering, fallback to a full read
        try:
            dataset_reader = dataset.dataset(filename, format=self._get_file_extension())
            time_filter = self._build_arrow_time_filter(timerange)

            if time_filter is not None and timerange is not None:
                tradesdata = dataset_reader.to_table(filter=time_filter).to_pandas()
                start_desc = timerange.startts if timerange.startts > 0 else "unbounded"
                stop_desc = timerange.stopts if timerange.stopts > 0 else "unbounded"
                logger.debug(
                    f"Loaded {len(tradesdata)} trades for {pair} "
                    f"(filtered start={start_desc}, stop={stop_desc})"
                )
            else:
                tradesdata = dataset_reader.to_table().to_pandas()
                logger.debug(f"Loaded {len(tradesdata)} trades for {pair} (unfiltered)")

        except (ImportError, AttributeError, ValueError) as e:
            # Fallback: load entire file
            logger.warning(f"Unable to use Arrow filtering, loading entire trades file: {e}")
            tradesdata = self._load_dataframe(filename)

        return tradesdata
