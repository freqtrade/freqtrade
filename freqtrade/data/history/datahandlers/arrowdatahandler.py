import logging
from abc import abstractmethod
from datetime import timedelta
from pathlib import Path

from pandas import DataFrame
from pyarrow import ArrowException, dataset

from freqtrade.candle_columns import get_candle_columns, get_candle_dtypes
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange import timeframe_to_seconds

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class ArrowDataHandler(IDataHandler):
    """
    Base class for columnar data handlers backed by pyarrow (feather / parquet).

    Subclasses need to implement the format-specific read/write of a single
    DataFrame (_load_dataframe / _store_dataframe) and _get_file_extension.
    """

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
        :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
        :return: None
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)

        self._store_dataframe(
            data.reset_index(drop=True).loc[:, get_candle_columns(candle_type)], filename
        )

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
        :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        if not filename.exists():
            # Fallback mode for 1M files
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            if not filename.exists():
                return self._empty_ohlcv_df(candle_type)
        try:
            pairdata = self._load_ohlcv_dataframe(filename, timeframe, timerange)
            if pairdata.empty:
                return self._empty_ohlcv_df(candle_type)

            pairdata = self._normalize_columns(pairdata, pair, candle_type)
            pairdata = pairdata.astype(dtype=get_candle_dtypes(candle_type))
            pairdata["date"] = pairdata["date"].dt.as_unit("ms")
            return pairdata
        except Exception as e:
            # Also covers files in an unreadable layout - one broken file must not abort
            # loading of all other data.
            logger.exception(
                f"Error loading data from {filename}. Exception: {e}. Returning empty dataframe."
            )
            return self._empty_ohlcv_df(candle_type)

    def _build_arrow_ohlcv_filter(self, timerange: TimeRange | None, timeframe: str):
        """
        Build Arrow predicate filter on the "date" column for ohlcv data.

        Both bounds are widened by one candle. `ohlcv_load` trims the dataframe to the
        exact timerange afterwards, but inspects the untrimmed dataframe first to warn
        about missing data and to decide whether the last candle may be incomplete.
        The extra candle on each side keeps that information intact.

        :param timerange: TimeRange object with start/stop dates
        :param timeframe: Timeframe of the data (e.g. "5m")
        :return: Arrow filter expression or None if unbounded
        """
        if not timerange:
            return None

        widen = timedelta(seconds=timeframe_to_seconds(timeframe))
        date_field = dataset.field("date")
        exprs = []

        if timerange.starttype == "date" and (startdt := timerange.startdt):
            exprs.append(date_field >= startdt - widen)
        if timerange.stoptype == "date" and (stopdt := timerange.stopdt):
            exprs.append(date_field <= stopdt + widen)

        if not exprs:
            return None
        elif len(exprs) == 1:
            return exprs[0]
        else:
            return exprs[0] & exprs[1]

    def _load_ohlcv_dataframe(
        self, filename: Path, timeframe: str, timerange: TimeRange | None
    ) -> DataFrame:
        """
        Load ohlcv data through Arrow, pushing the timerange filter down where possible
        to avoid reading candles outside of the requested range.
        Falls back to reading the full file - callers trim the result either way.
        :param filename: File to load
        :param timeframe: Timeframe of the data (e.g. "5m")
        :param timerange: Timerange to limit the data to
        :return: DataFrame as stored on disk, limited to (roughly) timerange
        """
        time_filter = self._build_arrow_ohlcv_filter(timerange, timeframe)
        try:
            dataset_reader = dataset.dataset(filename, format=self._get_file_extension())
            pairdata = dataset_reader.to_table(filter=time_filter).to_pandas()
        except (ImportError, AttributeError, ValueError, ArrowException) as e:
            # Fallback: load entire file
            logger.warning(f"Unable to use Arrow filtering, loading entire ohlcv file: {e}")
            return self._load_dataframe(filename)

        if time_filter is not None and pairdata.empty:
            # No candles in the requested range - reload everything, so callers can tell
            # "this file has no data at all" apart from "the file has no data for this
            # timerange" (and can point at the data that is available).
            return self._load_dataframe(filename)
        return pairdata

    def ohlcv_append(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
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

    def _build_arrow_trades_filter(self, timerange: TimeRange | None):
        """
        Build Arrow predicate filter for timerange filtering.
        Treats 0 as unbounded (no filter on that side).
        :param timerange: TimeRange object with start/stop timestamps
        :return: Arrow filter expression or None if fully unbounded
        """
        if not timerange:
            return None

        ts_field = dataset.field("timestamp")
        exprs = []

        # Treat 0 as unbounded
        if timerange.startts and timerange.startts > 0:
            exprs.append(ts_field >= timerange.startts)
        if timerange.stopts and timerange.stopts > 0:
            exprs.append(ts_field <= timerange.stopts)

        if not exprs:
            return None
        elif len(exprs) == 1:
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
            time_filter = self._build_arrow_trades_filter(timerange)
            tradesdata = dataset_reader.to_table(filter=time_filter).to_pandas()
            logger.debug(
                f"Loaded {len(tradesdata)} trades for {pair} "
                f"({f'filter: {time_filter}' if time_filter is not None else 'unfiltered'})"
            )
        except (ImportError, AttributeError, ValueError, ArrowException) as e:
            # Fallback: load entire file
            logger.warning(f"Unable to use Arrow filtering, loading entire trades file: {e}")
            tradesdata = self._load_dataframe(filename)

        return tradesdata
