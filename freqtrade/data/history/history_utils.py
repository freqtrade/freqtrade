import logging
import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arrow
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.data.converter import (ohlcv_to_dataframe,
                                      trades_remove_duplicates,
                                      trades_to_ohlcv)
from freqtrade.data.history.idatahandler import IDataHandler, get_datahandler
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.misc import format_ms_time

logger = logging.getLogger(__name__)


def load_pair_history(pair: str,
                      timeframe: str,
                      datadir: Path, *,
                      timerange: Optional[TimeRange] = None,
                      fill_up_missing: bool = True,
                      drop_incomplete: bool = True,
                      startup_candles: int = 0,
                      data_format: str = None,
                      data_handler: IDataHandler = None,
                      ) -> DataFrame:
    """
    Load cached ohlcv history for the given pair.

    :param pair: Pair to load data for
    :param timeframe: Timeframe (e.g. "5m")
    :param datadir: Path to the data storage location.
    :param data_format: Format of the data. Ignored if data_handler is set.
    :param timerange: Limit data to be loaded to this timerange
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param drop_incomplete: Drop last candle assuming it may be incomplete.
    :param startup_candles: Additional candles to load at the start of the period
    :param data_handler: Initialized data-handler to use.
                         Will be initialized from data_format if not set
    :return: DataFrame with ohlcv data, or empty DataFrame
    """
    data_handler = get_datahandler(datadir, data_format, data_handler)

    return data_handler.ohlcv_load(pair=pair,
                                   timeframe=timeframe,
                                   timerange=timerange,
                                   fill_missing=fill_up_missing,
                                   drop_incomplete=drop_incomplete,
                                   startup_candles=startup_candles,
                                   )


def load_data(datadir: Path,
              timeframe: str,
              pairs: List[str], *,
              timerange: Optional[TimeRange] = None,
              fill_up_missing: bool = True,
              startup_candles: int = 0,
              fail_without_data: bool = False,
              data_format: str = 'json',
              ) -> Dict[str, DataFrame]:
    """
    Load ohlcv history data for a list of pairs.

    :param datadir: Path to the data storage location.
    :param timeframe: Timeframe (e.g. "5m")
    :param pairs: List of pairs to load
    :param timerange: Limit data to be loaded to this timerange
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param startup_candles: Additional candles to load at the start of the period
    :param fail_without_data: Raise OperationalException if no data is found.
    :param data_format: Data format which should be used. Defaults to json
    :return: dict(<pair>:<Dataframe>)
    """
    result: Dict[str, DataFrame] = {}
    if startup_candles > 0 and timerange:
        logger.info(f'Using indicator startup period: {startup_candles} ...')

    data_handler = get_datahandler(datadir, data_format)

    for pair in pairs:
        hist = load_pair_history(pair=pair, timeframe=timeframe,
                                 datadir=datadir, timerange=timerange,
                                 fill_up_missing=fill_up_missing,
                                 startup_candles=startup_candles,
                                 data_handler=data_handler
                                 )
        if not hist.empty:
            result[pair] = hist

    if fail_without_data and not result:
        raise OperationalException("No data found. Terminating.")
    return result


def refresh_data(datadir: Path,
                 timeframe: str,
                 pairs: List[str],
                 exchange: Exchange,
                 data_format: str = None,
                 timerange: Optional[TimeRange] = None,
                 ) -> None:
    """
    Refresh ohlcv history data for a list of pairs.

    :param datadir: Path to the data storage location.
    :param timeframe: Timeframe (e.g. "5m")
    :param pairs: List of pairs to load
    :param exchange: Exchange object
    :param timerange: Limit data to be loaded to this timerange
    """
    data_handler = get_datahandler(datadir, data_format)
    for pair in pairs:
        _download_pair_history(pair=pair, timeframe=timeframe,
                               datadir=datadir, timerange=timerange,
                               exchange=exchange, data_handler=data_handler)


def _load_cached_data_for_updating(pair: str, timeframe: str, timerange: Optional[TimeRange],
                                   data_handler: IDataHandler) -> Tuple[DataFrame, Optional[int]]:
    """
    Load cached data to download more data.
    If timerange is passed in, checks whether data from an before the stored data will be
    downloaded.
    If that's the case then what's available should be completely overwritten.
    Otherwise downloads always start at the end of the available data to avoid data gaps.
    Note: Only used by download_pair_history().
    """
    start = None
    if timerange:
        if timerange.starttype == 'date':
            # TODO: convert to date for conversion
            start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)

    # Intentionally don't pass timerange in - since we need to load the full dataset.
    data = data_handler.ohlcv_load(pair, timeframe=timeframe,
                                   timerange=None, fill_missing=False,
                                   drop_incomplete=True, warn_no_data=False)
    if not data.empty:
        if start and start < data.iloc[0]['date']:
            # Earlier data than existing data requested, redownload all
            data = DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)
        else:
            start = data.iloc[-1]['date']

    start_ms = int(start.timestamp() * 1000) if start else None
    return data, start_ms


def _download_pair_history(datadir: Path,
                           exchange: Exchange,
                           pair: str, *,
                           timeframe: str = '5m',
                           timerange: Optional[TimeRange] = None,
                           data_handler: IDataHandler = None) -> bool:
    """
    Download latest candles from the exchange for the pair and timeframe passed in parameters
    The data is downloaded starting from the last correct data that
    exists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data

    :param pair: pair to download
    :param timeframe: Timeframe (e.g "5m")
    :param timerange: range of time to download
    :return: bool with success state
    """
    data_handler = get_datahandler(datadir, data_handler=data_handler)

    try:
        logger.info(
            f'Download history data for pair: "{pair}", timeframe: {timeframe} '
            f'and store in {datadir}.'
        )

        # data, since_ms = _load_cached_data_for_updating_old(datadir, pair, timeframe, timerange)
        data, since_ms = _load_cached_data_for_updating(pair, timeframe, timerange,
                                                        data_handler=data_handler)

        logger.debug("Current Start: %s",
                     f"{data.iloc[0]['date']:%Y-%m-%d %H:%M:%S}" if not data.empty else 'None')
        logger.debug("Current End: %s",
                     f"{data.iloc[-1]['date']:%Y-%m-%d %H:%M:%S}" if not data.empty else 'None')

        # Default since_ms to 30 days if nothing is given
        new_data = exchange.get_historic_ohlcv(pair=pair,
                                               timeframe=timeframe,
                                               since_ms=since_ms if since_ms else
                                               int(arrow.utcnow().shift(
                                                   days=-30).float_timestamp) * 1000
                                               )
        # TODO: Maybe move parsing to exchange class (?)
        new_dataframe = ohlcv_to_dataframe(new_data, timeframe, pair,
                                           fill_missing=False, drop_incomplete=True)
        if data.empty:
            data = new_dataframe
        else:
            data = data.append(new_dataframe)

        logger.debug("New  Start: %s",
                     f"{data.iloc[0]['date']:%Y-%m-%d %H:%M:%S}" if not data.empty else 'None')
        logger.debug("New End: %s",
                     f"{data.iloc[-1]['date']:%Y-%m-%d %H:%M:%S}" if not data.empty else 'None')

        data_handler.ohlcv_store(pair, timeframe, data=data)
        return True

    except Exception as e:
        logger.error(
            f'Failed to download history data for pair: "{pair}", timeframe: {timeframe}. '
            f'Error: {e}'
        )
        return False


def refresh_backtest_ohlcv_data(exchange: Exchange, pairs: List[str], timeframes: List[str],
                                datadir: Path, timerange: Optional[TimeRange] = None,
                                erase: bool = False, data_format: str = None) -> List[str]:
    """
    Refresh stored ohlcv data for backtesting and hyperopt operations.
    Used by freqtrade download-data subcommand.
    :return: List of pairs that are not available.
    """
    pairs_not_available = []
    data_handler = get_datahandler(datadir, data_format)
    for pair in pairs:
        if pair not in exchange.markets:
            pairs_not_available.append(pair)
            logger.info(f"Skipping pair {pair}...")
            continue
        for timeframe in timeframes:

            if erase:
                if data_handler.ohlcv_purge(pair, timeframe):
                    logger.info(
                        f'Deleting existing data for pair {pair}, interval {timeframe}.')

            logger.info(f'Downloading pair {pair}, interval {timeframe}.')
            _download_pair_history(datadir=datadir, exchange=exchange,
                                   pair=pair, timeframe=str(timeframe),
                                   timerange=timerange, data_handler=data_handler)
    return pairs_not_available


def _download_trades_history(exchange: Exchange,
                             pair: str, *,
                             timerange: Optional[TimeRange] = None,
                             data_handler: IDataHandler
                             ) -> bool:
    """
    Download trade history from the exchange.
    Appends to previously downloaded trades data.
    """
    try:

        since = timerange.startts * 1000 if \
            (timerange and timerange.starttype == 'date') else int(arrow.utcnow().shift(
                days=-30).float_timestamp) * 1000

        trades = data_handler.trades_load(pair)

        # TradesList columns are defined in constants.DEFAULT_TRADES_COLUMNS
        # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
        # DEFAULT_TRADES_COLUMNS: 1 -> id

        from_id = trades[-1][1] if trades else None
        if trades and since < trades[-1][0]:
            # Reset since to the last available point
            # - 5 seconds (to ensure we're getting all trades)
            since = trades[-1][0] - (5 * 1000)
            logger.info(f"Using last trade date -5s - Downloading trades for {pair} "
                        f"since: {format_ms_time(since)}.")

        logger.debug(f"Current Start: {format_ms_time(trades[0][0]) if trades else 'None'}")
        logger.debug(f"Current End: {format_ms_time(trades[-1][0]) if trades else 'None'}")
        logger.info(f"Current Amount of trades: {len(trades)}")

        # Default since_ms to 30 days if nothing is given
        new_trades = exchange.get_historic_trades(pair=pair,
                                                  since=since,
                                                  from_id=from_id,
                                                  )
        trades.extend(new_trades[1])
        # Remove duplicates to make sure we're not storing data we don't need
        trades = trades_remove_duplicates(trades)
        data_handler.trades_store(pair, data=trades)

        logger.debug(f"New Start: {format_ms_time(trades[0][0])}")
        logger.debug(f"New End: {format_ms_time(trades[-1][0])}")
        logger.info(f"New Amount of trades: {len(trades)}")
        return True

    except Exception as e:
        logger.error(
            f'Failed to download historic trades for pair: "{pair}". '
            f'Error: {e}'
        )
        return False


def refresh_backtest_trades_data(exchange: Exchange, pairs: List[str], datadir: Path,
                                 timerange: TimeRange, erase: bool = False,
                                 data_format: str = 'jsongz') -> List[str]:
    """
    Refresh stored trades data for backtesting and hyperopt operations.
    Used by freqtrade download-data subcommand.
    :return: List of pairs that are not available.
    """
    pairs_not_available = []
    data_handler = get_datahandler(datadir, data_format=data_format)
    for pair in pairs:
        if pair not in exchange.markets:
            pairs_not_available.append(pair)
            logger.info(f"Skipping pair {pair}...")
            continue

        if erase:
            if data_handler.trades_purge(pair):
                logger.info(f'Deleting existing data for pair {pair}.')

        logger.info(f'Downloading trades for pair {pair}.')
        _download_trades_history(exchange=exchange,
                                 pair=pair,
                                 timerange=timerange,
                                 data_handler=data_handler)
    return pairs_not_available


def convert_trades_to_ohlcv(pairs: List[str], timeframes: List[str],
                            datadir: Path, timerange: TimeRange, erase: bool = False,
                            data_format_ohlcv: str = 'json',
                            data_format_trades: str = 'jsongz') -> None:
    """
    Convert stored trades data to ohlcv data
    """
    data_handler_trades = get_datahandler(datadir, data_format=data_format_trades)
    data_handler_ohlcv = get_datahandler(datadir, data_format=data_format_ohlcv)

    for pair in pairs:
        trades = data_handler_trades.trades_load(pair)
        for timeframe in timeframes:
            if erase:
                if data_handler_ohlcv.ohlcv_purge(pair, timeframe):
                    logger.info(f'Deleting existing data for pair {pair}, interval {timeframe}.')
            ohlcv = trades_to_ohlcv(trades, timeframe)
            # Store ohlcv
            data_handler_ohlcv.ohlcv_store(pair, timeframe, data=ohlcv)


def get_timerange(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
    """
    Get the maximum common timerange for the given backtest data.

    :param data: dictionary with preprocessed backtesting data
    :return: tuple containing min_date, max_date
    """
    timeranges = [
        (arrow.get(frame['date'].min()), arrow.get(frame['date'].max()))
        for frame in data.values()
    ]
    return (min(timeranges, key=operator.itemgetter(0))[0],
            max(timeranges, key=operator.itemgetter(1))[1])


def validate_backtest_data(data: DataFrame, pair: str, min_date: datetime,
                           max_date: datetime, timeframe_min: int) -> bool:
    """
    Validates preprocessed backtesting data for missing values and shows warnings about it that.

    :param data: preprocessed backtesting data (as DataFrame)
    :param pair: pair used for log output.
    :param min_date: start-date of the data
    :param max_date: end-date of the data
    :param timeframe_min: Timeframe in minutes
    """
    # total difference in minutes / timeframe-minutes
    expected_frames = int((max_date - min_date).total_seconds() // 60 // timeframe_min)
    found_missing = False
    dflen = len(data)
    if dflen < expected_frames:
        found_missing = True
        logger.warning("%s has missing frames: expected %s, got %s, that's %s missing values",
                       pair, expected_frames, dflen, expected_frames - dflen)
    return found_missing
