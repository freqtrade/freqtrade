"""
Handle historic data (ohlcv).

Includes:
* load data for a pair (or a list of pairs) from disk
* download data from exchange and store to disk
"""

import logging
import operator
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arrow
from pandas import DataFrame

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.data.converter import parse_ticker_dataframe, trades_to_ohlcv
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import (Exchange, timeframe_to_minutes,
                                timeframe_to_seconds)

logger = logging.getLogger(__name__)


def trim_tickerlist(tickerlist: List[Dict], timerange: TimeRange) -> List[Dict]:
    """
    Trim tickerlist based on given timerange
    """
    if not tickerlist:
        return tickerlist

    start_index = 0
    stop_index = len(tickerlist)

    if timerange.starttype == 'date':
        while (start_index < len(tickerlist) and
               tickerlist[start_index][0] < timerange.startts * 1000):
            start_index += 1

    if timerange.stoptype == 'date':
        while (stop_index > 0 and
               tickerlist[stop_index-1][0] > timerange.stopts * 1000):
            stop_index -= 1

    if start_index > stop_index:
        raise ValueError(f'The timerange [{timerange.startts},{timerange.stopts}] is incorrect')

    return tickerlist[start_index:stop_index]


def trim_dataframe(df: DataFrame, timerange: TimeRange, df_date_col: str = 'date') -> DataFrame:
    """
    Trim dataframe based on given timerange
    :param df: Dataframe to trim
    :param timerange: timerange (use start and end date if available)
    :param: df_date_col: Column in the dataframe to use as Date column
    :return: trimmed dataframe
    """
    if timerange.starttype == 'date':
        start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)
        df = df.loc[df[df_date_col] >= start, :]
    if timerange.stoptype == 'date':
        stop = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)
        df = df.loc[df[df_date_col] <= stop, :]
    return df


def load_tickerdata_file(datadir: Path, pair: str, timeframe: str,
                         timerange: Optional[TimeRange] = None) -> List[Dict]:
    """
    Load a pair from file, either .json.gz or .json
    :return: tickerlist or None if unsuccessful
    """
    filename = pair_data_filename(datadir, pair, timeframe)
    pairdata = misc.file_load_json(filename)
    if not pairdata:
        return []

    if timerange:
        pairdata = trim_tickerlist(pairdata, timerange)
    return pairdata


def store_tickerdata_file(datadir: Path, pair: str,
                          timeframe: str, data: list, is_zip: bool = False):
    """
    Stores tickerdata to file
    """
    filename = pair_data_filename(datadir, pair, timeframe)
    misc.file_dump_json(filename, data, is_zip=is_zip)


def load_trades_file(datadir: Path, pair: str,
                     timerange: Optional[TimeRange] = None) -> List[Dict]:
    """
    Load a pair from file, either .json.gz or .json
    :return: tradelist or empty list if unsuccesful
    """
    filename = pair_trades_filename(datadir, pair)
    tradesdata = misc.file_load_json(filename)
    if not tradesdata:
        return []

    return tradesdata


def store_trades_file(datadir: Path, pair: str,
                      data: list, is_zip: bool = True):
    """
    Stores tickerdata to file
    """
    filename = pair_trades_filename(datadir, pair)
    misc.file_dump_json(filename, data, is_zip=is_zip)


def _validate_pairdata(pair, pairdata, timerange: TimeRange):
    if timerange.starttype == 'date' and pairdata[0][0] > timerange.startts * 1000:
        logger.warning('Missing data at start for pair %s, data starts at %s',
                       pair, arrow.get(pairdata[0][0] // 1000).strftime('%Y-%m-%d %H:%M:%S'))
    if timerange.stoptype == 'date' and pairdata[-1][0] < timerange.stopts * 1000:
        logger.warning('Missing data at end for pair %s, data ends at %s',
                       pair, arrow.get(pairdata[-1][0] // 1000).strftime('%Y-%m-%d %H:%M:%S'))


def load_pair_history(pair: str,
                      timeframe: str,
                      datadir: Path,
                      timerange: Optional[TimeRange] = None,
                      fill_up_missing: bool = True,
                      drop_incomplete: bool = True,
                      startup_candles: int = 0,
                      ) -> DataFrame:
    """
    Load cached ticker history for the given pair.

    :param pair: Pair to load data for
    :param timeframe: Ticker timeframe (e.g. "5m")
    :param datadir: Path to the data storage location.
    :param timerange: Limit data to be loaded to this timerange
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param drop_incomplete: Drop last candle assuming it may be incomplete.
    :param startup_candles: Additional candles to load at the start of the period
    :return: DataFrame with ohlcv data, or empty DataFrame
    """
    timerange_startup = deepcopy(timerange)
    if startup_candles > 0 and timerange_startup:
        timerange_startup.subtract_start(timeframe_to_seconds(timeframe) * startup_candles)

    pairdata = load_tickerdata_file(datadir, pair, timeframe, timerange=timerange_startup)

    if pairdata:
        if timerange_startup:
            _validate_pairdata(pair, pairdata, timerange_startup)
        return parse_ticker_dataframe(pairdata, timeframe, pair=pair,
                                      fill_missing=fill_up_missing,
                                      drop_incomplete=drop_incomplete)
    else:
        logger.warning(
            f'No history data for pair: "{pair}", timeframe: {timeframe}. '
            'Use `freqtrade download-data` to download the data'
        )
        return DataFrame()


def load_data(datadir: Path,
              timeframe: str,
              pairs: List[str],
              timerange: Optional[TimeRange] = None,
              fill_up_missing: bool = True,
              startup_candles: int = 0,
              fail_without_data: bool = False
              ) -> Dict[str, DataFrame]:
    """
    Load ticker history data for a list of pairs.

    :param datadir: Path to the data storage location.
    :param timeframe: Ticker Timeframe (e.g. "5m")
    :param pairs: List of pairs to load
    :param timerange: Limit data to be loaded to this timerange
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param startup_candles: Additional candles to load at the start of the period
    :param fail_without_data: Raise OperationalException if no data is found.
    :return: dict(<pair>:<Dataframe>)
    """
    result: Dict[str, DataFrame] = {}
    if startup_candles > 0 and timerange:
        logger.info(f'Using indicator startup period: {startup_candles} ...')

    for pair in pairs:
        hist = load_pair_history(pair=pair, timeframe=timeframe,
                                 datadir=datadir, timerange=timerange,
                                 fill_up_missing=fill_up_missing,
                                 startup_candles=startup_candles)
        if not hist.empty:
            result[pair] = hist

    if fail_without_data and not result:
        raise OperationalException("No data found. Terminating.")
    return result


def refresh_data(datadir: Path,
                 timeframe: str,
                 pairs: List[str],
                 exchange: Exchange,
                 timerange: Optional[TimeRange] = None,
                 ) -> None:
    """
    Refresh ticker history data for a list of pairs.

    :param datadir: Path to the data storage location.
    :param timeframe: Ticker Timeframe (e.g. "5m")
    :param pairs: List of pairs to load
    :param exchange: Exchange object
    :param timerange: Limit data to be loaded to this timerange
    """
    for pair in pairs:
        _download_pair_history(pair=pair, timeframe=timeframe,
                               datadir=datadir, timerange=timerange,
                               exchange=exchange)


def pair_data_filename(datadir: Path, pair: str, timeframe: str) -> Path:
    pair_s = pair.replace("/", "_")
    filename = datadir.joinpath(f'{pair_s}-{timeframe}.json')
    return filename


def pair_trades_filename(datadir: Path, pair: str) -> Path:
    pair_s = pair.replace("/", "_")
    filename = datadir.joinpath(f'{pair_s}-trades.json.gz')
    return filename


def _load_cached_data_for_updating(datadir: Path, pair: str, timeframe: str,
                                   timerange: Optional[TimeRange]) -> Tuple[List[Any],
                                                                            Optional[int]]:
    """
    Load cached data to download more data.
    If timerange is passed in, checks whether data from an before the stored data will be
    downloaded.
    If that's the case then what's available should be completely overwritten.
    Only used by download_pair_history().
    """

    since_ms = None

    # user sets timerange, so find the start time
    if timerange:
        if timerange.starttype == 'date':
            since_ms = timerange.startts * 1000
        elif timerange.stoptype == 'line':
            num_minutes = timerange.stopts * timeframe_to_minutes(timeframe)
            since_ms = arrow.utcnow().shift(minutes=num_minutes).timestamp * 1000

    # read the cached file
    # Intentionally don't pass timerange in - since we need to load the full dataset.
    data = load_tickerdata_file(datadir, pair, timeframe)
    # remove the last item, could be incomplete candle
    if data:
        data.pop()
    else:
        data = []

    if data:
        if since_ms and since_ms < data[0][0]:
            # Earlier data than existing data requested, redownload all
            data = []
        else:
            # a part of the data was already downloaded, so download unexist data only
            since_ms = data[-1][0] + 1

    return (data, since_ms)


def _download_pair_history(datadir: Path,
                           exchange: Exchange,
                           pair: str,
                           timeframe: str = '5m',
                           timerange: Optional[TimeRange] = None) -> bool:
    """
    Download latest candles from the exchange for the pair and timeframe passed in parameters
    The data is downloaded starting from the last correct data that
    exists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data

    :param pair: pair to download
    :param timeframe: Ticker Timeframe (e.g 5m)
    :param timerange: range of time to download
    :return: bool with success state
    """
    try:
        logger.info(
            f'Download history data for pair: "{pair}", timeframe: {timeframe} '
            f'and store in {datadir}.'
        )

        data, since_ms = _load_cached_data_for_updating(datadir, pair, timeframe, timerange)

        logger.debug("Current Start: %s", misc.format_ms_time(data[1][0]) if data else 'None')
        logger.debug("Current End: %s", misc.format_ms_time(data[-1][0]) if data else 'None')

        # Default since_ms to 30 days if nothing is given
        new_data = exchange.get_historic_ohlcv(pair=pair,
                                               timeframe=timeframe,
                                               since_ms=since_ms if since_ms else
                                               int(arrow.utcnow().shift(
                                                   days=-30).float_timestamp) * 1000
                                               )
        data.extend(new_data)

        logger.debug("New Start: %s", misc.format_ms_time(data[0][0]))
        logger.debug("New End: %s", misc.format_ms_time(data[-1][0]))

        store_tickerdata_file(datadir, pair, timeframe, data=data)
        return True

    except Exception as e:
        logger.error(
            f'Failed to download history data for pair: "{pair}", timeframe: {timeframe}. '
            f'Error: {e}'
        )
        return False


def refresh_backtest_ohlcv_data(exchange: Exchange, pairs: List[str], timeframes: List[str],
                                datadir: Path, timerange: Optional[TimeRange] = None,
                                erase=False) -> List[str]:
    """
    Refresh stored ohlcv data for backtesting and hyperopt operations.
    Used by freqtrade download-data subcommand.
    :return: List of pairs that are not available.
    """
    pairs_not_available = []
    for pair in pairs:
        if pair not in exchange.markets:
            pairs_not_available.append(pair)
            logger.info(f"Skipping pair {pair}...")
            continue
        for timeframe in timeframes:

            dl_file = pair_data_filename(datadir, pair, timeframe)
            if erase and dl_file.exists():
                logger.info(
                    f'Deleting existing data for pair {pair}, interval {timeframe}.')
                dl_file.unlink()

            logger.info(f'Downloading pair {pair}, interval {timeframe}.')
            _download_pair_history(datadir=datadir, exchange=exchange,
                                   pair=pair, timeframe=str(timeframe),
                                   timerange=timerange)
    return pairs_not_available


def _download_trades_history(datadir: Path,
                             exchange: Exchange,
                             pair: str,
                             timerange: Optional[TimeRange] = None) -> bool:
    """
    Download trade history from the exchange.
    Appends to previously downloaded trades data.
    """
    try:

        since = timerange.startts * 1000 if timerange and timerange.starttype == 'date' else None

        trades = load_trades_file(datadir, pair)

        from_id = trades[-1]['id'] if trades else None

        logger.debug("Current Start: %s", trades[0]['datetime'] if trades else 'None')
        logger.debug("Current End: %s", trades[-1]['datetime'] if trades else 'None')

        # Default since_ms to 30 days if nothing is given
        new_trades = exchange.get_historic_trades(pair=pair,
                                                  since=since if since else
                                                  int(arrow.utcnow().shift(
                                                      days=-30).float_timestamp) * 1000,
                                                  from_id=from_id,
                                                  )
        trades.extend(new_trades[1])
        store_trades_file(datadir, pair, trades)

        logger.debug("New Start: %s", trades[0]['datetime'])
        logger.debug("New End: %s", trades[-1]['datetime'])
        logger.info(f"New Amount of trades: {len(trades)}")
        return True

    except Exception as e:
        logger.error(
            f'Failed to download historic trades for pair: "{pair}". '
            f'Error: {e}'
        )
        return False


def refresh_backtest_trades_data(exchange: Exchange, pairs: List[str], datadir: Path,
                                 timerange: TimeRange, erase=False) -> List[str]:
    """
    Refresh stored trades data for backtesting and hyperopt operations.
    Used by freqtrade download-data subcommand.
    :return: List of pairs that are not available.
    """
    pairs_not_available = []
    for pair in pairs:
        if pair not in exchange.markets:
            pairs_not_available.append(pair)
            logger.info(f"Skipping pair {pair}...")
            continue

        dl_file = pair_trades_filename(datadir, pair)
        if erase and dl_file.exists():
            logger.info(
                f'Deleting existing data for pair {pair}.')
            dl_file.unlink()

        logger.info(f'Downloading trades for pair {pair}.')
        _download_trades_history(datadir=datadir, exchange=exchange,
                                 pair=pair,
                                 timerange=timerange)
    return pairs_not_available


def convert_trades_to_ohlcv(pairs: List[str], timeframes: List[str],
                            datadir: Path, timerange: TimeRange, erase=False) -> None:
    """
    Convert stored trades data to ohlcv data
    """
    for pair in pairs:
        trades = load_trades_file(datadir, pair)
        for timeframe in timeframes:
            ohlcv_file = pair_data_filename(datadir, pair, timeframe)
            if erase and ohlcv_file.exists():
                logger.info(f'Deleting existing data for pair {pair}, interval {timeframe}.')
                ohlcv_file.unlink()
            ohlcv = trades_to_ohlcv(trades, timeframe)
            # Store ohlcv
            store_tickerdata_file(datadir, pair, timeframe, data=ohlcv)


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
    :param timeframe_min: ticker Timeframe in minutes
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
