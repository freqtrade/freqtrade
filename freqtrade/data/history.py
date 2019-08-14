"""
Handle historic data (ohlcv).

Includes:
* load data for a pair (or a list of pairs) from disk
* download data from exchange and store to disk
"""

import logging
import operator
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arrow
from pandas import DataFrame

from freqtrade import OperationalException, misc
from freqtrade.configuration import TimeRange
from freqtrade.data.converter import parse_ticker_dataframe
from freqtrade.exchange import Exchange, timeframe_to_minutes

logger = logging.getLogger(__name__)


def trim_tickerlist(tickerlist: List[Dict], timerange: TimeRange) -> List[Dict]:
    """
    Trim tickerlist based on given timerange
    """
    if not tickerlist:
        return tickerlist

    start_index = 0
    stop_index = len(tickerlist)

    if timerange.starttype == 'line':
        stop_index = timerange.startts
    if timerange.starttype == 'index':
        start_index = timerange.startts
    elif timerange.starttype == 'date':
        while (start_index < len(tickerlist) and
               tickerlist[start_index][0] < timerange.startts * 1000):
            start_index += 1

    if timerange.stoptype == 'line':
        start_index = len(tickerlist) + timerange.stopts
    if timerange.stoptype == 'index':
        stop_index = timerange.stopts
    elif timerange.stoptype == 'date':
        while (stop_index > 0 and
               tickerlist[stop_index-1][0] > timerange.stopts * 1000):
            stop_index -= 1

    if start_index > stop_index:
        raise ValueError(f'The timerange [{timerange.startts},{timerange.stopts}] is incorrect')

    return tickerlist[start_index:stop_index]


def load_tickerdata_file(datadir: Optional[Path], pair: str, ticker_interval: str,
                         timerange: Optional[TimeRange] = None) -> Optional[list]:
    """
    Load a pair from file, either .json.gz or .json
    :return: tickerlist or None if unsuccesful
    """
    filename = pair_data_filename(datadir, pair, ticker_interval)
    pairdata = misc.file_load_json(filename)
    if not pairdata:
        return []

    if timerange:
        pairdata = trim_tickerlist(pairdata, timerange)
    return pairdata


def load_pair_history(pair: str,
                      ticker_interval: str,
                      datadir: Optional[Path],
                      timerange: TimeRange = TimeRange(None, None, 0, 0),
                      refresh_pairs: bool = False,
                      exchange: Optional[Exchange] = None,
                      fill_up_missing: bool = True,
                      drop_incomplete: bool = True
                      ) -> DataFrame:
    """
    Loads cached ticker history for the given pair.
    :param pair: Pair to load data for
    :param ticker_interval: Ticker-interval (e.g. "5m")
    :param datadir: Path to the data storage location.
    :param timerange: Limit data to be loaded to this timerange
    :param refresh_pairs: Refresh pairs from exchange.
        (Note: Requires exchange to be passed as well.)
    :param exchange: Exchange object (needed when using "refresh_pairs")
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param drop_incomplete: Drop last candle assuming it may be incomplete.
    :return: DataFrame with ohlcv data
    """

    # The user forced the refresh of pairs
    if refresh_pairs:
        download_pair_history(datadir=datadir,
                              exchange=exchange,
                              pair=pair,
                              ticker_interval=ticker_interval,
                              timerange=timerange)

    pairdata = load_tickerdata_file(datadir, pair, ticker_interval, timerange=timerange)

    if pairdata:
        if timerange.starttype == 'date' and pairdata[0][0] > timerange.startts * 1000:
            logger.warning('Missing data at start for pair %s, data starts at %s',
                           pair, arrow.get(pairdata[0][0] // 1000).strftime('%Y-%m-%d %H:%M:%S'))
        if timerange.stoptype == 'date' and pairdata[-1][0] < timerange.stopts * 1000:
            logger.warning('Missing data at end for pair %s, data ends at %s',
                           pair,
                           arrow.get(pairdata[-1][0] // 1000).strftime('%Y-%m-%d %H:%M:%S'))
        return parse_ticker_dataframe(pairdata, ticker_interval, pair=pair,
                                      fill_missing=fill_up_missing,
                                      drop_incomplete=drop_incomplete)
    else:
        logger.warning(
            f'No history data for pair: "{pair}", interval: {ticker_interval}. '
            'Use --refresh-pairs-cached option or download_backtest_data.py '
            'script to download the data'
        )
        return None


def load_data(datadir: Optional[Path],
              ticker_interval: str,
              pairs: List[str],
              refresh_pairs: bool = False,
              exchange: Optional[Exchange] = None,
              timerange: TimeRange = TimeRange(None, None, 0, 0),
              fill_up_missing: bool = True,
              live: bool = False
              ) -> Dict[str, DataFrame]:
    """
    Loads ticker history data for a list of pairs the given parameters
    :return: dict(<pair>:<tickerlist>)
    """
    result: Dict[str, DataFrame] = {}
    if live:
        if exchange:
            logger.info('Live: Downloading data for all defined pairs ...')
            exchange.refresh_latest_ohlcv([(pair, ticker_interval) for pair in pairs])
            result = {key[0]: value for key, value in exchange._klines.items() if value is not None}
        else:
            raise OperationalException(
                "Exchange needs to be initialized when using live data."
            )
    else:
        logger.info('Using local backtesting data ...')

        for pair in pairs:
            hist = load_pair_history(pair=pair, ticker_interval=ticker_interval,
                                     datadir=datadir, timerange=timerange,
                                     refresh_pairs=refresh_pairs,
                                     exchange=exchange,
                                     fill_up_missing=fill_up_missing)
            if hist is not None:
                result[pair] = hist
    return result


def make_testdata_path(datadir: Optional[Path]) -> Path:
    """Return the path where testdata files are stored"""
    return datadir or (Path(__file__).parent.parent / "tests" / "testdata").resolve()


def pair_data_filename(datadir: Optional[Path], pair: str, ticker_interval: str) -> Path:
    path = make_testdata_path(datadir)
    pair_s = pair.replace("/", "_")
    filename = path.joinpath(f'{pair_s}-{ticker_interval}.json')
    return filename


def load_cached_data_for_updating(filename: Path, ticker_interval: str,
                                  timerange: Optional[TimeRange]) -> Tuple[List[Any],
                                                                           Optional[int]]:
    """
    Load cached data and choose what part of the data should be updated
    Only used by download_pair_history().
    """

    since_ms = None

    # user sets timerange, so find the start time
    if timerange:
        if timerange.starttype == 'date':
            since_ms = timerange.startts * 1000
        elif timerange.stoptype == 'line':
            num_minutes = timerange.stopts * timeframe_to_minutes(ticker_interval)
            since_ms = arrow.utcnow().shift(minutes=num_minutes).timestamp * 1000

    # read the cached file
    if filename.is_file():
        with open(filename, "rt") as file:
            data = misc.json_load(file)
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


def download_pair_history(datadir: Optional[Path],
                          exchange: Optional[Exchange],
                          pair: str,
                          ticker_interval: str = '5m',
                          timerange: Optional[TimeRange] = None) -> bool:
    """
    Download the latest ticker intervals from the exchange for the pair passed in parameters
    The data is downloaded starting from the last correct ticker interval data that
    exists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data

    :param pair: pair to download
    :param ticker_interval: ticker interval
    :param timerange: range of time to download
    :return: bool with success state
    """
    if not exchange:
        raise OperationalException(
            "Exchange needs to be initialized when downloading pair history data"
        )

    try:
        filename = pair_data_filename(datadir, pair, ticker_interval)

        logger.info(
            f'Download history data for pair: "{pair}", interval: {ticker_interval} '
            f'and store in {datadir}.'
        )

        data, since_ms = load_cached_data_for_updating(filename, ticker_interval, timerange)

        logger.debug("Current Start: %s", misc.format_ms_time(data[1][0]) if data else 'None')
        logger.debug("Current End: %s", misc.format_ms_time(data[-1][0]) if data else 'None')

        # Default since_ms to 30 days if nothing is given
        new_data = exchange.get_history(pair=pair, ticker_interval=ticker_interval,
                                        since_ms=since_ms if since_ms
                                        else
                                        int(arrow.utcnow().shift(days=-30).float_timestamp) * 1000)
        data.extend(new_data)

        logger.debug("New Start: %s", misc.format_ms_time(data[0][0]))
        logger.debug("New End: %s", misc.format_ms_time(data[-1][0]))

        misc.file_dump_json(filename, data)
        return True

    except Exception as e:
        logger.error(
            f'Failed to download history data for pair: "{pair}", interval: {ticker_interval}. '
            f'Error: {e}'
        )
        return False


def get_timeframe(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
    """
    Get the maximum timeframe for the given backtest data
    :param data: dictionary with preprocessed backtesting data
    :return: tuple containing min_date, max_date
    """
    timeframe = [
        (arrow.get(frame['date'].min()), arrow.get(frame['date'].max()))
        for frame in data.values()
    ]
    return min(timeframe, key=operator.itemgetter(0))[0], \
        max(timeframe, key=operator.itemgetter(1))[1]


def validate_backtest_data(data: DataFrame, pair: str, min_date: datetime,
                           max_date: datetime, ticker_interval_mins: int) -> bool:
    """
    Validates preprocessed backtesting data for missing values and shows warnings about it that.

    :param data: preprocessed backtesting data (as DataFrame)
    :param pair: pair used for log output.
    :param min_date: start-date of the data
    :param max_date: end-date of the data
    :param ticker_interval_mins: ticker interval in minutes
    """
    # total difference in minutes / interval-minutes
    expected_frames = int((max_date - min_date).total_seconds() // 60 // ticker_interval_mins)
    found_missing = False
    dflen = len(data)
    if dflen < expected_frames:
        found_missing = True
        logger.warning("%s has missing frames: expected %s, got %s, that's %s missing values",
                       pair, expected_frames, dflen, expected_frames - dflen)
    return found_missing
