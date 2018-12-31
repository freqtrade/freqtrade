"""
Handle historic data (ohlcv).
includes:
* load data for a pair (or a list of pairs) from disk
* download data from exchange and store to disk
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import arrow
from pandas import DataFrame

from freqtrade import misc, constants, OperationalException
from freqtrade.data.converter import parse_ticker_dataframe
from freqtrade.exchange import Exchange
from freqtrade.arguments import TimeRange

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


def load_tickerdata_file(
        datadir: Optional[Path], pair: str,
        ticker_interval: str,
        timerange: Optional[TimeRange] = None) -> Optional[list]:
    """
    Load a pair from file, either .json.gz or .json
    :return tickerlist or None if unsuccesful
    """
    path = make_testdata_path(datadir)
    pair_s = pair.replace('/', '_')
    file = path.joinpath(f'{pair_s}-{ticker_interval}.json')

    pairdata = misc.file_load_json(file)

    if not pairdata:
        return None

    if timerange:
        pairdata = trim_tickerlist(pairdata, timerange)
    return pairdata


def load_pair_history(pair: str,
                      ticker_interval: str,
                      datadir: Optional[Path],
                      timerange: TimeRange = TimeRange(None, None, 0, 0),
                      refresh_pairs: bool = False,
                      exchange: Optional[Exchange] = None,
                      fill_up_missing: bool = True
                      ) -> DataFrame:
    """
    Loads cached ticker history for the given pair.
    :return: DataFrame with ohlcv data
    """

    pairdata = load_tickerdata_file(datadir, pair, ticker_interval, timerange=timerange)
    # If the user force the refresh of pairs
    if refresh_pairs:
        if not exchange:
            raise OperationalException("Exchange needs to be initialized when "
                                       "calling load_data with refresh_pairs=True")

        logger.info('Download data for all pairs and store them in %s', datadir)
        download_pair_history(datadir=datadir,
                              exchange=exchange,
                              pair=pair,
                              tick_interval=ticker_interval,
                              timerange=timerange)

    if pairdata:
        if timerange.starttype == 'date' and pairdata[0][0] > timerange.startts * 1000:
            logger.warning('Missing data at start for pair %s, data starts at %s',
                           pair, arrow.get(pairdata[0][0] // 1000).strftime('%Y-%m-%d %H:%M:%S'))
        if timerange.stoptype == 'date' and pairdata[-1][0] < timerange.stopts * 1000:
            logger.warning('Missing data at end for pair %s, data ends at %s',
                           pair,
                           arrow.get(pairdata[-1][0] // 1000).strftime('%Y-%m-%d %H:%M:%S'))
        return parse_ticker_dataframe(pairdata, ticker_interval, fill_up_missing)
    else:
        logger.warning('No data for pair: "%s", Interval: %s. '
                       'Use --refresh-pairs-cached to download the data',
                       pair, ticker_interval)
        return None


def load_data(datadir: Optional[Path],
              ticker_interval: str,
              pairs: List[str],
              refresh_pairs: bool = False,
              exchange: Optional[Exchange] = None,
              timerange: TimeRange = TimeRange(None, None, 0, 0),
              fill_up_missing: bool = True) -> Dict[str, DataFrame]:
    """
    Loads ticker history data for a list of pairs the given parameters
    :return: dict(<pair>:<tickerlist>)
    """
    result = {}

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


def load_cached_data_for_updating(filename: Path, tick_interval: str,
                                  timerange: Optional[TimeRange]) -> Tuple[List[Any],
                                                                           Optional[int]]:
    """
    Load cached data and choose what part of the data should be updated
    """

    since_ms = None

    # user sets timerange, so find the start time
    if timerange:
        if timerange.starttype == 'date':
            since_ms = timerange.startts * 1000
        elif timerange.stoptype == 'line':
            num_minutes = timerange.stopts * constants.TICKER_INTERVAL_MINUTES[tick_interval]
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
                          exchange: Exchange,
                          pair: str,
                          tick_interval: str = '5m',
                          timerange: Optional[TimeRange] = None) -> bool:
    """
    Download the latest ticker intervals from the exchange for the pair passed in parameters
    The data is downloaded starting from the last correct ticker interval data that
    exists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data
    :param pair: pair to download
    :param tick_interval: ticker interval
    :param timerange: range of time to download
    :return: bool with success state

    """
    try:
        path = make_testdata_path(datadir)
        filepair = pair.replace("/", "_")
        filename = path.joinpath(f'{filepair}-{tick_interval}.json')

        logger.info('Download the pair: "%s", Interval: %s', pair, tick_interval)

        data, since_ms = load_cached_data_for_updating(filename, tick_interval, timerange)

        logger.debug("Current Start: %s", misc.format_ms_time(data[1][0]) if data else 'None')
        logger.debug("Current End: %s", misc.format_ms_time(data[-1][0]) if data else 'None')

        # Default since_ms to 30 days if nothing is given
        new_data = exchange.get_history(pair=pair, tick_interval=tick_interval,
                                        since_ms=since_ms if since_ms
                                        else
                                        int(arrow.utcnow().shift(days=-30).float_timestamp) * 1000)
        data.extend(new_data)

        logger.debug("New Start: %s", misc.format_ms_time(data[0][0]))
        logger.debug("New End: %s", misc.format_ms_time(data[-1][0]))

        misc.file_dump_json(filename, data)
        return True
    except BaseException:
            logger.info('Failed to download the pair: "%s", Interval: %s',
                        pair, tick_interval)
            return False
