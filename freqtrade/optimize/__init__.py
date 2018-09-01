# pragma pylint: disable=missing-docstring

import gzip
try:
    import ujson as json
    _UJSON = True
except ImportError:
    # see mypy/issues/1153
    import json  # type: ignore
    _UJSON = False
import logging
import os
from typing import Optional, List, Dict, Tuple, Any
import arrow

from freqtrade import misc, constants, OperationalException
from freqtrade.exchange import Exchange
from freqtrade.arguments import TimeRange

logger = logging.getLogger(__name__)


def json_load(data):
    """Try to load data with ujson"""
    if _UJSON:
        return json.load(data, precise_float=True)
    else:
        return json.load(data)


def trim_tickerlist(tickerlist: List[Dict], timerange: TimeRange) -> List[Dict]:
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
        datadir: str, pair: str,
        ticker_interval: str,
        timerange: Optional[TimeRange] = None) -> Optional[List[Dict]]:
    """
    Load a pair from file,
    :return dict OR empty if unsuccesful
    """
    path = make_testdata_path(datadir)
    pair_s = pair.replace('/', '_')
    file = os.path.join(path, f'{pair_s}-{ticker_interval}.json')
    gzipfile = file + '.gz'

    # If the file does not exist we download it when None is returned.
    # If file exists, read the file, load the json
    if os.path.isfile(gzipfile):
        logger.debug('Loading ticker data from file %s', gzipfile)
        with gzip.open(gzipfile) as tickerdata:
            pairdata = json.load(tickerdata)
    elif os.path.isfile(file):
        logger.debug('Loading ticker data from file %s', file)
        with open(file) as tickerdata:
            pairdata = json.load(tickerdata)
    else:
        return None

    if timerange:
        pairdata = trim_tickerlist(pairdata, timerange)
    return pairdata


def load_data(datadir: str,
              ticker_interval: str,
              pairs: List[str],
              refresh_pairs: Optional[bool] = False,
              exchange: Optional[Exchange] = None,
              timerange: TimeRange = TimeRange(None, None, 0, 0)) -> Dict[str, List]:
    """
    Loads ticker history data for the given parameters
    :return: dict
    """
    result = {}

    # If the user force the refresh of pairs
    if refresh_pairs:
        logger.info('Download data for all pairs and store them in %s', datadir)
        if not exchange:
            raise OperationalException("Exchange needs to be initialized when "
                                       "calling load_data with refresh_pairs=True")
        download_pairs(datadir, exchange, pairs, ticker_interval, timerange=timerange)

    for pair in pairs:
        pairdata = load_tickerdata_file(datadir, pair, ticker_interval, timerange=timerange)
        if pairdata:
            result[pair] = pairdata
        else:
            logger.warning(
                'No data for pair: "%s", Interval: %s. '
                'Use --refresh-pairs-cached to download the data',
                pair,
                ticker_interval
            )

    return result


def make_testdata_path(datadir: str) -> str:
    """Return the path where testdata files are stored"""
    return datadir or os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '..', 'tests', 'testdata'
        )
    )


def download_pairs(datadir, exchange: Exchange, pairs: List[str],
                   ticker_interval: str,
                   timerange: TimeRange = TimeRange(None, None, 0, 0)) -> bool:
    """For each pairs passed in parameters, download the ticker intervals"""
    for pair in pairs:
        try:
            download_backtesting_testdata(datadir,
                                          exchange=exchange,
                                          pair=pair,
                                          tick_interval=ticker_interval,
                                          timerange=timerange)
        except BaseException:
            logger.info(
                'Failed to download the pair: "%s", Interval: %s',
                pair,
                ticker_interval
            )
            return False
    return True


def load_cached_data_for_updating(filename: str,
                                  tick_interval: str,
                                  timerange: Optional[TimeRange]) -> Tuple[
                                                                                  List[Any],
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
    if os.path.isfile(filename):
        with open(filename, "rt") as file:
            data = json_load(file)
            # remove the last item, because we are not sure if it is correct
            # it could be fetched when the candle was incompleted
            if data:
                data.pop()
    else:
        data = []

    if data:
        if since_ms and since_ms < data[0][0]:
            # the data is requested for earlier period than the cache has
            # so fully redownload all the data
            data = []
        else:
            # a part of the data was already downloaded, so
            # download unexist data only
            since_ms = data[-1][0] + 1

    return (data, since_ms)


def download_backtesting_testdata(datadir: str,
                                  exchange: Exchange,
                                  pair: str,
                                  tick_interval: str = '5m',
                                  timerange: Optional[TimeRange] = None) -> None:

    """
    Download the latest ticker intervals from the exchange for the pairs passed in parameters
    The data is downloaded starting from the last correct ticker interval data that
    esists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data
    :param pairs: list of pairs to download
    :param tick_interval: ticker interval
    :param timerange: range of time to download
    :return: None

    """

    path = make_testdata_path(datadir)
    filepair = pair.replace("/", "_")
    filename = os.path.join(path, f'{filepair}-{tick_interval}.json')

    logger.info(
        'Download the pair: "%s", Interval: %s',
        pair,
        tick_interval
    )

    data, since_ms = load_cached_data_for_updating(filename, tick_interval, timerange)

    logger.debug("Current Start: %s", misc.format_ms_time(data[1][0]) if data else 'None')
    logger.debug("Current End: %s", misc.format_ms_time(data[-1][0]) if data else 'None')

    new_data = exchange.get_candle_history(pair=pair, tick_interval=tick_interval,
                                           since_ms=since_ms)
    data.extend(new_data)

    logger.debug("New Start: %s", misc.format_ms_time(data[0][0]))
    logger.debug("New End: %s", misc.format_ms_time(data[-1][0]))

    misc.file_dump_json(filename, data)
