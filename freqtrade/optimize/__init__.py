# pragma pylint: disable=missing-docstring

import gzip
import json
import logging
import os
import arrow
from typing import Optional, List, Dict, Tuple

from freqtrade import misc
from freqtrade.exchange import get_ticker_history
from freqtrade.constants import Constants

from user_data.hyperopt_conf import hyperopt_optimize_conf

logger = logging.getLogger(__name__)


def trim_tickerlist(tickerlist: List[Dict], timerange: Tuple[Tuple, int, int]) -> List[Dict]:
    if not tickerlist:
        return tickerlist

    stype, start, stop = timerange

    start_index = 0
    stop_index = len(tickerlist)

    if stype[0] == 'line':
        stop_index = start
    if stype[0] == 'index':
        start_index = start
    elif stype[0] == 'date':
        while tickerlist[start_index][0] < start * 1000:
            start_index += 1

    if stype[1] == 'line':
        start_index = len(tickerlist) + stop
    if stype[1] == 'index':
        stop_index = stop
    elif stype[1] == 'date':
        while tickerlist[stop_index-1][0] > stop * 1000:
            stop_index -= 1

    if start_index > stop_index:
        raise ValueError(f'The timerange [{start},{stop}] is incorrect')

    return tickerlist[start_index:stop_index]


def load_tickerdata_file(
        datadir: str, pair: str,
        ticker_interval: str,
        timerange: Optional[Tuple[Tuple, int, int]] = None) -> Optional[List[Dict]]:
    """
    Load a pair from file,
    :return dict OR empty if unsuccesful
    """
    path = make_testdata_path(datadir)
    pair_file_string = pair.replace('/', '_')
    file = os.path.join(path, '{pair}-{ticker_interval}.json'.format(
        pair=pair_file_string,
        ticker_interval=ticker_interval,
    ))
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
              pairs: Optional[List[str]] = None,
              refresh_pairs: Optional[bool] = False,
              timerange: Optional[Tuple[Tuple, int, int]] = None) -> Dict[str, List]:
    """
    Loads ticker history data for the given parameters
    :return: dict
    """
    result = {}

    _pairs = pairs or hyperopt_optimize_conf()['exchange']['pair_whitelist']

    # If the user force the refresh of pairs
    if refresh_pairs:
        logger.info('Download data for all pairs and store them in %s', datadir)
        download_pairs(datadir, _pairs, ticker_interval, timerange=timerange)

    for pair in _pairs:
        pairdata = load_tickerdata_file(datadir, pair, ticker_interval, timerange=timerange)
        if not pairdata:
            # download the tickerdata from exchange
            download_backtesting_testdata(datadir,
                                          pair=pair,
                                          tick_interval=ticker_interval,
                                          timerange=timerange)
            # and retry reading the pair
            pairdata = load_tickerdata_file(datadir, pair, ticker_interval, timerange=timerange)
        result[pair] = pairdata
    return result


def make_testdata_path(datadir: str) -> str:
    """Return the path where testdata files are stored"""
    return datadir or os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '..', 'tests', 'testdata'
        )
    )


def download_pairs(datadir, pairs: List[str],
                   ticker_interval: str,
                   timerange: Optional[Tuple[Tuple, int, int]] = None) -> bool:
    """For each pairs passed in parameters, download the ticker intervals"""
    for pair in pairs:
        try:
            download_backtesting_testdata(datadir,
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
                                  timerange: Optional[Tuple[Tuple, int, int]]) -> Tuple[list, int]:
    """
    Load cached data and choose what part of the data should be updated
    """

    since_ms = None

    # user sets timerange, so find the start time
    if timerange:
        if timerange[0][0] == 'date':
            since_ms = timerange[1] * 1000
        elif timerange[0][1] == 'line':
            num_minutes = timerange[2] * Constants.TICKER_INTERVAL_MINUTES[tick_interval]
            since_ms = arrow.utcnow().shift(minutes=num_minutes).timestamp * 1000

    # read the cached file
    if os.path.isfile(filename):
        with open(filename, "rt") as file:
            data = json.load(file)
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


# FIX: 20180110, suggest rename interval to tick_interval
def download_backtesting_testdata(datadir: str,
                                  pair: str,
                                  tick_interval: str = '5m',
                                  timerange: Optional[Tuple[Tuple, int, int]] = None) -> bool:
    """
    Download the latest ticker intervals from the exchange for the pairs passed in parameters
    The data is downloaded starting from the last correct ticker interval data that
    esists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data
    :param pairs: list of pairs to download
    :param tick_interval: ticker interval
    :param timerange: range of time to download
    :return: bool
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

    new_data = get_ticker_history(pair=pair, tick_interval=tick_interval, since_ms=since_ms)
    data.extend(new_data)

    logger.debug("New Start: %s", misc.format_ms_time(data[0][0]))
    logger.debug("New End: %s", misc.format_ms_time(data[-1][0]))

    misc.file_dump_json(filename, data)

    return True
