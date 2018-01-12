# pragma pylint: disable=missing-docstring

import logging
import json
import os
from typing import Optional, List, Dict
from pandas import DataFrame
from freqtrade.exchange import get_ticker_history
from freqtrade.optimize.hyperopt_conf import hyperopt_optimize_conf
from freqtrade.analyze import populate_indicators, parse_ticker_dataframe

logger = logging.getLogger(__name__)


def load_tickerdata_file(datadir, pair, ticker_interval):
    """
    Load a pair from file,
    :return dict OR empty if unsuccesful
    """
    path = make_testdata_path(datadir)
    file = '{abspath}/{pair}-{ticker_interval}.json'.format(
        abspath=path,
        pair=pair,
        ticker_interval=ticker_interval,
    )
    # The file does not exist we download it
    if not os.path.isfile(file):
        return None

    # Read the file, load the json
    with open(file) as tickerdata:
        pairdata = json.load(tickerdata)
    return pairdata


def load_data(datadir: str, ticker_interval: int, pairs: Optional[List[str]] = None,
              refresh_pairs: Optional[bool] = False) -> Dict[str, List]:
    """
    Loads ticker history data for the given parameters
    :param ticker_interval: ticker interval in minutes
    :param pairs: list of pairs
    :return: dict
    """
    result = {}

    _pairs = pairs or hyperopt_optimize_conf()['exchange']['pair_whitelist']

    # If the user force the refresh of pairs
    if refresh_pairs:
        logger.info('Download data for all pairs and store them in %s', datadir)
        download_pairs(datadir, _pairs)

    for pair in _pairs:
        pairdata = load_tickerdata_file(datadir, pair, ticker_interval)
        if not pairdata:
            # download the tickerdata from exchange
            download_backtesting_testdata(datadir, pair=pair, interval=ticker_interval)
            # and retry reading the pair
            pairdata = load_tickerdata_file(datadir, pair, ticker_interval)
        result[pair] = pairdata
    return result


def preprocess(tickerdata: Dict[str, List]) -> Dict[str, DataFrame]:
    """Creates a dataframe and populates indicators for given ticker data"""
    return {pair: populate_indicators(parse_ticker_dataframe(pair_data))
            for pair, pair_data in tickerdata.items()}


def make_testdata_path(datadir: str) -> str:
    """Return the path where testdata files are stored"""
    return datadir or os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', 'tests', 'testdata'))


def download_pairs(datadir, pairs: List[str]) -> bool:
    """For each pairs passed in parameters, download 1 and 5 ticker intervals"""
    for pair in pairs:
        try:
            for interval in [1, 5, 30, 60, 1440]:
                download_backtesting_testdata(datadir, pair=pair, interval=interval)
        except BaseException:
            logger.info('Failed to download the pair: "{pair}", Interval: {interval} min'.format(
                pair=pair,
                interval=interval,
            ))
            return False
    return True


def download_backtesting_testdata(datadir: str, pair: str, interval: int = 5) -> bool:
    """
    Download the latest 1 and 5 ticker intervals from Bittrex for the pairs passed in parameters
    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data
    :param pairs: list of pairs to download
    :return: bool
    """

    path = make_testdata_path(datadir)
    logger.info('Download the pair: "{pair}", Interval: {interval} min'.format(
        pair=pair,
        interval=interval,
    ))

    filepair = pair.replace("-", "_")
    filename = os.path.join(path, '{pair}-{interval}.json'.format(
        pair=filepair,
        interval=interval,
    ))
    filename = filename.replace('USDT_BTC', 'BTC_FAKEBULL')

    if os.path.isfile(filename):
        with open(filename, "rt") as fp:
            data = json.load(fp)
        logger.debug("Current Start: {}".format(data[1]['T']))
        logger.debug("Current End: {}".format(data[-1:][0]['T']))
    else:
        data = []
        logger.debug("Current Start: None")
        logger.debug("Current End: None")

    new_data = get_ticker_history(pair=pair, tick_interval=int(interval))
    for row in new_data:
        if row not in data:
            data.append(row)
    logger.debug("New Start: {}".format(data[1]['T']))
    logger.debug("New End: {}".format(data[-1:][0]['T']))
    data = sorted(data, key=lambda data: data['T'])

    with open(filename, "wt") as fp:
        json.dump(data, fp)

    return True
