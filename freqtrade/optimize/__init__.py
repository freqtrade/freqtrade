# pragma pylint: disable=missing-docstring

import logging
import json
import os
from typing import Optional, List, Dict
from freqtrade.exchange import get_ticker_history

from pandas import DataFrame

from freqtrade.analyze import populate_indicators, parse_ticker_dataframe

logger = logging.getLogger(__name__)


def load_data(pairs: List[str], ticker_interval: int = 5,
              refresh_pairs: Optional[bool] = False) -> Dict[str, List]:
    """
    Loads ticker history data for the given parameters
    :param ticker_interval: ticker interval in minutes
    :param pairs: list of pairs
    :return: dict
    """
    path = testdata_path()
    result = {}

    # If the user force the refresh of pairs
    if refresh_pairs:
        logger.info('Download data for all pairs and store them in freqtrade/tests/testsdata')
        download_pairs(pairs)

    for pair in pairs:
        file = '{abspath}/{pair}-{ticker_interval}.json'.format(
            abspath=path,
            pair=pair,
            ticker_interval=ticker_interval,
        )
        # The file does not exist we download it
        if not os.path.isfile(file):
            download_backtesting_testdata(pair=pair, interval=ticker_interval)

        # Read the file, load the json
        with open(file) as tickerdata:
            result[pair] = json.load(tickerdata)
    return result


def preprocess(tickerdata: Dict[str, List]) -> Dict[str, DataFrame]:
    """Creates a dataframe and populates indicators for given ticker data"""
    processed = {}
    for pair, pair_data in tickerdata.items():
        processed[pair] = populate_indicators(parse_ticker_dataframe(pair_data))
    return processed


def testdata_path() -> str:
    """Return the path where testdata files are stored"""
    return os.path.abspath(os.path.dirname(__file__)) + '/../tests/testdata'


def download_pairs(pairs: List[str]) -> bool:
    """For each pairs passed in parameters, download 1 and 5 ticker intervals"""
    for pair in pairs:
        try:
            for interval in [1, 5]:
                download_backtesting_testdata(pair=pair, interval=interval)
        except BaseException:
            logger.info('Failed to download the pair: "{pair}", Interval: {interval} min'.format(
                pair=pair,
                interval=interval,
            ))
            return False
    return True


def download_backtesting_testdata(pair: str, interval: int = 5) -> bool:
    """
    Download the latest 1 and 5 ticker intervals from Bittrex for the pairs passed in parameters
    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data
    :param pairs: list of pairs to download
    :return: bool
    """

    path = testdata_path()
    logger.info('Download the pair: "{pair}", Interval: {interval} min'.format(
        pair=pair,
        interval=interval,
    ))

    filepair = pair.replace("-", "_")
    filename = os.path.join(path, '{}-{}.json'.format(
        filepair,
        interval,
    ))
    filename = filename.replace('USDT_BTC', 'BTC_FAKEBULL')

    if os.path.isfile(filename):
        with open(filename, "rt") as fp:
            data = json.load(fp)
        logger.debug("Current Start:", data[1]['T'])
        logger.debug("Current End: ", data[-1:][0]['T'])
    else:
        data = []
        logger.debug("Current Start: None")
        logger.debug("Current End: None")

    new_data = get_ticker_history(pair=pair, tick_interval=int(interval))
    for row in new_data:
        if row not in data:
            data.append(row)
    logger.debug("New Start:", data[1]['T'])
    logger.debug("New End: ", data[-1:][0]['T'])
    data = sorted(data, key=lambda data: data['T'])

    with open(filename, "wt") as fp:
        json.dump(data, fp)

    return True
