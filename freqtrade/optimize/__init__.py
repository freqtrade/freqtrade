# pragma pylint: disable=missing-docstring


import json
import os
from typing import Optional, List, Dict

from pandas import DataFrame

from freqtrade.analyze import populate_indicators, parse_ticker_dataframe


def load_data(ticker_interval: int = 5, pairs: Optional[List[str]] = None) -> Dict[str, List]:
    """
    Loads ticker history data for the given parameters
    :param ticker_interval: ticker interval in minutes
    :param pairs: list of pairs
    :return: dict
    """
    path = os.path.abspath(os.path.dirname(__file__))
    result = {}
    _pairs = pairs or [
        'BTC_BCC', 'BTC_ETH', 'BTC_DASH', 'BTC_POWR', 'BTC_ETC',
        'BTC_VTC', 'BTC_WAVES', 'BTC_LSK', 'BTC_XLM', 'BTC_OK',
    ]
    for pair in _pairs:
        with open('{abspath}/../tests/testdata/{pair}-{ticker_interval}.json'.format(
            abspath=path,
            pair=pair,
            ticker_interval=ticker_interval,
        )) as tickerdata:
            result[pair] = json.load(tickerdata)
    return result


def preprocess(tickerdata: Dict[str, List]) -> Dict[str, DataFrame]:
    """Creates a dataframe and populates indicators for given ticker data"""
    processed = {}
    for pair, pair_data in tickerdata.items():
        processed[pair] = populate_indicators(parse_ticker_dataframe(pair_data))
    return processed
