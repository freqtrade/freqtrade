# pragma pylint: disable=missing-docstring
import json
import os
from typing import Optional, List


def load_backtesting_data(ticker_interval: int = 5, pairs: Optional[List[str]] = None):
    path = os.path.abspath(os.path.dirname(__file__))
    result = {}
    _pairs = pairs or [
        'BTC_BCC', 'BTC_ETH', 'BTC_DASH', 'BTC_POWR', 'BTC_ETC',
        'BTC_VTC', 'BTC_WAVES', 'BTC_LSK', 'BTC_XLM', 'BTC_OK',
    ]
    for pair in _pairs:
        with open('{abspath}/testdata/{pair}-{ticker_interval}.json'.format(
            abspath=path,
            pair=pair,
            ticker_interval=ticker_interval,
        )) as tickerdata:
            result[pair] = json.load(tickerdata)
    return result
