# pragma pylint: disable=missing-docstring
import json
import os


def load_backtesting_data(ticker_interval: int = 5):
    path = os.path.abspath(os.path.dirname(__file__))
    result = {}
    pairs = [
        'BTC_BCC', 'BTC_ETH', 'BTC_DASH', 'BTC_POWR', 'BTC_ETC',
        'BTC_VTC', 'BTC_WAVES', 'BTC_LSK', 'BTC_XLM', 'BTC_OK',
    ]
    for pair in pairs:
        with open('{abspath}/testdata/{pair}-{ticker_interval}.json'.format(
            abspath=path,
            pair=pair,
            ticker_interval=ticker_interval,
        )) as tickerdata:
            result[pair] = json.load(tickerdata)
    return result
