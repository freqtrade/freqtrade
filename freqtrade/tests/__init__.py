import json
import os


def load_backtesting_data(ticker_interval: int = 5):
    path = os.path.abspath(os.path.dirname(__file__))
    result = {}
    pairs = [
        'BTC_BCC', 'BTC_ETH', 'BTC_MER', 'BTC_POWR', 'BTC_ETC',
        'BTC_OK', 'BTC_NEO', 'BTC_EMC2', 'BTC_DASH', 'BTC_LSK',
        'BTC_LTC', 'BTC_XZC', 'BTC_OMG', 'BTC_STRAT', 'BTC_XRP',
        'BTC_QTUM', 'BTC_WAVES', 'BTC_VTC', 'BTC_XLM', 'BTC_MCO'
    ]
    for pair in pairs:
        with open('{abspath}/testdata/{pair}-{ticker_interval}.json'.format(
                abspath=path,
                pair=pair,
                ticker_interval=ticker_interval,
        )) as fp:
            result[pair] = json.load(fp)
    return result
