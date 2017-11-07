# pragma pylint: disable=missing-docstring
import json
import pytest


@pytest.fixture(scope="module")
def conf():
    return {
        "minimal_roi": {
            "40":  0.0,
            "30":  0.01,
            "20":  0.02,
            "0":  0.04
        },
        "stoploss": -0.05
    }


@pytest.fixture(scope="module")
def backdata():
    result = {}
    for pair in ['btc-neo', 'btc-eth', 'btc-omg', 'btc-edg', 'btc-pay',
                 'btc-pivx', 'btc-qtum', 'btc-mtl', 'btc-etc', 'btc-ltc']:
        with open('freqtrade/tests/testdata/' + pair + '.json') as data_file:
            result[pair] = json.load(data_file)
    return result
