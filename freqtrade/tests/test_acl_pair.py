
# whitelist, blacklist, filtering, all of that will
# eventually become some rules to run on a generic ACL engine

# perhaps try to anticipate that by using some python package

import pytest
from unittest.mock import MagicMock
import copy

from freqtrade.main import refresh_whitelist
#from freqtrade.exchange import Exchanges
from freqtrade import exchange

# "deep equal"
def assert_list_equal (l1, l2):
    for pair in l1:
        assert pair in l2
    for pair in l2:
        assert pair in l1

def whitelist_conf():
    return {
        "stake_currency":"BTC",
        "exchange": {
            "pair_whitelist": [
                "BTC_ETH",
                "BTC_TKN",
                "BTC_TRST",
                "BTC_SWT",
                "BTC_BCC"
            ],
        },
    }

def get_health():
    return [{'Currency': 'ETH',
             'IsActive': True
            },
            {'Currency': 'TKN',
             'IsActive': True
            }]

def get_health_empty():
    return []

# below three test could be merged into a single
# test that ran randomlly generated health lists

def test_refresh_whitelist(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_wallet_health=get_health)
    # no argument: use the whitelist provided by config
    refresh_whitelist()
    whitelist = ['BTC_ETH', 'BTC_TKN']
    pairslist = conf['exchange']['pair_whitelist']
    # Ensure all except those in whitelist are removed
    assert_list_equal(whitelist, pairslist)

def test_refresh_whitelist_dynamic(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_wallet_health=get_health)
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['BTC_ETH', 'BTC_TKN']
    refresh_whitelist(whitelist)
    pairslist = conf['exchange']['pair_whitelist']
    assert_list_equal(whitelist, pairslist)

def test_refresh_whitelist_dynamic_empty(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_wallet_health=get_health_empty)
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    conf['exchange']['pair_whitelist'] = []
    refresh_whitelist(whitelist)
    pairslist = conf['exchange']['pair_whitelist']
    assert_list_equal(whitelist, pairslist)
