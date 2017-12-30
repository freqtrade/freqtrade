from freqtrade.main import refresh_whitelist

# whitelist, blacklist, filtering, all of that will
# eventually become some rules to run on a generic ACL engine
# perhaps try to anticipate that by using some python package


def whitelist_conf():
    return {
        "stake_currency": "BTC",
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
    assert set(whitelist) == set(pairslist)


def test_refresh_whitelist_dynamic(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_wallet_health=get_health)
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['BTC_ETH', 'BTC_TKN']
    refresh_whitelist(whitelist)
    pairslist = conf['exchange']['pair_whitelist']
    assert set(whitelist) == set(pairslist)


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
    assert set(whitelist) == set(pairslist)
