from freqtrade.main import refresh_whitelist, gen_pair_whitelist

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
            "pair_blacklist": [
                "BTC_BLK"
            ],
        },
    }


def get_market_summaries():
    return [{
        "MarketName": "BTC-TKN",
        "High": 0.00000919,
        "Low": 0.00000820,
        "Volume": 74339.61396015,
        "Last": 0.00000820,
        "BaseVolume": 1664,
        "TimeStamp": "2014-07-09T07:19:30.15",
        "Bid": 0.00000820,
        "Ask": 0.00000831,
        "OpenBuyOrders": 15,
        "OpenSellOrders": 15,
        "PrevDay": 0.00000821,
        "Created": "2014-03-20T06:00:00",
        "DisplayMarketName": ""
    }, {
        "MarketName": "BTC-ETH",
        "High": 0.00000072,
        "Low": 0.00000001,
        "Volume": 166340678.42280999,
        "Last": 0.00000005,
        "BaseVolume": 42,
        "TimeStamp": "2014-07-09T07:21:40.51",
        "Bid": 0.00000004,
        "Ask": 0.00000005,
        "OpenBuyOrders": 18,
        "OpenSellOrders": 18,
        "PrevDay": 0.00000002,
        "Created": "2014-05-30T07:57:49.637",
        "DisplayMarketName": ""
    }, {
        "MarketName": "BTC-BLK",
        "High": 0.00000072,
        "Low": 0.00000001,
        "Volume": 166340678.42280999,
        "Last": 0.00000005,
        "BaseVolume": 3,
        "TimeStamp": "2014-07-09T07:21:40.51",
        "Bid": 0.00000004,
        "Ask": 0.00000005,
        "OpenBuyOrders": 18,
        "OpenSellOrders": 18,
        "PrevDay": 0.00000002,
        "Created": "2014-05-30T07:57:49.637",
        "DisplayMarketName": ""
    }
    ]


def get_health():
    return [{'Currency': 'ETH',
             'IsActive': True
             },
            {'Currency': 'TKN',
             'IsActive': True
             },
            {'Currency': 'BLK',
             'IsActive': True
             }
            ]


def get_health_empty():
    return []


def test_refresh_market_pair_not_in_whitelist(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_wallet_health=get_health)
    refreshedwhitelist = refresh_whitelist(conf['exchange']['pair_whitelist'] + ['BTC_XXX'])
    # List ordered by BaseVolume
    whitelist = ['BTC_ETH', 'BTC_TKN']
    # Ensure all except those in whitelist are removed
    assert whitelist == refreshedwhitelist


def test_refresh_whitelist(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_wallet_health=get_health)
    refreshedwhitelist = refresh_whitelist(conf['exchange']['pair_whitelist'])
    # List ordered by BaseVolume
    whitelist = ['BTC_ETH', 'BTC_TKN']
    # Ensure all except those in whitelist are removed
    assert whitelist == refreshedwhitelist


def test_refresh_whitelist_dynamic(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_wallet_health=get_health)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_market_summaries=get_market_summaries)
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['BTC_TKN', 'BTC_ETH']
    refreshedwhitelist = refresh_whitelist(gen_pair_whitelist(conf['stake_currency']))
    assert whitelist == refreshedwhitelist


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
