# pragma pylint: disable=missing-docstring,C0103

from freqtrade.main import refresh_whitelist, gen_pair_whitelist


# whitelist, blacklist, filtering, all of that will
# eventually become some rules to run on a generic ACL engine
# perhaps try to anticipate that by using some python package


def whitelist_conf():
    return {
        'stake_currency': 'BTC',
        'exchange': {
            'pair_whitelist': [
                'ETH/BTC',
                'TKN/BTC',
                'TRST/BTC',
                'SWT/BTC',
                'BCC/BTC'
            ],
            'pair_blacklist': [
                'BLK/BTC'
            ],
        },
    }


def get_markets():
    return [
        {
            'id': 'ethbtc',
            'symbol': 'ETH/BTC',
            'base': 'ETH',
            'quote': 'BTC',
            'active': True,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,

            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': 500000,
            },
            'info': '',
        },
        {
            'id': 'tknbtc',
            'symbol': 'TKN/BTC',
            'base': 'TKN',
            'quote': 'BTC',
            'active': True,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,

            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': 500000,
            },
            'info': '',
        },
        {
            'id': 'blkbtc',
            'symbol': 'BLK/BTC',
            'base': 'BLK',
            'quote': 'BTC',
            'active': True,
            'precision': {
                'price': 8,
                'amount': 8,
                'cost': 8,
            },
            'lot': 0.00000001,

            'limits': {
                'amount': {
                    'min': 0.01,
                    'max': 1000,
                },
                'price': 500000,
                'cost': 500000,
            },
            'info': '',
        }
    ]


def get_markets_empty():
    return []


def test_refresh_market_pair_not_in_whitelist(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_markets=get_markets)
    refreshedwhitelist = refresh_whitelist(
        conf['exchange']['pair_whitelist'] + ['XXX/BTC'])
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert whitelist == refreshedwhitelist


def test_refresh_whitelist(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_markets=get_markets)
    refreshedwhitelist = refresh_whitelist(conf['exchange']['pair_whitelist'])
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert whitelist == refreshedwhitelist


def test_refresh_whitelist_dynamic(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_markets=get_markets)
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['TKN/BTC', 'ETH/BTC']
    refreshedwhitelist = refresh_whitelist(
        gen_pair_whitelist(conf['stake_currency']))
    assert whitelist == refreshedwhitelist


def test_refresh_whitelist_dynamic_empty(mocker):
    conf = whitelist_conf()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_markets=get_markets_empty)
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    conf['exchange']['pair_whitelist'] = []
    refresh_whitelist(whitelist)
    pairslist = conf['exchange']['pair_whitelist']
    assert set(whitelist) == set(pairslist)
