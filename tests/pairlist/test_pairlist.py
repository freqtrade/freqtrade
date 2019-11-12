# pragma pylint: disable=missing-docstring,C0103,protected-access

from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade import OperationalException
from freqtrade.constants import AVAILABLE_PAIRLISTS
from freqtrade.resolvers import PairListResolver
from freqtrade.pairlist.pairlistmanager import PairListManager
from tests.conftest import get_patched_freqtradebot, log_has_re

# whitelist, blacklist


@pytest.fixture(scope="function")
def whitelist_conf(default_conf):
    default_conf['stake_currency'] = 'BTC'
    default_conf['exchange']['pair_whitelist'] = [
        'ETH/BTC',
        'TKN/BTC',
        'TRST/BTC',
        'SWT/BTC',
        'BCC/BTC'
    ]
    default_conf['exchange']['pair_blacklist'] = [
        'BLK/BTC'
    ]
    default_conf['pairlists'] = [
        {
            "method": "VolumePairList",
            "config": {
                "number_assets": 5,
                "sort_key": "quoteVolume",
                }
        },
    ]
    return default_conf


@pytest.fixture(scope="function")
def static_pl_conf(whitelist_conf):
    whitelist_conf['pairlists'] = [
        {
            "method": "StaticPairList",
        },
    ]
    return whitelist_conf


def test_load_pairlist_noexist(mocker, markets, default_conf):
    bot = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    plm = PairListManager(bot.exchange, default_conf)
    with pytest.raises(OperationalException,
                       match=r"Impossible to load Pairlist 'NonexistingPairList'. "
                             r"This class does not exist or contains Python code errors."):
        PairListResolver('NonexistingPairList', bot.exchange, plm, default_conf, {}, 1)


def test_refresh_market_pair_not_in_whitelist(mocker, markets, static_pl_conf):

    freqtradebot = get_patched_freqtradebot(mocker, static_pl_conf)

    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    freqtradebot.pairlists.refresh_pairlist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtradebot.pairlists.whitelist)
    # Ensure config dict hasn't been changed
    assert (static_pl_conf['exchange']['pair_whitelist'] ==
            freqtradebot.config['exchange']['pair_whitelist'])


def test_refresh_static_pairlist(mocker, markets, static_pl_conf):
    freqtradebot = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtradebot.pairlists.refresh_pairlist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtradebot.pairlists.whitelist)
    assert static_pl_conf['exchange']['pair_blacklist'] == freqtradebot.pairlists.blacklist


def test_refresh_pairlist_dynamic(mocker, shitcoinmarkets, tickers, whitelist_conf):

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True),
    )
    bot = get_patched_freqtradebot(mocker, whitelist_conf)
    # Remock markets with shitcoinmarkets since get_patched_freqtradebot uses the markets fixture
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=shitcoinmarkets),
     )
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'HOT/BTC', 'FUEL/BTC']
    bot.pairlists.refresh_pairlist()

    assert whitelist == bot.pairlists.whitelist

    whitelist_conf['pairlists'] = [{'method': 'VolumePairList',
                                    'config': {}
                                    }
                                   ]

    with pytest.raises(OperationalException,
                       match=r'`number_assets` not specified. Please check your configuration '
                             r'for "pairlist.config.number_assets"'):
        PairListManager(bot.exchange, whitelist_conf)


def test_VolumePairList_refresh_empty(mocker, markets_empty, whitelist_conf):
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        exchange_has=MagicMock(return_value=True),
    )
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets_empty))

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    whitelist_conf['exchange']['pair_whitelist'] = []
    freqtradebot.pairlists.refresh_pairlist()
    pairslist = whitelist_conf['exchange']['pair_whitelist']

    assert set(whitelist) == set(pairslist)


@pytest.mark.parametrize("pairlists,base_currency,whitelist_result", [
    ([{"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "quoteVolume"}}],
        "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'HOT/BTC', 'FUEL/BTC']),
    # Different sorting depending on quote or bid volume
    ([{"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "bidVolume"}}],
        "BTC",  ['HOT/BTC', 'FUEL/BTC', 'LTC/BTC', 'TKN/BTC', 'ETH/BTC']),
    ([{"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "quoteVolume"}}],
        "USDT", ['ETH/USDT']),
    # No pair for ETH ...
    ([{"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "quoteVolume"}}],
     "ETH", []),
    # Precisionfilter and quote volume
    ([{"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "quoteVolume"}},
      {"method": "PrecisionFilter"}], "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'FUEL/BTC']),
    # Precisionfilter bid
    ([{"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "bidVolume"}},
      {"method": "PrecisionFilter"}], "BTC", ['FUEL/BTC', 'LTC/BTC', 'TKN/BTC', 'ETH/BTC']),
    # Lowpricefilter and VolumePairList
    ([{"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "quoteVolume"}},
      {"method": "LowPriceFilter", "config": {"low_price_percent": 0.03}}],
        "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'FUEL/BTC']),
    # Hot is removed by precision_filter, Fuel by low_price_filter.
    ([{"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "quoteVolume"}},
      {"method": "PrecisionFilter"},
      {"method": "LowPriceFilter", "config": {"low_price_percent": 0.02}}
      ], "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']),
    # StaticPairlist Only
    ([{"method": "StaticPairList"},
      ], "BTC", ['ETH/BTC', 'TKN/BTC']),
    # Static Pairlist before VolumePairList - sorting changes
    ([{"method": "StaticPairList"},
      {"method": "VolumePairList", "config": {"number_assets": 5, "sort_key": "bidVolume"}},
      ], "BTC", ['TKN/BTC', 'ETH/BTC']),
])
def test_VolumePairList_whitelist_gen(mocker, whitelist_conf, shitcoinmarkets, tickers,
                                      pairlists, base_currency, whitelist_result,
                                      caplog) -> None:
    whitelist_conf['pairlists'] = pairlists

    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)

    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          get_tickers=tickers,
                          markets=PropertyMock(return_value=shitcoinmarkets),
                          )

    freqtrade.config['stake_currency'] = base_currency
    freqtrade.pairlists.refresh_pairlist()
    whitelist = freqtrade.pairlists.whitelist

    assert whitelist == whitelist_result
    for pairlist in pairlists:
        if pairlist['method'] == 'PrecisionFilter':
            assert log_has_re(r'^Removed .* from whitelist, because stop price .* '
                              r'would be <= stop limit.*', caplog)
        if pairlist['method'] == 'LowPriceFilter':
            assert log_has_re(r'^Removed .* from whitelist, because 1 unit is .*%$', caplog)


def test_gen_pair_whitelist_not_supported(mocker, default_conf, tickers) -> None:
    default_conf['pairlists'] = [{'method': 'VolumePairList',
                                  'config': {'number_assets': 10}
                                  }]

    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          get_tickers=tickers,
                          exchange_has=MagicMock(return_value=False),
                          )

    with pytest.raises(OperationalException):
        get_patched_freqtradebot(mocker, default_conf)


@pytest.mark.parametrize("pairlist", AVAILABLE_PAIRLISTS)
def test_pairlist_class(mocker, whitelist_conf, markets, pairlist):
    whitelist_conf['pairlists'][0]['method'] = pairlist
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          markets=PropertyMock(return_value=markets),
                          exchange_has=MagicMock(return_value=True)
                          )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)

    assert freqtrade.pairlists.name_list == [pairlist]
    assert pairlist in str(freqtrade.pairlists.short_desc())
    assert isinstance(freqtrade.pairlists.whitelist, list)
    assert isinstance(freqtrade.pairlists.blacklist, list)


@pytest.mark.parametrize("pairlist", AVAILABLE_PAIRLISTS)
@pytest.mark.parametrize("whitelist,log_message", [
    (['ETH/BTC', 'TKN/BTC'], ""),
    # TRX/ETH not in markets
    (['ETH/BTC', 'TKN/BTC', 'TRX/ETH'], "is not compatible with exchange"),
    # wrong stake
    (['ETH/BTC', 'TKN/BTC', 'ETH/USDT'], "is not compatible with your stake currency"),
    # BCH/BTC not available
    (['ETH/BTC', 'TKN/BTC', 'BCH/BTC'], "is not compatible with exchange"),
    # BLK/BTC in blacklist
    (['ETH/BTC', 'TKN/BTC', 'BLK/BTC'], "in your blacklist. Removing "),
    # BTT/BTC is inactive
    (['ETH/BTC', 'TKN/BTC', 'BTT/BTC'], "Market is not active")
])
def test__whitelist_for_active_markets(mocker, whitelist_conf, markets, pairlist, whitelist, caplog,
                                       log_message, tickers):
    whitelist_conf['pairlists'][0]['method'] = pairlist
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          markets=PropertyMock(return_value=markets),
                          exchange_has=MagicMock(return_value=True),
                          get_tickers=tickers
                          )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    caplog.clear()

    # Assign starting whitelist
    new_whitelist = freqtrade.pairlists._pairlists[0]._whitelist_for_active_markets(whitelist)

    assert set(new_whitelist) == set(['ETH/BTC', 'TKN/BTC'])
    assert log_message in caplog.text


def test_volumepairlist_invalid_sortvalue(mocker, markets, whitelist_conf):
    whitelist_conf['pairlists'][0]['config'].update({"sort_key": "asdf"})

    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    with pytest.raises(OperationalException,
                       match=r"key asdf not in .*"):
        get_patched_freqtradebot(mocker, whitelist_conf)


def test_volumepairlist_caching(mocker, markets, whitelist_conf, tickers):

    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          markets=PropertyMock(return_value=markets),
                          exchange_has=MagicMock(return_value=True),
                          get_tickers=tickers
                          )
    bot = get_patched_freqtradebot(mocker, whitelist_conf)
    assert bot.pairlists._pairlists[0]._last_refresh == 0
    assert tickers.call_count == 0
    bot.pairlists.refresh_pairlist()
    assert tickers.call_count == 1

    assert bot.pairlists._pairlists[0]._last_refresh != 0
    lrf = bot.pairlists._pairlists[0]._last_refresh
    bot.pairlists.refresh_pairlist()
    assert tickers.call_count == 1
    # Time should not be updated.
    assert bot.pairlists._pairlists[0]._last_refresh == lrf


def test_pairlistmanager_no_pairlist(mocker, markets, whitelist_conf, caplog):
    del whitelist_conf['pairlists'][0]['method']
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    with pytest.raises(OperationalException,
                       match=r"No Pairlist defined!"):
        get_patched_freqtradebot(mocker, whitelist_conf)
    assert log_has_re("No method in .*", caplog)

    whitelist_conf['pairlists'] = []

    with pytest.raises(OperationalException,
                       match=r"No Pairlist defined!"):
        get_patched_freqtradebot(mocker, whitelist_conf)
