# pragma pylint: disable=missing-docstring,C0103,protected-access

from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.constants import AVAILABLE_PAIRLISTS
from freqtrade.exceptions import OperationalException
from freqtrade.pairlist.pairlistmanager import PairListManager
from freqtrade.resolvers import PairListResolver
from tests.conftest import get_patched_freqtradebot, log_has, log_has_re


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
            "number_assets": 5,
            "sort_key": "quoteVolume",
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


def test_log_on_refresh(mocker, static_pl_conf, markets, tickers):
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          markets=PropertyMock(return_value=markets),
                          exchange_has=MagicMock(return_value=True),
                          get_tickers=tickers
                          )
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    logmock = MagicMock()
    # Assign starting whitelist
    pl = freqtrade.pairlists._pairlist_handlers[0]
    pl.log_on_refresh(logmock, 'Hello world')
    assert logmock.call_count == 1
    pl.log_on_refresh(logmock, 'Hello world')
    assert logmock.call_count == 1
    assert pl._log_cache.currsize == 1
    assert ('Hello world',) in pl._log_cache._Cache__data

    pl.log_on_refresh(logmock, 'Hello world2')
    assert logmock.call_count == 2
    assert pl._log_cache.currsize == 2


def test_load_pairlist_noexist(mocker, markets, default_conf):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    plm = PairListManager(freqtrade.exchange, default_conf)
    with pytest.raises(OperationalException,
                       match=r"Impossible to load Pairlist 'NonexistingPairList'. "
                             r"This class does not exist or contains Python code errors."):
        PairListResolver.load_pairlist('NonexistingPairList', freqtrade.exchange, plm,
                                       default_conf, {}, 1)


def test_refresh_market_pair_not_in_whitelist(mocker, markets, static_pl_conf):

    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)

    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    freqtrade.pairlists.refresh_pairlist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    # Ensure config dict hasn't been changed
    assert (static_pl_conf['exchange']['pair_whitelist'] ==
            freqtrade.config['exchange']['pair_whitelist'])


def test_refresh_static_pairlist(mocker, markets, static_pl_conf):
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf['exchange']['pair_blacklist'] == freqtrade.pairlists.blacklist


def test_refresh_pairlist_dynamic(mocker, shitcoinmarkets, tickers, whitelist_conf):

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True),
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    # Remock markets with shitcoinmarkets since get_patched_freqtradebot uses the markets fixture
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=shitcoinmarkets),
     )
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC', 'HOT/BTC']
    freqtrade.pairlists.refresh_pairlist()

    assert whitelist == freqtrade.pairlists.whitelist

    whitelist_conf['pairlists'] = [{'method': 'VolumePairList',
                                    'config': {}
                                    }
                                   ]

    with pytest.raises(OperationalException,
                       match=r'`number_assets` not specified. Please check your configuration '
                             r'for "pairlist.config.number_assets"'):
        PairListManager(freqtrade.exchange, whitelist_conf)


def test_VolumePairList_refresh_empty(mocker, markets_empty, whitelist_conf):
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        exchange_has=MagicMock(return_value=True),
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets_empty))

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    whitelist_conf['exchange']['pair_whitelist'] = []
    freqtrade.pairlists.refresh_pairlist()
    pairslist = whitelist_conf['exchange']['pair_whitelist']

    assert set(whitelist) == set(pairslist)


@pytest.mark.parametrize("pairlists,base_currency,whitelist_result", [
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"}],
        "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC', 'HOT/BTC']),
    # Different sorting depending on quote or bid volume
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "bidVolume"}],
        "BTC",  ['HOT/BTC', 'FUEL/BTC', 'XRP/BTC', 'LTC/BTC', 'TKN/BTC']),
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"}],
        "USDT", ['ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT']),
    # No pair for ETH ...
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"}],
     "ETH", []),
    # Precisionfilter and quote volume
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
      {"method": "PrecisionFilter"}], "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC']),
    # Precisionfilter bid
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "bidVolume"},
      {"method": "PrecisionFilter"}], "BTC", ['FUEL/BTC', 'XRP/BTC', 'LTC/BTC', 'TKN/BTC']),
    # PriceFilter and VolumePairList
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
      {"method": "PriceFilter", "low_price_ratio": 0.03}],
        "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC']),
    # PriceFilter and VolumePairList
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
      {"method": "PriceFilter", "low_price_ratio": 0.03}],
        "USDT", ['ETH/USDT', 'NANO/USDT']),
    # Hot is removed by precision_filter, Fuel by low_price_filter.
    ([{"method": "VolumePairList", "number_assets": 6, "sort_key": "quoteVolume"},
      {"method": "PrecisionFilter"},
      {"method": "PriceFilter", "low_price_ratio": 0.02}
      ], "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC']),
    # HOT and XRP are removed because below 1250 quoteVolume
    ([{"method": "VolumePairList", "number_assets": 5,
       "sort_key": "quoteVolume", "min_value": 1250}],
        "BTC", ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']),
    # StaticPairlist Only
    ([{"method": "StaticPairList"},
      ], "BTC", ['ETH/BTC', 'TKN/BTC']),
    # Static Pairlist before VolumePairList - sorting changes
    ([{"method": "StaticPairList"},
      {"method": "VolumePairList", "number_assets": 5, "sort_key": "bidVolume"},
      ], "BTC", ['TKN/BTC', 'ETH/BTC']),
    # SpreadFilter
    ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
      {"method": "SpreadFilter", "max_spread": 0.005}
      ], "USDT", ['ETH/USDT']),
])
def test_VolumePairList_whitelist_gen(mocker, whitelist_conf, shitcoinmarkets, tickers,
                                      pairlists, base_currency, whitelist_result,
                                      caplog) -> None:
    whitelist_conf['pairlists'] = pairlists
    whitelist_conf['stake_currency'] = base_currency

    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)

    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          get_tickers=tickers,
                          markets=PropertyMock(return_value=shitcoinmarkets),
                          )

    freqtrade.pairlists.refresh_pairlist()
    whitelist = freqtrade.pairlists.whitelist

    assert whitelist == whitelist_result
    for pairlist in pairlists:
        if pairlist['method'] == 'PrecisionFilter':
            assert log_has_re(r'^Removed .* from whitelist, because stop price .* '
                              r'would be <= stop limit.*', caplog)
        if pairlist['method'] == 'PriceFilter':
            assert (log_has_re(r'^Removed .* from whitelist, because 1 unit is .*%$', caplog) or
                    log_has_re(r"^Removed .* from whitelist, because ticker\['last'\] is empty.*",
                               caplog))
        if pairlist['method'] == 'VolumePairList':
            logmsg = ("DEPRECATED: using any key other than quoteVolume for "
                      "VolumePairList is deprecated.")
            if pairlist['sort_key'] != 'quoteVolume':
                assert log_has(logmsg, caplog)
            else:
                assert not log_has(logmsg, caplog)


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
    pairlist_handler = freqtrade.pairlists._pairlist_handlers[0]
    new_whitelist = pairlist_handler._whitelist_for_active_markets(whitelist)

    assert set(new_whitelist) == set(['ETH/BTC', 'TKN/BTC'])
    assert log_message in caplog.text


def test_volumepairlist_invalid_sortvalue(mocker, markets, whitelist_conf):
    whitelist_conf['pairlists'][0].update({"sort_key": "asdf"})

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
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    assert freqtrade.pairlists._pairlist_handlers[0]._last_refresh == 0
    assert tickers.call_count == 0
    freqtrade.pairlists.refresh_pairlist()
    assert tickers.call_count == 1

    assert freqtrade.pairlists._pairlist_handlers[0]._last_refresh != 0
    lrf = freqtrade.pairlists._pairlist_handlers[0]._last_refresh
    freqtrade.pairlists.refresh_pairlist()
    assert tickers.call_count == 1
    # Time should not be updated.
    assert freqtrade.pairlists._pairlist_handlers[0]._last_refresh == lrf


def test_pairlistmanager_no_pairlist(mocker, markets, whitelist_conf, caplog):
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))

    whitelist_conf['pairlists'] = []

    with pytest.raises(OperationalException,
                       match=r"No Pairlist Handlers defined"):
        get_patched_freqtradebot(mocker, whitelist_conf)
