# pragma pylint: disable=missing-docstring,C0103,protected-access

from unittest.mock import MagicMock, PropertyMock

from freqtrade import OperationalException
from freqtrade.constants import AVAILABLE_PAIRLISTS
from freqtrade.resolvers import PairListResolver
from freqtrade.tests.conftest import get_patched_freqtradebot, log_has
import pytest

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
    default_conf['pairlist'] = {'method': 'StaticPairList',
                                'config': {'number_assets': 3}
                                }

    return default_conf


def test_load_pairlist_noexist(mocker, markets, default_conf):
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    with pytest.raises(ImportError,
                       match=r"Impossible to load Pairlist 'NonexistingPairList'."
                             r" This class does not exist or contains Python code errors"):
        PairListResolver('NonexistingPairList', freqtradebot, default_conf).pairlist


def test_refresh_market_pair_not_in_whitelist(mocker, markets, whitelist_conf):

    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)

    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    freqtradebot.pairlists.refresh_pairlist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtradebot.pairlists.whitelist)
    # Ensure config dict hasn't been changed
    assert (whitelist_conf['exchange']['pair_whitelist'] ==
            freqtradebot.config['exchange']['pair_whitelist'])


def test_refresh_pairlists(mocker, markets, whitelist_conf):
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)

    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    freqtradebot.pairlists.refresh_pairlist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtradebot.pairlists.whitelist)
    assert whitelist_conf['exchange']['pair_blacklist'] == freqtradebot.pairlists.blacklist


def test_refresh_pairlist_dynamic(mocker, markets, tickers, whitelist_conf):
    whitelist_conf['pairlist'] = {'method': 'VolumePairList',
                                  'config': {'number_assets': 5}
                                  }
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets),
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True)
    )
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['ETH/BTC', 'TKN/BTC', 'BTT/BTC']
    freqtradebot.pairlists.refresh_pairlist()

    assert whitelist == freqtradebot.pairlists.whitelist

    whitelist_conf['pairlist'] = {'method': 'VolumePairList',
                                  'config': {}
                                  }
    with pytest.raises(OperationalException,
                       match=r'`number_assets` not specified. Please check your configuration '
                             r'for "pairlist.config.number_assets"'):
        PairListResolver('VolumePairList', freqtradebot, whitelist_conf).pairlist


def test_VolumePairList_refresh_empty(mocker, markets_empty, whitelist_conf):
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets_empty))

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    whitelist_conf['exchange']['pair_whitelist'] = []
    freqtradebot.pairlists.refresh_pairlist()
    pairslist = whitelist_conf['exchange']['pair_whitelist']

    assert set(whitelist) == set(pairslist)


@pytest.mark.parametrize("precision_filter,base_currency,key,whitelist_result", [
    (False, "BTC", "quoteVolume", ['ETH/BTC', 'TKN/BTC', 'BTT/BTC']),
    (False, "BTC", "bidVolume", ['BTT/BTC', 'TKN/BTC', 'ETH/BTC']),
    (False, "USDT", "quoteVolume", ['ETH/USDT', 'LTC/USDT']),
    (False, "ETH", "quoteVolume", []),  # this replaces tests that were removed from test_exchange
    (True, "BTC", "quoteVolume", ["ETH/BTC", "TKN/BTC"]),
    (True, "BTC", "bidVolume", ["TKN/BTC", "ETH/BTC"])
])
def test_VolumePairList_whitelist_gen(mocker, whitelist_conf, markets, tickers, base_currency, key,
                                      whitelist_result, precision_filter) -> None:
    whitelist_conf['pairlist']['method'] = 'VolumePairList'
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)
    mocker.patch('freqtrade.exchange.Exchange.symbol_price_prec', lambda s, p, r: round(r, 8))

    freqtrade.pairlists._precision_filter = precision_filter
    freqtrade.config['stake_currency'] = base_currency
    whitelist = freqtrade.pairlists._gen_pair_whitelist(base_currency=base_currency, key=key)
    assert whitelist == whitelist_result


def test_gen_pair_whitelist_not_supported(mocker, default_conf, tickers) -> None:
    default_conf['pairlist'] = {'method': 'VolumePairList',
                                'config': {'number_assets': 10}
                                }
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=False))

    with pytest.raises(OperationalException):
        get_patched_freqtradebot(mocker, default_conf)


@pytest.mark.parametrize("pairlist", AVAILABLE_PAIRLISTS)
def test_pairlist_class(mocker, whitelist_conf, markets, pairlist):
    whitelist_conf['pairlist']['method'] = pairlist
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)

    assert freqtrade.pairlists.name == pairlist
    assert pairlist in freqtrade.pairlists.short_desc()
    assert isinstance(freqtrade.pairlists.whitelist, list)
    assert isinstance(freqtrade.pairlists.blacklist, list)


@pytest.mark.parametrize("pairlist", AVAILABLE_PAIRLISTS)
@pytest.mark.parametrize("whitelist,log_message", [
    (['ETH/BTC', 'TKN/BTC'], ""),
    (['ETH/BTC', 'TKN/BTC', 'TRX/ETH'], "is not compatible with exchange"),  # TRX/ETH wrong stake
    (['ETH/BTC', 'TKN/BTC', 'BCH/BTC'], "is not compatible with exchange"),  # BCH/BTC not available
    (['ETH/BTC', 'TKN/BTC', 'BLK/BTC'], "is not compatible with exchange"),  # BLK/BTC in blacklist
    (['ETH/BTC', 'TKN/BTC', 'LTC/BTC'], "Market is not active")  # LTC/BTC is inactive
])
def test_validate_whitelist(mocker, whitelist_conf, markets, pairlist, whitelist, caplog,
                            log_message):
    whitelist_conf['pairlist']['method'] = pairlist
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    caplog.clear()

    new_whitelist = freqtrade.pairlists._validate_whitelist(whitelist)

    assert set(new_whitelist) == set(['ETH/BTC', 'TKN/BTC'])
    assert log_message in caplog.text
