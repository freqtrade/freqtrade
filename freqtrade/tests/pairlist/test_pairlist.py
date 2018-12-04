# pragma pylint: disable=missing-docstring,C0103,protected-access

from unittest.mock import MagicMock

from freqtrade import OperationalException
from freqtrade.constants import AVAILABLE_PAIRLISTS
from freqtrade.tests.conftest import get_patched_freqtradebot
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


def test_refresh_market_pair_not_in_whitelist(mocker, markets, whitelist_conf):

    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)

    mocker.patch('freqtrade.exchange.Exchange.get_markets', markets)
    freqtradebot.pairlists.refresh_whitelist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtradebot.pairlists.whitelist)
    # Ensure config dict hasn't been changed
    assert (whitelist_conf['exchange']['pair_whitelist'] ==
            freqtradebot.config['exchange']['pair_whitelist'])


def test_refresh_pairlists(mocker, markets, whitelist_conf):
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)

    mocker.patch('freqtrade.exchange.Exchange.get_markets', markets)
    freqtradebot.pairlists.refresh_whitelist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtradebot.pairlists.whitelist)
    assert whitelist_conf['exchange']['pair_blacklist'] == freqtradebot.pairlists.blacklist


def test_refresh_whitelist_dynamic(mocker, markets, tickers, whitelist_conf):
    whitelist_conf['pairlist'] = {'method': 'VolumePairList',
                                  'config': {'number_assets': 5}
                                  }
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_markets=markets,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True)
    )
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    freqtradebot.pairlists.refresh_whitelist()

    assert whitelist == freqtradebot.pairlists.whitelist


def test_VolumePairList_refresh_empty(mocker, markets_empty, whitelist_conf):
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_markets', markets_empty)

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    whitelist_conf['exchange']['pair_whitelist'] = []
    freqtradebot.pairlists.refresh_whitelist()
    pairslist = whitelist_conf['exchange']['pair_whitelist']

    assert set(whitelist) == set(pairslist)


def test_VolumePairList_whitelist_gen(mocker, whitelist_conf, markets, tickers) -> None:
    whitelist_conf['pairlist']['method'] = 'VolumePairList'
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_markets', markets)
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)

    # Test to retrieved BTC sorted on quoteVolume (default)
    whitelist = freqtrade.pairlists._gen_pair_whitelist(base_currency='BTC', key='quoteVolume')
    assert whitelist == ['ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC']

    # Test to retrieve BTC sorted on bidVolume
    whitelist = freqtrade.pairlists._gen_pair_whitelist(base_currency='BTC', key='bidVolume')
    assert whitelist == ['LTC/BTC', 'TKN/BTC', 'ETH/BTC', 'BLK/BTC']

    # Test with USDT sorted on quoteVolume (default)
    whitelist = freqtrade.pairlists._gen_pair_whitelist(base_currency='USDT', key='quoteVolume')
    assert whitelist == ['TKN/USDT', 'ETH/USDT', 'LTC/USDT', 'BLK/USDT']

    # Test with ETH (our fixture does not have ETH, so result should be empty)
    whitelist = freqtrade.pairlists._gen_pair_whitelist(base_currency='ETH', key='quoteVolume')
    assert whitelist == []


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
    mocker.patch('freqtrade.exchange.Exchange.get_markets', markets)
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)

    assert freqtrade.pairlists.name == pairlist
    assert pairlist in freqtrade.pairlists.short_desc()
    assert isinstance(freqtrade.pairlists.whitelist, list)
    assert isinstance(freqtrade.pairlists.blacklist, list)

    whitelist = ['ETH/BTC', 'TKN/BTC']
    new_whitelist = freqtrade.pairlists._validate_whitelist(whitelist)

    assert set(whitelist) == set(new_whitelist)

    whitelist = ['ETH/BTC', 'TKN/BTC', 'TRX/ETH']
    new_whitelist = freqtrade.pairlists._validate_whitelist(whitelist)
    # TRX/ETH was removed
    assert set(['ETH/BTC', 'TKN/BTC']) == set(new_whitelist)

    whitelist = ['ETH/BTC', 'TKN/BTC', 'BLK/BTC']
    new_whitelist = freqtrade.pairlists._validate_whitelist(whitelist)
    # BLK/BTC is in blacklist ...
    assert set(['ETH/BTC', 'TKN/BTC']) == set(new_whitelist)
