# pragma pylint: disable=missing-docstring,C0103,protected-access

from unittest.mock import MagicMock

from freqtrade import OperationalException
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
    whitelist_conf['dynamic_whitelist'] = 5
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_markets=markets,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True)
    )

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    freqtradebot.pairlists.refresh_whitelist()

    assert whitelist == freqtradebot.pairlists.whitelist


def test_refresh_whitelist_dynamic_empty(mocker, markets_empty, whitelist_conf):
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_markets', markets_empty)

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    whitelist_conf['exchange']['pair_whitelist'] = []
    freqtradebot.pairlists.refresh_whitelist()
    pairslist = whitelist_conf['exchange']['pair_whitelist']

    assert set(whitelist) == set(pairslist)


def test_gen_pair_whitelist(mocker, default_conf, tickers) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))

    # Test to retrieved BTC sorted on quoteVolume (default)
    freqtrade.pairlists.refresh_whitelist()

    whitelist = freqtrade.pairlists.whitelist
    assert whitelist == ['ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC']

    # Test to retrieve BTC sorted on bidVolume
    whitelist = freqtrade._gen_pair_whitelist(base_currency='BTC', key='bidVolume')
    assert whitelist == ['LTC/BTC', 'TKN/BTC', 'ETH/BTC', 'BLK/BTC']

    # Test with USDT sorted on quoteVolume (default)
    whitelist = freqtrade._gen_pair_whitelist(base_currency='USDT')
    assert whitelist == ['TKN/USDT', 'ETH/USDT', 'LTC/USDT', 'BLK/USDT']

    # Test with ETH (our fixture does not have ETH, so result should be empty)
    whitelist = freqtrade._gen_pair_whitelist(base_currency='ETH')
    assert whitelist == []


def test_gen_pair_whitelist_not_supported(mocker, default_conf, tickers) -> None:
    default_conf['whitelist'] = {'method': 'VolumePairList',
                                 'config': {'number_assets': 10}
                                 }
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=False))

    with pytest.raises(OperationalException):
        get_patched_freqtradebot(mocker, default_conf)
