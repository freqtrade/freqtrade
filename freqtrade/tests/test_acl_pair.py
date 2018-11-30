# pragma pylint: disable=missing-docstring,C0103,protected-access

from unittest.mock import MagicMock

from freqtrade.tests.conftest import get_patched_freqtradebot

import pytest

# whitelist, blacklist,


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
    assert whitelist == freqtradebot.pairlists.whitelist
    # Ensure config dict hasn't been changed
    assert (whitelist_conf['exchange']['pair_whitelist'] ==
            freqtradebot.config['exchange']['pair_whitelist'])


def test_refresh_whitelist(mocker, markets, whitelist_conf):
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)

    mocker.patch('freqtrade.exchange.Exchange.get_markets', markets)
    freqtradebot.pairlists.refresh_whitelist()
    # List ordered by BaseVolume
    whitelist = ['ETH/BTC', 'TKN/BTC']
    # Ensure all except those in whitelist are removed
    assert whitelist == freqtradebot.pairlists.whitelist


def test_refresh_whitelist_dynamic(mocker, markets, tickers, whitelist_conf):
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_markets=markets,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True)
    )

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ['ETH/BTC', 'TKN/BTC']

    freqtradebot._refresh_whitelist(
        freqtradebot._gen_pair_whitelist(whitelist_conf['stake_currency'])
    )

    assert whitelist == freqtradebot.pairlists.whitelist


def test_refresh_whitelist_dynamic_empty(mocker, markets_empty, whitelist_conf):
    freqtradebot = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_markets', markets_empty)

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    whitelist_conf['exchange']['pair_whitelist'] = []
    freqtradebot._refresh_whitelist(whitelist)
    pairslist = whitelist_conf['exchange']['pair_whitelist']

    assert set(whitelist) == set(pairslist)
