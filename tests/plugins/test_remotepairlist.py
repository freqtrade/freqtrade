from unittest.mock import MagicMock

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.RemotePairList import RemotePairList
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import get_patched_exchange, get_patched_freqtradebot


@pytest.fixture(scope="function")
def rpl_config(default_conf):
    default_conf['stake_currency'] = 'USDT'

    default_conf['exchange']['pair_whitelist'] = [
        'ETH/USDT',
        'BTC/USDT',
    ]
    default_conf['exchange']['pair_blacklist'] = [
        'BLK/USDT'
    ]
    return default_conf


def test_fetch_pairlist_mock_response_html(mocker, rpl_config):
    mock_response = MagicMock()
    mock_response.headers = {'content-type': 'text/html'}
    mocker.patch('requests.get', return_value=mock_response)

    rpl_config['pairlists'] = [
        {
            "method": "RemotePairList",
            "pairlist_url": "http://example.com/pairlist",
            "number_assets": 10,
            "read_timeout": 10,
            "keep_pairlist_on_failure": True,
        }
    ]

    exchange = get_patched_exchange(mocker, rpl_config)
    pairlistmanager = PairListManager(exchange, rpl_config)

    mocker.patch("freqtrade.plugins.pairlist.RemotePairList.requests.get",
                 return_value=mock_response)
    remote_pairlist = RemotePairList(exchange, pairlistmanager, rpl_config,
                                     rpl_config['pairlists'][0], 0)

    with pytest.raises(OperationalException, match='RemotePairList is not of type JSON abort'):
        remote_pairlist.fetch_pairlist()


def test_remote_pairlist_init_no_pairlist_url(mocker, rpl_config):

    rpl_config['pairlists'] = [
        {
            "method": "RemotePairList",
            "number_assets": 10,
            "keep_pairlist_on_failure": True,
        }
    ]

    get_patched_exchange(mocker, rpl_config)
    with pytest.raises(OperationalException, match=r'`pairlist_url` not specified.'
                       r' Please check your configuration for "pairlist.config.pairlist_url"'):
        get_patched_freqtradebot(mocker, rpl_config)


def test_remote_pairlist_init_no_number_assets(mocker, rpl_config):

    rpl_config['pairlists'] = [
        {
            "method": "RemotePairList",
            "pairlist_url": "http://example.com/pairlist",
            "keep_pairlist_on_failure": True,
        }
    ]

    get_patched_exchange(mocker, rpl_config)

    with pytest.raises(OperationalException, match=r'`number_assets` not specified. '
                       'Please check your configuration for "pairlist.config.number_assets"'):
        get_patched_freqtradebot(mocker, rpl_config)


def test_fetch_pairlist_mock_response_valid(mocker, rpl_config):

    rpl_config['pairlists'] = [
        {
            "method": "RemotePairList",
            "pairlist_url": "http://example.com/pairlist",
            "number_assets": 10,
            "refresh_period": 10,
            "read_timeout": 10,
            "keep_pairlist_on_failure": True,
        }
    ]

    mock_response = MagicMock()

    mock_response.json.return_value = {
        "pairs": ["ETH/BTC", "XRP/BTC", "LTC/BTC", "EOS/BTC"],
        "info": "Mock pairlist response",
        "refresh_period": 60
    }

    mock_response.headers = {
        "content-type": "application/json"
    }

    mock_response.elapsed.total_seconds.return_value = 0.4
    mocker.patch("freqtrade.plugins.pairlist.RemotePairList.requests.get",
                 return_value=mock_response)

    exchange = get_patched_exchange(mocker, rpl_config)
    pairlistmanager = PairListManager(exchange, rpl_config)
    remote_pairlist = RemotePairList(exchange, pairlistmanager, rpl_config,
                                     rpl_config['pairlists'][0], 0)
    pairs, time_elapsed, info = remote_pairlist.fetch_pairlist()

    assert pairs == ["ETH/BTC", "XRP/BTC", "LTC/BTC", "EOS/BTC"]
    assert time_elapsed == 0.4
    assert info == "Mock pairlist response"
    assert remote_pairlist._refresh_period == 60
