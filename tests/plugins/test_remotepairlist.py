import json
from unittest.mock import MagicMock

import pytest
import requests

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.RemotePairList import RemotePairList
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import get_patched_exchange, get_patched_freqtradebot, log_has


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


def test_gen_pairlist_with_local_file(mocker, rpl_config):

    mock_file = MagicMock()
    mock_file.read.return_value = '{"pairs": ["TKN/USDT","ETH/USDT","NANO/USDT"]}'
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.open', return_value=mock_file)

    mock_file_path = mocker.patch('freqtrade.plugins.pairlist.RemotePairList.Path')
    mock_file_path.exists.return_value = True

    jsonparse = json.loads(mock_file.read.return_value)
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.json.load', return_value=jsonparse)

    rpl_config['pairlists'] = [
        {
            "method": "RemotePairList",
            'number_assets': 2,
            'refresh_period': 1800,
            'keep_pairlist_on_failure': True,
            'pairlist_url': 'file:///pairlist.json',
            'bearer_token': '',
            'read_timeout': 60
        }
    ]

    exchange = get_patched_exchange(mocker, rpl_config)
    pairlistmanager = PairListManager(exchange, rpl_config)

    remote_pairlist = RemotePairList(exchange, pairlistmanager, rpl_config,
                                     rpl_config['pairlists'][0], 0)

    result = remote_pairlist.gen_pairlist([])

    assert result == ['TKN/USDT', 'ETH/USDT']


def test_fetch_pairlist_mock_response_html(mocker, rpl_config):
    mock_response = MagicMock()
    mock_response.headers = {'content-type': 'text/html'}

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

    with pytest.raises(OperationalException, match='RemotePairList is not of type JSON, abort.'):
        remote_pairlist.fetch_pairlist()


def test_fetch_pairlist_timeout_keep_last_pairlist(mocker, rpl_config, caplog):
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
                 side_effect=requests.exceptions.RequestException)

    remote_pairlist = RemotePairList(exchange, pairlistmanager, rpl_config,
                                     rpl_config['pairlists'][0], 0)

    remote_pairlist._last_pairlist = ["BTC/USDT", "ETH/USDT", "LTC/USDT"]

    pairs, time_elapsed = remote_pairlist.fetch_pairlist()
    assert log_has(f"Was not able to fetch pairlist from: {remote_pairlist._pairlist_url}", caplog)
    assert log_has("Keeping last fetched pairlist", caplog)
    assert pairs == ["BTC/USDT", "ETH/USDT", "LTC/USDT"]


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
        "pairs": ["ETH/USDT", "XRP/USDT", "LTC/USDT", "EOS/USDT"],
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
    pairs, time_elapsed = remote_pairlist.fetch_pairlist()

    assert pairs == ["ETH/USDT", "XRP/USDT", "LTC/USDT", "EOS/USDT"]
    assert time_elapsed == 0.4
    assert remote_pairlist._refresh_period == 60
