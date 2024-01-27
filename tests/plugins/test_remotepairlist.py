import json
from unittest.mock import MagicMock, PropertyMock

import pytest
import requests

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.RemotePairList import RemotePairList
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, get_patched_exchange, get_patched_freqtradebot, log_has


@pytest.fixture(scope="function")
def rpl_config(default_conf):
    default_conf['stake_currency'] = 'USDT'

    default_conf['exchange']['pair_whitelist'] = [
        'ETH/USDT',
        'XRP/USDT',
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
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.rapidjson.load', return_value=jsonparse)

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

    with pytest.raises(OperationalException, match='RemotePairList is not of type JSON.'):
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
    remote_pairlist._init_done = True
    pairlist_url = rpl_config['pairlists'][0]['pairlist_url']
    pairs, _time_elapsed = remote_pairlist.fetch_pairlist()

    assert log_has(f'Error: Was not able to fetch pairlist from: ' f'{pairlist_url}', caplog)
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


def test_remote_pairlist_init_wrong_mode(mocker, rpl_config):
    rpl_config['pairlists'] = [
        {
            "method": "RemotePairList",
            "mode": "blacklis",
            "number_assets": 20,
            "pairlist_url": "http://example.com/pairlist",
            "keep_pairlist_on_failure": True,
        }
    ]

    with pytest.raises(
        OperationalException,
        match=r'`mode` not configured correctly. Supported Modes are "whitelist","blacklist"'
    ):
        get_patched_freqtradebot(mocker, rpl_config)

    rpl_config['pairlists'] = [
        {
            "method": "RemotePairList",
            "mode": "blacklist",
            "number_assets": 20,
            "pairlist_url": "http://example.com/pairlist",
            "keep_pairlist_on_failure": True,
        }
    ]

    with pytest.raises(
            OperationalException,
            match=r'A `blacklist` mode RemotePairList can not be.*first.*'
    ):
        get_patched_freqtradebot(mocker, rpl_config)


def test_remote_pairlist_init_wrong_proc_mode(mocker, rpl_config):
    rpl_config['pairlists'] = [
        {
            "method": "RemotePairList",
            "processing_mode": "filler",
            "mode": "whitelist",
            "number_assets": 20,
            "pairlist_url": "http://example.com/pairlist",
            "keep_pairlist_on_failure": True,
        }
    ]

    get_patched_exchange(mocker, rpl_config)
    with pytest.raises(
        OperationalException,
        match=r'`processing_mode` not configured correctly. Supported Modes are "filter","append"'
    ):
        get_patched_freqtradebot(mocker, rpl_config)


def test_remote_pairlist_blacklist(mocker, rpl_config, caplog, markets, tickers):

    mock_response = MagicMock()

    mock_response.json.return_value = {
        "pairs": ["XRP/USDT"],
        "refresh_period": 60
    }

    mock_response.headers = {
        "content-type": "application/json"
    }

    rpl_config['pairlists'] = [
        {
            "method": "StaticPairList",
        },
        {
            "method": "RemotePairList",
            "mode": "blacklist",
            "pairlist_url": "http://example.com/pairlist",
            "number_assets": 3
        }
    ]

    mocker.patch.multiple(EXMS,
                          markets=PropertyMock(return_value=markets),
                          exchange_has=MagicMock(return_value=True),
                          get_tickers=tickers
                          )

    mocker.patch("freqtrade.plugins.pairlist.RemotePairList.requests.get",
                 return_value=mock_response)

    exchange = get_patched_exchange(mocker, rpl_config)

    pairlistmanager = PairListManager(exchange, rpl_config)

    remote_pairlist = RemotePairList(exchange, pairlistmanager, rpl_config,
                                     rpl_config["pairlists"][1], 1)

    pairs, _time_elapsed = remote_pairlist.fetch_pairlist()

    assert pairs == ["XRP/USDT"]

    whitelist = remote_pairlist.filter_pairlist(rpl_config['exchange']['pair_whitelist'], {})
    assert whitelist == ["ETH/USDT"]

    assert log_has(f"Blacklist - Filtered out pairs: {pairs}", caplog)


@pytest.mark.parametrize("processing_mode", ["filter", "append"])
def test_remote_pairlist_whitelist(mocker, rpl_config, processing_mode, markets, tickers):

    mock_response = MagicMock()

    mock_response.json.return_value = {
        "pairs": ["XRP/USDT"],
        "refresh_period": 60
    }

    mock_response.headers = {
        "content-type": "application/json"
    }

    rpl_config['pairlists'] = [
        {
            "method": "StaticPairList",
        },
        {
            "method": "RemotePairList",
            "mode": "whitelist",
            "processing_mode": processing_mode,
            "pairlist_url": "http://example.com/pairlist",
            "number_assets": 3
        }
    ]

    mocker.patch.multiple(EXMS,
                          markets=PropertyMock(return_value=markets),
                          exchange_has=MagicMock(return_value=True),
                          get_tickers=tickers
                          )

    mocker.patch("freqtrade.plugins.pairlist.RemotePairList.requests.get",
                 return_value=mock_response)

    exchange = get_patched_exchange(mocker, rpl_config)

    pairlistmanager = PairListManager(exchange, rpl_config)

    remote_pairlist = RemotePairList(exchange, pairlistmanager, rpl_config,
                                     rpl_config["pairlists"][1], 1)

    pairs, _time_elapsed = remote_pairlist.fetch_pairlist()

    assert pairs == ["XRP/USDT"]

    whitelist = remote_pairlist.filter_pairlist(rpl_config['exchange']['pair_whitelist'], {})
    assert whitelist == (["XRP/USDT"] if processing_mode == "filter" else ['ETH/USDT', 'XRP/USDT'])
