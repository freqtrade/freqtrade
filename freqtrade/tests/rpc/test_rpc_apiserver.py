"""
Unit test file for rpc/api_server.py
"""

from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.__init__ import __version__
from freqtrade.rpc.api_server import ApiServer
from freqtrade.state import State
from freqtrade.tests.conftest import get_patched_freqtradebot, patch_apiserver, patch_get_signal


@pytest.fixture
def botclient(default_conf, mocker):
    default_conf.update({"api_server":{"enabled": True,
                                       "listen_ip_address": "127.0.0.1",
                                       "listen_port": "8080"}})
    ftbot = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.rpc.api_server.ApiServer.run', MagicMock())
    apiserver = ApiServer(ftbot)
    yield ftbot, apiserver.app.test_client()
    # Cleanup ... ?


def response_success_assert(response):
    assert response.status_code == 200
    assert response.content_type == "application/json"


def test_api_stop_workflow(botclient):
    ftbot, client = botclient
    assert ftbot.state == State.RUNNING
    rc = client.post("/stop")
    response_success_assert(rc)
    assert rc.json == {'status': 'stopping trader ...'}
    assert ftbot.state == State.STOPPED

    # Stop bot again
    rc = client.post("/stop")
    response_success_assert(rc)
    assert rc.json == {'status': 'already stopped'}

    # Start bot
    rc = client.post("/start")
    response_success_assert(rc)
    assert rc.json == {'status': 'starting trader ...'}
    assert ftbot.state == State.RUNNING

    # Call start again
    rc = client.post("/start")
    response_success_assert(rc)
    assert rc.json == {'status': 'already running'}


def test_api__init__(default_conf, mocker):
    """
    Test __init__() method
    """
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.api_server.ApiServer.run', MagicMock())

    apiserver = ApiServer(get_patched_freqtradebot(mocker, default_conf))
    assert apiserver._config == default_conf


def test_api_reloadconf(botclient):
    ftbot, client = botclient

    rc = client.post("/reload_conf")
    response_success_assert(rc)
    assert rc.json == {'status': 'reloading config ...'}
    assert ftbot.state == State.RELOAD_CONF


def test_api_stopbuy(botclient):
    ftbot, client = botclient
    assert ftbot.config['max_open_trades'] != 0

    rc = client.post("/stopbuy")
    response_success_assert(rc)
    assert rc.json == {'status': 'No more buy will occur from now. Run /reload_conf to reset.'}
    assert ftbot.config['max_open_trades'] == 0


def test_api_version(botclient):
    ftbot, client = botclient

    rc = client.get("/version")
    response_success_assert(rc)
    assert rc.json == {"version": __version__}


def test_api_balance(botclient, mocker, rpc_balance):
    ftbot, client = botclient

    def mock_ticker(symbol, refresh):
        if symbol == 'BTC/USDT':
            return {
                'bid': 10000.00,
                'ask': 10000.00,
                'last': 10000.00,
            }
        elif symbol == 'XRP/BTC':
            return {
                'bid': 0.00001,
                'ask': 0.00001,
                'last': 0.00001,
            }
        return {
            'bid': 0.1,
            'ask': 0.1,
            'last': 0.1,
        }
    mocker.patch('freqtrade.exchange.Exchange.get_balances', return_value=rpc_balance)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', side_effect=mock_ticker)

    rc = client.get("/balance")
    response_success_assert(rc)
    assert "currencies" in rc.json
    assert len(rc.json["currencies"]) == 5
    assert rc.json['currencies'][0] == {
        'currency': 'BTC',
        'available': 12.0,
        'balance': 12.0,
        'pending': 0.0,
        'est_btc': 12.0,
    }


def test_api_count(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    rc = client.get("/count")
    response_success_assert(rc)

    assert rc.json["current"] == 0
    assert rc.json["max"] == 1.0

    # Create some test data
    ftbot.create_trade()
    rc = client.get("/count")
    response_success_assert(rc)
    assert rc.json["current"] == 1.0
    assert rc.json["max"] == 1.0
