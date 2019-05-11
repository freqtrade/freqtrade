"""
Unit test file for rpc/api_server.py
"""

from unittest.mock import MagicMock

import pytest

from freqtrade.rpc.api_server import ApiServer
from freqtrade.state import State
from freqtrade.tests.conftest import get_patched_freqtradebot, patch_apiserver


@pytest.fixture
def botclient(default_conf, mocker):
    ftbot = get_patched_freqtradebot(mocker, default_conf)
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
