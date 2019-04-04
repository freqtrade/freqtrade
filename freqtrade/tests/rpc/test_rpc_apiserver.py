"""
Unit test file for rpc/api_server.py
"""

from unittest.mock import MagicMock

from freqtrade.rpc.api_server import ApiServer
from freqtrade.state import State
from freqtrade.tests.conftest import get_patched_freqtradebot, patch_apiserver


def test__init__(default_conf, mocker):
    """
    Test __init__() method
    """
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.api_server.ApiServer.run', MagicMock())

    apiserver = ApiServer(get_patched_freqtradebot(mocker, default_conf))
    assert apiserver._config == default_conf


def test_start_endpoint(default_conf, mocker):
    """Test /start endpoint"""
    patch_apiserver(mocker)
    bot = get_patched_freqtradebot(mocker, default_conf)
    apiserver = ApiServer(bot)

    bot.state = State.STOPPED
    assert bot.state == State.STOPPED
    result = apiserver.start()
    assert result == '{"status": "starting trader ..."}'
    assert bot.state == State.RUNNING

    result = apiserver.start()
    assert result == '{"status": "already running"}'


def test_stop_endpoint(default_conf, mocker):
    """Test /stop endpoint"""
    patch_apiserver(mocker)
    bot = get_patched_freqtradebot(mocker, default_conf)
    apiserver = ApiServer(bot)

    bot.state = State.RUNNING
    assert bot.state == State.RUNNING
    result = apiserver.stop()
    assert result == '{"status": "stopping trader ..."}'
    assert bot.state == State.STOPPED

    result = apiserver.stop()
    assert result == '{"status": "already stopped"}'
