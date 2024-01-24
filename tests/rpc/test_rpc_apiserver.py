"""
Unit test file for rpc/api_server.py
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock

import pandas as pd
import pytest
import rapidjson
import uvicorn
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.exceptions import HTTPException
from fastapi.testclient import TestClient
from requests.auth import _basic_auth_str
from sqlalchemy import select

from freqtrade.__init__ import __version__
from freqtrade.enums import CandleType, RunMode, State, TradingMode
from freqtrade.exceptions import DependencyException, ExchangeError, OperationalException
from freqtrade.loggers import setup_logging, setup_logging_pre
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import PairLocks, Trade
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server import ApiServer
from freqtrade.rpc.api_server.api_auth import create_token, get_user_from_token
from freqtrade.rpc.api_server.uvicorn_threaded import UvicornServer
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from tests.conftest import (CURRENT_TEST_STRATEGY, EXMS, create_mock_trades, get_mock_coro,
                            get_patched_freqtradebot, log_has, log_has_re, patch_get_signal)


BASE_URI = "/api/v1"
_TEST_USER = "FreqTrader"
_TEST_PASS = "SuperSecurePassword1!"
_TEST_WS_TOKEN = "secret_Ws_t0ken"


@pytest.fixture
def botclient(default_conf, mocker):
    setup_logging_pre()
    setup_logging(default_conf)
    default_conf['runmode'] = RunMode.DRY_RUN
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": 8080,
                                        "CORS_origins": ['http://example.com'],
                                        "username": _TEST_USER,
                                        "password": _TEST_PASS,
                                        "ws_token": _TEST_WS_TOKEN
                                        }})

    ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    mocker.patch('freqtrade.rpc.api_server.ApiServer.start_api', MagicMock())
    apiserver = None
    try:
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(rpc)
        # We need to use the TestClient as a context manager to
        # handle lifespan events correctly
        with TestClient(apiserver.app) as client:
            yield ftbot, client
        # Cleanup ... ?
    finally:
        if apiserver:
            apiserver.cleanup()
        ApiServer.shutdown()


def client_post(client: TestClient, url, data={}):

    return client.post(url,
                       json=data,
                       headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
                                'Origin': 'http://example.com',
                                'content-type': 'application/json'
                                })


def client_patch(client: TestClient, url, data={}):

    return client.patch(url,
                        json=data,
                        headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
                                 'Origin': 'http://example.com',
                                 'content-type': 'application/json'
                                 })


def client_get(client: TestClient, url):
    # Add fake Origin to ensure CORS kicks in
    return client.get(url, headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
                                    'Origin': 'http://example.com'})


def client_delete(client: TestClient, url):
    # Add fake Origin to ensure CORS kicks in
    return client.delete(url, headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
                                       'Origin': 'http://example.com'})


def assert_response(response, expected_code=200, needs_cors=True):
    assert response.status_code == expected_code
    assert response.headers.get('content-type') == "application/json"
    if needs_cors:
        assert ('access-control-allow-credentials', 'true') in response.headers.items()
        assert ('access-control-allow-origin', 'http://example.com') in response.headers.items()


def test_api_not_found(botclient):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/invalid_url")
    assert_response(rc, 404)
    assert rc.json() == {"detail": "Not Found"}


def test_api_ui_fallback(botclient, mocker):
    _ftbot, client = botclient

    rc = client_get(client, "/favicon.ico")
    assert rc.status_code == 200

    rc = client_get(client, "/fallback_file.html")
    assert rc.status_code == 200
    assert '`freqtrade install-ui`' in rc.text

    # Forwarded to fallback_html or index.html (depending if it's installed or not)
    rc = client_get(client, "/something")
    assert rc.status_code == 200

    rc = client_get(client, "/something.js")
    assert rc.status_code == 200

    # Test directory traversal without mock
    rc = client_get(client, '%2F%2F%2Fetc/passwd')
    assert rc.status_code == 200
    # Allow both fallback or real UI
    assert '`freqtrade install-ui`' in rc.text or '<!DOCTYPE html>' in rc.text

    mocker.patch.object(Path, 'is_file', MagicMock(side_effect=[True, False]))
    rc = client_get(client, '%2F%2F%2Fetc/passwd')
    assert rc.status_code == 200

    assert '`freqtrade install-ui`' in rc.text


def test_api_ui_version(botclient, mocker):
    _ftbot, client = botclient

    mocker.patch('freqtrade.commands.deploy_commands.read_ui_version', return_value='0.1.2')
    rc = client_get(client, "/ui_version")
    assert rc.status_code == 200
    assert rc.json()['version'] == '0.1.2'


def test_api_auth():
    with pytest.raises(ValueError):
        create_token({'identity': {'u': 'Freqtrade'}}, 'secret1234', token_type="NotATokenType")

    token = create_token({'identity': {'u': 'Freqtrade'}}, 'secret1234')
    assert isinstance(token, str)

    u = get_user_from_token(token, 'secret1234')
    assert u == 'Freqtrade'
    with pytest.raises(HTTPException):
        get_user_from_token(token, 'secret1234', token_type='refresh')
    # Create invalid token
    token = create_token({'identity': {'u1': 'Freqrade'}}, 'secret1234')
    with pytest.raises(HTTPException):
        get_user_from_token(token, 'secret1234')

    with pytest.raises(HTTPException):
        get_user_from_token(b'not_a_token', 'secret1234')


def test_api_ws_auth(botclient):
    ftbot, client = botclient
    def url(token): return f"/api/v1/message/ws?token={token}"

    bad_token = "bad-ws_token"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(url(bad_token)) as websocket:
            websocket.receive()

    good_token = _TEST_WS_TOKEN
    with client.websocket_connect(url(good_token)) as websocket:
        pass

    jwt_secret = ftbot.config['api_server'].get('jwt_secret_key', 'super-secret')
    jwt_token = create_token({'identity': {'u': 'Freqtrade'}}, jwt_secret)
    with client.websocket_connect(url(jwt_token)) as websocket:
        pass


def test_api_unauthorized(botclient):
    ftbot, client = botclient
    rc = client.get(f"{BASE_URI}/ping")
    assert_response(rc, needs_cors=False)
    assert rc.json() == {'status': 'pong'}

    # Don't send user/pass information
    rc = client.get(f"{BASE_URI}/version")
    assert_response(rc, 401, needs_cors=False)
    assert rc.json() == {'detail': 'Unauthorized'}

    # Change only username
    ftbot.config['api_server']['username'] = 'Ftrader'
    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json() == {'detail': 'Unauthorized'}

    # Change only password
    ftbot.config['api_server']['username'] = _TEST_USER
    ftbot.config['api_server']['password'] = 'WrongPassword'
    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json() == {'detail': 'Unauthorized'}

    ftbot.config['api_server']['username'] = 'Ftrader'
    ftbot.config['api_server']['password'] = 'WrongPassword'

    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json() == {'detail': 'Unauthorized'}


def test_api_token_login(botclient):
    _ftbot, client = botclient
    rc = client.post(f"{BASE_URI}/token/login",
                     data=None,
                     headers={'Authorization': _basic_auth_str('WRONG_USER', 'WRONG_PASS'),
                              'Origin': 'http://example.com'})
    assert_response(rc, 401)
    rc = client_post(client, f"{BASE_URI}/token/login")
    assert_response(rc)
    assert 'access_token' in rc.json()
    assert 'refresh_token' in rc.json()

    # test Authentication is working with JWT tokens too
    rc = client.get(f"{BASE_URI}/count",
                    headers={'Authorization': f'Bearer {rc.json()["access_token"]}',
                             'Origin': 'http://example.com'})
    assert_response(rc)


def test_api_token_refresh(botclient):
    _ftbot, client = botclient
    rc = client_post(client, f"{BASE_URI}/token/login")
    assert_response(rc)
    rc = client.post(f"{BASE_URI}/token/refresh",
                     data=None,
                     headers={'Authorization': f'Bearer {rc.json()["refresh_token"]}',
                              'Origin': 'http://example.com'})
    assert_response(rc)
    assert 'access_token' in rc.json()
    assert 'refresh_token' not in rc.json()


def test_api_stop_workflow(botclient):
    ftbot, client = botclient
    assert ftbot.state == State.RUNNING
    rc = client_post(client, f"{BASE_URI}/stop")
    assert_response(rc)
    assert rc.json() == {'status': 'stopping trader ...'}
    assert ftbot.state == State.STOPPED

    # Stop bot again
    rc = client_post(client, f"{BASE_URI}/stop")
    assert_response(rc)
    assert rc.json() == {'status': 'already stopped'}

    # Start bot
    rc = client_post(client, f"{BASE_URI}/start")
    assert_response(rc)
    assert rc.json() == {'status': 'starting trader ...'}
    assert ftbot.state == State.RUNNING

    # Call start again
    rc = client_post(client, f"{BASE_URI}/start")
    assert_response(rc)
    assert rc.json() == {'status': 'already running'}


def test_api__init__(default_conf, mocker):
    """
    Test __init__() method
    """
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": 8080,
                                        "username": "TestUser",
                                        "password": "testPass",
                                        }})
    mocker.patch('freqtrade.rpc.telegram.Telegram._init')
    mocker.patch('freqtrade.rpc.api_server.webserver.ApiServer.start_api', MagicMock())
    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
    assert apiserver._config == default_conf
    with pytest.raises(OperationalException, match="RPC Handler already attached."):
        apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))

    apiserver.cleanup()
    ApiServer.shutdown()


def test_api_UvicornServer(mocker):
    thread_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.threading.Thread')
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert thread_mock.call_count == 0

    # Fake started to avoid sleeping forever
    s.started = True
    s.run_in_thread()
    assert thread_mock.call_count == 1

    s.cleanup()
    assert s.should_exit is True


def test_api_UvicornServer_run(mocker):
    serve_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve',
                              get_mock_coro(None))
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert serve_mock.call_count == 0

    # Fake started to avoid sleeping forever
    s.started = True
    s.run()
    assert serve_mock.call_count == 1


def test_api_UvicornServer_run_no_uvloop(mocker, import_fails):
    serve_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve',
                              get_mock_coro(None))
    asyncio.set_event_loop(asyncio.new_event_loop())
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert serve_mock.call_count == 0

    # Fake started to avoid sleeping forever
    s.started = True
    s.run()
    assert serve_mock.call_count == 1


def test_api_run(default_conf, mocker, caplog):
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": 8080,
                                        "username": "TestUser",
                                        "password": "testPass",
                                        }})
    mocker.patch('freqtrade.rpc.telegram.Telegram._init')

    server_inst_mock = MagicMock()
    server_inst_mock.run_in_thread = MagicMock()
    server_inst_mock.run = MagicMock()
    server_mock = MagicMock(return_value=server_inst_mock)
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer', server_mock)

    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))

    assert server_mock.call_count == 1
    assert apiserver._config == default_conf
    apiserver.start_api()
    assert server_mock.call_count == 2
    assert server_inst_mock.run_in_thread.call_count == 2
    assert server_inst_mock.run.call_count == 0
    assert server_mock.call_args_list[0][0][0].host == "127.0.0.1"
    assert server_mock.call_args_list[0][0][0].port == 8080
    assert isinstance(server_mock.call_args_list[0][0][0].app, FastAPI)

    assert log_has("Starting HTTP Server at 127.0.0.1:8080", caplog)
    assert log_has("Starting Local Rest Server.", caplog)

    # Test binding to public
    caplog.clear()
    server_mock.reset_mock()
    apiserver._config.update({"api_server": {"enabled": True,
                                             "listen_ip_address": "0.0.0.0",
                                             "listen_port": 8089,
                                             "password": "",
                                             }})
    apiserver.start_api()

    assert server_mock.call_count == 1
    assert server_inst_mock.run_in_thread.call_count == 1
    assert server_inst_mock.run.call_count == 0
    assert server_mock.call_args_list[0][0][0].host == "0.0.0.0"
    assert server_mock.call_args_list[0][0][0].port == 8089
    assert isinstance(server_mock.call_args_list[0][0][0].app, FastAPI)
    assert log_has("Starting HTTP Server at 0.0.0.0:8089", caplog)
    assert log_has("Starting Local Rest Server.", caplog)
    assert log_has("SECURITY WARNING - Local Rest Server listening to external connections",
                   caplog)
    assert log_has("SECURITY WARNING - This is insecure please set to your loopback,"
                   "e.g 127.0.0.1 in config.json", caplog)
    assert log_has("SECURITY WARNING - No password for local REST Server defined. "
                   "Please make sure that this is intentional!", caplog)
    assert log_has_re("SECURITY WARNING - `jwt_secret_key` seems to be default.*", caplog)

    server_mock.reset_mock()
    apiserver._standalone = True
    apiserver.start_api()
    assert server_inst_mock.run_in_thread.call_count == 0
    assert server_inst_mock.run.call_count == 1

    apiserver1 = ApiServer(default_conf)
    assert id(apiserver1) == id(apiserver)

    apiserver._standalone = False

    # Test crashing API server
    caplog.clear()
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer',
                 MagicMock(side_effect=Exception))
    apiserver.start_api()
    assert log_has("Api server failed to start.", caplog)
    apiserver.cleanup()
    ApiServer.shutdown()


def test_api_cleanup(default_conf, mocker, caplog):
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": 8080,
                                        "username": "TestUser",
                                        "password": "testPass",
                                        }})
    mocker.patch('freqtrade.rpc.telegram.Telegram._init')

    server_mock = MagicMock()
    server_mock.cleanup = MagicMock()
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer', server_mock)

    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))

    apiserver.cleanup()
    assert apiserver._server.cleanup.call_count == 1
    assert log_has("Stopping API Server", caplog)
    ApiServer.shutdown()


def test_api_reloadconf(botclient):
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/reload_config")
    assert_response(rc)
    assert rc.json() == {'status': 'Reloading config ...'}
    assert ftbot.state == State.RELOAD_CONFIG


def test_api_stopentry(botclient):
    ftbot, client = botclient
    assert ftbot.config['max_open_trades'] != 0

    rc = client_post(client, f"{BASE_URI}/stopbuy")
    assert_response(rc)
    assert rc.json() == {
        'status': 'No more entries will occur from now. Run /reload_config to reset.'}
    assert ftbot.config['max_open_trades'] == 0

    rc = client_post(client, f"{BASE_URI}/stopentry")
    assert_response(rc)
    assert rc.json() == {
        'status': 'No more entries will occur from now. Run /reload_config to reset.'}
    assert ftbot.config['max_open_trades'] == 0


def test_api_balance(botclient, mocker, rpc_balance, tickers):
    ftbot, client = botclient

    ftbot.config['dry_run'] = False
    mocker.patch(f'{EXMS}.get_balances', return_value=rpc_balance)
    mocker.patch(f'{EXMS}.get_tickers', tickers)
    mocker.patch(f'{EXMS}.get_valid_pair_combination',
                 side_effect=lambda a, b: f"{a}/{b}")
    ftbot.wallets.update()

    rc = client_get(client, f"{BASE_URI}/balance")
    assert_response(rc)
    response = rc.json()
    assert "currencies" in response
    assert len(response["currencies"]) == 5
    assert response['currencies'][0] == {
        'currency': 'BTC',
        'free': 12.0,
        'balance': 12.0,
        'used': 0.0,
        'bot_owned': pytest.approx(11.879999),
        'est_stake': 12.0,
        'est_stake_bot': pytest.approx(11.879999),
        'stake': 'BTC',
        'is_position': False,
        'leverage': 1.0,
        'position': 0.0,
        'side': 'long',
        'is_bot_managed': True,
    }
    assert response['total'] == 12.159513094
    assert response['total_bot'] == pytest.approx(11.879999)
    assert 'starting_capital' in response
    assert 'starting_capital_fiat' in response
    assert 'starting_capital_pct' in response
    assert 'starting_capital_ratio' in response


@pytest.mark.parametrize('is_short', [True, False])
def test_api_count(botclient, mocker, ticker, fee, markets, is_short):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/count")
    assert_response(rc)

    assert rc.json()["current"] == 0
    assert rc.json()["max"] == 1

    # Create some test data
    create_mock_trades(fee, is_short=is_short)
    rc = client_get(client, f"{BASE_URI}/count")
    assert_response(rc)
    assert rc.json()["current"] == 4
    assert rc.json()["max"] == 1

    ftbot.config['max_open_trades'] = float('inf')
    rc = client_get(client, f"{BASE_URI}/count")
    assert rc.json()["max"] == -1


def test_api_locks(botclient):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/locks")
    assert_response(rc)

    assert 'locks' in rc.json()

    assert rc.json()['lock_count'] == 0
    assert rc.json()['lock_count'] == len(rc.json()['locks'])

    PairLocks.lock_pair('ETH/BTC', datetime.now(timezone.utc) + timedelta(minutes=4), 'randreason')
    PairLocks.lock_pair('XRP/BTC', datetime.now(timezone.utc) + timedelta(minutes=20), 'deadbeef')

    rc = client_get(client, f"{BASE_URI}/locks")
    assert_response(rc)

    assert rc.json()['lock_count'] == 2
    assert rc.json()['lock_count'] == len(rc.json()['locks'])
    assert 'ETH/BTC' in (rc.json()['locks'][0]['pair'], rc.json()['locks'][1]['pair'])
    assert 'randreason' in (rc.json()['locks'][0]['reason'], rc.json()['locks'][1]['reason'])
    assert 'deadbeef' in (rc.json()['locks'][0]['reason'], rc.json()['locks'][1]['reason'])

    # Test deletions
    rc = client_delete(client, f"{BASE_URI}/locks/1")
    assert_response(rc)
    assert rc.json()['lock_count'] == 1

    rc = client_post(client, f"{BASE_URI}/locks/delete",
                     data={"pair": "XRP/BTC"})
    assert_response(rc)
    assert rc.json()['lock_count'] == 0


def test_api_show_config(botclient):
    ftbot, client = botclient
    patch_get_signal(ftbot)

    rc = client_get(client, f"{BASE_URI}/show_config")
    assert_response(rc)
    response = rc.json()
    assert 'dry_run' in response
    assert response['exchange'] == 'binance'
    assert response['timeframe'] == '5m'
    assert response['timeframe_ms'] == 300000
    assert response['timeframe_min'] == 5
    assert response['state'] == 'running'
    assert response['bot_name'] == 'freqtrade'
    assert response['trading_mode'] == 'spot'
    assert response['strategy_version'] is None
    assert not response['trailing_stop']
    assert 'entry_pricing' in response
    assert 'exit_pricing' in response
    assert 'unfilledtimeout' in response
    assert 'version' in response
    assert 'api_version' in response
    assert 2.1 <= response['api_version'] < 3.0


def test_api_daily(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/daily")
    assert_response(rc)
    assert len(rc.json()['data']) == 7
    assert rc.json()['stake_currency'] == 'BTC'
    assert rc.json()['fiat_display_currency'] == 'USD'
    assert rc.json()['data'][0]['date'] == str(datetime.now(timezone.utc).date())


def test_api_weekly(botclient, mocker, ticker, fee, markets, time_machine):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    time_machine.move_to("2023-03-31 21:45:05 +00:00")
    rc = client_get(client, f"{BASE_URI}/weekly")
    assert_response(rc)
    assert len(rc.json()['data']) == 4
    assert rc.json()['stake_currency'] == 'BTC'
    assert rc.json()['fiat_display_currency'] == 'USD'
    # Moved to monday
    assert rc.json()['data'][0]['date'] == '2023-03-27'
    assert rc.json()['data'][1]['date'] == '2023-03-20'


def test_api_monthly(botclient, mocker, ticker, fee, markets, time_machine):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    time_machine.move_to("2023-03-31 21:45:05 +00:00")
    rc = client_get(client, f"{BASE_URI}/monthly")
    assert_response(rc)
    assert len(rc.json()['data']) == 3
    assert rc.json()['stake_currency'] == 'BTC'
    assert rc.json()['fiat_display_currency'] == 'USD'
    assert rc.json()['data'][0]['date'] == '2023-03-01'
    assert rc.json()['data'][1]['date'] == '2023-02-01'


@pytest.mark.parametrize('is_short', [True, False])
def test_api_trades(botclient, mocker, fee, markets, is_short):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/trades")
    assert_response(rc)
    assert len(rc.json()) == 4
    assert rc.json()['trades_count'] == 0
    assert rc.json()['total_trades'] == 0
    assert rc.json()['offset'] == 0

    create_mock_trades(fee, is_short=is_short)
    Trade.session.flush()

    rc = client_get(client, f"{BASE_URI}/trades")
    assert_response(rc)
    assert len(rc.json()['trades']) == 2
    assert rc.json()['trades_count'] == 2
    assert rc.json()['total_trades'] == 2
    assert rc.json()['trades'][0]['is_short'] == is_short
    rc = client_get(client, f"{BASE_URI}/trades?limit=1")
    assert_response(rc)
    assert len(rc.json()['trades']) == 1
    assert rc.json()['trades_count'] == 1
    assert rc.json()['total_trades'] == 2


@pytest.mark.parametrize('is_short', [True, False])
def test_api_trade_single(botclient, mocker, fee, ticker, markets, is_short):
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        fetch_ticker=ticker,
    )
    rc = client_get(client, f"{BASE_URI}/trade/3")
    assert_response(rc, 404)
    assert rc.json()['detail'] == 'Trade not found.'

    Trade.rollback()
    create_mock_trades(fee, is_short=is_short)

    rc = client_get(client, f"{BASE_URI}/trade/3")
    assert_response(rc)
    assert rc.json()['trade_id'] == 3
    assert rc.json()['is_short'] == is_short


@pytest.mark.parametrize('is_short', [True, False])
def test_api_delete_trade(botclient, mocker, fee, markets, is_short):
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        cancel_order=cancel_mock,
        cancel_stoploss_order=stoploss_mock,
    )

    create_mock_trades(fee, is_short=is_short)

    ftbot.strategy.order_types['stoploss_on_exchange'] = True
    trades = Trade.session.scalars(select(Trade)).all()
    Trade.commit()
    assert len(trades) > 2

    rc = client_delete(client, f"{BASE_URI}/trades/1")
    assert_response(rc)
    assert rc.json()['result_msg'] == 'Deleted trade 1. Closed 1 open orders.'
    assert len(trades) - 1 == len(Trade.session.scalars(select(Trade)).all())
    assert cancel_mock.call_count == 1

    cancel_mock.reset_mock()
    rc = client_delete(client, f"{BASE_URI}/trades/1")
    # Trade is gone now.
    assert_response(rc, 502)
    assert cancel_mock.call_count == 0

    assert len(trades) - 1 == len(Trade.session.scalars(select(Trade)).all())
    rc = client_delete(client, f"{BASE_URI}/trades/5")
    assert_response(rc)
    assert rc.json()['result_msg'] == 'Deleted trade 5. Closed 1 open orders.'
    assert len(trades) - 2 == len(Trade.session.scalars(select(Trade)).all())
    assert stoploss_mock.call_count == 1

    rc = client_delete(client, f"{BASE_URI}/trades/502")
    # Error - trade won't exist.
    assert_response(rc, 502)


@pytest.mark.parametrize('is_short', [True, False])
def test_api_delete_open_order(botclient, mocker, fee, markets, ticker, is_short):
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        fetch_ticker=ticker,
        cancel_order=cancel_mock,
        cancel_stoploss_order=stoploss_mock,
    )

    rc = client_delete(client, f"{BASE_URI}/trades/10/open-order")
    assert_response(rc, 502)
    assert 'Invalid trade_id.' in rc.json()['error']

    create_mock_trades(fee, is_short=is_short)
    Trade.commit()

    rc = client_delete(client, f"{BASE_URI}/trades/5/open-order")
    assert_response(rc, 502)
    assert 'No open order for trade_id' in rc.json()['error']
    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f'{EXMS}.fetch_order', side_effect=ExchangeError)
    rc = client_delete(client, f"{BASE_URI}/trades/6/open-order")
    assert_response(rc, 502)
    assert 'Order not found.' in rc.json()['error']

    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f'{EXMS}.fetch_order', return_value=trade.orders[-1].to_ccxt_object())

    rc = client_delete(client, f"{BASE_URI}/trades/6/open-order")
    assert_response(rc)
    assert cancel_mock.call_count == 1


@pytest.mark.parametrize('is_short', [True, False])
def test_api_trade_reload_trade(botclient, mocker, fee, markets, ticker, is_short):
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    ftbot.handle_onexchange_order = MagicMock()
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        fetch_ticker=ticker,
        cancel_order=cancel_mock,
        cancel_stoploss_order=stoploss_mock,
    )

    rc = client_post(client, f"{BASE_URI}/trades/10/reload")
    assert_response(rc, 502)
    assert 'Could not find trade with id 10.' in rc.json()['error']
    assert ftbot.handle_onexchange_order.call_count == 0

    create_mock_trades(fee, is_short=is_short)
    Trade.commit()

    rc = client_post(client, f"{BASE_URI}/trades/5/reload")
    assert ftbot.handle_onexchange_order.call_count == 1


def test_api_logs(botclient):
    _ftbot, client = botclient
    rc = client_get(client, f"{BASE_URI}/logs")
    assert_response(rc)
    assert len(rc.json()) == 2
    assert 'logs' in rc.json()
    # Using a fixed comparison here would make this test fail!
    assert rc.json()['log_count'] > 1
    assert len(rc.json()['logs']) == rc.json()['log_count']

    assert isinstance(rc.json()['logs'][0], list)
    # date
    assert isinstance(rc.json()['logs'][0][0], str)
    # created_timestamp
    assert isinstance(rc.json()['logs'][0][1], float)
    assert isinstance(rc.json()['logs'][0][2], str)
    assert isinstance(rc.json()['logs'][0][3], str)
    assert isinstance(rc.json()['logs'][0][4], str)

    rc1 = client_get(client, f"{BASE_URI}/logs?limit=5")
    assert_response(rc1)
    assert len(rc1.json()) == 2
    assert 'logs' in rc1.json()
    # Using a fixed comparison here would make this test fail!
    if rc1.json()['log_count'] < 5:
        # Help debugging random test failure
        print(f"rc={rc.json()}")
        print(f"rc1={rc1.json()}")
    assert rc1.json()['log_count'] > 2
    assert len(rc1.json()['logs']) == rc1.json()['log_count']


def test_api_edge_disabled(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/edge")
    assert_response(rc, 502)
    assert rc.json() == {"error": "Error querying /api/v1/edge: Edge is not enabled."}


@pytest.mark.parametrize('is_short,expected', [
    (
        True,
        {'best_pair': 'ETC/BTC', 'best_rate': -0.5, 'best_pair_profit_ratio': -0.005,
         'profit_all_coin': 45.561959,
         'profit_all_fiat': 562462.39126200, 'profit_all_percent_mean': 66.41,
         'profit_all_ratio_mean': 0.664109545, 'profit_all_percent_sum': 398.47,
         'profit_all_ratio_sum': 3.98465727, 'profit_all_percent': 4.56,
         'profit_all_ratio': 0.04556147, 'profit_closed_coin': -0.00673913,
         'profit_closed_fiat': -83.19455985, 'profit_closed_ratio_mean': -0.0075,
         'profit_closed_percent_mean': -0.75, 'profit_closed_ratio_sum': -0.015,
         'profit_closed_percent_sum': -1.5, 'profit_closed_ratio': -6.739057628404269e-06,
         'profit_closed_percent': -0.0, 'winning_trades': 0, 'losing_trades': 2,
         'profit_factor': 0.0, 'winrate': 0.0, 'expectancy': -0.0033695635,
         'expectancy_ratio': -1.0, 'trading_volume': 75.945,
         }
    ),
    (
        False,
        {'best_pair': 'XRP/BTC', 'best_rate': 1.0, 'best_pair_profit_ratio': 0.01,
         'profit_all_coin': -45.79641127,
         'profit_all_fiat': -565356.69712815, 'profit_all_percent_mean': -66.41,
         'profit_all_ratio_mean': -0.6641100666666667, 'profit_all_percent_sum': -398.47,
         'profit_all_ratio_sum': -3.9846604, 'profit_all_percent': -4.58,
         'profit_all_ratio': -0.045796261934205953, 'profit_closed_coin': 0.00073913,
         'profit_closed_fiat': 9.124559849999999, 'profit_closed_ratio_mean': 0.0075,
         'profit_closed_percent_mean': 0.75, 'profit_closed_ratio_sum': 0.015,
         'profit_closed_percent_sum': 1.5, 'profit_closed_ratio': 7.391275897987988e-07,
         'profit_closed_percent': 0.0, 'winning_trades': 2, 'losing_trades': 0,
         'profit_factor': None, 'winrate': 1.0, 'expectancy': 0.0003695635,
         'expectancy_ratio': 100, 'trading_volume': 75.945,
         }
    ),
    (
        None,
        {'best_pair': 'XRP/BTC', 'best_rate': 1.0, 'best_pair_profit_ratio': 0.01,
         'profit_all_coin': -14.94732578,
         'profit_all_fiat': -184524.7367541, 'profit_all_percent_mean': 0.08,
         'profit_all_ratio_mean': 0.000835751666666662, 'profit_all_percent_sum': 0.5,
         'profit_all_ratio_sum': 0.005014509999999972, 'profit_all_percent': -1.49,
         'profit_all_ratio': -0.014947184841095841, 'profit_closed_coin': -0.00542913,
         'profit_closed_fiat': -67.02260985, 'profit_closed_ratio_mean': 0.0025,
         'profit_closed_percent_mean': 0.25, 'profit_closed_ratio_sum': 0.005,
         'profit_closed_percent_sum': 0.5, 'profit_closed_ratio': -5.429078808526421e-06,
         'profit_closed_percent': -0.0, 'winning_trades': 1, 'losing_trades': 1,
         'profit_factor': 0.02775724835771106, 'winrate': 0.5,
         'expectancy': -0.0027145635000000003, 'expectancy_ratio': -0.48612137582114445,
         'trading_volume': 75.945,
         }
    )
])
def test_api_profit(botclient, mocker, ticker, fee, markets, is_short, expected):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc, 200)
    assert rc.json()['trade_count'] == 0

    create_mock_trades(fee, is_short=is_short)
    # Simulate fulfilled LIMIT_BUY order for trade

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc)
    # raise ValueError(rc.json())
    assert rc.json() == {
        'avg_duration': ANY,
        'best_pair': expected['best_pair'],
        'best_pair_profit_ratio': expected['best_pair_profit_ratio'],
        'best_rate': expected['best_rate'],
        'first_trade_date': ANY,
        'first_trade_humanized': ANY,
        'first_trade_timestamp': ANY,
        'latest_trade_date': ANY,
        'latest_trade_humanized': '5 minutes ago',
        'latest_trade_timestamp': ANY,
        'profit_all_coin': pytest.approx(expected['profit_all_coin']),
        'profit_all_fiat': pytest.approx(expected['profit_all_fiat']),
        'profit_all_percent_mean': pytest.approx(expected['profit_all_percent_mean']),
        'profit_all_ratio_mean': pytest.approx(expected['profit_all_ratio_mean']),
        'profit_all_percent_sum': pytest.approx(expected['profit_all_percent_sum']),
        'profit_all_ratio_sum': pytest.approx(expected['profit_all_ratio_sum']),
        'profit_all_percent': pytest.approx(expected['profit_all_percent']),
        'profit_all_ratio': pytest.approx(expected['profit_all_ratio']),
        'profit_closed_coin': pytest.approx(expected['profit_closed_coin']),
        'profit_closed_fiat': pytest.approx(expected['profit_closed_fiat']),
        'profit_closed_ratio_mean': pytest.approx(expected['profit_closed_ratio_mean']),
        'profit_closed_percent_mean': pytest.approx(expected['profit_closed_percent_mean']),
        'profit_closed_ratio_sum': pytest.approx(expected['profit_closed_ratio_sum']),
        'profit_closed_percent_sum': pytest.approx(expected['profit_closed_percent_sum']),
        'profit_closed_ratio': pytest.approx(expected['profit_closed_ratio']),
        'profit_closed_percent': pytest.approx(expected['profit_closed_percent']),
        'trade_count': 6,
        'closed_trade_count': 2,
        'winning_trades': expected['winning_trades'],
        'losing_trades': expected['losing_trades'],
        'profit_factor': expected['profit_factor'],
        'winrate': expected['winrate'],
        'expectancy': expected['expectancy'],
        'expectancy_ratio': expected['expectancy_ratio'],
        'max_drawdown': ANY,
        'max_drawdown_abs': ANY,
        'max_drawdown_start': ANY,
        'max_drawdown_start_timestamp': ANY,
        'max_drawdown_end': ANY,
        'max_drawdown_end_timestamp': ANY,
        'trading_volume': expected['trading_volume'],
        'bot_start_timestamp': 0,
        'bot_start_date': '',
    }


@pytest.mark.parametrize('is_short', [True, False])
def test_api_stats(botclient, mocker, ticker, fee, markets, is_short):
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    rc = client_get(client, f"{BASE_URI}/stats")
    assert_response(rc, 200)
    assert 'durations' in rc.json()
    assert 'exit_reasons' in rc.json()

    create_mock_trades(fee, is_short=is_short)

    rc = client_get(client, f"{BASE_URI}/stats")
    assert_response(rc, 200)
    assert 'durations' in rc.json()
    assert 'exit_reasons' in rc.json()

    assert 'wins' in rc.json()['durations']
    assert 'losses' in rc.json()['durations']
    assert 'draws' in rc.json()['durations']


def test_api_performance(botclient, fee):
    ftbot, client = botclient
    patch_get_signal(ftbot)

    trade = Trade(
        pair='LTC/ETH',
        amount=1,
        exchange='binance',
        stake_amount=1,
        open_rate=0.245441,
        is_open=False,
        fee_close=fee.return_value,
        fee_open=fee.return_value,
        close_rate=0.265441,
        leverage=1.0,
    )
    trade.close_profit = trade.calc_profit_ratio(trade.close_rate)
    trade.close_profit_abs = trade.calc_profit(trade.close_rate)
    Trade.session.add(trade)

    trade = Trade(
        pair='XRP/ETH',
        amount=5,
        stake_amount=1,
        exchange='binance',
        open_rate=0.412,
        is_open=False,
        fee_close=fee.return_value,
        fee_open=fee.return_value,
        close_rate=0.391,
        leverage=1.0,
    )
    trade.close_profit = trade.calc_profit_ratio(trade.close_rate)
    trade.close_profit_abs = trade.calc_profit(trade.close_rate)

    Trade.session.add(trade)
    Trade.commit()

    rc = client_get(client, f"{BASE_URI}/performance")
    assert_response(rc)
    assert len(rc.json()) == 2
    assert rc.json() == [{'count': 1, 'pair': 'LTC/ETH', 'profit': 7.61, 'profit_pct': 7.61,
                          'profit_ratio': 0.07609203, 'profit_abs': 0.0187228},
                         {'count': 1, 'pair': 'XRP/ETH', 'profit': -5.57, 'profit_pct': -5.57,
                          'profit_ratio': -0.05570419, 'profit_abs': -0.1150375}]


def test_api_entries(botclient, fee):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    # Empty
    rc = client_get(client, f"{BASE_URI}/entries")
    assert_response(rc)
    assert len(rc.json()) == 0

    create_mock_trades(fee)
    rc = client_get(client, f"{BASE_URI}/entries")
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp['enter_tag'] == 'TEST1'
    assert resp['count'] == 1
    assert resp['profit_pct'] == 0.5


def test_api_exits(botclient, fee):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    # Empty
    rc = client_get(client, f"{BASE_URI}/exits")
    assert_response(rc)
    assert len(rc.json()) == 0

    create_mock_trades(fee)
    rc = client_get(client, f"{BASE_URI}/exits")
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp['exit_reason'] == 'sell_signal'
    assert resp['count'] == 1
    assert resp['profit_pct'] == 0.5


def test_api_mix_tag(botclient, fee):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    # Empty
    rc = client_get(client, f"{BASE_URI}/mix_tags")
    assert_response(rc)
    assert len(rc.json()) == 0

    create_mock_trades(fee)
    rc = client_get(client, f"{BASE_URI}/mix_tags")
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp['mix_tag'] == 'TEST1 sell_signal'
    assert resp['count'] == 1
    assert resp['profit_pct'] == 0.5


@pytest.mark.parametrize(
    'is_short,current_rate,open_trade_value',
    [(True, 1.098e-05, 15.0911775),
     (False, 1.099e-05, 15.1668225)])
def test_api_status(botclient, mocker, ticker, fee, markets, is_short,
                    current_rate, open_trade_value):
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        fetch_order=MagicMock(return_value={}),
    )

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc, 200)
    assert rc.json() == []
    create_mock_trades(fee, is_short=is_short)

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc)
    assert len(rc.json()) == 4
    assert rc.json()[0] == {
        'amount': 123.0,
        'amount_requested': 123.0,
        'close_date': None,
        'close_timestamp': None,
        'close_profit': None,
        'close_profit_pct': None,
        'close_profit_abs': None,
        'close_rate': None,
        'profit_ratio': ANY,
        'profit_pct': ANY,
        'profit_abs': ANY,
        'profit_fiat': ANY,
        'total_profit_abs': ANY,
        'total_profit_fiat': ANY,
        'total_profit_ratio': ANY,
        'realized_profit': 0.0,
        'realized_profit_ratio': None,
        'current_rate': current_rate,
        'open_date': ANY,
        'open_timestamp': ANY,
        'open_rate': 0.123,
        'pair': 'ETH/BTC',
        'base_currency': 'ETH',
        'quote_currency': 'BTC',
        'stake_amount': 0.001,
        'max_stake_amount': ANY,
        'stop_loss_abs': ANY,
        'stop_loss_pct': ANY,
        'stop_loss_ratio': ANY,
        'stoploss_order_id': None,
        'stoploss_last_update': ANY,
        'stoploss_last_update_timestamp': ANY,
        'initial_stop_loss_abs': 0.0,
        'initial_stop_loss_pct': ANY,
        'initial_stop_loss_ratio': ANY,
        'stoploss_current_dist': ANY,
        'stoploss_current_dist_ratio': ANY,
        'stoploss_current_dist_pct': ANY,
        'stoploss_entry_dist': ANY,
        'stoploss_entry_dist_ratio': ANY,
        'trade_id': 1,
        'close_rate_requested': ANY,
        'fee_close': 0.0025,
        'fee_close_cost': None,
        'fee_close_currency': None,
        'fee_open': 0.0025,
        'fee_open_cost': None,
        'fee_open_currency': None,
        'is_open': True,
        "is_short": is_short,
        'max_rate': ANY,
        'min_rate': ANY,
        'open_rate_requested': ANY,
        'open_trade_value': open_trade_value,
        'exit_reason': None,
        'exit_order_status': None,
        'strategy': CURRENT_TEST_STRATEGY,
        'enter_tag': None,
        'timeframe': 5,
        'exchange': 'binance',
        'leverage': 1.0,
        'interest_rate': 0.0,
        'liquidation_price': None,
        'funding_fees': None,
        'trading_mode': ANY,
        'amount_precision': None,
        'price_precision': None,
        'precision_mode': None,
        'orders': [ANY],
        'has_open_orders': True,
    }

    mocker.patch(f'{EXMS}.get_rate',
                 MagicMock(side_effect=ExchangeError("Pair 'ETH/BTC' not available")))

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc)
    resp_values = rc.json()
    assert len(resp_values) == 4
    assert resp_values[0]['profit_abs'] == 0.0


def test_api_version(botclient):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc)
    assert rc.json() == {"version": __version__}


def test_api_blacklist(botclient, mocker):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/blacklist")
    assert_response(rc)
    # DOGE and HOT are not in the markets mock!
    assert rc.json() == {"blacklist": ["DOGE/BTC", "HOT/BTC"],
                         "blacklist_expanded": [],
                         "length": 2,
                         "method": ["StaticPairList"],
                         "errors": {},
                         }

    # Add ETH/BTC to blacklist
    rc = client_post(client, f"{BASE_URI}/blacklist",
                     data={"blacklist": ["ETH/BTC"]})
    assert_response(rc)
    assert rc.json() == {"blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC"],
                         "blacklist_expanded": ["ETH/BTC"],
                         "length": 3,
                         "method": ["StaticPairList"],
                         "errors": {},
                         }

    rc = client_post(client, f"{BASE_URI}/blacklist",
                     data={"blacklist": ["XRP/.*"]})
    assert_response(rc)
    assert rc.json() == {"blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC", "XRP/.*"],
                         "blacklist_expanded": ["ETH/BTC", "XRP/BTC", "XRP/USDT"],
                         "length": 4,
                         "method": ["StaticPairList"],
                         "errors": {},
                         }

    rc = client_delete(client, f"{BASE_URI}/blacklist?pairs_to_delete=DOGE/BTC")
    assert_response(rc)
    assert rc.json() == {"blacklist": ["HOT/BTC", "ETH/BTC", "XRP/.*"],
                         "blacklist_expanded": ["ETH/BTC", "XRP/BTC", "XRP/USDT"],
                         "length": 3,
                         "method": ["StaticPairList"],
                         "errors": {},
                         }

    rc = client_delete(client, f"{BASE_URI}/blacklist?pairs_to_delete=NOTHING/BTC")
    assert_response(rc)
    assert rc.json() == {"blacklist": ["HOT/BTC", "ETH/BTC", "XRP/.*"],
                         "blacklist_expanded": ["ETH/BTC", "XRP/BTC", "XRP/USDT"],
                         "length": 3,
                         "method": ["StaticPairList"],
                         "errors": {
                             "NOTHING/BTC": {
                                 "error_msg": "Pair NOTHING/BTC is not in the current blacklist."
                             }
    },
    }
    rc = client_delete(
        client,
        f"{BASE_URI}/blacklist?pairs_to_delete=HOT/BTC&pairs_to_delete=ETH/BTC")
    assert_response(rc)
    assert rc.json() == {"blacklist": ["XRP/.*"],
                         "blacklist_expanded": ["XRP/BTC", "XRP/USDT"],
                         "length": 1,
                         "method": ["StaticPairList"],
                         "errors": {},
                         }


def test_api_whitelist(botclient):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/whitelist")
    assert_response(rc)
    assert rc.json() == {
        "whitelist": ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC'],
        "length": 4,
        "method": ["StaticPairList"]
    }


@pytest.mark.parametrize('endpoint', [
    'forcebuy',
    'forceenter',
])
def test_api_force_entry(botclient, mocker, fee, endpoint):
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/{endpoint}",
                     data={"pair": "ETH/BTC"})
    assert_response(rc, 502)
    assert rc.json() == {"error": f"Error querying /api/v1/{endpoint}: Force_entry not enabled."}

    # enable forcebuy
    ftbot.config['force_entry_enable'] = True

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch("freqtrade.rpc.rpc.RPC._rpc_force_entry", fbuy_mock)
    rc = client_post(client, f"{BASE_URI}/{endpoint}",
                     data={"pair": "ETH/BTC"})
    assert_response(rc)
    assert rc.json() == {"status": "Error entering long trade for pair ETH/BTC."}

    # Test creating trade
    fbuy_mock = MagicMock(return_value=Trade(
        pair='ETH/BTC',
        amount=1,
        amount_requested=1,
        exchange='binance',
        stake_amount=1,
        open_rate=0.245441,
        open_date=datetime.now(timezone.utc),
        is_open=False,
        is_short=False,
        fee_close=fee.return_value,
        fee_open=fee.return_value,
        close_rate=0.265441,
        id=22,
        timeframe=5,
        strategy=CURRENT_TEST_STRATEGY,
        trading_mode=TradingMode.SPOT
    ))
    mocker.patch("freqtrade.rpc.rpc.RPC._rpc_force_entry", fbuy_mock)

    rc = client_post(client, f"{BASE_URI}/{endpoint}",
                     data={"pair": "ETH/BTC"})
    assert_response(rc)
    assert rc.json() == {
        'amount': 1.0,
        'amount_requested': 1.0,
        'trade_id': 22,
        'close_date': None,
        'close_timestamp': None,
        'close_rate': 0.265441,
        'open_date': ANY,
        'open_timestamp': ANY,
        'open_rate': 0.245441,
        'pair': 'ETH/BTC',
        'base_currency': 'ETH',
        'quote_currency': 'BTC',
        'stake_amount': 1,
        'max_stake_amount': ANY,
        'stop_loss_abs': None,
        'stop_loss_pct': None,
        'stop_loss_ratio': None,
        'stoploss_order_id': None,
        'stoploss_last_update': None,
        'stoploss_last_update_timestamp': None,
        'initial_stop_loss_abs': None,
        'initial_stop_loss_pct': None,
        'initial_stop_loss_ratio': None,
        'close_profit': None,
        'close_profit_pct': None,
        'close_profit_abs': None,
        'close_rate_requested': None,
        'profit_ratio': None,
        'profit_pct': None,
        'profit_abs': None,
        'profit_fiat': None,
        'realized_profit': 0.0,
        'realized_profit_ratio': None,
        'fee_close': 0.0025,
        'fee_close_cost': None,
        'fee_close_currency': None,
        'fee_open': 0.0025,
        'fee_open_cost': None,
        'fee_open_currency': None,
        'is_open': False,
        'is_short': False,
        'max_rate': None,
        'min_rate': None,
        'open_rate_requested': None,
        'open_trade_value': 0.24605460,
        'exit_reason': None,
        'exit_order_status': None,
        'strategy': CURRENT_TEST_STRATEGY,
        'enter_tag': None,
        'timeframe': 5,
        'exchange': 'binance',
        'leverage': None,
        'interest_rate': None,
        'liquidation_price': None,
        'funding_fees': None,
        'trading_mode': 'spot',
        'amount_precision': None,
        'price_precision': None,
        'precision_mode': None,
        'has_open_orders': False,
        'orders': [],
    }


def test_api_forceexit(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    patch_get_signal(ftbot)

    rc = client_post(client, f"{BASE_URI}/forceexit",
                     data={"tradeid": "1"})
    assert_response(rc, 502)
    assert rc.json() == {"error": "Error querying /api/v1/forceexit: invalid argument"}
    Trade.rollback()

    create_mock_trades(fee)
    trade = Trade.get_trades([Trade.id == 5]).first()
    assert pytest.approx(trade.amount) == 123
    rc = client_post(client, f"{BASE_URI}/forceexit",
                     data={"tradeid": "5", "ordertype": "market", "amount": 23})
    assert_response(rc)
    assert rc.json() == {'result': 'Created exit order for trade 5.'}
    Trade.rollback()

    trade = Trade.get_trades([Trade.id == 5]).first()
    assert pytest.approx(trade.amount) == 100
    assert trade.is_open is True

    rc = client_post(client, f"{BASE_URI}/forceexit",
                     data={"tradeid": "5"})
    assert_response(rc)
    assert rc.json() == {'result': 'Created exit order for trade 5.'}
    Trade.rollback()

    trade = Trade.get_trades([Trade.id == 5]).first()
    assert trade.is_open is False


def test_api_pair_candles(botclient, ohlcv_history):
    ftbot, client = botclient
    timeframe = '5m'
    amount = 3

    # No pair
    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?limit={amount}&timeframe={timeframe}")
    assert_response(rc, 422)

    # No timeframe
    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?pair=XRP%2FBTC")
    assert_response(rc, 422)

    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}")
    assert_response(rc)
    assert 'columns' in rc.json()
    assert 'data_start_ts' in rc.json()
    assert 'data_start' in rc.json()
    assert 'data_stop' in rc.json()
    assert 'data_stop_ts' in rc.json()
    assert len(rc.json()['data']) == 0
    ohlcv_history['sma'] = ohlcv_history['close'].rolling(2).mean()
    ohlcv_history['enter_long'] = 0
    ohlcv_history.loc[1, 'enter_long'] = 1
    ohlcv_history['exit_long'] = 0
    ohlcv_history['enter_short'] = 0
    ohlcv_history['exit_short'] = 0

    ftbot.dataprovider._set_cached_df("XRP/BTC", timeframe, ohlcv_history, CandleType.SPOT)

    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}")
    assert_response(rc)
    assert 'strategy' in rc.json()
    assert rc.json()['strategy'] == CURRENT_TEST_STRATEGY
    assert 'columns' in rc.json()
    assert 'data_start_ts' in rc.json()
    assert 'data_start' in rc.json()
    assert 'data_stop' in rc.json()
    assert 'data_stop_ts' in rc.json()
    assert rc.json()['data_start'] == '2017-11-26 08:50:00+00:00'
    assert rc.json()['data_start_ts'] == 1511686200000
    assert rc.json()['data_stop'] == '2017-11-26 09:00:00+00:00'
    assert rc.json()['data_stop_ts'] == 1511686800000
    assert isinstance(rc.json()['columns'], list)
    assert set(rc.json()['columns']) == {
        'date', 'open', 'high', 'low', 'close', 'volume',
        'sma', 'enter_long', 'exit_long', 'enter_short', 'exit_short', '__date_ts',
        '_enter_long_signal_close', '_exit_long_signal_close',
        '_enter_short_signal_close', '_exit_short_signal_close'
    }
    assert 'pair' in rc.json()
    assert rc.json()['pair'] == 'XRP/BTC'

    assert 'data' in rc.json()
    assert len(rc.json()['data']) == amount

    assert (rc.json()['data'] ==
            [['2017-11-26T08:50:00Z', 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869,
              None, 0, 0, 0, 0, 1511686200000, None, None, None, None],
             ['2017-11-26T08:55:00Z', 8.88e-05, 8.942e-05, 8.88e-05,
                 8.893e-05, 0.05874751, 8.886500000000001e-05, 1, 0, 0, 0, 1511686500000, 8.893e-05,
                 None, None, None],
             ['2017-11-26T09:00:00Z', 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05,
                 0.7039405, 8.885e-05, 0, 0, 0, 0, 1511686800000, None, None, None, None]

             ])
    ohlcv_history['exit_long'] = ohlcv_history['exit_long'].astype('float64')
    ohlcv_history.at[0, 'exit_long'] = float('inf')
    ohlcv_history['date1'] = ohlcv_history['date']
    ohlcv_history.at[0, 'date1'] = pd.NaT

    ftbot.dataprovider._set_cached_df("XRP/BTC", timeframe, ohlcv_history, CandleType.SPOT)
    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}")
    assert_response(rc)
    assert (rc.json()['data'] ==
            [['2017-11-26T08:50:00Z', 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869,
              None, 0, None, 0, 0, None, 1511686200000, None, None, None, None],
             ['2017-11-26T08:55:00Z', 8.88e-05, 8.942e-05, 8.88e-05,
                 8.893e-05, 0.05874751, 8.886500000000001e-05, 1, 0.0, 0, 0, '2017-11-26T08:55:00Z',
                 1511686500000, 8.893e-05, None, None, None],
             ['2017-11-26T09:00:00Z', 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05,
                 0.7039405, 8.885e-05, 0, 0.0, 0, 0, '2017-11-26T09:00:00Z', 1511686800000,
                 None, None, None, None]
             ])


def test_api_pair_history(botclient, mocker):
    _ftbot, client = botclient
    timeframe = '5m'
    lfm = mocker.patch('freqtrade.strategy.interface.IStrategy.load_freqAI_model')
    # No pair
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?timeframe={timeframe}"
                    f"&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}")
    assert_response(rc, 422)

    # No Timeframe
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC"
                    f"&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}")
    assert_response(rc, 422)

    # No timerange
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    f"&strategy={CURRENT_TEST_STRATEGY}")
    assert_response(rc, 422)

    # No strategy
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&timerange=20180111-20180112")
    assert_response(rc, 422)

    # Invalid strategy
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}11")
    assert_response(rc, 502)

    # Working
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    f"&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}")
    assert_response(rc, 200)
    result = rc.json()
    assert result['length'] == 289
    assert len(result['data']) == result['length']
    assert 'columns' in result
    assert 'data' in result
    data = result['data']
    assert len(data) == 289
    # analyed DF has 30 columns
    assert len(result['columns']) == 30
    assert len(data[0]) == 30
    date_col_idx = [idx for idx, c in enumerate(result['columns']) if c == 'date'][0]
    rsi_col_idx = [idx for idx, c in enumerate(result['columns']) if c == 'rsi'][0]

    assert data[0][date_col_idx] == '2018-01-11T00:00:00Z'
    assert data[0][rsi_col_idx] is not None
    assert data[0][rsi_col_idx] > 0
    assert lfm.call_count == 1
    assert result['pair'] == 'UNITTEST/BTC'
    assert result['strategy'] == CURRENT_TEST_STRATEGY
    assert result['data_start'] == '2018-01-11 00:00:00+00:00'
    assert result['data_start_ts'] == 1515628800000
    assert result['data_stop'] == '2018-01-12 00:00:00+00:00'
    assert result['data_stop_ts'] == 1515715200000

    # No data found
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    f"&timerange=20200111-20200112&strategy={CURRENT_TEST_STRATEGY}")
    assert_response(rc, 502)
    assert rc.json()['detail'] == ("No data for UNITTEST/BTC, 5m in 20200111-20200112 found.")


def test_api_plot_config(botclient, mocker):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)
    assert rc.json() == {}

    ftbot.strategy.plot_config = {
        'main_plot': {'sma': {}},
        'subplots': {'RSI': {'rsi': {'color': 'red'}}}
    }
    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)
    assert rc.json() == ftbot.strategy.plot_config
    assert isinstance(rc.json()['main_plot'], dict)
    assert isinstance(rc.json()['subplots'], dict)

    ftbot.strategy.plot_config = {'main_plot': {'sma': {}}}
    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)

    assert isinstance(rc.json()['main_plot'], dict)
    assert isinstance(rc.json()['subplots'], dict)

    rc = client_get(client, f"{BASE_URI}/plot_config?strategy=freqai_test_classifier")
    assert_response(rc)
    res = rc.json()
    assert 'target_roi' in res['subplots']
    assert 'do_predict' in res['subplots']

    rc = client_get(client, f"{BASE_URI}/plot_config?strategy=HyperoptableStrategy")
    assert_response(rc)
    assert rc.json()['subplots'] == {}

    rc = client_get(client, f"{BASE_URI}/plot_config?strategy=NotAStrategy")
    assert_response(rc, 502)
    assert rc.json()['detail'] is not None

    mocker.patch('freqtrade.rpc.api_server.api_v1.get_rpc_optional', return_value=None)

    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)


def test_api_strategies(botclient, tmp_path):
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path

    rc = client_get(client, f"{BASE_URI}/strategies")

    assert_response(rc)

    assert rc.json() == {'strategies': [
        'HyperoptableStrategy',
        'HyperoptableStrategyV2',
        'InformativeDecoratorTest',
        'StrategyTestV2',
        'StrategyTestV3',
        'StrategyTestV3CustomEntryPrice',
        'StrategyTestV3Futures',
        'freqai_rl_test_strat',
        'freqai_test_classifier',
        'freqai_test_multimodel_classifier_strat',
        'freqai_test_multimodel_strat',
        'freqai_test_strat',
        'strategy_test_v3_recursive_issue'
    ]}


def test_api_strategy(botclient):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/strategy/{CURRENT_TEST_STRATEGY}")

    assert_response(rc)
    assert rc.json()['strategy'] == CURRENT_TEST_STRATEGY

    data = (Path(__file__).parents[1] / "strategy/strats/strategy_test_v3.py").read_text()
    assert rc.json()['code'] == data

    rc = client_get(client, f"{BASE_URI}/strategy/NoStrat")
    assert_response(rc, 404)

    # Disallow base64 strategies
    rc = client_get(client, f"{BASE_URI}/strategy/xx:cHJpbnQoImhlbGxvIHdvcmxkIik=")
    assert_response(rc, 500)


def test_api_exchanges(botclient):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/exchanges")
    assert_response(rc)
    response = rc.json()
    assert isinstance(response['exchanges'], list)
    assert len(response['exchanges']) > 20
    okx = [x for x in response['exchanges'] if x['name'] == 'okx'][0]
    assert okx == {
        "name": "okx",
        "valid": True,
        "supported": True,
        "comment": "",
        "trade_modes": [
            {
                "trading_mode": "spot",
                "margin_mode": ""
            },
            {
                "trading_mode": "futures",
                "margin_mode": "isolated"
            }
        ]
    }

    mexc = [x for x in response['exchanges'] if x['name'] == 'mexc'][0]
    assert mexc == {
        "name": "mexc",
        "valid": True,
        "supported": False,
        "comment": "",
        "trade_modes": [
                {
                    "trading_mode": "spot",
                    "margin_mode": ""
                }
            ]
    }


def test_api_freqaimodels(botclient, tmp_path, mocker):
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path
    mocker.patch(
        "freqtrade.resolvers.freqaimodel_resolver.FreqaiModelResolver.search_all_objects",
        return_value=[
            {'name': 'LightGBMClassifier'},
            {'name': 'LightGBMClassifierMultiTarget'},
            {'name': 'LightGBMRegressor'},
            {'name': 'LightGBMRegressorMultiTarget'},
            {'name': 'ReinforcementLearner'},
            {'name': 'ReinforcementLearner_multiproc'},
            {'name': 'SKlearnRandomForestClassifier'},
            {'name': 'XGBoostClassifier'},
            {'name': 'XGBoostRFClassifier'},
            {'name': 'XGBoostRFRegressor'},
            {'name': 'XGBoostRegressor'},
            {'name': 'XGBoostRegressorMultiTarget'},
        ])

    rc = client_get(client, f"{BASE_URI}/freqaimodels")

    assert_response(rc)

    assert rc.json() == {'freqaimodels': [
        'LightGBMClassifier',
        'LightGBMClassifierMultiTarget',
        'LightGBMRegressor',
        'LightGBMRegressorMultiTarget',
        'ReinforcementLearner',
        'ReinforcementLearner_multiproc',
        'SKlearnRandomForestClassifier',
        'XGBoostClassifier',
        'XGBoostRFClassifier',
        'XGBoostRFRegressor',
        'XGBoostRegressor',
        'XGBoostRegressorMultiTarget'
    ]}


def test_api_pairlists_available(botclient, tmp_path):
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path

    rc = client_get(client, f"{BASE_URI}/pairlists/available")

    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'

    ftbot.config['runmode'] = RunMode.WEBSERVER

    rc = client_get(client, f"{BASE_URI}/pairlists/available")
    assert_response(rc)
    response = rc.json()
    assert isinstance(response['pairlists'], list)
    assert len(response['pairlists']) > 0

    assert len([r for r in response['pairlists'] if r['name'] == 'AgeFilter']) == 1
    assert len([r for r in response['pairlists'] if r['name'] == 'VolumePairList']) == 1
    assert len([r for r in response['pairlists'] if r['name'] == 'StaticPairList']) == 1

    volumepl = [r for r in response['pairlists'] if r['name'] == 'VolumePairList'][0]
    assert volumepl['is_pairlist_generator'] is True
    assert len(volumepl['params']) > 1
    age_pl = [r for r in response['pairlists'] if r['name'] == 'AgeFilter'][0]
    assert age_pl['is_pairlist_generator'] is False
    assert len(volumepl['params']) > 2


def test_api_pairlists_evaluate(botclient, tmp_path, mocker):
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path

    rc = client_get(client, f"{BASE_URI}/pairlists/evaluate/randomJob")

    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'

    ftbot.config['runmode'] = RunMode.WEBSERVER

    rc = client_get(client, f"{BASE_URI}/pairlists/evaluate/randomJob")
    assert_response(rc, 404)
    assert rc.json()['detail'] == 'Job not found.'

    body = {
        "pairlists": [
            {"method": "StaticPairList", },
        ],
        "blacklist": [
        ],
        "stake_currency": "BTC"
    }
    # Fail, already running
    ApiBG.pairlist_running = True
    rc = client_post(client, f"{BASE_URI}/pairlists/evaluate", body)
    assert_response(rc, 400)
    assert rc.json()['detail'] == 'Pairlist evaluation is already running.'

    # should start the run
    ApiBG.pairlist_running = False
    rc = client_post(client, f"{BASE_URI}/pairlists/evaluate", body)
    assert_response(rc)
    assert rc.json()['status'] == 'Pairlist evaluation started in background.'
    job_id = rc.json()['job_id']

    rc = client_get(client, f"{BASE_URI}/background/RandomJob")
    assert_response(rc, 404)
    assert rc.json()['detail'] == 'Job not found.'

    rc = client_get(client, f"{BASE_URI}/background/{job_id}")
    assert_response(rc)
    response = rc.json()
    assert response['job_id'] == job_id
    assert response['job_category'] == 'pairlist'

    rc = client_get(client, f"{BASE_URI}/pairlists/evaluate/{job_id}")
    assert_response(rc)
    response = rc.json()
    assert response['result']['whitelist'] == ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC']
    assert response['result']['length'] == 4

    # Restart with additional filter, reducing the list to 2
    body['pairlists'].append({"method": "OffsetFilter", "number_assets": 2})
    rc = client_post(client, f"{BASE_URI}/pairlists/evaluate", body)
    assert_response(rc)
    assert rc.json()['status'] == 'Pairlist evaluation started in background.'
    job_id = rc.json()['job_id']

    rc = client_get(client, f"{BASE_URI}/pairlists/evaluate/{job_id}")
    assert_response(rc)
    response = rc.json()
    assert response['result']['whitelist'] == ['ETH/BTC', 'LTC/BTC', ]
    assert response['result']['length'] == 2
    # Patch __run_pairlists
    plm = mocker.patch('freqtrade.rpc.api_server.api_background_tasks.__run_pairlist',
                       return_value=None)
    body = {
        "pairlists": [
            {"method": "StaticPairList", },
        ],
        "blacklist": [
        ],
        "stake_currency": "BTC",
        "exchange": "randomExchange",
        "trading_mode": "futures",
        "margin_mode": "isolated",
    }
    rc = client_post(client, f"{BASE_URI}/pairlists/evaluate", body)
    assert_response(rc)
    assert plm.call_count == 1
    call_config = plm.call_args_list[0][0][1]
    assert call_config['exchange']['name'] == 'randomExchange'
    assert call_config['trading_mode'] == 'futures'
    assert call_config['margin_mode'] == 'isolated'


def test_list_available_pairs(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/available_pairs")

    assert_response(rc)
    assert rc.json()['length'] == 12
    assert isinstance(rc.json()['pairs'], list)

    rc = client_get(client, f"{BASE_URI}/available_pairs?timeframe=5m")
    assert_response(rc)
    assert rc.json()['length'] == 12

    rc = client_get(client, f"{BASE_URI}/available_pairs?stake_currency=ETH")
    assert_response(rc)
    assert rc.json()['length'] == 1
    assert rc.json()['pairs'] == ['XRP/ETH']
    assert len(rc.json()['pair_interval']) == 2

    rc = client_get(client, f"{BASE_URI}/available_pairs?stake_currency=ETH&timeframe=5m")
    assert_response(rc)
    assert rc.json()['length'] == 1
    assert rc.json()['pairs'] == ['XRP/ETH']
    assert len(rc.json()['pair_interval']) == 1

    ftbot.config['trading_mode'] = 'futures'
    rc = client_get(
        client, f"{BASE_URI}/available_pairs?timeframe=1h")
    assert_response(rc)
    assert rc.json()['length'] == 1
    assert rc.json()['pairs'] == ['XRP/USDT:USDT']

    rc = client_get(
        client, f"{BASE_URI}/available_pairs?timeframe=1h&candletype=mark")
    assert_response(rc)
    assert rc.json()['length'] == 2
    assert rc.json()['pairs'] == ['UNITTEST/USDT:USDT', 'XRP/USDT:USDT']
    assert len(rc.json()['pair_interval']) == 2


def test_sysinfo(botclient):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/sysinfo")
    assert_response(rc)
    result = rc.json()
    assert 'cpu_pct' in result
    assert 'ram_pct' in result


def test_api_backtesting(botclient, mocker, fee, caplog, tmp_path):
    try:
        ftbot, client = botclient
        mocker.patch(f'{EXMS}.get_fee', fee)

        rc = client_get(client, f"{BASE_URI}/backtest")
        # Backtest prevented in default mode
        assert_response(rc, 503)
        assert rc.json()['detail'] == 'Bot is not in the correct state.'

        ftbot.config['runmode'] = RunMode.WEBSERVER
        # Backtesting not started yet
        rc = client_get(client, f"{BASE_URI}/backtest")
        assert_response(rc)

        result = rc.json()
        assert result['status'] == 'not_started'
        assert not result['running']
        assert result['status_msg'] == 'Backtest not yet executed'
        assert result['progress'] == 0

        # Reset backtesting
        rc = client_delete(client, f"{BASE_URI}/backtest")
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'reset'
        assert not result['running']
        assert result['status_msg'] == 'Backtest reset'
        ftbot.config['export'] = 'trades'
        ftbot.config['backtest_cache'] = 'day'
        ftbot.config['user_data_dir'] = tmp_path
        ftbot.config['exportfilename'] = tmp_path / "backtest_results"
        ftbot.config['exportfilename'].mkdir()

        # start backtesting
        data = {
            "strategy": CURRENT_TEST_STRATEGY,
            "timeframe": "5m",
            "timerange": "20180110-20180111",
            "max_open_trades": 3,
            "stake_amount": 100,
            "dry_run_wallet": 1000,
            "enable_protections": False
        }
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert_response(rc)
        result = rc.json()

        assert result['status'] == 'running'
        assert result['progress'] == 0
        assert result['running']
        assert result['status_msg'] == 'Backtest started'

        rc = client_get(client, f"{BASE_URI}/backtest")
        assert_response(rc)

        result = rc.json()
        assert result['status'] == 'ended'
        assert not result['running']
        assert result['status_msg'] == 'Backtest ended'
        assert result['progress'] == 1
        assert result['backtest_result']

        rc = client_get(client, f"{BASE_URI}/backtest/abort")
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'not_running'
        assert not result['running']
        assert result['status_msg'] == 'Backtest ended'

        # Simulate running backtest
        ApiBG.bgtask_running = True
        rc = client_get(client, f"{BASE_URI}/backtest/abort")
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'stopping'
        assert not result['running']
        assert result['status_msg'] == 'Backtest ended'

        # Get running backtest...
        rc = client_get(client, f"{BASE_URI}/backtest")
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'running'
        assert result['running']
        assert result['step'] == "backtest"
        assert result['status_msg'] == "Backtest running"

        # Try delete with task still running
        rc = client_delete(client, f"{BASE_URI}/backtest")
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'running'

        # Post to backtest that's still running
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert_response(rc, 502)
        result = rc.json()
        assert 'Bot Background task already running' in result['error']

        ApiBG.bgtask_running = False

        # Rerun backtest (should get previous result)
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert_response(rc)
        result = rc.json()
        assert log_has_re('Reusing result of previous backtest.*', caplog)

        data['stake_amount'] = 101

        mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest_one_strategy',
                     side_effect=DependencyException('DeadBeef'))
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert log_has("Backtesting caused an error: DeadBeef", caplog)

        rc = client_get(client, f"{BASE_URI}/backtest")
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'error'
        assert 'Backtest failed' in result['status_msg']

        # Delete backtesting to avoid leakage since the backtest-object may stick around.
        rc = client_delete(client, f"{BASE_URI}/backtest")
        assert_response(rc)

        result = rc.json()
        assert result['status'] == 'reset'
        assert not result['running']
        assert result['status_msg'] == 'Backtest reset'

        # Disallow base64 strategies
        data['strategy'] = "xx:cHJpbnQoImhlbGxvIHdvcmxkIik="
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert_response(rc, 500)
    finally:
        Backtesting.cleanup()


def test_api_backtest_history(botclient, mocker, testdatadir):
    ftbot, client = botclient
    mocker.patch('freqtrade.data.btanalysis._get_backtest_files',
                 return_value=[
                     testdatadir / 'backtest_results/backtest-result_multistrat.json',
                     testdatadir / 'backtest_results/backtest-result.json'
                     ])

    rc = client_get(client, f"{BASE_URI}/backtest/history")
    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'

    ftbot.config['user_data_dir'] = testdatadir
    ftbot.config['runmode'] = RunMode.WEBSERVER

    rc = client_get(client, f"{BASE_URI}/backtest/history")
    assert_response(rc)
    result = rc.json()
    assert len(result) == 3
    fn = result[0]['filename']
    assert fn == "backtest-result_multistrat"
    assert result[0]['notes'] == ''
    strategy = result[0]['strategy']
    rc = client_get(client, f"{BASE_URI}/backtest/history/result?filename={fn}&strategy={strategy}")
    assert_response(rc)
    result2 = rc.json()
    assert result2
    assert result2['status'] == 'ended'
    assert not result2['running']
    assert result2['progress'] == 1
    # Only one strategy loaded - even though we use multiresult
    assert len(result2['backtest_result']['strategy']) == 1
    assert result2['backtest_result']['strategy'][strategy]


def test_api_delete_backtest_history_entry(botclient, tmp_path: Path):
    ftbot, client = botclient

    # Create a temporary directory and file
    bt_results_base = tmp_path / "backtest_results"
    bt_results_base.mkdir()
    file_path = bt_results_base / "test.json"
    file_path.touch()
    meta_path = file_path.with_suffix('.meta.json')
    meta_path.touch()

    rc = client_delete(client, f"{BASE_URI}/backtest/history/randomFile.json")
    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'

    ftbot.config['user_data_dir'] = tmp_path
    ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_delete(client, f"{BASE_URI}/backtest/history/randomFile.json")
    assert rc.status_code == 404
    assert rc.json()['detail'] == 'File not found.'

    rc = client_delete(client, f"{BASE_URI}/backtest/history/{file_path.name}")
    assert rc.status_code == 200

    assert not file_path.exists()
    assert not meta_path.exists()


def test_api_patch_backtest_history_entry(botclient, tmp_path: Path):
    ftbot, client = botclient

    # Create a temporary directory and file
    bt_results_base = tmp_path / "backtest_results"
    bt_results_base.mkdir()
    file_path = bt_results_base / "test.json"
    file_path.touch()
    meta_path = file_path.with_suffix('.meta.json')
    with meta_path.open('w') as metafile:
        rapidjson.dump({
            CURRENT_TEST_STRATEGY: {
                "run_id": "6e542efc8d5e62cef6e5be0ffbc29be81a6e751d",
                "backtest_start_time": 1690176003}
            }, metafile)

    def read_metadata():
        with meta_path.open('r') as metafile:
            return rapidjson.load(metafile)

    rc = client_patch(client, f"{BASE_URI}/backtest/history/randomFile.json")
    assert_response(rc, 503)

    ftbot.config['user_data_dir'] = tmp_path
    ftbot.config['runmode'] = RunMode.WEBSERVER

    rc = client_patch(client, f"{BASE_URI}/backtest/history/randomFile.json", {
        "strategy": CURRENT_TEST_STRATEGY,
    })
    assert rc.status_code == 404

    # Nonexisting strategy
    rc = client_patch(client, f"{BASE_URI}/backtest/history/{file_path.name}", {
        "strategy": f"{CURRENT_TEST_STRATEGY}xxx",
    })
    assert rc.status_code == 400
    assert rc.json()['detail'] == 'Strategy not in metadata.'

    # no Notes
    rc = client_patch(client, f"{BASE_URI}/backtest/history/{file_path.name}", {
        "strategy": CURRENT_TEST_STRATEGY,
    })
    assert rc.status_code == 200
    res = rc.json()
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]['strategy'] == CURRENT_TEST_STRATEGY
    assert res[0]['notes'] == ''

    fileres = read_metadata()
    assert fileres[CURRENT_TEST_STRATEGY]['run_id'] == res[0]['run_id']
    assert fileres[CURRENT_TEST_STRATEGY]['notes'] == ''

    rc = client_patch(client, f"{BASE_URI}/backtest/history/{file_path.name}", {
        "strategy": CURRENT_TEST_STRATEGY,
        "notes": "FooBar",
    })
    assert rc.status_code == 200
    res = rc.json()
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]['strategy'] == CURRENT_TEST_STRATEGY
    assert res[0]['notes'] == 'FooBar'

    fileres = read_metadata()
    assert fileres[CURRENT_TEST_STRATEGY]['run_id'] == res[0]['run_id']
    assert fileres[CURRENT_TEST_STRATEGY]['notes'] == 'FooBar'


def test_health(botclient):
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/health")

    assert_response(rc)
    ret = rc.json()
    assert ret["last_process_ts"] is None
    assert ret["last_process"] is None


def test_api_ws_subscribe(botclient, mocker):
    _ftbot, client = botclient
    ws_url = f"/api/v1/message/ws?token={_TEST_WS_TOKEN}"

    sub_mock = mocker.patch('freqtrade.rpc.api_server.ws.WebSocketChannel.set_subscriptions')

    with client.websocket_connect(ws_url) as ws:
        ws.send_json({'type': 'subscribe', 'data': ['whitelist']})
        time.sleep(0.2)

    # Check call count is now 1 as we sent a valid subscribe request
    assert sub_mock.call_count == 1

    with client.websocket_connect(ws_url) as ws:
        ws.send_json({'type': 'subscribe', 'data': 'whitelist'})
        time.sleep(0.2)

    # Call count hasn't changed as the subscribe request was invalid
    assert sub_mock.call_count == 1


def test_api_ws_requests(botclient, caplog):
    caplog.set_level(logging.DEBUG)

    _ftbot, client = botclient
    ws_url = f"/api/v1/message/ws?token={_TEST_WS_TOKEN}"

    # Test whitelist request
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({"type": "whitelist", "data": None})
        response = ws.receive_json()

    assert log_has_re(r"Request of type whitelist from.+", caplog)
    assert response['type'] == "whitelist"

    # Test analyzed_df request
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({"type": "analyzed_df", "data": {}})
        response = ws.receive_json()

    assert log_has_re(r"Request of type analyzed_df from.+", caplog)
    assert response['type'] == "analyzed_df"

    caplog.clear()
    # Test analyzed_df request with data
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({"type": "analyzed_df", "data": {"limit": 100}})
        response = ws.receive_json()

    assert log_has_re(r"Request of type analyzed_df from.+", caplog)
    assert response['type'] == "analyzed_df"


def test_api_ws_send_msg(default_conf, mocker, caplog):
    try:
        caplog.set_level(logging.DEBUG)

        default_conf.update({"api_server": {"enabled": True,
                                            "listen_ip_address": "127.0.0.1",
                                            "listen_port": 8080,
                                            "CORS_origins": ['http://example.com'],
                                            "username": _TEST_USER,
                                            "password": _TEST_PASS,
                                            "ws_token": _TEST_WS_TOKEN
                                            }})
        mocker.patch('freqtrade.rpc.telegram.Telegram._init')
        mocker.patch('freqtrade.rpc.api_server.ApiServer.start_api')
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))

        # Start test client context manager to run lifespan events
        with TestClient(apiserver.app):
            # Test message is published on the Message Stream
            test_message = {"type": "status", "data": "test"}
            first_waiter = apiserver._message_stream._waiter
            apiserver.send_msg(test_message)
            assert first_waiter.result()[0] == test_message

            second_waiter = apiserver._message_stream._waiter
            apiserver.send_msg(test_message)
            assert first_waiter != second_waiter

    finally:
        ApiServer.shutdown()
        ApiServer.shutdown()
