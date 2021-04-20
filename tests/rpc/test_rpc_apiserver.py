"""
Unit test file for rpc/api_server.py
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.testclient import TestClient
from numpy import isnan
from requests.auth import _basic_auth_str

from freqtrade.__init__ import __version__
from freqtrade.exceptions import ExchangeError
from freqtrade.loggers import setup_logging, setup_logging_pre
from freqtrade.persistence import PairLocks, Trade
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server import ApiServer
from freqtrade.rpc.api_server.api_auth import create_token, get_user_from_token
from freqtrade.rpc.api_server.uvicorn_threaded import UvicornServer
from freqtrade.state import RunMode, State
from tests.conftest import (create_mock_trades, get_mock_coro, get_patched_freqtradebot, log_has,
                            log_has_re, patch_get_signal)


BASE_URI = "/api/v1"
_TEST_USER = "FreqTrader"
_TEST_PASS = "SuperSecurePassword1!"


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
                                        }})

    ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    mocker.patch('freqtrade.rpc.api_server.ApiServer.start_api', MagicMock())
    apiserver = ApiServer(rpc, default_conf)
    yield ftbot, TestClient(apiserver.app)
    # Cleanup ... ?


def client_post(client, url, data={}):
    return client.post(url,
                       data=data,
                       headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
                                'Origin': 'http://example.com',
                                'content-type': 'application/json'
                                })


def client_get(client, url):
    # Add fake Origin to ensure CORS kicks in
    return client.get(url, headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
                                    'Origin': 'http://example.com'})


def client_delete(client, url):
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
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/invalid_url")
    assert_response(rc, 404)
    assert rc.json() == {"detail": "Not Found"}


def test_api_ui_fallback(botclient):
    ftbot, client = botclient

    rc = client_get(client, "/favicon.ico")
    assert rc.status_code == 200

    rc = client_get(client, "/fallback_file.html")
    assert rc.status_code == 200
    assert '`freqtrade install-ui`' in rc.text

    # Forwarded to fallback_html or index.html (depending if it's installed or not)
    rc = client_get(client, "/something")
    assert rc.status_code == 200


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
    ftbot, client = botclient
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
    ftbot, client = botclient
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
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.api_server.webserver.ApiServer.start_api', MagicMock())
    apiserver = ApiServer(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)
    assert apiserver._config == default_conf


def test_api_UvicornServer(mocker):
    thread_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.threading.Thread')
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert thread_mock.call_count == 0

    s.install_signal_handlers()
    # Original implementation starts a thread - make sure that's not the case
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

    s.install_signal_handlers()
    # Original implementation starts a thread - make sure that's not the case
    assert serve_mock.call_count == 0

    # Fake started to avoid sleeping forever
    s.started = True
    s.run()
    assert serve_mock.call_count == 1


def test_api_UvicornServer_run_no_uvloop(mocker, import_fails):
    serve_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve',
                              get_mock_coro(None))
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert serve_mock.call_count == 0

    s.install_signal_handlers()
    # Original implementation starts a thread - make sure that's not the case
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
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())

    server_mock = MagicMock()
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer', server_mock)

    apiserver = ApiServer(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)

    assert server_mock.call_count == 1
    assert apiserver._config == default_conf
    apiserver.start_api()
    assert server_mock.call_count == 2
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

    # Test crashing API server
    caplog.clear()
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer',
                 MagicMock(side_effect=Exception))
    apiserver.start_api()
    assert log_has("Api server failed to start.", caplog)


def test_api_cleanup(default_conf, mocker, caplog):
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": 8080,
                                        "username": "TestUser",
                                        "password": "testPass",
                                        }})
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())

    server_mock = MagicMock()
    server_mock.cleanup = MagicMock()
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer', server_mock)

    apiserver = ApiServer(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)

    apiserver.cleanup()
    assert apiserver._server.cleanup.call_count == 1
    assert log_has("Stopping API Server", caplog)


def test_api_reloadconf(botclient):
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/reload_config")
    assert_response(rc)
    assert rc.json() == {'status': 'Reloading config ...'}
    assert ftbot.state == State.RELOAD_CONFIG


def test_api_stopbuy(botclient):
    ftbot, client = botclient
    assert ftbot.config['max_open_trades'] != 0

    rc = client_post(client, f"{BASE_URI}/stopbuy")
    assert_response(rc)
    assert rc.json() == {'status': 'No more buy will occur from now. Run /reload_config to reset.'}
    assert ftbot.config['max_open_trades'] == 0


def test_api_balance(botclient, mocker, rpc_balance):
    ftbot, client = botclient

    ftbot.config['dry_run'] = False
    mocker.patch('freqtrade.exchange.Exchange.get_balances', return_value=rpc_balance)
    mocker.patch('freqtrade.exchange.Exchange.get_valid_pair_combination',
                 side_effect=lambda a, b: f"{a}/{b}")
    ftbot.wallets.update()

    rc = client_get(client, f"{BASE_URI}/balance")
    assert_response(rc)
    assert "currencies" in rc.json()
    assert len(rc.json()["currencies"]) == 5
    assert rc.json()['currencies'][0] == {
        'currency': 'BTC',
        'free': 12.0,
        'balance': 12.0,
        'used': 0.0,
        'est_stake': 12.0,
        'stake': 'BTC',
    }


def test_api_count(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
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
    create_mock_trades(fee)
    rc = client_get(client, f"{BASE_URI}/count")
    assert_response(rc)
    assert rc.json()["current"] == 4
    assert rc.json()["max"] == 1

    ftbot.config['max_open_trades'] = float('inf')
    rc = client_get(client, f"{BASE_URI}/count")
    assert rc.json()["max"] == -1


def test_api_locks(botclient):
    ftbot, client = botclient

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
                     data='{"pair": "XRP/BTC"}')
    assert_response(rc)
    assert rc.json()['lock_count'] == 0


def test_api_show_config(botclient, mocker):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))

    rc = client_get(client, f"{BASE_URI}/show_config")
    assert_response(rc)
    assert 'dry_run' in rc.json()
    assert rc.json()['exchange'] == 'binance'
    assert rc.json()['timeframe'] == '5m'
    assert rc.json()['timeframe_ms'] == 300000
    assert rc.json()['timeframe_min'] == 5
    assert rc.json()['state'] == 'running'
    assert rc.json()['bot_name'] == 'freqtrade'
    assert not rc.json()['trailing_stop']
    assert 'bid_strategy' in rc.json()
    assert 'ask_strategy' in rc.json()


def test_api_daily(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
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
    assert rc.json()['data'][0]['date'] == str(datetime.utcnow().date())


def test_api_trades(botclient, mocker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/trades")
    assert_response(rc)
    assert len(rc.json()) == 3
    assert rc.json()['trades_count'] == 0
    assert rc.json()['total_trades'] == 0

    create_mock_trades(fee)
    Trade.query.session.flush()

    rc = client_get(client, f"{BASE_URI}/trades")
    assert_response(rc)
    assert len(rc.json()['trades']) == 2
    assert rc.json()['trades_count'] == 2
    assert rc.json()['total_trades'] == 2
    rc = client_get(client, f"{BASE_URI}/trades?limit=1")
    assert_response(rc)
    assert len(rc.json()['trades']) == 1
    assert rc.json()['trades_count'] == 1
    assert rc.json()['total_trades'] == 2


def test_api_trade_single(botclient, mocker, fee, ticker, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets),
        fetch_ticker=ticker,
    )
    rc = client_get(client, f"{BASE_URI}/trade/3")
    assert_response(rc, 404)
    assert rc.json()['detail'] == 'Trade not found.'

    create_mock_trades(fee)
    Trade.query.session.flush()

    rc = client_get(client, f"{BASE_URI}/trade/3")
    assert_response(rc)
    assert rc.json()['trade_id'] == 3


def test_api_delete_trade(botclient, mocker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets),
        cancel_order=cancel_mock,
        cancel_stoploss_order=stoploss_mock,
    )
    rc = client_delete(client, f"{BASE_URI}/trades/1")
    # Error - trade won't exist yet.
    assert_response(rc, 502)

    create_mock_trades(fee)
    Trade.query.session.flush()
    ftbot.strategy.order_types['stoploss_on_exchange'] = True
    trades = Trade.query.all()
    trades[1].stoploss_order_id = '1234'
    assert len(trades) > 2

    rc = client_delete(client, f"{BASE_URI}/trades/1")
    assert_response(rc)
    assert rc.json()['result_msg'] == 'Deleted trade 1. Closed 1 open orders.'
    assert len(trades) - 1 == len(Trade.query.all())
    assert cancel_mock.call_count == 1

    cancel_mock.reset_mock()
    rc = client_delete(client, f"{BASE_URI}/trades/1")
    # Trade is gone now.
    assert_response(rc, 502)
    assert cancel_mock.call_count == 0

    assert len(trades) - 1 == len(Trade.query.all())
    rc = client_delete(client, f"{BASE_URI}/trades/2")
    assert_response(rc)
    assert rc.json()['result_msg'] == 'Deleted trade 2. Closed 2 open orders.'
    assert len(trades) - 2 == len(Trade.query.all())
    assert stoploss_mock.call_count == 1


def test_api_logs(botclient):
    ftbot, client = botclient
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
    assert rc1.json()['log_count'] == 5
    assert len(rc1.json()['logs']) == rc1.json()['log_count']


def test_api_edge_disabled(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/edge")
    assert_response(rc, 502)
    assert rc.json() == {"error": "Error querying /api/v1/edge: Edge is not enabled."}


@pytest.mark.usefixtures("init_persistence")
def test_api_profit(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc, 200)
    assert rc.json()['trade_count'] == 0

    create_mock_trades(fee)
    # Simulate fulfilled LIMIT_BUY order for trade

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc)
    assert rc.json() == {'avg_duration': ANY,
                         'best_pair': 'XRP/BTC',
                         'best_rate': 1.0,
                         'first_trade_date': ANY,
                         'first_trade_timestamp': ANY,
                         'latest_trade_date': '5 minutes ago',
                         'latest_trade_timestamp': ANY,
                         'profit_all_coin': -44.0631579,
                         'profit_all_fiat': -543959.6842755,
                         'profit_all_percent_mean': -66.41,
                         'profit_all_ratio_mean': -0.6641100666666667,
                         'profit_all_percent_sum': -398.47,
                         'profit_all_ratio_sum': -3.9846604,
                         'profit_closed_coin': 0.00073913,
                         'profit_closed_fiat': 9.124559849999999,
                         'profit_closed_ratio_mean': 0.0075,
                         'profit_closed_percent_mean': 0.75,
                         'profit_closed_ratio_sum': 0.015,
                         'profit_closed_percent_sum': 1.5,
                         'trade_count': 6,
                         'closed_trade_count': 2,
                         'winning_trades': 2,
                         'losing_trades': 0,
                         }


@pytest.mark.usefixtures("init_persistence")
def test_api_stats(botclient, mocker, ticker, fee, markets,):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    rc = client_get(client, f"{BASE_URI}/stats")
    assert_response(rc, 200)
    assert 'durations' in rc.json()
    assert 'sell_reasons' in rc.json()

    create_mock_trades(fee)

    rc = client_get(client, f"{BASE_URI}/stats")
    assert_response(rc, 200)
    assert 'durations' in rc.json()
    assert 'sell_reasons' in rc.json()

    assert 'wins' in rc.json()['durations']
    assert 'losses' in rc.json()['durations']
    assert 'draws' in rc.json()['durations']


def test_api_performance(botclient, mocker, ticker, fee):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))

    trade = Trade(
        pair='LTC/ETH',
        amount=1,
        exchange='binance',
        stake_amount=1,
        open_rate=0.245441,
        open_order_id="123456",
        is_open=False,
        fee_close=fee.return_value,
        fee_open=fee.return_value,
        close_rate=0.265441,

    )
    trade.close_profit = trade.calc_profit_ratio()
    Trade.query.session.add(trade)

    trade = Trade(
        pair='XRP/ETH',
        amount=5,
        stake_amount=1,
        exchange='binance',
        open_rate=0.412,
        open_order_id="123456",
        is_open=False,
        fee_close=fee.return_value,
        fee_open=fee.return_value,
        close_rate=0.391
    )
    trade.close_profit = trade.calc_profit_ratio()
    Trade.query.session.add(trade)
    Trade.query.session.flush()

    rc = client_get(client, f"{BASE_URI}/performance")
    assert_response(rc)
    assert len(rc.json()) == 2
    assert rc.json() == [{'count': 1, 'pair': 'LTC/ETH', 'profit': 7.61},
                         {'count': 1, 'pair': 'XRP/ETH', 'profit': -5.57}]


def test_api_status(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        fetch_order=MagicMock(return_value={}),
    )

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc, 200)
    assert rc.json() == []
    create_mock_trades(fee)

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
        'current_profit': ANY,
        'current_profit_pct': ANY,
        'current_profit_abs': ANY,
        'profit_ratio': ANY,
        'profit_pct': ANY,
        'profit_abs': ANY,
        'profit_fiat': ANY,
        'current_rate': 1.099e-05,
        'open_date': ANY,
        'open_timestamp': ANY,
        'open_order': None,
        'open_rate': 0.123,
        'pair': 'ETH/BTC',
        'stake_amount': 0.001,
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
        'max_rate': ANY,
        'min_rate': ANY,
        'open_order_id': 'dry_run_buy_12345',
        'open_rate_requested': ANY,
        'open_trade_value': 15.1668225,
        'sell_reason': None,
        'sell_order_status': None,
        'strategy': 'DefaultStrategy',
        'timeframe': 5,
        'exchange': 'binance',
    }

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_sell_rate',
                 MagicMock(side_effect=ExchangeError("Pair 'ETH/BTC' not available")))

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc)
    resp_values = rc.json()
    assert len(resp_values) == 4
    assert isnan(resp_values[0]['profit_abs'])


def test_api_version(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc)
    assert rc.json() == {"version": __version__}


def test_api_blacklist(botclient, mocker):
    ftbot, client = botclient

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
                     data='{"blacklist": ["ETH/BTC"]}')
    assert_response(rc)
    assert rc.json() == {"blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC"],
                         "blacklist_expanded": ["ETH/BTC"],
                         "length": 3,
                         "method": ["StaticPairList"],
                         "errors": {},
                         }

    rc = client_post(client, f"{BASE_URI}/blacklist",
                     data='{"blacklist": ["XRP/.*"]}')
    assert_response(rc)
    assert rc.json() == {"blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC", "XRP/.*"],
                         "blacklist_expanded": ["ETH/BTC", "XRP/BTC"],
                         "length": 4,
                         "method": ["StaticPairList"],
                         "errors": {},
                         }


def test_api_whitelist(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/whitelist")
    assert_response(rc)
    assert rc.json() == {
        "whitelist": ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC'],
        "length": 4,
        "method": ["StaticPairList"]
        }


def test_api_forcebuy(botclient, mocker, fee):
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/forcebuy",
                     data='{"pair": "ETH/BTC"}')
    assert_response(rc, 502)
    assert rc.json() == {"error": "Error querying /api/v1/forcebuy: Forcebuy not enabled."}

    # enable forcebuy
    ftbot.config['forcebuy_enable'] = True

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch("freqtrade.rpc.RPC._rpc_forcebuy", fbuy_mock)
    rc = client_post(client, f"{BASE_URI}/forcebuy",
                     data='{"pair": "ETH/BTC"}')
    assert_response(rc)
    assert rc.json() == {"status": "Error buying pair ETH/BTC."}

    # Test creating trade
    fbuy_mock = MagicMock(return_value=Trade(
        pair='ETH/ETH',
        amount=1,
        amount_requested=1,
        exchange='binance',
        stake_amount=1,
        open_rate=0.245441,
        open_order_id="123456",
        open_date=datetime.utcnow(),
        is_open=False,
        fee_close=fee.return_value,
        fee_open=fee.return_value,
        close_rate=0.265441,
        id=22,
        timeframe=5,
        strategy="DefaultStrategy"
    ))
    mocker.patch("freqtrade.rpc.RPC._rpc_forcebuy", fbuy_mock)

    rc = client_post(client, f"{BASE_URI}/forcebuy",
                     data='{"pair": "ETH/BTC"}')
    assert_response(rc)
    assert rc.json() == {
        'amount': 1,
        'amount_requested': 1,
        'trade_id': 22,
        'close_date': None,
        'close_timestamp': None,
        'close_rate': 0.265441,
        'open_date': ANY,
        'open_timestamp': ANY,
        'open_rate': 0.245441,
        'pair': 'ETH/ETH',
        'stake_amount': 1,
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
        'fee_close': 0.0025,
        'fee_close_cost': None,
        'fee_close_currency': None,
        'fee_open': 0.0025,
        'fee_open_cost': None,
        'fee_open_currency': None,
        'is_open': False,
        'max_rate': None,
        'min_rate': None,
        'open_order_id': '123456',
        'open_rate_requested': None,
        'open_trade_value': 0.24605460,
        'sell_reason': None,
        'sell_order_status': None,
        'strategy': 'DefaultStrategy',
        'timeframe': 5,
        'exchange': 'binance',
        }


def test_api_forcesell(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    patch_get_signal(ftbot, (True, False))

    rc = client_post(client, f"{BASE_URI}/forcesell",
                     data='{"tradeid": "1"}')
    assert_response(rc, 502)
    assert rc.json() == {"error": "Error querying /api/v1/forcesell: invalid argument"}

    ftbot.enter_positions()

    rc = client_post(client, f"{BASE_URI}/forcesell",
                     data='{"tradeid": "1"}')
    assert_response(rc)
    assert rc.json() == {'result': 'Created sell order for trade 1.'}


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
    ohlcv_history['buy'] = 0
    ohlcv_history.loc[1, 'buy'] = 1
    ohlcv_history['sell'] = 0

    ftbot.dataprovider._set_cached_df("XRP/BTC", timeframe, ohlcv_history)

    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}")
    assert_response(rc)
    assert 'strategy' in rc.json()
    assert rc.json()['strategy'] == 'DefaultStrategy'
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
    assert rc.json()['columns'] == ['date', 'open', 'high',
                                    'low', 'close', 'volume', 'sma', 'buy', 'sell',
                                    '__date_ts', '_buy_signal_open', '_sell_signal_open']
    assert 'pair' in rc.json()
    assert rc.json()['pair'] == 'XRP/BTC'

    assert 'data' in rc.json()
    assert len(rc.json()['data']) == amount

    assert (rc.json()['data'] ==
            [['2017-11-26 08:50:00', 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869,
              None, 0, 0, 1511686200000, None, None],
             ['2017-11-26 08:55:00', 8.88e-05, 8.942e-05, 8.88e-05,
                 8.893e-05, 0.05874751, 8.886500000000001e-05, 1, 0, 1511686500000, 8.88e-05, None],
             ['2017-11-26 09:00:00', 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05,
                 0.7039405, 8.885e-05, 0, 0, 1511686800000, None, None]

             ])


def test_api_pair_history(botclient, ohlcv_history):
    ftbot, client = botclient
    timeframe = '5m'

    # No pair
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?timeframe={timeframe}"
                    "&timerange=20180111-20180112&strategy=DefaultStrategy")
    assert_response(rc, 422)

    # No Timeframe
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC"
                    "&timerange=20180111-20180112&strategy=DefaultStrategy")
    assert_response(rc, 422)

    # No timerange
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&strategy=DefaultStrategy")
    assert_response(rc, 422)

    # No strategy
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&timerange=20180111-20180112")
    assert_response(rc, 422)

    # Working
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&timerange=20180111-20180112&strategy=DefaultStrategy")
    assert_response(rc, 200)
    assert rc.json()['length'] == 289
    assert len(rc.json()['data']) == rc.json()['length']
    assert 'columns' in rc.json()
    assert 'data' in rc.json()
    assert rc.json()['pair'] == 'UNITTEST/BTC'
    assert rc.json()['strategy'] == 'DefaultStrategy'
    assert rc.json()['data_start'] == '2018-01-11 00:00:00+00:00'
    assert rc.json()['data_start_ts'] == 1515628800000
    assert rc.json()['data_stop'] == '2018-01-12 00:00:00+00:00'
    assert rc.json()['data_stop_ts'] == 1515715200000

    # No data found
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&timerange=20200111-20200112&strategy=DefaultStrategy")
    assert_response(rc, 502)
    assert rc.json()['error'] == ("Error querying /api/v1/pair_history: "
                                  "No data for UNITTEST/BTC, 5m in 20200111-20200112 found.")


def test_api_plot_config(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)
    assert rc.json() == {}

    ftbot.strategy.plot_config = {'main_plot': {'sma': {}},
                                  'subplots': {'RSI': {'rsi': {'color': 'red'}}}}
    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)
    assert rc.json() == ftbot.strategy.plot_config
    assert isinstance(rc.json()['main_plot'], dict)


def test_api_strategies(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/strategies")

    assert_response(rc)
    assert rc.json() == {'strategies': [
        'DefaultStrategy',
        'HyperoptableStrategy',
        'TestStrategyLegacy'
        ]}


def test_api_strategy(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/strategy/DefaultStrategy")

    assert_response(rc)
    assert rc.json()['strategy'] == 'DefaultStrategy'

    data = (Path(__file__).parents[1] / "strategy/strats/default_strategy.py").read_text()
    assert rc.json()['code'] == data

    rc = client_get(client, f"{BASE_URI}/strategy/NoStrat")
    assert_response(rc, 404)


def test_list_available_pairs(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/available_pairs")

    assert_response(rc)
    assert rc.json()['length'] == 13
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
