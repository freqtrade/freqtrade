"""
Unit test file for rpc/api_server.py
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest
from flask import Flask
from requests.auth import _basic_auth_str

from freqtrade.__init__ import __version__
from freqtrade.loggers import setup_logging, setup_logging_pre
from freqtrade.persistence import PairLocks, Trade
from freqtrade.rpc.api_server import BASE_URI, ApiServer
from freqtrade.state import RunMode, State
from tests.conftest import create_mock_trades, get_patched_freqtradebot, log_has, patch_get_signal


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
    mocker.patch('freqtrade.rpc.api_server.ApiServer.run', MagicMock())
    apiserver = ApiServer(ftbot)
    yield ftbot, apiserver.app.test_client()
    # Cleanup ... ?


def client_post(client, url, data={}):
    return client.post(url,
                       content_type="application/json",
                       data=data,
                       headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
                                'Origin': 'http://example.com'})


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
    assert response.content_type == "application/json"
    if needs_cors:
        assert ('Access-Control-Allow-Credentials', 'true') in response.headers._list
        assert ('Access-Control-Allow-Origin', 'http://example.com') in response.headers._list


def test_api_not_found(botclient):
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/invalid_url")
    assert_response(rc, 404)
    assert rc.json == {"status": "error",
                       "reason": f"There's no API call for http://localhost{BASE_URI}/invalid_url.",
                       "code": 404
                       }


def test_api_unauthorized(botclient):
    ftbot, client = botclient
    rc = client.get(f"{BASE_URI}/ping")
    assert_response(rc, needs_cors=False)
    assert rc.json == {'status': 'pong'}

    # Don't send user/pass information
    rc = client.get(f"{BASE_URI}/version")
    assert_response(rc, 401, needs_cors=False)
    assert rc.json == {'error': 'Unauthorized'}

    # Change only username
    ftbot.config['api_server']['username'] = 'Ftrader'
    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json == {'error': 'Unauthorized'}

    # Change only password
    ftbot.config['api_server']['username'] = _TEST_USER
    ftbot.config['api_server']['password'] = 'WrongPassword'
    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json == {'error': 'Unauthorized'}

    ftbot.config['api_server']['username'] = 'Ftrader'
    ftbot.config['api_server']['password'] = 'WrongPassword'

    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json == {'error': 'Unauthorized'}


def test_api_token_login(botclient):
    ftbot, client = botclient
    rc = client_post(client, f"{BASE_URI}/token/login")
    assert_response(rc)
    assert 'access_token' in rc.json
    assert 'refresh_token' in rc.json

    # test Authentication is working with JWT tokens too
    rc = client.get(f"{BASE_URI}/count",
                    content_type="application/json",
                    headers={'Authorization': f'Bearer {rc.json["access_token"]}',
                             'Origin': 'http://example.com'})
    assert_response(rc)


def test_api_token_refresh(botclient):
    ftbot, client = botclient
    rc = client_post(client, f"{BASE_URI}/token/login")
    assert_response(rc)
    rc = client.post(f"{BASE_URI}/token/refresh",
                     content_type="application/json",
                     data=None,
                     headers={'Authorization': f'Bearer {rc.json["refresh_token"]}',
                              'Origin': 'http://example.com'})
    assert_response(rc)
    assert 'access_token' in rc.json
    assert 'refresh_token' not in rc.json


def test_api_stop_workflow(botclient):
    ftbot, client = botclient
    assert ftbot.state == State.RUNNING
    rc = client_post(client, f"{BASE_URI}/stop")
    assert_response(rc)
    assert rc.json == {'status': 'stopping trader ...'}
    assert ftbot.state == State.STOPPED

    # Stop bot again
    rc = client_post(client, f"{BASE_URI}/stop")
    assert_response(rc)
    assert rc.json == {'status': 'already stopped'}

    # Start bot
    rc = client_post(client, f"{BASE_URI}/start")
    assert_response(rc)
    assert rc.json == {'status': 'starting trader ...'}
    assert ftbot.state == State.RUNNING

    # Call start again
    rc = client_post(client, f"{BASE_URI}/start")
    assert_response(rc)
    assert rc.json == {'status': 'already running'}


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
    mocker.patch('freqtrade.rpc.api_server.ApiServer.run', MagicMock())

    apiserver = ApiServer(get_patched_freqtradebot(mocker, default_conf))
    assert apiserver._config == default_conf


def test_api_run(default_conf, mocker, caplog):
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": 8080,
                                        "username": "TestUser",
                                        "password": "testPass",
                                        }})
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.api_server.threading.Thread', MagicMock())

    server_mock = MagicMock()
    mocker.patch('freqtrade.rpc.api_server.make_server', server_mock)

    apiserver = ApiServer(get_patched_freqtradebot(mocker, default_conf))

    assert apiserver._config == default_conf
    apiserver.run()
    assert server_mock.call_count == 1
    assert server_mock.call_args_list[0][0][0] == "127.0.0.1"
    assert server_mock.call_args_list[0][0][1] == 8080
    assert isinstance(server_mock.call_args_list[0][0][2], Flask)
    assert hasattr(apiserver, "srv")

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
    apiserver.run()

    assert server_mock.call_count == 1
    assert server_mock.call_args_list[0][0][0] == "0.0.0.0"
    assert server_mock.call_args_list[0][0][1] == 8089
    assert isinstance(server_mock.call_args_list[0][0][2], Flask)
    assert log_has("Starting HTTP Server at 0.0.0.0:8089", caplog)
    assert log_has("Starting Local Rest Server.", caplog)
    assert log_has("SECURITY WARNING - Local Rest Server listening to external connections",
                   caplog)
    assert log_has("SECURITY WARNING - This is insecure please set to your loopback,"
                   "e.g 127.0.0.1 in config.json", caplog)
    assert log_has("SECURITY WARNING - No password for local REST Server defined. "
                   "Please make sure that this is intentional!", caplog)

    # Test crashing flask
    caplog.clear()
    mocker.patch('freqtrade.rpc.api_server.make_server', MagicMock(side_effect=Exception))
    apiserver.run()
    assert log_has("Api server failed to start.", caplog)


def test_api_cleanup(default_conf, mocker, caplog):
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": 8080,
                                        "username": "TestUser",
                                        "password": "testPass",
                                        }})
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.api_server.threading.Thread', MagicMock())
    mocker.patch('freqtrade.rpc.api_server.make_server', MagicMock())

    apiserver = ApiServer(get_patched_freqtradebot(mocker, default_conf))
    apiserver.run()
    stop_mock = MagicMock()
    stop_mock.shutdown = MagicMock()
    apiserver.srv = stop_mock

    apiserver.cleanup()
    assert stop_mock.shutdown.call_count == 1
    assert log_has("Stopping API Server", caplog)


def test_api_reloadconf(botclient):
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/reload_config")
    assert_response(rc)
    assert rc.json == {'status': 'Reloading config ...'}
    assert ftbot.state == State.RELOAD_CONFIG


def test_api_stopbuy(botclient):
    ftbot, client = botclient
    assert ftbot.config['max_open_trades'] != 0

    rc = client_post(client, f"{BASE_URI}/stopbuy")
    assert_response(rc)
    assert rc.json == {'status': 'No more buy will occur from now. Run /reload_config to reset.'}
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
    assert "currencies" in rc.json
    assert len(rc.json["currencies"]) == 5
    assert rc.json['currencies'][0] == {
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

    assert rc.json["current"] == 0
    assert rc.json["max"] == 1.0

    # Create some test data
    ftbot.enter_positions()
    rc = client_get(client, f"{BASE_URI}/count")
    assert_response(rc)
    assert rc.json["current"] == 1.0
    assert rc.json["max"] == 1.0


def test_api_locks(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/locks")
    assert_response(rc)

    assert 'locks' in rc.json

    assert rc.json['lock_count'] == 0
    assert rc.json['lock_count'] == len(rc.json['locks'])

    PairLocks.lock_pair('ETH/BTC', datetime.now(timezone.utc) + timedelta(minutes=4), 'randreason')
    PairLocks.lock_pair('XRP/BTC', datetime.now(timezone.utc) + timedelta(minutes=20), 'deadbeef')

    rc = client_get(client, f"{BASE_URI}/locks")
    assert_response(rc)

    assert rc.json['lock_count'] == 2
    assert rc.json['lock_count'] == len(rc.json['locks'])
    assert 'ETH/BTC' in (rc.json['locks'][0]['pair'], rc.json['locks'][1]['pair'])
    assert 'randreason' in (rc.json['locks'][0]['reason'], rc.json['locks'][1]['reason'])
    assert 'deadbeef' in (rc.json['locks'][0]['reason'], rc.json['locks'][1]['reason'])


def test_api_show_config(botclient, mocker):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))

    rc = client_get(client, f"{BASE_URI}/show_config")
    assert_response(rc)
    assert 'dry_run' in rc.json
    assert rc.json['exchange'] == 'bittrex'
    assert rc.json['timeframe'] == '5m'
    assert rc.json['timeframe_ms'] == 300000
    assert rc.json['timeframe_min'] == 5
    assert rc.json['state'] == 'running'
    assert not rc.json['trailing_stop']
    assert 'bid_strategy' in rc.json
    assert 'ask_strategy' in rc.json


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
    assert len(rc.json['data']) == 7
    assert rc.json['stake_currency'] == 'BTC'
    assert rc.json['fiat_display_currency'] == 'USD'
    assert rc.json['data'][0]['date'] == str(datetime.utcnow().date())


def test_api_trades(botclient, mocker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/trades")
    assert_response(rc)
    assert len(rc.json) == 2
    assert rc.json['trades_count'] == 0

    create_mock_trades(fee)

    rc = client_get(client, f"{BASE_URI}/trades")
    assert_response(rc)
    assert len(rc.json['trades']) == 2
    assert rc.json['trades_count'] == 2
    rc = client_get(client, f"{BASE_URI}/trades?limit=1")
    assert_response(rc)
    assert len(rc.json['trades']) == 1
    assert rc.json['trades_count'] == 1


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
    ftbot.strategy.order_types['stoploss_on_exchange'] = True
    trades = Trade.query.all()
    trades[1].stoploss_order_id = '1234'
    assert len(trades) > 2

    rc = client_delete(client, f"{BASE_URI}/trades/1")
    assert_response(rc)
    assert rc.json['result_msg'] == 'Deleted trade 1. Closed 1 open orders.'
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
    assert rc.json['result_msg'] == 'Deleted trade 2. Closed 2 open orders.'
    assert len(trades) - 2 == len(Trade.query.all())
    assert stoploss_mock.call_count == 1


def test_api_logs(botclient):
    ftbot, client = botclient
    rc = client_get(client, f"{BASE_URI}/logs")
    assert_response(rc)
    assert len(rc.json) == 2
    assert 'logs' in rc.json
    # Using a fixed comparison here would make this test fail!
    assert rc.json['log_count'] > 1
    assert len(rc.json['logs']) == rc.json['log_count']

    assert isinstance(rc.json['logs'][0], list)
    # date
    assert isinstance(rc.json['logs'][0][0], str)
    # created_timestamp
    assert isinstance(rc.json['logs'][0][1], float)
    assert isinstance(rc.json['logs'][0][2], str)
    assert isinstance(rc.json['logs'][0][3], str)
    assert isinstance(rc.json['logs'][0][4], str)

    rc = client_get(client, f"{BASE_URI}/logs?limit=5")
    assert_response(rc)
    assert len(rc.json) == 2
    assert 'logs' in rc.json
    # Using a fixed comparison here would make this test fail!
    assert rc.json['log_count'] == 5
    assert len(rc.json['logs']) == rc.json['log_count']


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
    assert rc.json == {"error": "Error querying _edge: Edge is not enabled."}


@pytest.mark.usefixtures("init_persistence")
def test_api_profit(botclient, mocker, ticker, fee, markets, limit_buy_order, limit_sell_order):
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
    assert rc.json['trade_count'] == 0

    ftbot.enter_positions()
    trade = Trade.query.first()

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)
    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc, 200)
    # One open trade
    assert rc.json['trade_count'] == 1
    assert rc.json['best_pair'] == ''
    assert rc.json['best_rate'] == 0

    trade = Trade.query.first()
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc)
    assert rc.json == {'avg_duration': '0:00:00',
                       'best_pair': 'ETH/BTC',
                       'best_rate': 6.2,
                       'first_trade_date': 'just now',
                       'first_trade_timestamp': ANY,
                       'latest_trade_date': 'just now',
                       'latest_trade_timestamp': ANY,
                       'profit_all_coin': 6.217e-05,
                       'profit_all_fiat': 0.76748865,
                       'profit_all_percent': 6.2,
                       'profit_all_percent_mean': 6.2,
                       'profit_all_ratio_mean': 0.06201058,
                       'profit_all_percent_sum': 6.2,
                       'profit_all_ratio_sum': 0.06201058,
                       'profit_closed_coin': 6.217e-05,
                       'profit_closed_fiat': 0.76748865,
                       'profit_closed_percent': 6.2,
                       'profit_closed_ratio_mean': 0.06201058,
                       'profit_closed_percent_mean': 6.2,
                       'profit_closed_ratio_sum': 0.06201058,
                       'profit_closed_percent_sum': 6.2,
                       'trade_count': 1,
                       'closed_trade_count': 1,
                       'winning_trades': 1,
                       'losing_trades': 0,
                       }


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
    Trade.session.add(trade)

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
    Trade.session.add(trade)
    Trade.session.flush()

    rc = client_get(client, f"{BASE_URI}/performance")
    assert_response(rc)
    assert len(rc.json) == 2
    assert rc.json == [{'count': 1, 'pair': 'LTC/ETH', 'profit': 7.61},
                       {'count': 1, 'pair': 'XRP/ETH', 'profit': -5.57}]


def test_api_status(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc, 200)
    assert rc.json == []

    ftbot.enter_positions()
    trades = Trade.get_open_trades()
    trades[0].open_order_id = None
    ftbot.exit_positions(trades)

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc)
    assert len(rc.json) == 1
    assert rc.json == [{'amount': 91.07468123,
                        'amount_requested': 91.07468123,
                        'base_currency': 'BTC',
                        'close_date': None,
                        'close_date_hum': None,
                        'close_timestamp': None,
                        'close_profit': None,
                        'close_profit_pct': None,
                        'close_profit_abs': None,
                        'close_rate': None,
                        'current_profit': -0.00408133,
                        'current_profit_pct': -0.41,
                        'current_profit_abs': -4.09e-06,
                        'profit_ratio': -0.00408133,
                        'profit_pct': -0.41,
                        'profit_abs': -4.09e-06,
                        'current_rate': 1.099e-05,
                        'open_date': ANY,
                        'open_date_hum': 'just now',
                        'open_timestamp': ANY,
                        'open_order': None,
                        'open_rate': 1.098e-05,
                        'pair': 'ETH/BTC',
                        'stake_amount': 0.001,
                        'stop_loss_abs': 9.882e-06,
                        'stop_loss_pct': -10.0,
                        'stop_loss_ratio': -0.1,
                        'stoploss_order_id': None,
                        'stoploss_last_update': ANY,
                        'stoploss_last_update_timestamp': ANY,
                        'initial_stop_loss_abs': 9.882e-06,
                        'initial_stop_loss_pct': -10.0,
                        'initial_stop_loss_ratio': -0.1,
                        'stoploss_current_dist': -1.1080000000000002e-06,
                        'stoploss_current_dist_ratio': -0.10081893,
                        'stoploss_current_dist_pct': -10.08,
                        'stoploss_entry_dist': -0.00010475,
                        'stoploss_entry_dist_ratio': -0.10448878,
                        'trade_id': 1,
                        'close_rate_requested': None,
                        'current_rate': 1.099e-05,
                        'fee_close': 0.0025,
                        'fee_close_cost': None,
                        'fee_close_currency': None,
                        'fee_open': 0.0025,
                        'fee_open_cost': None,
                        'fee_open_currency': None,
                        'open_date': ANY,
                        'is_open': True,
                        'max_rate': 1.099e-05,
                        'min_rate': 1.098e-05,
                        'open_order_id': None,
                        'open_rate_requested': 1.098e-05,
                        'open_trade_price': 0.0010025,
                        'sell_reason': None,
                        'sell_order_status': None,
                        'strategy': 'DefaultStrategy',
                        'timeframe': 5,
                        'exchange': 'bittrex',
                        }]


def test_api_version(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc)
    assert rc.json == {"version": __version__}


def test_api_blacklist(botclient, mocker):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/blacklist")
    assert_response(rc)
    assert rc.json == {"blacklist": ["DOGE/BTC", "HOT/BTC"],
                       "length": 2,
                       "method": ["StaticPairList"],
                       "errors": {},
                       }

    # Add ETH/BTC to blacklist
    rc = client_post(client, f"{BASE_URI}/blacklist",
                     data='{"blacklist": ["ETH/BTC"]}')
    assert_response(rc)
    assert rc.json == {"blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC"],
                       "length": 3,
                       "method": ["StaticPairList"],
                       "errors": {},
                       }


def test_api_whitelist(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/whitelist")
    assert_response(rc)
    assert rc.json == {"whitelist": ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC'],
                       "length": 4,
                       "method": ["StaticPairList"]}


def test_api_forcebuy(botclient, mocker, fee):
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/forcebuy",
                     data='{"pair": "ETH/BTC"}')
    assert_response(rc, 502)
    assert rc.json == {"error": "Error querying _forcebuy: Forcebuy not enabled."}

    # enable forcebuy
    ftbot.config['forcebuy_enable'] = True

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch("freqtrade.rpc.RPC._rpc_forcebuy", fbuy_mock)
    rc = client_post(client, f"{BASE_URI}/forcebuy",
                     data='{"pair": "ETH/BTC"}')
    assert_response(rc)
    assert rc.json == {"status": "Error buying pair ETH/BTC."}

    # Test creating trae
    fbuy_mock = MagicMock(return_value=Trade(
        pair='ETH/ETH',
        amount=1,
        amount_requested=1,
        exchange='bittrex',
        stake_amount=1,
        open_rate=0.245441,
        open_order_id="123456",
        open_date=datetime.utcnow(),
        is_open=False,
        fee_close=fee.return_value,
        fee_open=fee.return_value,
        close_rate=0.265441,
    ))
    mocker.patch("freqtrade.rpc.RPC._rpc_forcebuy", fbuy_mock)

    rc = client_post(client, f"{BASE_URI}/forcebuy",
                     data='{"pair": "ETH/BTC"}')
    assert_response(rc)
    assert rc.json == {'amount': 1,
                       'amount_requested': 1,
                       'trade_id': None,
                       'close_date': None,
                       'close_date_hum': None,
                       'close_timestamp': None,
                       'close_rate': 0.265441,
                       'open_date': ANY,
                       'open_date_hum': 'just now',
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
                       'open_trade_price': 0.24605460,
                       'sell_reason': None,
                       'sell_order_status': None,
                       'strategy': None,
                       'timeframe': None,
                       'exchange': 'bittrex',
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
    assert rc.json == {"error": "Error querying _forcesell: invalid argument"}

    ftbot.enter_positions()

    rc = client_post(client, f"{BASE_URI}/forcesell",
                     data='{"tradeid": "1"}')
    assert_response(rc)
    assert rc.json == {'result': 'Created sell order for trade 1.'}


def test_api_pair_candles(botclient, ohlcv_history):
    ftbot, client = botclient
    timeframe = '5m'
    amount = 2

    # No pair
    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?limit={amount}&timeframe={timeframe}")
    assert_response(rc, 400)

    # No timeframe
    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?pair=XRP%2FBTC")
    assert_response(rc, 400)

    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}")
    assert_response(rc)
    assert 'columns' in rc.json
    assert 'data_start_ts' in rc.json
    assert 'data_start' in rc.json
    assert 'data_stop' in rc.json
    assert 'data_stop_ts' in rc.json
    assert len(rc.json['data']) == 0
    ohlcv_history['sma'] = ohlcv_history['close'].rolling(2).mean()
    ohlcv_history['buy'] = 0
    ohlcv_history.loc[1, 'buy'] = 1
    ohlcv_history['sell'] = 0

    ftbot.dataprovider._set_cached_df("XRP/BTC", timeframe, ohlcv_history)

    rc = client_get(client,
                    f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}")
    assert_response(rc)
    assert 'strategy' in rc.json
    assert rc.json['strategy'] == 'DefaultStrategy'
    assert 'columns' in rc.json
    assert 'data_start_ts' in rc.json
    assert 'data_start' in rc.json
    assert 'data_stop' in rc.json
    assert 'data_stop_ts' in rc.json
    assert rc.json['data_start'] == '2017-11-26 08:50:00+00:00'
    assert rc.json['data_start_ts'] == 1511686200000
    assert rc.json['data_stop'] == '2017-11-26 08:55:00+00:00'
    assert rc.json['data_stop_ts'] == 1511686500000
    assert isinstance(rc.json['columns'], list)
    assert rc.json['columns'] == ['date', 'open', 'high',
                                  'low', 'close', 'volume', 'sma', 'buy', 'sell',
                                  '__date_ts', '_buy_signal_open', '_sell_signal_open']
    assert 'pair' in rc.json
    assert rc.json['pair'] == 'XRP/BTC'

    assert 'data' in rc.json
    assert len(rc.json['data']) == amount

    assert (rc.json['data'] ==
            [['2017-11-26 08:50:00', 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869,
              None, 0, 0, 1511686200000, None, None],
             ['2017-11-26 08:55:00', 8.88e-05, 8.942e-05, 8.88e-05,
                 8.893e-05, 0.05874751, 8.886500000000001e-05, 1, 0, 1511686500000, 8.88e-05, None]
             ])


def test_api_pair_history(botclient, ohlcv_history):
    ftbot, client = botclient
    timeframe = '5m'

    # No pair
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?timeframe={timeframe}"
                    "&timerange=20180111-20180112&strategy=DefaultStrategy")
    assert_response(rc, 400)

    # No Timeframe
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC"
                    "&timerange=20180111-20180112&strategy=DefaultStrategy")
    assert_response(rc, 400)

    # No timerange
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&strategy=DefaultStrategy")
    assert_response(rc, 400)

    # No strategy
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&timerange=20180111-20180112")
    assert_response(rc, 400)

    # Working
    rc = client_get(client,
                    f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                    "&timerange=20180111-20180112&strategy=DefaultStrategy")
    assert_response(rc, 200)
    assert rc.json['length'] == 289
    assert len(rc.json['data']) == rc.json['length']
    assert 'columns' in rc.json
    assert 'data' in rc.json
    assert rc.json['pair'] == 'UNITTEST/BTC'
    assert rc.json['strategy'] == 'DefaultStrategy'
    assert rc.json['data_start'] == '2018-01-11 00:00:00+00:00'
    assert rc.json['data_start_ts'] == 1515628800000
    assert rc.json['data_stop'] == '2018-01-12 00:00:00+00:00'
    assert rc.json['data_stop_ts'] == 1515715200000


def test_api_plot_config(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)
    assert rc.json == {}

    ftbot.strategy.plot_config = {'main_plot': {'sma': {}},
                                  'subplots': {'RSI': {'rsi': {'color': 'red'}}}}
    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)
    assert rc.json == ftbot.strategy.plot_config
    assert isinstance(rc.json['main_plot'], dict)


def test_api_strategies(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/strategies")

    assert_response(rc)
    assert rc.json == {'strategies': ['DefaultStrategy', 'TestStrategyLegacy']}


def test_api_strategy(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/strategy/DefaultStrategy")

    assert_response(rc)
    assert rc.json['strategy'] == 'DefaultStrategy'

    data = (Path(__file__).parents[1] / "strategy/strats/default_strategy.py").read_text()
    assert rc.json['code'] == data

    rc = client_get(client, f"{BASE_URI}/strategy/NoStrat")
    assert_response(rc, 404)


def test_list_available_pairs(botclient):
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/available_pairs")

    assert_response(rc)
    assert rc.json['length'] == 12
    assert isinstance(rc.json['pairs'], list)

    rc = client_get(client, f"{BASE_URI}/available_pairs?timeframe=5m")
    assert_response(rc)
    assert rc.json['length'] == 12

    rc = client_get(client, f"{BASE_URI}/available_pairs?stake_currency=ETH")
    assert_response(rc)
    assert rc.json['length'] == 1
    assert rc.json['pairs'] == ['XRP/ETH']
    assert len(rc.json['pair_interval']) == 2

    rc = client_get(client, f"{BASE_URI}/available_pairs?stake_currency=ETH&timeframe=5m")
    assert_response(rc)
    assert rc.json['length'] == 1
    assert rc.json['pairs'] == ['XRP/ETH']
    assert len(rc.json['pair_interval']) == 1
