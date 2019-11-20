"""
Unit test file for rpc/api_server.py
"""

from datetime import datetime
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest
from flask import Flask
from requests.auth import _basic_auth_str

from freqtrade.__init__ import __version__
from freqtrade.persistence import Trade
from freqtrade.rpc.api_server import BASE_URI, ApiServer
from freqtrade.state import State
from tests.conftest import get_patched_freqtradebot, log_has, patch_get_signal

_TEST_USER = "FreqTrader"
_TEST_PASS = "SuperSecurePassword1!"


@pytest.fixture
def botclient(default_conf, mocker):
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": "8080",
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
                       headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS)})


def client_get(client, url):
    return client.get(url, headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS)})


def assert_response(response, expected_code=200):
    assert response.status_code == expected_code
    assert response.content_type == "application/json"


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
    assert_response(rc)
    assert rc.json == {'status': 'pong'}

    # Don't send user/pass information
    rc = client.get(f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json == {'error': 'Unauthorized'}

    # Change only username
    ftbot.config['api_server']['username'] = "Ftrader"
    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json == {'error': 'Unauthorized'}

    # Change only password
    ftbot.config['api_server']['username'] = _TEST_USER
    ftbot.config['api_server']['password'] = "WrongPassword"
    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json == {'error': 'Unauthorized'}

    ftbot.config['api_server']['username'] = "Ftrader"
    ftbot.config['api_server']['password'] = "WrongPassword"

    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json == {'error': 'Unauthorized'}


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
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.api_server.ApiServer.run', MagicMock())

    apiserver = ApiServer(get_patched_freqtradebot(mocker, default_conf))
    assert apiserver._config == default_conf


def test_api_run(default_conf, mocker, caplog):
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": "8080"}})
    mocker.patch('freqtrade.rpc.telegram.Updater', MagicMock())
    mocker.patch('freqtrade.rpc.api_server.threading.Thread', MagicMock())

    server_mock = MagicMock()
    mocker.patch('freqtrade.rpc.api_server.make_server', server_mock)

    apiserver = ApiServer(get_patched_freqtradebot(mocker, default_conf))

    assert apiserver._config == default_conf
    apiserver.run()
    assert server_mock.call_count == 1
    assert server_mock.call_args_list[0][0][0] == "127.0.0.1"
    assert server_mock.call_args_list[0][0][1] == "8080"
    assert isinstance(server_mock.call_args_list[0][0][2], Flask)
    assert hasattr(apiserver, "srv")

    assert log_has("Starting HTTP Server at 127.0.0.1:8080", caplog)
    assert log_has("Starting Local Rest Server.", caplog)

    # Test binding to public
    caplog.clear()
    server_mock.reset_mock()
    apiserver._config.update({"api_server": {"enabled": True,
                                             "listen_ip_address": "0.0.0.0",
                                             "listen_port": "8089",
                                             "password": "",
                                             }})
    apiserver.run()

    assert server_mock.call_count == 1
    assert server_mock.call_args_list[0][0][0] == "0.0.0.0"
    assert server_mock.call_args_list[0][0][1] == "8089"
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
                                        "listen_port": "8080"}})
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

    rc = client_post(client, f"{BASE_URI}/reload_conf")
    assert_response(rc)
    assert rc.json == {'status': 'reloading config ...'}
    assert ftbot.state == State.RELOAD_CONF


def test_api_stopbuy(botclient):
    ftbot, client = botclient
    assert ftbot.config['max_open_trades'] != 0

    rc = client_post(client, f"{BASE_URI}/stopbuy")
    assert_response(rc)
    assert rc.json == {'status': 'No more buy will occur from now. Run /reload_conf to reset.'}
    assert ftbot.config['max_open_trades'] == 0


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
    mocker.patch('freqtrade.exchange.Exchange.get_valid_pair_combination',
                 side_effect=lambda a, b: f"{a}/{b}")

    rc = client_get(client, f"{BASE_URI}/balance")
    assert_response(rc)
    assert "currencies" in rc.json
    assert len(rc.json["currencies"]) == 5
    assert rc.json['currencies'][0] == {
        'currency': 'BTC',
        'free': 12.0,
        'balance': 12.0,
        'used': 0.0,
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
    rc = client_get(client, f"{BASE_URI}/count")
    assert_response(rc)

    assert rc.json["current"] == 0
    assert rc.json["max"] == 1.0

    # Create some test data
    ftbot.create_trades()
    rc = client_get(client, f"{BASE_URI}/count")
    assert_response(rc)
    assert rc.json["current"] == 1.0
    assert rc.json["max"] == 1.0


def test_api_show_config(botclient, mocker):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))

    rc = client_get(client, f"{BASE_URI}/show_config")
    assert_response(rc)
    assert 'dry_run' in rc.json
    assert rc.json['exchange'] == 'bittrex'
    assert rc.json['ticker_interval'] == '5m'
    assert not rc.json['trailing_stop']


def test_api_daily(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/daily")
    assert_response(rc)
    assert len(rc.json) == 7
    assert rc.json[0][0] == str(datetime.utcnow().date())


def test_api_edge_disabled(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    rc = client_get(client, f"{BASE_URI}/edge")
    assert_response(rc, 502)
    assert rc.json == {"error": "Error querying _edge: Edge is not enabled."}


def test_api_profit(botclient, mocker, ticker, fee, markets, limit_buy_order, limit_sell_order):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc, 502)
    assert len(rc.json) == 1
    assert rc.json == {"error": "Error querying _profit: no closed trade"}

    ftbot.create_trades()
    trade = Trade.query.first()

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)
    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc, 502)
    assert rc.json == {"error": "Error querying _profit: no closed trade"}

    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc)
    assert rc.json == {'avg_duration': '0:00:00',
                       'best_pair': 'ETH/BTC',
                       'best_rate': 6.2,
                       'first_trade_date': 'just now',
                       'latest_trade_date': 'just now',
                       'profit_all_coin': 6.217e-05,
                       'profit_all_fiat': 0,
                       'profit_all_percent': 6.2,
                       'profit_closed_coin': 6.217e-05,
                       'profit_closed_fiat': 0,
                       'profit_closed_percent': 6.2,
                       'trade_count': 1
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
    trade.close_profit = trade.calc_profit_percent()
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
    trade.close_profit = trade.calc_profit_percent()
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
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc, 200)
    assert rc.json == []

    ftbot.create_trades()
    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc)
    assert len(rc.json) == 1
    assert rc.json == [{'amount': 90.99181074,
                        'base_currency': 'BTC',
                        'close_date': None,
                        'close_date_hum': None,
                        'close_profit': None,
                        'close_rate': None,
                        'current_profit': -0.59,
                        'current_rate': 1.098e-05,
                        'initial_stop_loss': 0.0,
                        'initial_stop_loss_pct': None,
                        'open_date': ANY,
                        'open_date_hum': 'just now',
                        'open_order': '(limit buy rem=0.00000000)',
                        'open_rate': 1.099e-05,
                        'pair': 'ETH/BTC',
                        'stake_amount': 0.001,
                        'stop_loss': 0.0,
                        'stop_loss_pct': None,
                        'trade_id': 1}]


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
                       "method": ["StaticPairList"]}

    # Add ETH/BTC to blacklist
    rc = client_post(client, f"{BASE_URI}/blacklist",
                     data='{"blacklist": ["ETH/BTC"]}')
    assert_response(rc)
    assert rc.json == {"blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC"],
                       "length": 3,
                       "method": ["StaticPairList"]}


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
    ftbot.config["forcebuy_enable"] = True

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
                       'close_date': None,
                       'close_date_hum': None,
                       'close_rate': 0.265441,
                       'initial_stop_loss': None,
                       'initial_stop_loss_pct': None,
                       'open_date': ANY,
                       'open_date_hum': 'just now',
                       'open_rate': 0.245441,
                       'pair': 'ETH/ETH',
                       'stake_amount': 1,
                       'stop_loss': None,
                       'stop_loss_pct': None,
                       'trade_id': None}


def test_api_forcesell(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    patch_get_signal(ftbot, (True, False))

    rc = client_post(client, f"{BASE_URI}/forcesell",
                     data='{"tradeid": "1"}')
    assert_response(rc, 502)
    assert rc.json == {"error": "Error querying _forcesell: invalid argument"}

    ftbot.create_trades()

    rc = client_post(client, f"{BASE_URI}/forcesell",
                     data='{"tradeid": "1"}')
    assert_response(rc)
    assert rc.json == {'result': 'Created sell order for trade 1.'}
