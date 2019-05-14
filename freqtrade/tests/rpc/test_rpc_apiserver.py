"""
Unit test file for rpc/api_server.py
"""

from datetime import datetime
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest

from freqtrade.__init__ import __version__
from freqtrade.persistence import Trade
from freqtrade.rpc.api_server import ApiServer
from freqtrade.state import State
from freqtrade.tests.conftest import (get_patched_freqtradebot, log_has,
                                      patch_get_signal)


@pytest.fixture
def botclient(default_conf, mocker):
    default_conf.update({"api_server": {"enabled": True,
                                        "listen_ip_address": "127.0.0.1",
                                        "listen_port": "8080"}})
    ftbot = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.rpc.api_server.ApiServer.run', MagicMock())
    apiserver = ApiServer(ftbot)
    yield ftbot, apiserver.app.test_client()
    # Cleanup ... ?


def assert_response(response, expected_code=200):
    assert response.status_code == expected_code
    assert response.content_type == "application/json"


def test_api_not_found(botclient):
    ftbot, client = botclient

    rc = client.post("/invalid_url")
    assert_response(rc, 404)
    assert rc.json == {'status': 'error',
                       'reason': "There's no API call for http://localhost/invalid_url.",
                       'code': 404
                       }


def test_api_stop_workflow(botclient):
    ftbot, client = botclient
    assert ftbot.state == State.RUNNING
    rc = client.post("/stop")
    assert_response(rc)
    assert rc.json == {'status': 'stopping trader ...'}
    assert ftbot.state == State.STOPPED

    # Stop bot again
    rc = client.post("/stop")
    assert_response(rc)
    assert rc.json == {'status': 'already stopped'}

    # Start bot
    rc = client.post("/start")
    assert_response(rc)
    assert rc.json == {'status': 'starting trader ...'}
    assert ftbot.state == State.RUNNING

    # Call start again
    rc = client.post("/start")
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

    apiserver = ApiServer(get_patched_freqtradebot(mocker, default_conf))

    # Monkey patch flask app
    run_mock = MagicMock()
    apiserver.app = MagicMock()
    apiserver.app.run = run_mock

    assert apiserver._config == default_conf
    apiserver.run()
    assert run_mock.call_count == 1
    assert run_mock.call_args_list[0][1]["host"] == "127.0.0.1"
    assert run_mock.call_args_list[0][1]["port"] == "8080"

    assert log_has("Starting HTTP Server at 127.0.0.1:8080", caplog.record_tuples)
    assert log_has("Starting Local Rest Server", caplog.record_tuples)

    # Test binding to public
    caplog.clear()
    run_mock.reset_mock()
    apiserver._config.update({"api_server": {"enabled": True,
                                             "listen_ip_address": "0.0.0.0",
                                             "listen_port": "8089"}})
    apiserver.run()

    assert run_mock.call_count == 1
    assert run_mock.call_args_list[0][1]["host"] == "0.0.0.0"
    assert run_mock.call_args_list[0][1]["port"] == "8089"
    assert log_has("Starting HTTP Server at 0.0.0.0:8089", caplog.record_tuples)
    assert log_has("Starting Local Rest Server", caplog.record_tuples)
    assert log_has("SECURITY WARNING - Local Rest Server listening to external connections",
                   caplog.record_tuples)
    assert log_has("SECURITY WARNING - This is insecure please set to your loopback,"
                   "e.g 127.0.0.1 in config.json",
                   caplog.record_tuples)


def test_api_reloadconf(botclient):
    ftbot, client = botclient

    rc = client.post("/reload_conf")
    assert_response(rc)
    assert rc.json == {'status': 'reloading config ...'}
    assert ftbot.state == State.RELOAD_CONF


def test_api_stopbuy(botclient):
    ftbot, client = botclient
    assert ftbot.config['max_open_trades'] != 0

    rc = client.post("/stopbuy")
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

    rc = client.get("/balance")
    assert_response(rc)
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
    assert_response(rc)

    assert rc.json["current"] == 0
    assert rc.json["max"] == 1.0

    # Create some test data
    ftbot.create_trade()
    rc = client.get("/count")
    assert_response(rc)
    assert rc.json["current"] == 1.0
    assert rc.json["max"] == 1.0


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
    rc = client.get("/daily")
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
    rc = client.get("/edge")
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

    rc = client.get("/profit")
    assert_response(rc, 502)
    assert len(rc.json) == 1
    assert rc.json == {"error": "Error querying _profit: no closed trade"}

    ftbot.create_trade()
    trade = Trade.query.first()

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)
    rc = client.get("/profit")
    assert_response(rc, 502)
    assert rc.json == {"error": "Error querying _profit: no closed trade"}

    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False

    rc = client.get("/profit")
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


def test_api_performance(botclient, mocker, ticker, fee, markets):
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

    rc = client.get("/performance")
    assert_response(rc)
    assert len(rc.json) == 2
    assert rc.json == [{'count': 1, 'pair': 'LTC/ETH', 'profit': 7.61},
                       {'count': 1, 'pair': 'XRP/ETH', 'profit': -5.57}]


def test_api_status(botclient, mocker, ticker, fee, markets, limit_buy_order, limit_sell_order):
    ftbot, client = botclient
    patch_get_signal(ftbot, (True, False))
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    rc = client.get("/status")
    assert_response(rc, 502)
    assert rc.json == {'error': 'Error querying _status: no active trade'}

    ftbot.create_trade()
    rc = client.get("/status")
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

    rc = client.get("/version")
    assert_response(rc)
    assert rc.json == {"version": __version__}


def test_api_blacklist(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient

    rc = client.get("/blacklist")
    assert_response(rc)
    assert rc.json == {"blacklist": ["DOGE/BTC", "HOT/BTC"],
                       "length": 2,
                       "method": "StaticPairList"}

    # Add ETH/BTC to blacklist
    rc = client.post("/blacklist", data='{"blacklist": ["ETH/BTC"]}',
                     content_type='application/json')
    assert_response(rc)
    assert rc.json == {"blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC"],
                       "length": 3,
                       "method": "StaticPairList"}


def test_api_whitelist(botclient, mocker, ticker, fee, markets):
    ftbot, client = botclient

    rc = client.get("/whitelist")
    assert_response(rc)
    assert rc.json == {"whitelist": ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC'],
                       "length": 4,
                       "method": "StaticPairList"}
