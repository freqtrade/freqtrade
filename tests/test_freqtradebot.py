# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

import logging
import time
from copy import deepcopy
from unittest.mock import MagicMock, PropertyMock

import arrow
import pytest
import requests

from freqtrade import (DependencyException, InvalidOrderException,
                       OperationalException, TemporaryError, constants)
from freqtrade.data.dataprovider import DataProvider
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.rpc import RPCMessageType
from freqtrade.state import RunMode, State
from freqtrade.strategy.interface import SellCheckTuple, SellType
from freqtrade.worker import Worker
from tests.conftest import (get_patched_freqtradebot, get_patched_worker,
                            log_has, log_has_re, patch_edge, patch_exchange,
                            patch_get_signal, patch_wallet)


def patch_RPCManager(mocker) -> MagicMock:
    """
    This function mock RPC manager to avoid repeating this code in almost every tests
    :param mocker: mocker to patch RPCManager class
    :return: RPCManager.send_msg MagicMock to track if this method is called
    """
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    rpc_mock = mocker.patch('freqtrade.freqtradebot.RPCManager.send_msg', MagicMock())
    return rpc_mock


# Unit tests

def test_freqtradebot_state(mocker, default_conf, markets) -> None:
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert freqtrade.state is State.RUNNING

    default_conf.pop('initial_state')
    freqtrade = FreqtradeBot(default_conf)
    assert freqtrade.state is State.STOPPED


def test_worker_state(mocker, default_conf, markets) -> None:
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    worker = get_patched_worker(mocker, default_conf)
    assert worker.state is State.RUNNING

    default_conf.pop('initial_state')
    worker = Worker(args=None, config=default_conf)
    assert worker.state is State.STOPPED


def test_cleanup(mocker, default_conf, caplog) -> None:
    mock_cleanup = MagicMock()
    mocker.patch('freqtrade.persistence.cleanup', mock_cleanup)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.cleanup()
    assert log_has('Cleaning up modules ...', caplog)
    assert mock_cleanup.call_count == 1


def test_worker_running(mocker, default_conf, caplog) -> None:
    mock_throttle = MagicMock()
    mocker.patch('freqtrade.worker.Worker._throttle', mock_throttle)
    mocker.patch('freqtrade.persistence.Trade.stoploss_reinitialization', MagicMock())

    worker = get_patched_worker(mocker, default_conf)

    state = worker._worker(old_state=None)
    assert state is State.RUNNING
    assert log_has('Changing state to: RUNNING', caplog)
    assert mock_throttle.call_count == 1
    # Check strategy is loaded, and received a dataprovider object
    assert worker.freqtrade.strategy
    assert worker.freqtrade.strategy.dp
    assert isinstance(worker.freqtrade.strategy.dp, DataProvider)


def test_worker_stopped(mocker, default_conf, caplog) -> None:
    mock_throttle = MagicMock()
    mocker.patch('freqtrade.worker.Worker._throttle', mock_throttle)
    mock_sleep = mocker.patch('time.sleep', return_value=None)

    worker = get_patched_worker(mocker, default_conf)
    worker.state = State.STOPPED
    state = worker._worker(old_state=State.RUNNING)
    assert state is State.STOPPED
    assert log_has('Changing state to: STOPPED', caplog)
    assert mock_throttle.call_count == 0
    assert mock_sleep.call_count == 1


def test_throttle(mocker, default_conf, caplog) -> None:
    def throttled_func():
        return 42

    caplog.set_level(logging.DEBUG)
    worker = get_patched_worker(mocker, default_conf)

    start = time.time()
    result = worker._throttle(throttled_func, min_secs=0.1)
    end = time.time()

    assert result == 42
    assert end - start > 0.1
    assert log_has('Throttling throttled_func for 0.10 seconds', caplog)

    result = worker._throttle(throttled_func, min_secs=-1)
    assert result == 42


def test_throttle_with_assets(mocker, default_conf) -> None:
    def throttled_func(nb_assets=-1):
        return nb_assets

    worker = get_patched_worker(mocker, default_conf)

    result = worker._throttle(throttled_func, min_secs=0.1, nb_assets=666)
    assert result == 666

    result = worker._throttle(throttled_func, min_secs=0.1)
    assert result == -1


def test_order_dict_dry_run(default_conf, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balance=MagicMock(return_value=default_conf['stake_amount'] * 2)
    )
    conf = default_conf.copy()
    conf['runmode'] = RunMode.DRY_RUN
    conf['order_types'] = {
        'buy': 'market',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }

    freqtrade = FreqtradeBot(conf)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']

    caplog.clear()
    # is left untouched
    conf = default_conf.copy()
    conf['runmode'] = RunMode.DRY_RUN
    conf['order_types'] = {
        'buy': 'market',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re(".*stoploss_on_exchange .* dry-run", caplog)


def test_order_dict_live(default_conf, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balance=MagicMock(return_value=default_conf['stake_amount'] * 2)
    )
    conf = default_conf.copy()
    conf['runmode'] = RunMode.LIVE
    conf['order_types'] = {
        'buy': 'market',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }

    freqtrade = FreqtradeBot(conf)
    assert not log_has_re(".*stoploss_on_exchange .* dry-run", caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']

    caplog.clear()
    # is left untouched
    conf = default_conf.copy()
    conf['runmode'] = RunMode.LIVE
    conf['order_types'] = {
        'buy': 'market',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re(".*stoploss_on_exchange .* dry-run", caplog)


def test_get_trade_stake_amount(default_conf, ticker, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balance=MagicMock(return_value=default_conf['stake_amount'] * 2)
    )

    freqtrade = FreqtradeBot(default_conf)

    result = freqtrade._get_trade_stake_amount('ETH/BTC')
    assert result == default_conf['stake_amount']


def test_get_trade_stake_amount_no_stake_amount(default_conf, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf['stake_amount'] * 0.5)
    freqtrade = FreqtradeBot(default_conf)

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        freqtrade._get_trade_stake_amount('ETH/BTC')


def test_get_trade_stake_amount_unlimited_amount(default_conf,
                                                 ticker,
                                                 limit_buy_order,
                                                 fee,
                                                 markets,
                                                 mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf['stake_amount'])
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee
    )

    conf = deepcopy(default_conf)
    conf['stake_amount'] = constants.UNLIMITED_STAKE_AMOUNT
    conf['max_open_trades'] = 2

    freqtrade = FreqtradeBot(conf)
    patch_get_signal(freqtrade)

    # no open trades, order amount should be 'balance / max_open_trades'
    result = freqtrade._get_trade_stake_amount('ETH/BTC')
    assert result == default_conf['stake_amount'] / conf['max_open_trades']

    # create one trade, order amount should be 'balance / (max_open_trades - num_open_trades)'
    freqtrade.execute_buy('ETH/BTC', result)

    result = freqtrade._get_trade_stake_amount('LTC/BTC')
    assert result == default_conf['stake_amount'] / (conf['max_open_trades'] - 1)

    # create 2 trades, order amount should be None
    freqtrade.execute_buy('LTC/BTC', result)

    result = freqtrade._get_trade_stake_amount('XRP/BTC')
    assert result is None

    # set max_open_trades = None, so do not trade
    conf['max_open_trades'] = 0
    freqtrade = FreqtradeBot(conf)
    result = freqtrade._get_trade_stake_amount('NEO/BTC')
    assert result is None


def test_edge_called_in_process(mocker, edge_conf) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)

    def _refresh_whitelist(list):
        return ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC']

    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.pairlists._validate_whitelist = _refresh_whitelist
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker, edge_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    freqtrade = FreqtradeBot(edge_conf)

    assert freqtrade._get_trade_stake_amount('NEO/BTC') == (999.9 * 0.5 * 0.01) / 0.20
    assert freqtrade._get_trade_stake_amount('LTC/BTC') == (999.9 * 0.5 * 0.01) / 0.21


def test_edge_overrides_stoploss(limit_buy_order, fee, markets, caplog, mocker, edge_conf) -> None:

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')

    # Strategy stoploss is -0.1 but Edge imposes a stoploss at -0.2
    # Thus, if price falls 21%, stoploss should be triggered
    #
    # mocking the ticker: price is falling ...
    buy_price = limit_buy_order['price']
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': buy_price * 0.79,
            'ask': buy_price * 0.79,
            'last': buy_price * 0.79
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    #############################################

    # Create a trade with "limit_buy_order" price
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.create_trades()
    trade = Trade.query.first()
    trade.update(limit_buy_order)
    #############################################

    # stoploss shoud be hit
    assert freqtrade.handle_trade(trade) is True
    assert log_has('executed sell, reason: SellType.STOP_LOSS', caplog)
    assert trade.sell_reason == SellType.STOP_LOSS.value


def test_edge_should_ignore_strategy_stoploss(limit_buy_order, fee, markets,
                                              mocker, edge_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')

    # Strategy stoploss is -0.1 but Edge imposes a stoploss at -0.2
    # Thus, if price falls 15%, stoploss should not be triggered
    #
    # mocking the ticker: price is falling ...
    buy_price = limit_buy_order['price']
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': buy_price * 0.85,
            'ask': buy_price * 0.85,
            'last': buy_price * 0.85
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )
    #############################################

    # Create a trade with "limit_buy_order" price
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.create_trades()
    trade = Trade.query.first()
    trade.update(limit_buy_order)
    #############################################

    # stoploss shoud not be hit
    assert freqtrade.handle_trade(trade) is False


def test_total_open_trades_stakes(mocker, default_conf, ticker,
                                  limit_buy_order, fee, markets) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf['stake_amount'] = 0.0000098751
    default_conf['max_open_trades'] = 2
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.create_trades()
    trade = Trade.query.first()

    assert trade is not None
    assert trade.stake_amount == 0.0000098751
    assert trade.is_open
    assert trade.open_date is not None

    freqtrade.create_trades()
    trade = Trade.query.order_by(Trade.id.desc()).first()

    assert trade is not None
    assert trade.stake_amount == 0.0000098751
    assert trade.is_open
    assert trade.open_date is not None

    assert Trade.total_open_trades_stakes() == 1.97502e-05


def test_get_min_pair_stake_amount(mocker, default_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.stoploss = -0.05
    markets = {'ETH/BTC': {'symbol': 'ETH/BTC'}}
    # no pair found
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    with pytest.raises(ValueError, match=r'.*get market information.*'):
        freqtrade._get_min_pair_stake_amount('BNB/BTC', 1)

    # no 'limits' section
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # empty 'limits' section
    markets["ETH/BTC"]["limits"] = {}
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # no cost Min
    markets["ETH/BTC"]["limits"] = {
        'cost': {"min": None},
        'amount': {}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # no amount Min
    markets["ETH/BTC"]["limits"] = {
        'cost': {},
        'amount': {"min": None}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # empty 'cost'/'amount' section
    markets["ETH/BTC"]["limits"] = {
        'cost': {},
        'amount': {}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # min cost is set
    markets["ETH/BTC"]["limits"] = {
        'cost': {'min': 2},
        'amount': {}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result == 2 / 0.9

    # min amount is set
    markets["ETH/BTC"]["limits"] = {
        'cost': {},
        'amount': {'min': 2}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 2)
    assert result == 2 * 2 / 0.9

    # min amount and cost are set (cost is minimal)
    markets["ETH/BTC"]["limits"] = {
        'cost': {'min': 2},
        'amount': {'min': 2}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 2)
    assert result == min(2, 2 * 2) / 0.9

    # min amount and cost are set (amount is minial)
    markets["ETH/BTC"]["limits"] = {
        'cost': {'min': 8},
        'amount': {'min': 2}
    }
    mocker.patch(
        'freqtrade.exchange.Exchange.markets',
        PropertyMock(return_value=markets)
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 2)
    assert result == min(8, 2 * 2) / 0.9


def test_create_trades(default_conf, ticker, limit_buy_order, fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade is not None
    assert trade.stake_amount == 0.001
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'bittrex'

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    assert trade.open_rate == 0.00001099
    assert trade.amount == 90.99181073

    assert whitelist == default_conf['exchange']['pair_whitelist']


def test_create_trades_no_stake_amount(default_conf, ticker, limit_buy_order,
                                       fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf['stake_amount'] * 0.5)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        freqtrade.create_trades()


def test_create_trades_minimal_amount(default_conf, ticker, limit_buy_order,
                                      fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    buy_mock = MagicMock(return_value={'id': limit_buy_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=buy_mock,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['stake_amount'] = 0.0005
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trades()
    rate, amount = buy_mock.call_args[1]['rate'], buy_mock.call_args[1]['amount']
    assert rate * amount >= default_conf['stake_amount']


def test_create_trades_too_small_stake_amount(default_conf, ticker, limit_buy_order,
                                              fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    buy_mock = MagicMock(return_value={'id': limit_buy_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=buy_mock,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    default_conf['stake_amount'] = 0.000000005
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    assert not freqtrade.create_trades()


def test_create_trades_limit_reached(default_conf, ticker, limit_buy_order,
                                     fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_balance=MagicMock(return_value=default_conf['stake_amount']),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['max_open_trades'] = 0
    default_conf['stake_amount'] = constants.UNLIMITED_STAKE_AMOUNT

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    assert not freqtrade.create_trades()
    assert freqtrade._get_trade_stake_amount('ETH/BTC') is None


def test_create_trades_no_pairs_let(default_conf, ticker, limit_buy_order, fee,
                                    markets, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    default_conf['exchange']['pair_whitelist'] = ["ETH/BTC"]
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    assert freqtrade.create_trades()
    assert not freqtrade.create_trades()
    assert log_has("No currency pair in whitelist, but checking to sell open trades.", caplog)


def test_create_trades_no_pairs_in_whitelist(default_conf, ticker, limit_buy_order, fee,
                                             markets, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['exchange']['pair_whitelist'] = []
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    assert not freqtrade.create_trades()
    assert log_has("Whitelist is empty.", caplog)


def test_create_trades_no_signal(default_conf, fee, mocker) -> None:
    default_conf['dry_run'] = True

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balance=MagicMock(return_value=20),
        get_fee=fee,
    )
    default_conf['stake_amount'] = 10
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade, value=(False, False))

    Trade.query = MagicMock()
    Trade.query.filter = MagicMock()
    assert not freqtrade.create_trades()


@pytest.mark.parametrize("max_open", range(0, 5))
def test_create_trades_multiple_trades(default_conf, ticker,
                                       fee, markets, mocker, max_open) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf['max_open_trades'] = max_open
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': "12355555"}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trades()

    trades = Trade.get_open_trades()
    assert len(trades) == max_open


def test_create_trades_preopen(default_conf, ticker, fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf['max_open_trades'] = 4
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': "12355555"}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create 2 existing trades
    freqtrade.execute_buy('ETH/BTC', default_conf['stake_amount'])
    freqtrade.execute_buy('NEO/BTC', default_conf['stake_amount'])

    assert len(Trade.get_open_trades()) == 2

    # Create 2 new trades using create_trades
    assert freqtrade.create_trades()

    trades = Trade.get_open_trades()
    assert len(trades) == 4


def test_process_trade_creation(default_conf, ticker, limit_buy_order,
                                markets, fee, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        markets=PropertyMock(return_value=markets),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_order=MagicMock(return_value=limit_buy_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades

    freqtrade.process()

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert trade.stake_amount == default_conf['stake_amount']
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'bittrex'
    assert trade.open_rate == 0.00001099
    assert trade.amount == 90.99181073703367

    assert log_has(
        'Buy signal found: about create a new trade with stake_amount: 0.001 ...', caplog
    )


def test_process_exchange_failures(default_conf, ticker, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        markets=PropertyMock(return_value=markets),
        buy=MagicMock(side_effect=TemporaryError)
    )
    sleep_mock = mocker.patch('time.sleep', side_effect=lambda _: None)

    worker = Worker(args=None, config=default_conf)
    patch_get_signal(worker.freqtrade)

    worker._process()
    assert sleep_mock.has_calls()


def test_process_operational_exception(default_conf, ticker, markets, mocker) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        markets=PropertyMock(return_value=markets),
        buy=MagicMock(side_effect=OperationalException)
    )
    worker = Worker(args=None, config=default_conf)
    patch_get_signal(worker.freqtrade)

    assert worker.state == State.RUNNING

    worker._process()
    assert worker.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]['status']


def test_process_trade_handling(
        default_conf, ticker, limit_buy_order, markets, fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        markets=PropertyMock(return_value=markets),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_order=MagicMock(return_value=limit_buy_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades
    freqtrade.process()

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1

    # Nothing happened ...
    freqtrade.process()
    assert len(trades) == 1


def test_process_trade_no_whitelist_pair(
        default_conf, ticker, limit_buy_order, markets, fee, mocker) -> None:
    """ Test process with trade not in pair list """
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        markets=PropertyMock(return_value=markets),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_order=MagicMock(return_value=limit_buy_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    pair = 'NOCLUE/BTC'
    # create open trade not in whitelist
    Trade.session.add(Trade(
        pair=pair,
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        amount=20,
        open_rate=0.01,
        exchange='bittrex',
    ))
    Trade.session.add(Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        amount=12,
        open_rate=0.001,
        exchange='bittrex',
    ))

    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    # Make sure each pair is only in the list once
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))


def test_process_informative_pairs_added(default_conf, ticker, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)

    def _refresh_whitelist(list):
        return ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC']

    refresh_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        markets=PropertyMock(return_value=markets),
        buy=MagicMock(side_effect=TemporaryError),
        refresh_latest_ohlcv=refresh_mock,
    )
    inf_pairs = MagicMock(return_value=[("BTC/ETH", '1m'), ("ETH/USDT", "1h")])
    mocker.patch('time.sleep', return_value=None)

    freqtrade = FreqtradeBot(default_conf)
    freqtrade.pairlists._validate_whitelist = _refresh_whitelist
    freqtrade.strategy.informative_pairs = inf_pairs
    # patch_get_signal(freqtrade)

    freqtrade.process()
    assert inf_pairs.call_count == 1
    assert refresh_mock.call_count == 1
    assert ("BTC/ETH", "1m") in refresh_mock.call_args[0][0]
    assert ("ETH/USDT", "1h") in refresh_mock.call_args[0][0]
    assert ("ETH/BTC", default_conf["ticker_interval"]) in refresh_mock.call_args[0][0]


def test_balance_fully_ask_side(mocker, default_conf) -> None:
    default_conf['bid_strategy']['ask_last_balance'] = 0.0
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={'ask': 20, 'last': 10}))

    assert freqtrade.get_target_bid('ETH/BTC') == 20


def test_balance_fully_last_side(mocker, default_conf) -> None:
    default_conf['bid_strategy']['ask_last_balance'] = 1.0
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={'ask': 20, 'last': 10}))

    assert freqtrade.get_target_bid('ETH/BTC') == 10


def test_balance_bigger_last_ask(mocker, default_conf) -> None:
    default_conf['bid_strategy']['ask_last_balance'] = 1.0
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={'ask': 5, 'last': 10}))
    assert freqtrade.get_target_bid('ETH/BTC') == 5


def test_execute_buy(mocker, default_conf, fee, markets, limit_buy_order) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf)
    stake_amount = 2
    bid = 0.11
    get_bid = MagicMock(return_value=bid)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        get_target_bid=get_bid,
        _get_min_pair_stake_amount=MagicMock(return_value=1)
        )
    buy_mm = MagicMock(return_value={'id': limit_buy_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=buy_mm,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    pair = 'ETH/BTC'

    assert freqtrade.execute_buy(pair, stake_amount)
    assert get_bid.call_count == 1
    assert buy_mm.call_count == 1
    call_args = buy_mm.call_args_list[0][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == bid
    assert call_args['amount'] == stake_amount / bid

    # Should create an open trade with an open order id
    # As the order is not fulfilled yet
    trade = Trade.query.first()
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id == limit_buy_order['id']

    # Test calling with price
    fix_price = 0.06
    assert freqtrade.execute_buy(pair, stake_amount, fix_price)
    # Make sure get_target_bid wasn't called again
    assert get_bid.call_count == 1

    assert buy_mm.call_count == 2
    call_args = buy_mm.call_args_list[1][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == fix_price
    assert call_args['amount'] == stake_amount / fix_price

    # In case of closed order
    limit_buy_order['status'] = 'closed'
    limit_buy_order['price'] = 10
    limit_buy_order['cost'] = 100
    mocker.patch('freqtrade.exchange.Exchange.buy', MagicMock(return_value=limit_buy_order))
    assert freqtrade.execute_buy(pair, stake_amount)
    trade = Trade.query.all()[2]
    assert trade
    assert trade.open_order_id is None
    assert trade.open_rate == 10
    assert trade.stake_amount == 100

    # In case of rejected or expired order and partially filled
    limit_buy_order['status'] = 'expired'
    limit_buy_order['amount'] = 90.99181073
    limit_buy_order['filled'] = 80.99181073
    limit_buy_order['remaining'] = 10.00
    limit_buy_order['price'] = 0.5
    limit_buy_order['cost'] = 40.495905365
    mocker.patch('freqtrade.exchange.Exchange.buy', MagicMock(return_value=limit_buy_order))
    assert freqtrade.execute_buy(pair, stake_amount)
    trade = Trade.query.all()[3]
    assert trade
    assert trade.open_order_id is None
    assert trade.open_rate == 0.5
    assert trade.stake_amount == 40.495905365

    # In case of the order is rejected and not filled at all
    limit_buy_order['status'] = 'rejected'
    limit_buy_order['amount'] = 90.99181073
    limit_buy_order['filled'] = 0.0
    limit_buy_order['remaining'] = 90.99181073
    limit_buy_order['price'] = 0.5
    limit_buy_order['cost'] = 0.0
    mocker.patch('freqtrade.exchange.Exchange.buy', MagicMock(return_value=limit_buy_order))
    assert not freqtrade.execute_buy(pair, stake_amount)


def test_add_stoploss_on_exchange(mocker, default_conf, limit_buy_order) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.get_order', return_value=limit_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 return_value=limit_buy_order['amount'])

    stoploss_limit = MagicMock(return_value={'id': 13434334})
    mocker.patch('freqtrade.exchange.Exchange.stoploss_limit', stoploss_limit)

    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    trade = MagicMock()
    trade.open_order_id = None
    trade.stoploss_order_id = None
    trade.is_open = True

    freqtrade.process_maybe_execute_sell(trade)
    assert trade.stoploss_order_id == '13434334'
    assert stoploss_limit.call_count == 1
    assert trade.is_open is True


def test_handle_stoploss_on_exchange(mocker, default_conf, fee, caplog,
                                     markets, limit_buy_order, limit_sell_order) -> None:
    stoploss_limit = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        stoploss_limit=stoploss_limit
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # First case: when stoploss is not yet set but the order is open
    # should get the stoploss order id immediately
    # and should return false as no trade actually happened
    trade = MagicMock()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = None

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss_limit.call_count == 1
    assert trade.stoploss_order_id == "13434334"

    # Second case: when stoploss is set but it is not yet hit
    # should do nothing and return false
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100

    hanging_stoploss_order = MagicMock(return_value={'status': 'open'})
    mocker.patch('freqtrade.exchange.Exchange.get_order', hanging_stoploss_order)

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.stoploss_order_id == 100

    # Third case: when stoploss was set but it was canceled for some reason
    # should set a stoploss immediately and return False
    caplog.clear()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100

    canceled_stoploss_order = MagicMock(return_value={'status': 'canceled'})
    mocker.patch('freqtrade.exchange.Exchange.get_order', canceled_stoploss_order)
    stoploss_limit.reset_mock()

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss_limit.call_count == 1
    assert trade.stoploss_order_id == "13434334"

    # Fourth case: when stoploss is set and it is hit
    # should unset stoploss_order_id and return true
    # as a trade actually happened
    caplog.clear()
    freqtrade.create_trades()
    trade = Trade.query.first()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100
    assert trade

    stoploss_order_hit = MagicMock(return_value={
        'status': 'closed',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2
    })
    mocker.patch('freqtrade.exchange.Exchange.get_order', stoploss_order_hit)
    assert freqtrade.handle_stoploss_on_exchange(trade) is True
    assert log_has('STOP_LOSS_LIMIT is hit for {}.'.format(trade), caplog)
    assert trade.stoploss_order_id is None
    assert trade.is_open is False

    mocker.patch(
        'freqtrade.exchange.Exchange.stoploss_limit',
        side_effect=DependencyException()
    )
    freqtrade.handle_stoploss_on_exchange(trade)
    assert log_has('Unable to place a stoploss order on exchange.', caplog)
    assert trade.stoploss_order_id is None

    # Fifth case: get_order returns InvalidOrder
    # It should try to add stoploss order
    trade.stoploss_order_id = 100
    stoploss_limit.reset_mock()
    mocker.patch('freqtrade.exchange.Exchange.get_order', side_effect=InvalidOrderException())
    mocker.patch('freqtrade.exchange.Exchange.stoploss_limit', stoploss_limit)
    freqtrade.handle_stoploss_on_exchange(trade)
    assert stoploss_limit.call_count == 1


def test_handle_sle_cancel_cant_recreate(mocker, default_conf, fee, caplog,
                                         markets, limit_buy_order, limit_sell_order) -> None:
    # Sixth case: stoploss order was cancelled but couldn't create new one
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        get_order=MagicMock(return_value={'status': 'canceled'}),
        stoploss_limit=MagicMock(side_effect=DependencyException()),
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trades()
    trade = Trade.query.first()
    trade.is_open = True
    trade.open_order_id = '12345'
    trade.stoploss_order_id = 100
    assert trade

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert log_has_re(r'Stoploss order was cancelled, but unable to recreate one.*', caplog)
    assert trade.stoploss_order_id is None
    assert trade.is_open is True


def test_create_stoploss_order_invalid_order(mocker, default_conf, caplog, fee,
                                             markets, limit_buy_order, limit_sell_order):
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    sell_mock = MagicMock(return_value={'id': limit_sell_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=sell_mock,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        get_order=MagicMock(return_value={'status': 'canceled'}),
        stoploss_limit=MagicMock(side_effect=InvalidOrderException()),
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    freqtrade.create_trades()
    trade = Trade.query.first()
    caplog.clear()
    freqtrade.create_stoploss_order(trade, 200, 199)
    assert trade.stoploss_order_id is None
    assert trade.sell_reason == SellType.EMERGENCY_SELL.value
    assert log_has("Unable to place a stoploss order on exchange. ", caplog)
    assert log_has("Selling the trade forcefully", caplog)

    # Should call a market sell
    assert sell_mock.call_count == 1
    assert sell_mock.call_args[1]['ordertype'] == 'market'
    assert sell_mock.call_args[1]['pair'] == trade.pair
    assert sell_mock.call_args[1]['amount'] == trade.amount

    # Rpc is sending first buy, then sell
    assert rpc_mock.call_count == 2
    assert rpc_mock.call_args_list[1][0][0]['sell_reason'] == SellType.EMERGENCY_SELL.value
    assert rpc_mock.call_args_list[1][0][0]['order_type'] == 'market'


def test_handle_stoploss_on_exchange_trailing(mocker, default_conf, fee, caplog,
                                              markets, limit_buy_order, limit_sell_order) -> None:
    # When trailing stoploss is set
    stoploss_limit = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        stoploss_limit=stoploss_limit
    )

    # enabling TSL
    default_conf['trailing_stop'] = True

    # disabling ROI
    default_conf['minimal_roi']['0'] = 999999999

    freqtrade = FreqtradeBot(default_conf)

    # enabling stoploss on exchange
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # setting stoploss
    freqtrade.strategy.stoploss = -0.05

    # setting stoploss_on_exchange_interval to 60 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60

    patch_get_signal(freqtrade)

    freqtrade.create_trades()
    trade = Trade.query.first()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100

    stoploss_order_hanging = MagicMock(return_value={
        'id': 100,
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'info': {
            'stopPrice': '0.000011134'
        }
    })

    mocker.patch('freqtrade.exchange.Exchange.get_order', stoploss_order_hanging)

    # stoploss initially at 5%
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # price jumped 2x
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', MagicMock(return_value={
        'bid': 0.00002344,
        'ask': 0.00002346,
        'last': 0.00002344
    }))

    cancel_order_mock = MagicMock()
    stoploss_order_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.cancel_order', cancel_order_mock)
    mocker.patch('freqtrade.exchange.Exchange.stoploss_limit', stoploss_order_mock)

    # stoploss should not be updated as the interval is 60 seconds
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    cancel_order_mock.assert_not_called()
    stoploss_order_mock.assert_not_called()

    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == 0.00002344 * 0.95

    # setting stoploss_on_exchange_interval to 0 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    cancel_order_mock.assert_called_once_with(100, 'ETH/BTC')
    stoploss_order_mock.assert_called_once_with(amount=85.25149190110828,
                                                pair='ETH/BTC',
                                                rate=0.00002344 * 0.95 * 0.99,
                                                stop_price=0.00002344 * 0.95)


def test_handle_stoploss_on_exchange_trailing_error(mocker, default_conf, fee, caplog,
                                                    markets, limit_buy_order,
                                                    limit_sell_order) -> None:
    # When trailing stoploss is set
    stoploss_limit = MagicMock(return_value={'id': 13434334})
    patch_exchange(mocker)

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        stoploss_limit=stoploss_limit
    )

    # enabling TSL
    default_conf['trailing_stop'] = True

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    # enabling stoploss on exchange
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # setting stoploss
    freqtrade.strategy.stoploss = -0.05

    # setting stoploss_on_exchange_interval to 60 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60
    patch_get_signal(freqtrade)
    freqtrade.create_trades()
    trade = Trade.query.first()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = "abcd"
    trade.stop_loss = 0.2
    trade.stoploss_last_update = arrow.utcnow().shift(minutes=-601).datetime.replace(tzinfo=None)

    stoploss_order_hanging = {
        'id': "abcd",
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'info': {
            'stopPrice': '0.1'
        }
    }
    mocker.patch('freqtrade.exchange.Exchange.cancel_order', side_effect=InvalidOrderException())
    mocker.patch('freqtrade.exchange.Exchange.get_order', stoploss_order_hanging)
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert log_has_re(r"Could not cancel stoploss order abcd for pair ETH/BTC.*", caplog)

    # Still try to create order
    assert stoploss_limit.call_count == 1

    # Fail creating stoploss order
    caplog.clear()
    cancel_mock = mocker.patch("freqtrade.exchange.Exchange.cancel_order", MagicMock())
    mocker.patch("freqtrade.exchange.Exchange.stoploss_limit", side_effect=DependencyException())
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert cancel_mock.call_count == 1
    assert log_has_re(r"Could not create trailing stoploss order for pair ETH/BTC\..*", caplog)


def test_tsl_on_exchange_compatible_with_edge(mocker, edge_conf, fee, caplog,
                                              markets, limit_buy_order, limit_sell_order) -> None:

    # When trailing stoploss is set
    stoploss_limit = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        stoploss_limit=stoploss_limit
    )

    # enabling TSL
    edge_conf['trailing_stop'] = True
    edge_conf['trailing_stop_positive'] = 0.01
    edge_conf['trailing_stop_positive_offset'] = 0.011

    # disabling ROI
    edge_conf['minimal_roi']['0'] = 999999999

    freqtrade = FreqtradeBot(edge_conf)

    # enabling stoploss on exchange
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # setting stoploss
    freqtrade.strategy.stoploss = -0.02

    # setting stoploss_on_exchange_interval to 0 second
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    patch_get_signal(freqtrade)

    freqtrade.active_pair_whitelist = freqtrade.edge.adjust(freqtrade.active_pair_whitelist)

    freqtrade.create_trades()
    trade = Trade.query.first()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100

    stoploss_order_hanging = MagicMock(return_value={
        'id': 100,
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'info': {
            'stopPrice': '0.000009384'
        }
    })

    mocker.patch('freqtrade.exchange.Exchange.get_order', stoploss_order_hanging)

    # stoploss initially at 20% as edge dictated it.
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.stop_loss == 0.000009384

    cancel_order_mock = MagicMock()
    stoploss_order_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.cancel_order', cancel_order_mock)
    mocker.patch('freqtrade.exchange.Exchange.stoploss_limit', stoploss_order_mock)

    # price goes down 5%
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', MagicMock(return_value={
        'bid': 0.00001172 * 0.95,
        'ask': 0.00001173 * 0.95,
        'last': 0.00001172 * 0.95
    }))

    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # stoploss should remain the same
    assert trade.stop_loss == 0.000009384

    # stoploss on exchange should not be canceled
    cancel_order_mock.assert_not_called()

    # price jumped 2x
    mocker.patch('freqtrade.exchange.Exchange.get_ticker', MagicMock(return_value={
        'bid': 0.00002344,
        'ask': 0.00002346,
        'last': 0.00002344
    }))

    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # stoploss should be set to 1% as trailing is on
    assert trade.stop_loss == 0.00002344 * 0.99
    cancel_order_mock.assert_called_once_with(100, 'NEO/BTC')
    stoploss_order_mock.assert_called_once_with(amount=2131074.168797954,
                                                pair='NEO/BTC',
                                                rate=0.00002344 * 0.99 * 0.99,
                                                stop_price=0.00002344 * 0.99)


def test_process_maybe_execute_buy(mocker, default_conf, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trades', MagicMock(return_value=False))
    freqtrade.process_maybe_execute_buy()
    assert log_has('Found no buy signals for whitelisted currencies. Trying again...', caplog)


def test_process_maybe_execute_buy_exception(mocker, default_conf, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.create_trades',
        MagicMock(side_effect=DependencyException)
    )
    freqtrade.process_maybe_execute_buy()
    assert log_has('Unable to create trade: ', caplog)


def test_process_maybe_execute_sell(mocker, default_conf, limit_buy_order, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.get_order', return_value=limit_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 return_value=limit_buy_order['amount'])

    trade = MagicMock()
    trade.open_order_id = '123'
    trade.open_fee = 0.001
    assert not freqtrade.process_maybe_execute_sell(trade)
    # Test amount not modified by fee-logic
    assert not log_has(
        'Applying fee to amount for Trade {} from 90.99181073 to 90.81'.format(trade), caplog
    )

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=90.81)
    # test amount modified by fee-logic
    assert not freqtrade.process_maybe_execute_sell(trade)


def test_process_maybe_execute_sell_exception(mocker, default_conf,
                                              limit_buy_order, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_order', return_value=limit_buy_order)

    trade = MagicMock()
    trade.open_order_id = '123'
    trade.open_fee = 0.001

    # Test raise of DependencyException exception
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.update_trade_state',
        side_effect=DependencyException()
    )
    freqtrade.process_maybe_execute_sell(trade)
    assert log_has('Unable to sell trade: ', caplog)


def test_update_trade_state(mocker, default_conf, limit_buy_order, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.get_order', return_value=limit_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 return_value=limit_buy_order['amount'])

    trade = Trade()
    # Mock session away
    Trade.session = MagicMock()
    trade.open_order_id = '123'
    trade.open_fee = 0.001
    freqtrade.update_trade_state(trade)
    # Test amount not modified by fee-logic
    assert not log_has_re(r'Applying fee to .*', caplog)
    assert trade.open_order_id is None
    assert trade.amount == limit_buy_order['amount']

    trade.open_order_id = '123'
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=90.81)
    assert trade.amount != 90.81
    # test amount modified by fee-logic
    freqtrade.update_trade_state(trade)
    assert trade.amount == 90.81
    assert trade.open_order_id is None

    trade.is_open = True
    trade.open_order_id = None
    # Assert we call handle_trade() if trade is feasible for execution
    freqtrade.update_trade_state(trade)

    assert log_has_re('Found open order for.*', caplog)


def test_update_trade_state_withorderdict(default_conf, trades_for_order, limit_buy_order, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    # get_order should not be called!!
    mocker.patch('freqtrade.exchange.Exchange.get_order', MagicMock(side_effect=ValueError))
    patch_exchange(mocker)
    Trade.session = MagicMock()
    amount = sum(x['amount'] for x in trades_for_order)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456",
        is_open=True,
    )
    freqtrade.update_trade_state(trade, limit_buy_order)
    assert trade.amount != amount
    assert trade.amount == limit_buy_order['amount']


def test_update_trade_state_exception(mocker, default_conf,
                                      limit_buy_order, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_order', return_value=limit_buy_order)

    trade = MagicMock()
    trade.open_order_id = '123'
    trade.open_fee = 0.001

    # Test raise of OperationalException exception
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
        side_effect=OperationalException()
    )
    freqtrade.update_trade_state(trade)
    assert log_has('Could not update trade amount: ', caplog)


def test_update_trade_state_orderexception(mocker, default_conf, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_order',
                 MagicMock(side_effect=InvalidOrderException))

    trade = MagicMock()
    trade.open_order_id = '123'
    trade.open_fee = 0.001

    # Test raise of OperationalException exception
    grm_mock = mocker.patch("freqtrade.freqtradebot.FreqtradeBot.get_real_amount", MagicMock())
    freqtrade.update_trade_state(trade)
    assert grm_mock.call_count == 0
    assert log_has(f'Unable to fetch order {trade.open_order_id}: ', caplog)


def test_update_trade_state_sell(default_conf, trades_for_order, limit_sell_order, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    # get_order should not be called!!
    mocker.patch('freqtrade.exchange.Exchange.get_order', MagicMock(side_effect=ValueError))
    wallet_mock = MagicMock()
    mocker.patch('freqtrade.wallets.Wallets.update', wallet_mock)

    patch_exchange(mocker)
    Trade.session = MagicMock()
    amount = limit_sell_order["amount"]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    wallet_mock.reset_mock()
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=0.0025,
        fee_close=0.0025,
        open_order_id="123456",
        is_open=True,
    )
    freqtrade.update_trade_state(trade, limit_sell_order)
    assert trade.amount == limit_sell_order['amount']
    # Wallet needs to be updated after closing a limit-sell order to reenable buying
    assert wallet_mock.call_count == 1
    assert not trade.is_open


def test_handle_trade(default_conf, limit_buy_order, limit_sell_order,
                      fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    time.sleep(0.01)  # Race condition fix
    trade.update(limit_buy_order)
    assert trade.is_open is True

    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.open_order_id == limit_sell_order['id']

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    assert trade.close_rate == 0.00001173
    assert trade.close_profit == 0.06201058
    assert trade.calc_profit() == 0.00006217
    assert trade.close_date is not None


def test_handle_overlpapping_signals(default_conf, ticker, limit_buy_order,
                                     fee, markets, mocker) -> None:
    default_conf.update({'experimental': {'use_sell_signal': True}})

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade, value=(True, True))
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.create_trades()

    # Buy and Sell triggering, so doing nothing ...
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 0

    # Buy is triggering, so buying ...
    patch_get_signal(freqtrade, value=(True, False))
    freqtrade.create_trades()
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Buy and Sell are not triggering, so doing nothing ...
    patch_get_signal(freqtrade, value=(False, False))
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Buy and Sell are triggering, so doing nothing ...
    patch_get_signal(freqtrade, value=(True, True))
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Sell is triggering, guess what : we are Selling!
    patch_get_signal(freqtrade, value=(False, True))
    trades = Trade.query.all()
    assert freqtrade.handle_trade(trades[0]) is True


def test_handle_trade_roi(default_conf, ticker, limit_buy_order,
                          fee, mocker, markets, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf.update({'experimental': {'use_sell_signal': True}})

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade, value=(True, False))
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.is_open = True

    # FIX: sniffing logs, suggest handle_trade should not execute_sell
    #      instead that responsibility should be moved out of handle_trade(),
    #      we might just want to check if we are in a sell condition without
    #      executing
    # if ROI is reached we must sell
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade)
    assert log_has("'ETH/BTC' - Required profit reached. Selling (sell_type=SellType.ROI)...",
                   caplog)


def test_handle_trade_experimental(
        default_conf, ticker, limit_buy_order, fee, mocker, markets, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf.update({'experimental': {'use_sell_signal': True}})
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.is_open = True

    patch_get_signal(freqtrade, value=(False, False))
    assert not freqtrade.handle_trade(trade)

    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade)
    assert log_has("'ETH/BTC' - Sell signal received. Selling (sell_type=SellType.SELL_SIGNAL)...",
                   caplog)


def test_close_trade(default_conf, ticker, limit_buy_order, limit_sell_order,
                     fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create trade and sell it
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    trade.update(limit_buy_order)
    trade.update(limit_sell_order)
    assert trade.is_open is False

    with pytest.raises(ValueError, match=r'.*closed trade.*'):
        freqtrade.handle_trade(trade)


def test_check_handle_timedout_buy(default_conf, ticker, limit_buy_order_old, fee, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_order=MagicMock(return_value=limit_buy_order_old),
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    trade_buy = Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='bittrex',
        open_order_id='123456789',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=True
    )

    Trade.session.add(trade_buy)

    # check it does cancel buy orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(trade_buy.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0


def test_check_handle_cancelled_buy(default_conf, ticker, limit_buy_order_old,
                                    fee, mocker, caplog) -> None:
    """ Handle Buy order cancelled on exchange"""
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    limit_buy_order_old.update({"status": "canceled"})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_order=MagicMock(return_value=limit_buy_order_old),
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    trade_buy = Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='bittrex',
        open_order_id='123456789',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=True
    )

    Trade.session.add(trade_buy)

    # check it does cancel buy orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(trade_buy.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    assert log_has_re("Buy order canceled on Exchange for Trade.*", caplog)


def test_check_handle_timedout_buy_exception(default_conf, ticker, limit_buy_order_old,
                                             fee, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_order=MagicMock(side_effect=DependencyException),
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    trade_buy = Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='bittrex',
        open_order_id='123456789',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=True
    )

    Trade.session.add(trade_buy)

    # check it does cancel buy orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 0
    trades = Trade.query.filter(Trade.open_order_id.is_(trade_buy.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1


def test_check_handle_timedout_sell(default_conf, ticker, limit_sell_order_old, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    trade_sell = Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='bittrex',
        open_order_id='123456789',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(hours=-5).datetime,
        close_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=False
    )

    Trade.session.add(trade_sell)

    # check it does cancel sell orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    assert trade_sell.is_open is True


def test_check_handle_cancelled_sell(default_conf, ticker, limit_sell_order_old,
                                     mocker, caplog) -> None:
    """ Handle sell order cancelled on exchange"""
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    limit_sell_order_old.update({"status": "canceled"})
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    trade_sell = Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='bittrex',
        open_order_id='123456789',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(hours=-5).datetime,
        close_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=False
    )

    Trade.session.add(trade_sell)

    # check it does cancel sell orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert trade_sell.is_open is True
    assert log_has_re("Sell order canceled on exchange for Trade.*", caplog)


def test_check_handle_timedout_partial(default_conf, ticker, limit_buy_order_old_partial,
                                       mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_order=MagicMock(return_value=limit_buy_order_old_partial),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    trade_buy = Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='bittrex',
        open_order_id='123456789',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=True
    )

    Trade.session.add(trade_buy)

    # check it does cancel buy orders over the time limit
    # note this is for a partially-complete buy order
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(trade_buy.open_order_id)).all()
    assert len(trades) == 1
    assert trades[0].amount == 23.0
    assert trades[0].stake_amount == trade_buy.open_rate * trades[0].amount


def test_check_handle_timedout_exception(default_conf, ticker, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()

    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        handle_timedout_limit_buy=MagicMock(),
        handle_timedout_limit_sell=MagicMock(),
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_order=MagicMock(side_effect=requests.exceptions.RequestException('Oh snap')),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    open_date = arrow.utcnow().shift(minutes=-601)
    trade_buy = Trade(
        pair='ETH/BTC',
        open_rate=0.00001099,
        exchange='bittrex',
        open_order_id='123456789',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=open_date.datetime,
        is_open=True
    )

    Trade.session.add(trade_buy)

    freqtrade.check_handle_timedout()
    assert log_has_re(f"Cannot query order for Trade\\(id=1, pair=ETH/BTC, amount=90.99181073, "
                      f"open_rate=0.00001099, open_since="
                      f"{open_date.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"\\(10 hours ago\\)\\) due to Traceback \\(most recent call last\\):\n*",
                      caplog)


def test_handle_timedout_limit_buy(mocker, default_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        cancel_order=cancel_order_mock
    )

    freqtrade = FreqtradeBot(default_conf)

    Trade.session = MagicMock()
    trade = MagicMock()
    order = {'remaining': 1,
             'amount': 1}
    assert freqtrade.handle_timedout_limit_buy(trade, order)
    assert cancel_order_mock.call_count == 1
    order['amount'] = 2
    assert not freqtrade.handle_timedout_limit_buy(trade, order)
    assert cancel_order_mock.call_count == 2


def test_handle_timedout_limit_sell(mocker, default_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        cancel_order=cancel_order_mock
    )

    freqtrade = FreqtradeBot(default_conf)

    trade = MagicMock()
    order = {'remaining': 1,
             'amount': 1,
             'status': "open"}
    assert freqtrade.handle_timedout_limit_sell(trade, order)
    assert cancel_order_mock.call_count == 1
    order['amount'] = 2
    assert not freqtrade.handle_timedout_limit_sell(trade, order)
    # Assert cancel_order was not called (callcount remains unchanged)
    assert cancel_order_mock.call_count == 1


def test_execute_sell_up(default_conf, ticker, fee, ticker_sell_up, markets, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_up
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'], sell_reason=SellType.ROI)

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'limit': 1.172e-05,
        'amount': 90.99181073703367,
        'order_type': 'limit',
        'open_rate': 1.099e-05,
        'current_rate': 1.172e-05,
        'profit_amount': 6.126e-05,
        'profit_percent': 0.0611052,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.ROI.value
    } == last_msg


def test_execute_sell_down(default_conf, ticker, fee, ticker_sell_down, markets, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_down
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_down()['bid'],
                           sell_reason=SellType.STOP_LOSS)

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.044e-05,
        'amount': 90.99181073703367,
        'order_type': 'limit',
        'open_rate': 1.099e-05,
        'current_rate': 1.044e-05,
        'profit_amount': -5.492e-05,
        'profit_percent': -0.05478342,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value
    } == last_msg


def test_execute_sell_down_stoploss_on_exchange_dry_run(default_conf, ticker, fee,
                                                        ticker_sell_down,
                                                        markets, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_down
    )

    default_conf['dry_run'] = True
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    # Setting trade stoploss to 0.01

    trade.stop_loss = 0.00001099 * 0.99
    freqtrade.execute_sell(trade=trade, limit=ticker_sell_down()['bid'],
                           sell_reason=SellType.STOP_LOSS)

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]

    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.08801e-05,
        'amount': 90.99181073703367,
        'order_type': 'limit',
        'open_rate': 1.099e-05,
        'current_rate': 1.044e-05,
        'profit_amount': -1.498e-05,
        'profit_percent': -0.01493766,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value

    } == last_msg


def test_execute_sell_sloe_cancel_exception(mocker, default_conf, ticker, fee,
                                            markets, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order', side_effect=InvalidOrderException())
    sellmock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        sell=sellmock
    )

    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade)
    freqtrade.create_trades()

    trade = Trade.query.first()
    Trade.session = MagicMock()

    freqtrade.config['dry_run'] = False
    trade.stoploss_order_id = "abcd"

    freqtrade.execute_sell(trade=trade, limit=1234,
                           sell_reason=SellType.STOP_LOSS)
    assert sellmock.call_count == 1
    assert log_has('Could not cancel stoploss order abcd', caplog)


def test_execute_sell_with_stoploss_on_exchange(default_conf,
                                                ticker, fee, ticker_sell_up,
                                                markets, mocker) -> None:

    default_conf['exchange']['name'] = 'binance'
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    stoploss_limit = MagicMock(return_value={
        'id': 123,
        'info': {
            'foo': 'bar'
        }
    })

    cancel_order = MagicMock(return_value=True)

    mocker.patch('freqtrade.exchange.Exchange.symbol_amount_prec', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.symbol_price_prec', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.stoploss_limit', stoploss_limit)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order', cancel_order)

    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    freqtrade.process_maybe_execute_sell(trade)

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_up
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'],
                           sell_reason=SellType.SELL_SIGNAL)

    trade = Trade.query.first()
    assert trade
    assert cancel_order.call_count == 1
    assert rpc_mock.call_count == 2


def test_may_execute_sell_after_stoploss_on_exchange_hit(default_conf,
                                                         ticker, fee,
                                                         limit_buy_order,
                                                         markets, mocker) -> None:
    default_conf['exchange']['name'] = 'binance'
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    stoploss_limit = MagicMock(return_value={
        'id': 123,
        'info': {
            'foo': 'bar'
        }
    })

    mocker.patch('freqtrade.exchange.Exchange.symbol_amount_prec', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Exchange.symbol_price_prec', lambda s, x, y: y)
    mocker.patch('freqtrade.exchange.Binance.stoploss_limit', stoploss_limit)

    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trades()
    trade = Trade.query.first()
    freqtrade.process_maybe_execute_sell(trade)
    assert trade
    assert trade.stoploss_order_id == '123'
    assert trade.open_order_id is None

    # Assuming stoploss on exchnage is hit
    # stoploss_order_id should become None
    # and trade should be sold at the price of stoploss
    stoploss_limit_executed = MagicMock(return_value={
        "id": "123",
        "timestamp": 1542707426845,
        "datetime": "2018-11-20T09:50:26.845Z",
        "lastTradeTimestamp": None,
        "symbol": "BTC/USDT",
        "type": "stop_loss_limit",
        "side": "sell",
        "price": 1.08801,
        "amount": 90.99181074,
        "cost": 99.0000000032274,
        "average": 1.08801,
        "filled": 90.99181074,
        "remaining": 0.0,
        "status": "closed",
        "fee": None,
        "trades": None
    })
    mocker.patch('freqtrade.exchange.Exchange.get_order', stoploss_limit_executed)

    freqtrade.process_maybe_execute_sell(trade)
    assert trade.stoploss_order_id is None
    assert trade.is_open is False
    assert trade.sell_reason == SellType.STOPLOSS_ON_EXCHANGE.value
    assert rpc_mock.call_count == 2


def test_execute_sell_market_order(default_conf, ticker, fee,
                                   ticker_sell_up, markets, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_up
    )
    freqtrade.config['order_types']['sell'] = 'market'

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'], sell_reason=SellType.ROI)

    assert not trade.is_open
    assert trade.close_profit == 0.0611052

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'limit': 1.172e-05,
        'amount': 90.99181073703367,
        'order_type': 'market',
        'open_rate': 1.099e-05,
        'current_rate': 1.172e-05,
        'profit_amount': 6.126e-05,
        'profit_percent': 0.0611052,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.ROI.value

    } == last_msg


def test_sell_profit_only_enable_profit(default_conf, limit_buy_order,
                                        fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00002172,
            'ask': 0.00002173,
            'last': 0.00002172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.SELL_SIGNAL.value


def test_sell_profit_only_disable_profit(default_conf, limit_buy_order,
                                         fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00002172,
            'ask': 0.00002173,
            'last': 0.00002172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.SELL_SIGNAL.value


def test_sell_profit_only_enable_loss(default_conf, limit_buy_order, fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00000172,
            'ask': 0.00000173,
            'last': 0.00000172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.stop_loss_reached = MagicMock(return_value=SellCheckTuple(
            sell_flag=False, sell_type=SellType.NONE))
    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is False


def test_sell_profit_only_disable_loss(default_conf, limit_buy_order, fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.0000172,
            'ask': 0.0000173,
            'last': 0.0000172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.SELL_SIGNAL.value


def test_locked_pairs(default_conf, ticker, fee, ticker_sell_down, markets, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_down
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_down()['bid'],
                           sell_reason=SellType.STOP_LOSS)
    trade.close(ticker_sell_down()['bid'])
    assert trade.pair in freqtrade.strategy._pair_locked_until
    assert freqtrade.strategy.is_pair_locked(trade.pair)

    # reinit - should buy other pair.
    caplog.clear()
    freqtrade.create_trades()

    assert log_has(f"Pair {trade.pair} is currently locked.", caplog)


def test_ignore_roi_if_buy_signal(default_conf, limit_buy_order, fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.0000172,
            'ask': 0.0000173,
            'last': 0.0000172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['experimental'] = {
        'ignore_roi_if_buy_signal': True
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(freqtrade, value=(True, True))
    assert freqtrade.handle_trade(trade) is False

    # Test if buy-signal is absent (should sell due to roi = true)
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.ROI.value


def test_trailing_stop_loss(default_conf, limit_buy_order, fee, markets, caplog, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001099,
            'ask': 0.00001099,
            'last': 0.00001099
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )
    default_conf['trailing_stop'] = True
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.create_trades()
    trade = Trade.query.first()
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker above buy price
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={
                     'bid': 0.00001099 * 1.5,
                     'ask': 0.00001099 * 1.5,
                     'last': 0.00001099 * 1.5
                 }))

    # Stoploss should be adjusted
    assert freqtrade.handle_trade(trade) is False

    # Price fell
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={
                     'bid': 0.00001099 * 1.1,
                     'ask': 0.00001099 * 1.1,
                     'last': 0.00001099 * 1.1
                 }))

    caplog.set_level(logging.DEBUG)
    # Sell as trailing-stop is reached
    assert freqtrade.handle_trade(trade) is True
    assert log_has(
        f"'ETH/BTC' - HIT STOP: current price at 0.000012, "
        f"stoploss is 0.000015, "
        f"initial stoploss was at 0.000010, trade opened at 0.000011", caplog)
    assert trade.sell_reason == SellType.TRAILING_STOP_LOSS.value


def test_trailing_stop_loss_positive(default_conf, limit_buy_order, fee, markets,
                                     caplog, mocker) -> None:
    buy_price = limit_buy_order['price']
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': buy_price - 0.000001,
            'ask': buy_price - 0.000001,
            'last': buy_price - 0.000001
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )
    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0.01
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    caplog.set_level(logging.DEBUG)
    # stop-loss not reached
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker above buy price
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.000003,
                     'ask': buy_price + 0.000003,
                     'last': buy_price + 0.000003
                 }))
    # stop-loss not reached, adjusted stoploss
    assert freqtrade.handle_trade(trade) is False
    assert log_has(f"'ETH/BTC' - Using positive stoploss: 0.01 offset: 0 profit: 0.2666%", caplog)
    assert log_has(f"'ETH/BTC' - Adjusting stoploss...", caplog)
    assert trade.stop_loss == 0.0000138501

    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.000002,
                     'ask': buy_price + 0.000002,
                     'last': buy_price + 0.000002
                 }))
    # Lower price again (but still positive)
    assert freqtrade.handle_trade(trade) is True
    assert log_has(
        f"'ETH/BTC' - HIT STOP: current price at {buy_price + 0.000002:.6f}, "
        f"stoploss is {trade.stop_loss:.6f}, "
        f"initial stoploss was at 0.000010, trade opened at 0.000011", caplog)


def test_trailing_stop_loss_offset(default_conf, limit_buy_order, fee,
                                   caplog, mocker, markets) -> None:
    buy_price = limit_buy_order['price']
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': buy_price - 0.000001,
            'ask': buy_price - 0.000001,
            'last': buy_price - 0.000001
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )

    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0.01
    default_conf['trailing_stop_positive_offset'] = 0.011
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    caplog.set_level(logging.DEBUG)
    # stop-loss not reached
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker above buy price
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.000003,
                     'ask': buy_price + 0.000003,
                     'last': buy_price + 0.000003
                 }))
    # stop-loss not reached, adjusted stoploss
    assert freqtrade.handle_trade(trade) is False
    assert log_has(f"'ETH/BTC' - Using positive stoploss: 0.01 offset: 0.011 profit: 0.2666%",
                   caplog)
    assert log_has(f"'ETH/BTC' - Adjusting stoploss...", caplog)
    assert trade.stop_loss == 0.0000138501

    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.000002,
                     'ask': buy_price + 0.000002,
                     'last': buy_price + 0.000002
                 }))
    # Lower price again (but still positive)
    assert freqtrade.handle_trade(trade) is True
    assert log_has(
        f"'ETH/BTC' - HIT STOP: current price at {buy_price + 0.000002:.6f}, "
        f"stoploss is {trade.stop_loss:.6f}, "
        f"initial stoploss was at 0.000010, trade opened at 0.000011", caplog)
    assert trade.sell_reason == SellType.TRAILING_STOP_LOSS.value


def test_tsl_only_offset_reached(default_conf, limit_buy_order, fee,
                                 caplog, mocker, markets) -> None:
    buy_price = limit_buy_order['price']
    # buy_price: 0.00001099

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': buy_price,
            'ask': buy_price,
            'last': buy_price
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )

    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0.05
    default_conf['trailing_stop_positive_offset'] = 0.055
    default_conf['trailing_only_offset_is_reached'] = True

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    caplog.set_level(logging.DEBUG)
    # stop-loss not reached
    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == 0.0000098910

    # Raise ticker above buy price
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.0000004,
                     'ask': buy_price + 0.0000004,
                     'last': buy_price + 0.0000004
                 }))

    # stop-loss should not be adjusted as offset is not reached yet
    assert freqtrade.handle_trade(trade) is False

    assert not log_has(f"'ETH/BTC' - Adjusting stoploss...", caplog)
    assert trade.stop_loss == 0.0000098910

    # price rises above the offset (rises 12% when the offset is 5.5%)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.0000014,
                     'ask': buy_price + 0.0000014,
                     'last': buy_price + 0.0000014
                 }))

    assert freqtrade.handle_trade(trade) is False
    assert log_has(f"'ETH/BTC' - Using positive stoploss: 0.05 offset: 0.055 profit: 0.1218%",
                   caplog)
    assert log_has(f"'ETH/BTC' - Adjusting stoploss...", caplog)
    assert trade.stop_loss == 0.0000117705


def test_disable_ignore_roi_if_buy_signal(default_conf, limit_buy_order,
                                          fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00000172,
            'ask': 0.00000173,
            'last': 0.00000172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    default_conf['experimental'] = {
        'ignore_roi_if_buy_signal': False
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.create_trades()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    # Sell due to min_roi_reached
    patch_get_signal(freqtrade, value=(True, True))
    assert freqtrade.handle_trade(trade) is True

    # Test if buy-signal is absent
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.STOP_LOSS.value


def test_get_real_amount_quote(default_conf, trades_for_order, buy_order_fee, caplog, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount - (amount * 0.001)
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.992) from Trades',
                   caplog)


def test_get_real_amount_no_trade(default_conf, buy_order_fee, caplog, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    amount = buy_order_fee['amount']
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) failed: myTrade-Dict empty found',
                   caplog)


def test_get_real_amount_stake(default_conf, trades_for_order, buy_order_fee, mocker):
    trades_for_order[0]['fee']['currency'] = 'ETH'

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_no_currency_in_fee(default_conf, trades_for_order, buy_order_fee, mocker):

    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['fee'] = {'cost': 0.004, 'currency': None}
    trades_for_order[0]['fee']['currency'] = None

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Amount does not change
    assert freqtrade.get_real_amount(trade, limit_buy_order) == amount


def test_get_real_amount_BNB(default_conf, trades_for_order, buy_order_fee, mocker):
    trades_for_order[0]['fee']['currency'] = 'BNB'
    trades_for_order[0]['fee']['cost'] = 0.00094518

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_multi(default_conf, trades_for_order2, buy_order_fee, caplog, mocker):
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order2)
    amount = float(sum(x['amount'] for x in trades_for_order2))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount - (amount * 0.001)
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.992) from Trades',
                   caplog)


def test_get_real_amount_fromorder(default_conf, trades_for_order, buy_order_fee, caplog, mocker):
    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['fee'] = {'cost': 0.004, 'currency': 'LTC'}

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order',
                 return_value=[trades_for_order])
    amount = float(sum(x['amount'] for x in trades_for_order))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, limit_buy_order) == amount - 0.004
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.996) from Order',
                   caplog)


def test_get_real_amount_invalid_order(default_conf, trades_for_order, buy_order_fee, mocker):
    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['fee'] = {'cost': 0.004}

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    amount = float(sum(x['amount'] for x in trades_for_order))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Amount does not change
    assert freqtrade.get_real_amount(trade, limit_buy_order) == amount


def test_get_real_amount_invalid(default_conf, trades_for_order, buy_order_fee, mocker):
    # Remove "Currency" from fee dict
    trades_for_order[0]['fee'] = {'cost': 0.008}

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_open_trade(default_conf, mocker):
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    amount = 12345
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    order = {
        'id': 'mocked_order',
        'amount': amount,
        'status': 'open',
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    assert freqtrade.get_real_amount(trade, order) == amount


def test_order_book_depth_of_market(default_conf, ticker, limit_buy_order, fee, markets, mocker,
                                    order_book_l2):
    default_conf['bid_strategy']['check_depth_of_market']['enabled'] = True
    default_conf['bid_strategy']['check_depth_of_market']['bids_to_ask_delta'] = 0.1
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_order_book', order_book_l2)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade is not None
    assert trade.stake_amount == 0.001
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'bittrex'

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    assert trade.open_rate == 0.00001099
    assert whitelist == default_conf['exchange']['pair_whitelist']


def test_order_book_depth_of_market_high_delta(default_conf, ticker, limit_buy_order,
                                               fee, markets, mocker, order_book_l2):
    default_conf['bid_strategy']['check_depth_of_market']['enabled'] = True
    # delta is 100 which is impossible to reach. hence check_depth_of_market will return false
    default_conf['bid_strategy']['check_depth_of_market']['bids_to_ask_delta'] = 100
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_order_book', order_book_l2)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    # Save state of current whitelist
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade is None


def test_order_book_bid_strategy1(mocker, default_conf, order_book_l2, markets) -> None:
    """
    test if function get_target_bid will return the order book price
    instead of the ask rate
    """
    patch_exchange(mocker)
    ticker_mock = MagicMock(return_value={'ask': 0.045, 'last': 0.046})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets),
        get_order_book=order_book_l2,
        get_ticker=ticker_mock,

    )
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['use_order_book'] = True
    default_conf['bid_strategy']['order_book_top'] = 2
    default_conf['bid_strategy']['ask_last_balance'] = 0
    default_conf['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf)
    assert freqtrade.get_target_bid('ETH/BTC') == 0.043935
    assert ticker_mock.call_count == 0


def test_order_book_bid_strategy2(mocker, default_conf, order_book_l2, markets) -> None:
    """
    test if function get_target_bid will return the ask rate (since its value is lower)
    instead of the order book rate (even if enabled)
    """
    patch_exchange(mocker)
    ticker_mock = MagicMock(return_value={'ask': 0.042, 'last': 0.046})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets),
        get_order_book=order_book_l2,
        get_ticker=ticker_mock,

    )
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['use_order_book'] = True
    default_conf['bid_strategy']['order_book_top'] = 2
    default_conf['bid_strategy']['ask_last_balance'] = 0
    default_conf['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf)
    # ordrebook shall be used even if tickers would be lower.
    assert freqtrade.get_target_bid('ETH/BTC', ) != 0.042
    assert ticker_mock.call_count == 0


def test_check_depth_of_market_buy(default_conf, mocker, order_book_l2, markets) -> None:
    """
    test check depth of market
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        markets=PropertyMock(return_value=markets),
        get_order_book=order_book_l2
    )
    default_conf['telegram']['enabled'] = False
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['check_depth_of_market']['enabled'] = True
    # delta is 100 which is impossible to reach. hence function will return false
    default_conf['bid_strategy']['check_depth_of_market']['bids_to_ask_delta'] = 100
    freqtrade = FreqtradeBot(default_conf)

    conf = default_conf['bid_strategy']['check_depth_of_market']
    assert freqtrade._check_depth_of_market_buy('ETH/BTC', conf) is False


def test_order_book_ask_strategy(default_conf, limit_buy_order, limit_sell_order,
                                 fee, markets, mocker, order_book_l2) -> None:
    """
    test order book ask strategy
    """
    mocker.patch('freqtrade.exchange.Exchange.get_order_book', order_book_l2)
    default_conf['exchange']['name'] = 'binance'
    default_conf['ask_strategy']['use_order_book'] = True
    default_conf['ask_strategy']['order_book_min'] = 1
    default_conf['ask_strategy']['order_book_max'] = 2
    default_conf['telegram']['enabled'] = False
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trades()

    trade = Trade.query.first()
    assert trade

    time.sleep(0.01)  # Race condition fix
    trade.update(limit_buy_order)
    assert trade.is_open is True

    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True


def test_get_sell_rate(default_conf, mocker, ticker, order_book_l2) -> None:

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_order_book=order_book_l2,
        get_ticker=ticker,
    )
    pair = "ETH/BTC"

    # Test regular mode
    ft = get_patched_freqtradebot(mocker, default_conf)
    rate = ft.get_sell_rate(pair, True)
    assert isinstance(rate, float)
    assert rate == 0.00001098

    # Test orderbook mode
    default_conf['ask_strategy']['use_order_book'] = True
    default_conf['ask_strategy']['order_book_min'] = 1
    default_conf['ask_strategy']['order_book_max'] = 2
    ft = get_patched_freqtradebot(mocker, default_conf)
    rate = ft.get_sell_rate(pair, True)
    assert isinstance(rate, float)
    assert rate == 0.043936


def test_startup_state(default_conf, mocker):
    default_conf['pairlist'] = {'method': 'VolumePairList',
                                'config': {'number_assets': 20}
                                }
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    worker = get_patched_worker(mocker, default_conf)
    assert worker.state is State.RUNNING


def test_startup_trade_reinit(default_conf, edge_conf, mocker):

    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    reinit_mock = MagicMock()
    mocker.patch('freqtrade.persistence.Trade.stoploss_reinitialization', reinit_mock)

    ftbot = get_patched_freqtradebot(mocker, default_conf)
    ftbot.startup()
    assert reinit_mock.call_count == 1

    reinit_mock.reset_mock()

    ftbot = get_patched_freqtradebot(mocker, edge_conf)
    ftbot.startup()
    assert reinit_mock.call_count == 0
