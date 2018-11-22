# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

import logging
import re
import time
from copy import deepcopy
from unittest.mock import MagicMock

import arrow
import pytest
import requests

from freqtrade import (DependencyException, OperationalException,
                       TemporaryError, constants)
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.rpc import RPCMessageType
from freqtrade.state import State
from freqtrade.strategy.interface import SellType, SellCheckTuple
from freqtrade.tests.conftest import log_has, patch_exchange, patch_edge


# Functions for recurrent object patching
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot:
    """
    This function patch _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: None
    """
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    mocker.patch('freqtrade.freqtradebot.persistence.init', MagicMock())
    patch_exchange(mocker)

    return FreqtradeBot(config)


def patch_get_signal(freqtrade: FreqtradeBot, value=(True, False)) -> None:
    """
    :param mocker: mocker to patch IStrategy class
    :param value: which value IStrategy.get_signal() must return
    :return: None
    """
    freqtrade.strategy.get_signal = lambda e, s, t: value
    freqtrade.exchange.refresh_tickers = lambda p, i: None


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

def test_freqtradebot(mocker, default_conf) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert freqtrade.state is State.RUNNING

    default_conf.pop('initial_state')
    freqtrade = FreqtradeBot(default_conf)
    assert freqtrade.state is State.STOPPED


def test_cleanup(mocker, default_conf, caplog) -> None:
    mock_cleanup = MagicMock()
    mocker.patch('freqtrade.persistence.cleanup', mock_cleanup)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.cleanup()
    assert log_has('Cleaning up modules ...', caplog.record_tuples)
    assert mock_cleanup.call_count == 1


def test_worker_running(mocker, default_conf, caplog) -> None:
    mock_throttle = MagicMock()
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._throttle', mock_throttle)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    state = freqtrade.worker(old_state=None)
    assert state is State.RUNNING
    assert log_has('Changing state to: RUNNING', caplog.record_tuples)
    assert mock_throttle.call_count == 1


def test_worker_stopped(mocker, default_conf, caplog) -> None:
    mock_throttle = MagicMock()
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._throttle', mock_throttle)
    mock_sleep = mocker.patch('time.sleep', return_value=None)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.state = State.STOPPED
    state = freqtrade.worker(old_state=State.RUNNING)
    assert state is State.STOPPED
    assert log_has('Changing state to: STOPPED', caplog.record_tuples)
    assert mock_throttle.call_count == 0
    assert mock_sleep.call_count == 1


def test_throttle(mocker, default_conf, caplog) -> None:
    def throttled_func():
        return 42

    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    start = time.time()
    result = freqtrade._throttle(throttled_func, min_secs=0.1)
    end = time.time()

    assert result == 42
    assert end - start > 0.1
    assert log_has('Throttling throttled_func for 0.10 seconds', caplog.record_tuples)

    result = freqtrade._throttle(throttled_func, min_secs=-1)
    assert result == 42


def test_throttle_with_assets(mocker, default_conf) -> None:
    def throttled_func(nb_assets=-1):
        return nb_assets

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    result = freqtrade._throttle(throttled_func, min_secs=0.1, nb_assets=666)
    assert result == 666

    result = freqtrade._throttle(throttled_func, min_secs=0.1)
    assert result == -1


def test_gen_pair_whitelist(mocker, default_conf, tickers) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))

    # Test to retrieved BTC sorted on quoteVolume (default)
    whitelist = freqtrade._gen_pair_whitelist(base_currency='BTC')
    assert whitelist == ['ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC']

    # Test to retrieve BTC sorted on bidVolume
    whitelist = freqtrade._gen_pair_whitelist(base_currency='BTC', key='bidVolume')
    assert whitelist == ['LTC/BTC', 'TKN/BTC', 'ETH/BTC', 'BLK/BTC']

    # Test with USDT sorted on quoteVolume (default)
    whitelist = freqtrade._gen_pair_whitelist(base_currency='USDT')
    assert whitelist == ['TKN/USDT', 'ETH/USDT', 'LTC/USDT', 'BLK/USDT']

    # Test with ETH (our fixture does not have ETH, so result should be empty)
    whitelist = freqtrade._gen_pair_whitelist(base_currency='ETH')
    assert whitelist == []


def test_gen_pair_whitelist_not_supported(mocker, default_conf, tickers) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=False))

    with pytest.raises(OperationalException):
        freqtrade._gen_pair_whitelist(base_currency='BTC')


def test_get_trade_stake_amount(default_conf, ticker, limit_buy_order, fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balance=MagicMock(return_value=default_conf['stake_amount'] * 2)
    )

    freqtrade = FreqtradeBot(default_conf)

    result = freqtrade._get_trade_stake_amount('ETH/BTC')
    assert result == default_conf['stake_amount']


def test_get_trade_stake_amount_no_stake_amount(default_conf,
                                                ticker,
                                                limit_buy_order,
                                                fee,
                                                mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balance=MagicMock(return_value=default_conf['stake_amount'] * 0.5)
    )
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
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_balance=MagicMock(return_value=default_conf['stake_amount']),
        get_fee=fee,
        get_markets=markets
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
    freqtrade.create_trade()

    result = freqtrade._get_trade_stake_amount('LTC/BTC')
    assert result == default_conf['stake_amount'] / (conf['max_open_trades'] - 1)

    # create 2 trades, order amount should be None
    freqtrade.create_trade()

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
    freqtrade._refresh_whitelist = _refresh_whitelist
    patch_get_signal(freqtrade)
    freqtrade._process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker, edge_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    freqtrade = FreqtradeBot(edge_conf)

    assert freqtrade._get_trade_stake_amount('NEO/BTC') == (0.001 * 0.01) / 0.20
    assert freqtrade._get_trade_stake_amount('LTC/BTC') == (0.001 * 0.01) / 0.20


def test_edge_overrides_stoploss(limit_buy_order, fee, markets, caplog, mocker, edge_conf) -> None:

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)

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
        get_markets=markets,
    )
    #############################################

    # Create a trade with "limit_buy_order" price
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False
    freqtrade.create_trade()
    trade = Trade.query.first()
    trade.update(limit_buy_order)
    #############################################

    # stoploss shoud be hit
    assert freqtrade.handle_trade(trade) is True
    assert log_has('executed sell, reason: SellType.STOP_LOSS', caplog.record_tuples)
    assert trade.sell_reason == SellType.STOP_LOSS.value


def test_edge_should_ignore_strategy_stoploss(limit_buy_order, fee, markets,
                                              mocker, edge_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)

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
        get_markets=markets,
    )
    #############################################

    # Create a trade with "limit_buy_order" price
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False
    freqtrade.create_trade()
    trade = Trade.query.first()
    trade.update(limit_buy_order)
    #############################################

    # stoploss shoud not be hit
    assert freqtrade.handle_trade(trade) is False


def test_get_min_pair_stake_amount(mocker, default_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.stoploss = -0.05
    # no pair found
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC'
        }])
    )
    with pytest.raises(ValueError, match=r'.*get market information.*'):
        freqtrade._get_min_pair_stake_amount('BNB/BTC', 1)

    # no 'limits' section
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC'
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # empty 'limits' section
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC',
            'limits': {}
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # no cost Min
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC',
            'limits': {
                'cost': {"min": None},
                'amount': {}
            }
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # no amount Min
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC',
            'limits': {
                'cost': {},
                'amount': {"min": None}
            }
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # empty 'cost'/'amount' section
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC',
            'limits': {
                'cost': {},
                'amount': {}
            }
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result is None

    # min cost is set
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC',
            'limits': {
                'cost': {'min': 2},
                'amount': {}
            }
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 1)
    assert result == 2 / 0.9

    # min amount is set
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC',
            'limits': {
                'cost': {},
                'amount': {'min': 2}
            }
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 2)
    assert result == 2 * 2 / 0.9

    # min amount and cost are set (cost is minimal)
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC',
            'limits': {
                'cost': {'min': 2},
                'amount': {'min': 2}
            }
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 2)
    assert result == min(2, 2 * 2) / 0.9

    # min amount and cost are set (amount is minial)
    mocker.patch(
        'freqtrade.exchange.Exchange.get_markets',
        MagicMock(return_value=[{
            'symbol': 'ETH/BTC',
            'limits': {
                'cost': {'min': 8},
                'amount': {'min': 2}
            }
        }])
    )
    result = freqtrade._get_min_pair_stake_amount('ETH/BTC', 2)
    assert result == min(8, 2 * 2) / 0.9


def test_create_trade(default_conf, ticker, limit_buy_order, fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        get_markets=markets
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.create_trade()

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


def test_create_trade_no_stake_amount(default_conf, ticker, limit_buy_order,
                                      fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_balance=MagicMock(return_value=default_conf['stake_amount'] * 0.5),
        get_fee=fee,
        get_markets=markets
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        freqtrade.create_trade()


def test_create_trade_minimal_amount(default_conf, ticker, limit_buy_order,
                                     fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    buy_mock = MagicMock(return_value={'id': limit_buy_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=buy_mock,
        get_fee=fee,
        get_markets=markets
    )
    default_conf['stake_amount'] = 0.0005
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trade()
    rate, amount = buy_mock.call_args[1]['rate'], buy_mock.call_args[1]['amount']
    assert rate * amount >= default_conf['stake_amount']


def test_create_trade_too_small_stake_amount(default_conf, ticker, limit_buy_order,
                                             fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    buy_mock = MagicMock(return_value={'id': limit_buy_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=buy_mock,
        get_fee=fee,
        get_markets=markets
    )

    default_conf['stake_amount'] = 0.000000005
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    result = freqtrade.create_trade()
    assert result is False


def test_create_trade_limit_reached(default_conf, ticker, limit_buy_order,
                                    fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_balance=MagicMock(return_value=default_conf['stake_amount']),
        get_fee=fee,
        get_markets=markets
    )
    default_conf['max_open_trades'] = 0
    default_conf['stake_amount'] = constants.UNLIMITED_STAKE_AMOUNT

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    assert freqtrade.create_trade() is False
    assert freqtrade._get_trade_stake_amount('ETH/BTC') is None


def test_create_trade_no_pairs(default_conf, ticker, limit_buy_order, fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        get_markets=markets
    )

    default_conf['exchange']['pair_whitelist'] = ["ETH/BTC"]
    default_conf['exchange']['pair_blacklist'] = ["ETH/BTC"]
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trade()

    with pytest.raises(DependencyException, match=r'.*No currency pairs in whitelist.*'):
        freqtrade.create_trade()


def test_create_trade_no_pairs_after_blacklist(default_conf, ticker,
                                               limit_buy_order, fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        get_markets=markets
    )
    default_conf['exchange']['pair_whitelist'] = ["ETH/BTC"]
    default_conf['exchange']['pair_blacklist'] = ["ETH/BTC"]
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trade()

    with pytest.raises(DependencyException, match=r'.*No currency pairs in whitelist.*'):
        freqtrade.create_trade()


def test_create_trade_no_signal(default_conf, fee, mocker) -> None:
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
    assert not freqtrade.create_trade()


def test_process_trade_creation(default_conf, ticker, limit_buy_order,
                                markets, fee, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_markets=markets,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_order=MagicMock(return_value=limit_buy_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades

    result = freqtrade._process()
    assert result is True

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
        'Buy signal found: about create a new trade with stake_amount: 0.001000 ...',
        caplog.record_tuples
    )


def test_process_exchange_failures(default_conf, ticker, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_markets=markets,
        buy=MagicMock(side_effect=TemporaryError)
    )
    sleep_mock = mocker.patch('time.sleep', side_effect=lambda _: None)

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    result = freqtrade._process()
    assert result is False
    assert sleep_mock.has_calls()


def test_process_operational_exception(default_conf, ticker, markets, mocker) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_markets=markets,
        buy=MagicMock(side_effect=OperationalException)
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    assert freqtrade.state == State.RUNNING

    result = freqtrade._process()
    assert result is False
    assert freqtrade.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]['status']


def test_process_trade_handling(
        default_conf, ticker, limit_buy_order, markets, fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_markets=markets,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_order=MagicMock(return_value=limit_buy_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades
    result = freqtrade._process()
    assert result is True
    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1

    result = freqtrade._process()
    assert result is False


def test_process_trade_no_whitelist_pair(
        default_conf, ticker, limit_buy_order, markets, fee, mocker) -> None:
    """ Test _process with trade not in pair list """
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_markets=markets,
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
    result = freqtrade._process()
    assert pair in freqtrade.active_pair_whitelist
    # Make sure each pair is only in the list once
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))
    assert result is True


def test_balance_fully_ask_side(mocker, default_conf) -> None:
    default_conf['bid_strategy']['ask_last_balance'] = 0.0
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    assert freqtrade.get_target_bid('ETH/BTC', {'ask': 20, 'last': 10}) == 20


def test_balance_fully_last_side(mocker, default_conf) -> None:
    default_conf['bid_strategy']['ask_last_balance'] = 1.0
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    assert freqtrade.get_target_bid('ETH/BTC', {'ask': 20, 'last': 10}) == 10


def test_balance_bigger_last_ask(mocker, default_conf) -> None:
    default_conf['bid_strategy']['ask_last_balance'] = 1.0
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    assert freqtrade.get_target_bid('ETH/BTC', {'ask': 5, 'last': 10}) == 5


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
        get_markets=markets
    )
    pair = 'ETH/BTC'
    print(buy_mm.call_args_list)

    assert freqtrade.execute_buy(pair, stake_amount)
    assert get_bid.call_count == 1
    assert buy_mm.call_count == 1
    call_args = buy_mm.call_args_list[0][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == bid
    assert call_args['amount'] == stake_amount / bid

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


def test_execute_buy_with_stoploss_on_exchange(mocker, default_conf,
                                               fee, markets, limit_buy_order) -> None:
    default_conf['exchange']['name'] = 'binance'
    patch_RPCManager(mocker)
    patch_exchange(mocker, id='binance')
    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.stoploss_on_exchange = True
    freqtrade.strategy.stoploss = -0.05
    stake_amount = 2
    bid = 0.11
    get_bid = MagicMock(return_value=bid)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        get_target_bid=get_bid,
        _get_min_pair_stake_amount=MagicMock(return_value=1)
    )
    buy_mm = MagicMock(return_value={'id': limit_buy_order['id']})
    stoploss_limit = MagicMock(return_value={'id': 13434334})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=buy_mm,
        get_fee=fee,
        get_markets=markets,
        stoploss_limit=stoploss_limit
    )
    pair = 'ETH/BTC'
    print(buy_mm.call_args_list)

    assert freqtrade.execute_buy(pair, stake_amount)
    assert stoploss_limit.call_count == 1
    assert get_bid.call_count == 1
    assert buy_mm.call_count == 1
    call_args = buy_mm.call_args_list[0][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == bid
    assert call_args['amount'] == stake_amount / bid

    call_args = stoploss_limit.call_args_list[0][1]
    assert call_args['pair'] == pair
    assert call_args['amount'] == stake_amount / bid
    assert call_args['stop_price'] == 0.11 * 0.95
    assert call_args['rate'] == 0.11 * 0.95 * 0.98

    trade = Trade.query.first()
    assert trade.is_open is True
    assert trade.stoploss_order_id == 13434334


def test_process_maybe_execute_buy(mocker, default_conf) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade', MagicMock(return_value=True))
    assert freqtrade.process_maybe_execute_buy()

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade', MagicMock(return_value=False))
    assert not freqtrade.process_maybe_execute_buy()


def test_process_maybe_execute_buy_exception(mocker, default_conf, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.create_trade',
        MagicMock(side_effect=DependencyException)
    )
    freqtrade.process_maybe_execute_buy()
    log_has('Unable to create trade:', caplog.record_tuples)


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
        'Applying fee to amount for Trade {} from 90.99181073 to 90.81'.format(trade),
        caplog.record_tuples
    )

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=90.81)
    # test amount modified by fee-logic
    assert not freqtrade.process_maybe_execute_sell(trade)

    trade.is_open = True
    trade.open_order_id = None
    # Assert we call handle_trade() if trade is feasible for execution
    assert freqtrade.process_maybe_execute_sell(trade)

    regexp = re.compile('Found open order for.*')
    assert filter(regexp.match, caplog.record_tuples)


def test_process_maybe_execute_sell_exception(mocker, default_conf,
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
    freqtrade.process_maybe_execute_sell(trade)
    assert log_has('could not update trade amount: ', caplog.record_tuples)

    # Test raise of DependencyException exception
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
        side_effect=DependencyException()
    )
    freqtrade.process_maybe_execute_sell(trade)
    assert log_has('Unable to sell trade: ', caplog.record_tuples)


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
        get_markets=markets
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trade()

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
        get_markets=markets
    )

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade, value=(True, True))
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False

    freqtrade.create_trade()

    # Buy and Sell triggering, so doing nothing ...
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 0

    # Buy is triggering, so buying ...
    patch_get_signal(freqtrade, value=(True, False))
    freqtrade.create_trade()
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
        get_markets=markets
    )

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade, value=(True, False))
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: True

    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.is_open = True

    # FIX: sniffing logs, suggest handle_trade should not execute_sell
    #      instead that responsibility should be moved out of handle_trade(),
    #      we might just want to check if we are in a sell condition without
    #      executing
    # if ROI is reached we must sell
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade)
    assert log_has('Required profit reached. Selling..', caplog.record_tuples)


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
        get_markets=markets
    )

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False
    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.is_open = True

    patch_get_signal(freqtrade, value=(False, False))
    assert not freqtrade.handle_trade(trade)

    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade)
    assert log_has('Sell signal received. Selling..', caplog.record_tuples)


def test_close_trade(default_conf, ticker, limit_buy_order, limit_sell_order,
                     fee, markets, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        get_markets=markets
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create trade and sell it
    freqtrade.create_trade()

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
    regexp = re.compile(
        'Cannot query order for Trade(id=1, pair=ETH/BTC, amount=90.99181073, '
        'open_rate=0.00001099, open_since=10 hours ago) due to Traceback (most '
        'recent call last):\n.*'
    )

    freqtrade.check_handle_timedout()
    assert filter(regexp.match, caplog.record_tuples)


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
             'amount': 1}
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
        get_markets=markets
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trade()

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
        'market_url': 'https://bittrex.com/Market/Index?MarketName=BTC-ETH',
        'limit': 1.172e-05,
        'amount': 90.99181073703367,
        'open_rate': 1.099e-05,
        'current_rate': 1.172e-05,
        'profit_amount': 6.126e-05,
        'profit_percent': 0.0611052,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
    } == last_msg


def test_execute_sell_down(default_conf, ticker, fee, ticker_sell_down, markets, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trade()

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
        'market_url': 'https://bittrex.com/Market/Index?MarketName=BTC-ETH',
        'limit': 1.044e-05,
        'amount': 90.99181073703367,
        'open_rate': 1.099e-05,
        'current_rate': 1.044e-05,
        'profit_amount': -5.492e-05,
        'profit_percent': -0.05478342,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
    } == last_msg


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
        get_markets=markets
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
    freqtrade.strategy.stoploss_on_exchange = True
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_up
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'], sell_reason=SellType.ROI)

    trade = Trade.query.first()
    assert trade
    assert cancel_order.call_count == 1
    assert rpc_mock.call_count == 2


def test_execute_sell_without_conf_sell_up(default_conf, ticker, fee,
                                           ticker_sell_up, markets, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_up
    )
    freqtrade.config = {}

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'], sell_reason=SellType.ROI)

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'market_url': 'https://bittrex.com/Market/Index?MarketName=BTC-ETH',
        'limit': 1.172e-05,
        'amount': 90.99181073703367,
        'open_rate': 1.099e-05,
        'current_rate': 1.172e-05,
        'profit_amount': 6.126e-05,
        'profit_percent': 0.0611052,
    } == last_msg


def test_execute_sell_without_conf_sell_down(default_conf, ticker, fee,
                                             ticker_sell_down, markets, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker_sell_down
    )

    freqtrade.config = {}
    freqtrade.execute_sell(trade=trade, limit=ticker_sell_down()['bid'],
                           sell_reason=SellType.STOP_LOSS)

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'market_url': 'https://bittrex.com/Market/Index?MarketName=BTC-ETH',
        'limit': 1.044e-05,
        'amount': 90.99181073703367,
        'open_rate': 1.099e-05,
        'current_rate': 1.044e-05,
        'profit_amount': -5.492e-05,
        'profit_percent': -0.05478342,
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
        get_markets=markets
    )
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False

    freqtrade.create_trade()

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
        get_markets=markets
    )
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False
    freqtrade.create_trade()

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
        get_markets=markets
    )
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.stop_loss_reached = \
        lambda current_rate, trade, current_time, force_stoploss, current_profit: SellCheckTuple(
            sell_flag=False, sell_type=SellType.NONE)
    freqtrade.create_trade()

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
        get_markets=markets
    )
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False

    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.SELL_SIGNAL.value


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
        get_markets=markets
    )
    default_conf['experimental'] = {
        'ignore_roi_if_buy_signal': True
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: True

    freqtrade.create_trade()

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
            'bid': 0.00000102,
            'ask': 0.00000103,
            'last': 0.00000102
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
        get_markets=markets,
    )
    default_conf['trailing_stop'] = True
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False

    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    caplog.set_level(logging.DEBUG)
    # Sell as trailing-stop is reached
    assert freqtrade.handle_trade(trade) is True
    assert log_has(
        f'HIT STOP: current price at 0.000001, stop loss is {trade.stop_loss:.6f}, '
        f'initial stop loss was at 0.000010, trade opened at 0.000011', caplog.record_tuples)
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
        get_markets=markets,
    )
    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0.01
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False
    freqtrade.create_trade()

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
    assert log_has(f'using positive stop loss mode: 0.01 with offset 0 '
                   f'since we have profit 0.2666%',
                   caplog.record_tuples)
    assert log_has(f'adjusted stop loss', caplog.record_tuples)
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
        f'HIT STOP: current price at {buy_price + 0.000002:.6f}, '
        f'stop loss is {trade.stop_loss:.6f}, '
        f'initial stop loss was at 0.000010, trade opened at 0.000011', caplog.record_tuples)


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
        get_markets=markets,
    )

    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0.01
    default_conf['trailing_stop_positive_offset'] = 0.011
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: False
    freqtrade.create_trade()

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
    assert log_has(f'using positive stop loss mode: 0.01 with offset 0.011 '
                   f'since we have profit 0.2666%',
                   caplog.record_tuples)
    assert log_has(f'adjusted stop loss', caplog.record_tuples)
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
        f'HIT STOP: current price at {buy_price + 0.000002:.6f}, '
        f'stop loss is {trade.stop_loss:.6f}, '
        f'initial stop loss was at 0.000010, trade opened at 0.000011', caplog.record_tuples)
    assert trade.sell_reason == SellType.TRAILING_STOP_LOSS.value


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
        get_markets=markets
    )
    default_conf['experimental'] = {
        'ignore_roi_if_buy_signal': False
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = lambda trade, current_profit, current_time: True

    freqtrade.create_trade()

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
                   caplog.record_tuples)


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
                   caplog.record_tuples)


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
                   caplog.record_tuples)


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
                   caplog.record_tuples)


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
        get_markets=markets
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.create_trade()

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
        get_markets=markets
    )
    # Save state of current whitelist
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade is None


def test_order_book_bid_strategy1(mocker, default_conf, order_book_l2, markets) -> None:
    """
    test if function get_target_bid will return the order book price
    instead of the ask rate
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_markets=markets,
        get_order_book=order_book_l2
    )
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['use_order_book'] = True
    default_conf['bid_strategy']['order_book_top'] = 2
    default_conf['bid_strategy']['ask_last_balance'] = 0
    default_conf['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf)
    assert freqtrade.get_target_bid('ETH/BTC', {'ask': 0.045, 'last': 0.046}) == 0.043935


def test_order_book_bid_strategy2(mocker, default_conf, order_book_l2, markets) -> None:
    """
    test if function get_target_bid will return the ask rate (since its value is lower)
    instead of the order book rate (even if enabled)
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_markets=markets,
        get_order_book=order_book_l2
    )
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['use_order_book'] = True
    default_conf['bid_strategy']['order_book_top'] = 2
    default_conf['bid_strategy']['ask_last_balance'] = 0
    default_conf['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf)
    assert freqtrade.get_target_bid('ETH/BTC', {'ask': 0.042, 'last': 0.046}) == 0.042


def test_order_book_bid_strategy3(default_conf, mocker, order_book_l2, markets) -> None:
    """
    test if function get_target_bid will return ask rate instead
    of the order book rate
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_markets=markets,
        get_order_book=order_book_l2
    )
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['use_order_book'] = True
    default_conf['bid_strategy']['order_book_top'] = 1
    default_conf['bid_strategy']['ask_last_balance'] = 0
    default_conf['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf)

    assert freqtrade.get_target_bid('ETH/BTC', {'ask': 0.03, 'last': 0.029}) == 0.03


def test_check_depth_of_market_buy(default_conf, mocker, order_book_l2, markets) -> None:
    """
    test check depth of market
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_markets=markets,
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
        get_markets=markets
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    time.sleep(0.01)  # Race condition fix
    trade.update(limit_buy_order)
    assert trade.is_open is True

    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True


def test_startup_messages(default_conf, mocker):
    default_conf['dynamic_whitelist'] = 20
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert freqtrade.state is State.RUNNING
