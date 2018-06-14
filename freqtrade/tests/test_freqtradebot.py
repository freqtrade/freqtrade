# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

"""
Unit test file for freqtradebot.py
"""

import logging
import re
import time
from copy import deepcopy
from unittest.mock import MagicMock

import arrow
import pytest
import requests

from freqtrade import DependencyException, OperationalException, TemporaryError
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.state import State
from freqtrade.tests.conftest import log_has, patch_coinmarketcap


# Functions for recurrent object patching
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot:
    """
    This function patch _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: None
    """
    mocker.patch('freqtrade.freqtradebot.Analyze', MagicMock())
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    mocker.patch('freqtrade.freqtradebot.persistence.init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.exchange.init', MagicMock())
    patch_coinmarketcap(mocker)

    return FreqtradeBot(config)


def patch_get_signal(mocker, value=(True, False)) -> None:
    """

    :param mocker: mocker to patch Analyze class
    :param value: which value Analyze.get_signal() must return
    :return: None
    """
    mocker.patch(
        'freqtrade.freqtradebot.Analyze.get_signal',
        side_effect=lambda s, t: value
    )


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
def test_freqtradebot_object() -> None:
    """
    Test the FreqtradeBot object has the mandatory public methods
    """
    assert hasattr(FreqtradeBot, 'worker')
    assert hasattr(FreqtradeBot, 'cleanup')
    assert hasattr(FreqtradeBot, 'create_trade')
    assert hasattr(FreqtradeBot, 'get_target_bid')
    assert hasattr(FreqtradeBot, 'process_maybe_execute_buy')
    assert hasattr(FreqtradeBot, 'process_maybe_execute_sell')
    assert hasattr(FreqtradeBot, 'handle_trade')
    assert hasattr(FreqtradeBot, 'check_handle_timedout')
    assert hasattr(FreqtradeBot, 'handle_timedout_limit_buy')
    assert hasattr(FreqtradeBot, 'handle_timedout_limit_sell')
    assert hasattr(FreqtradeBot, 'execute_sell')


def test_freqtradebot(mocker, default_conf) -> None:
    """
    Test __init__, _init_modules, update_state, and get_state methods
    """
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert freqtrade.state is State.RUNNING

    conf = deepcopy(default_conf)
    conf.pop('initial_state')
    freqtrade = FreqtradeBot(conf)
    assert freqtrade.state is State.STOPPED


def test_cleanup(mocker, default_conf, caplog) -> None:
    """
    Test clean() method
    """
    mock_cleanup = MagicMock()
    mocker.patch('freqtrade.persistence.cleanup', mock_cleanup)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.cleanup()
    assert log_has('Cleaning up modules ...', caplog.record_tuples)
    assert mock_cleanup.call_count == 1


def test_worker_running(mocker, default_conf, caplog) -> None:
    """
    Test worker() method. Test when we start the bot
    """
    mock_throttle = MagicMock()
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._throttle', mock_throttle)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    state = freqtrade.worker(old_state=None)
    assert state is State.RUNNING
    assert log_has('Changing state to: RUNNING', caplog.record_tuples)
    assert mock_throttle.call_count == 1


def test_worker_stopped(mocker, default_conf, caplog) -> None:
    """
    Test worker() method. Test when we stop the bot
    """
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
    """
    Test _throttle() method
    """
    def func():
        """
        Test function to throttle
        """
        return 42

    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    start = time.time()
    result = freqtrade._throttle(func, min_secs=0.1)
    end = time.time()

    assert result == 42
    assert end - start > 0.1
    assert log_has('Throttling func for 0.10 seconds', caplog.record_tuples)

    result = freqtrade._throttle(func, min_secs=-1)
    assert result == 42


def test_throttle_with_assets(mocker, default_conf) -> None:
    """
    Test _throttle() method when the function passed can have parameters
    """
    def func(nb_assets=-1):
        """
        Test function to throttle
        """
        return nb_assets

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    result = freqtrade._throttle(func, min_secs=0.1, nb_assets=666)
    assert result == 666

    result = freqtrade._throttle(func, min_secs=0.1)
    assert result == -1


def test_gen_pair_whitelist(mocker, default_conf, tickers) -> None:
    """
    Test _gen_pair_whitelist() method
    """
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.freqtradebot.exchange.get_tickers', tickers)
    mocker.patch('freqtrade.freqtradebot.exchange.exchange_has', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))

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


@pytest.mark.skip(reason="Test not implemented")
def test_refresh_whitelist() -> None:
    """
    Test _refresh_whitelist() method
    """
    pass


def test_create_trade(default_conf, ticker, limit_buy_order, fee, mocker) -> None:
    """
    Test create_trade() method
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf)
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


def test_create_trade_minimal_amount(default_conf, ticker, limit_buy_order, fee, mocker) -> None:
    """
    Test create_trade() method
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    buy_mock = MagicMock(return_value={'id': limit_buy_order['id']})
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=buy_mock,
        get_fee=fee,
    )

    conf = deepcopy(default_conf)
    conf['stake_amount'] = 0.0005
    freqtrade = FreqtradeBot(conf)

    freqtrade.create_trade()
    rate, amount = buy_mock.call_args[0][1], buy_mock.call_args[0][2]
    assert rate * amount >= conf['stake_amount']


def test_create_trade_no_stake_amount(default_conf, ticker, limit_buy_order, fee, mocker) -> None:
    """
    Test create_trade() method
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_balance=MagicMock(return_value=default_conf['stake_amount'] * 0.5),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        freqtrade.create_trade()


def test_create_trade_no_pairs(default_conf, ticker, limit_buy_order, fee, mocker) -> None:
    """
    Test create_trade() method
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )

    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'] = ["ETH/BTC"]
    conf['exchange']['pair_blacklist'] = ["ETH/BTC"]
    freqtrade = FreqtradeBot(conf)

    freqtrade.create_trade()

    with pytest.raises(DependencyException, match=r'.*No currency pairs in whitelist.*'):
        freqtrade.create_trade()


def test_create_trade_no_pairs_after_blacklist(default_conf, ticker,
                                               limit_buy_order, fee, mocker) -> None:
    """
    Test create_trade() method
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )

    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'] = ["ETH/BTC"]
    conf['exchange']['pair_blacklist'] = ["ETH/BTC"]
    freqtrade = FreqtradeBot(conf)

    freqtrade.create_trade()

    with pytest.raises(DependencyException, match=r'.*No currency pairs in whitelist.*'):
        freqtrade.create_trade()


def test_create_trade_no_signal(default_conf, fee, mocker) -> None:
    """
    Test create_trade() method
    """
    conf = deepcopy(default_conf)
    conf['dry_run'] = True

    patch_get_signal(mocker, value=(False, False))
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker_history=MagicMock(return_value=20),
        get_balance=MagicMock(return_value=20),
        get_fee=fee,
    )

    conf = deepcopy(default_conf)
    conf['stake_amount'] = 10
    freqtrade = FreqtradeBot(conf)

    Trade.query = MagicMock()
    Trade.query.filter = MagicMock()
    assert not freqtrade.create_trade()


def test_process_trade_creation(default_conf, ticker, limit_buy_order,
                                markets, fee, mocker, caplog) -> None:
    """
    Test the trade creation in _process() method
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker, value={'price_usd': 12345.0})
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_markets=markets,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_order=MagicMock(return_value=limit_buy_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)

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
        'Checking buy signals to create a new trade with stake_amount: 0.001000 ...',
        caplog.record_tuples
    )


def test_process_exchange_failures(default_conf, ticker, markets, mocker) -> None:
    """
    Test _process() method when a RequestException happens
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker, value={'price_usd': 12345.0})
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_markets=markets,
        buy=MagicMock(side_effect=TemporaryError)
    )
    sleep_mock = mocker.patch('time.sleep', side_effect=lambda _: None)

    freqtrade = FreqtradeBot(default_conf)
    result = freqtrade._process()
    assert result is False
    assert sleep_mock.has_calls()


def test_process_operational_exception(default_conf, ticker, markets, mocker) -> None:
    """
    Test _process() method when an OperationalException happens
    """
    patch_get_signal(mocker)
    msg_mock = patch_RPCManager(mocker)
    patch_coinmarketcap(mocker, value={'price_usd': 12345.0})
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_markets=markets,
        buy=MagicMock(side_effect=OperationalException)
    )
    freqtrade = FreqtradeBot(default_conf)
    assert freqtrade.state == State.RUNNING

    result = freqtrade._process()
    assert result is False
    assert freqtrade.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]


def test_process_trade_handling(
        default_conf, ticker, limit_buy_order, markets, fee, mocker) -> None:
    """
    Test _process()
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker, value={'price_usd': 12345.0})
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_markets=markets,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_order=MagicMock(return_value=limit_buy_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades
    result = freqtrade._process()
    assert result is True
    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1

    result = freqtrade._process()
    assert result is False


def test_balance_fully_ask_side(mocker) -> None:
    """
    Test get_target_bid() method
    """
    freqtrade = get_patched_freqtradebot(mocker, {'bid_strategy': {'ask_last_balance': 0.0}})

    assert freqtrade.get_target_bid({'ask': 20, 'last': 10}) == 20


def test_balance_fully_last_side(mocker) -> None:
    """
    Test get_target_bid() method
    """
    freqtrade = get_patched_freqtradebot(mocker, {'bid_strategy': {'ask_last_balance': 1.0}})

    assert freqtrade.get_target_bid({'ask': 20, 'last': 10}) == 10


def test_balance_bigger_last_ask(mocker) -> None:
    """
    Test get_target_bid() method
    """
    freqtrade = get_patched_freqtradebot(mocker, {'bid_strategy': {'ask_last_balance': 1.0}})

    assert freqtrade.get_target_bid({'ask': 5, 'last': 10}) == 5


def test_process_maybe_execute_buy(mocker, default_conf) -> None:
    """
    Test process_maybe_execute_buy() method
    """
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade', MagicMock(return_value=True))
    assert freqtrade.process_maybe_execute_buy()

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade', MagicMock(return_value=False))
    assert not freqtrade.process_maybe_execute_buy()


def test_process_maybe_execute_buy_exception(mocker, default_conf, caplog) -> None:
    """
    Test exception on process_maybe_execute_buy() method
    """
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.create_trade',
        MagicMock(side_effect=DependencyException)
    )
    freqtrade.process_maybe_execute_buy()
    log_has('Unable to create trade:', caplog.record_tuples)


def test_process_maybe_execute_sell(mocker, default_conf, limit_buy_order, caplog) -> None:
    """
    Test process_maybe_execute_sell() method
    """
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.freqtradebot.exchange.get_order', return_value=limit_buy_order)
    mocker.patch('freqtrade.freqtradebot.exchange.get_trades_for_order', return_value=[])
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
    """
    Test the exceptions in process_maybe_execute_sell()
    """
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.freqtradebot.exchange.get_order', return_value=limit_buy_order)

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


def test_handle_trade(default_conf, limit_buy_order, limit_sell_order, fee, mocker) -> None:
    """
    Test check_handle() method
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee
    )
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})

    freqtrade = FreqtradeBot(default_conf)

    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    time.sleep(0.01)  # Race condition fix
    trade.update(limit_buy_order)
    assert trade.is_open is True

    patch_get_signal(mocker, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.open_order_id == limit_sell_order['id']

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    assert trade.close_rate == 0.00001173
    assert trade.close_profit == 0.06201057
    assert trade.calc_profit() == 0.00006217
    assert trade.close_date is not None


def test_handle_overlpapping_signals(default_conf, ticker, limit_buy_order, fee, mocker) -> None:
    """
    Test check_handle() method
    """
    conf = deepcopy(default_conf)
    conf.update({'experimental': {'use_sell_signal': True}})

    patch_get_signal(mocker, value=(True, True))
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.Analyze.min_roi_reached', return_value=False)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )

    freqtrade = FreqtradeBot(conf)

    freqtrade.create_trade()

    # Buy and Sell triggering, so doing nothing ...
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 0

    # Buy is triggering, so buying ...
    patch_get_signal(mocker, value=(True, False))
    freqtrade.create_trade()
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Buy and Sell are not triggering, so doing nothing ...
    patch_get_signal(mocker, value=(False, False))
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Buy and Sell are triggering, so doing nothing ...
    patch_get_signal(mocker, value=(True, True))
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Sell is triggering, guess what : we are Selling!
    patch_get_signal(mocker, value=(False, True))
    trades = Trade.query.all()
    assert freqtrade.handle_trade(trades[0]) is True


def test_handle_trade_roi(default_conf, ticker, limit_buy_order, fee, mocker, caplog) -> None:
    """
    Test check_handle() method
    """
    caplog.set_level(logging.DEBUG)
    conf = deepcopy(default_conf)
    conf.update({'experimental': {'use_sell_signal': True}})

    patch_get_signal(mocker, value=(True, False))
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )

    mocker.patch('freqtrade.freqtradebot.Analyze.min_roi_reached', return_value=True)
    freqtrade = FreqtradeBot(conf)
    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.is_open = True

    # FIX: sniffing logs, suggest handle_trade should not execute_sell
    #      instead that responsibility should be moved out of handle_trade(),
    #      we might just want to check if we are in a sell condition without
    #      executing
    # if ROI is reached we must sell
    patch_get_signal(mocker, value=(False, True))
    assert freqtrade.handle_trade(trade)
    assert log_has('Required profit reached. Selling..', caplog.record_tuples)


def test_handle_trade_experimental(
        default_conf, ticker, limit_buy_order, fee, mocker, caplog) -> None:
    """
    Test check_handle() method
    """
    caplog.set_level(logging.DEBUG)
    conf = deepcopy(default_conf)
    conf.update({'experimental': {'use_sell_signal': True}})

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )
    mocker.patch('freqtrade.freqtradebot.Analyze.min_roi_reached', return_value=False)

    freqtrade = FreqtradeBot(conf)
    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.is_open = True

    patch_get_signal(mocker, value=(False, False))
    assert not freqtrade.handle_trade(trade)

    patch_get_signal(mocker, value=(False, True))
    assert freqtrade.handle_trade(trade)
    assert log_has('Sell signal received. Selling..', caplog.record_tuples)


def test_close_trade(default_conf, ticker, limit_buy_order, limit_sell_order, fee, mocker) -> None:
    """
    Test check_handle() method
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)

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
    """
    Test check_handle_timedout() method
    """
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
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
    freqtrade.check_handle_timedout(600)
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(trade_buy.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0


def test_check_handle_timedout_sell(default_conf, ticker, limit_sell_order_old, mocker) -> None:
    """
    Test check_handle_timedout() method
    """
    rpc_mock = patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
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
    freqtrade.check_handle_timedout(600)
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    assert trade_sell.is_open is True


def test_check_handle_timedout_partial(default_conf, ticker, limit_buy_order_old_partial,
                                       mocker) -> None:
    """
    Test check_handle_timedout() method
    """
    rpc_mock = patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
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
    freqtrade.check_handle_timedout(600)
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(trade_buy.open_order_id)).all()
    assert len(trades) == 1
    assert trades[0].amount == 23.0
    assert trades[0].stake_amount == trade_buy.open_rate * trades[0].amount


def test_check_handle_timedout_exception(default_conf, ticker, mocker, caplog) -> None:
    """
    Test check_handle_timedout() method when get_order throw an exception
    """
    patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_coinmarketcap(mocker)

    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        handle_timedout_limit_buy=MagicMock(),
        handle_timedout_limit_sell=MagicMock(),
    )
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
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

    freqtrade.check_handle_timedout(600)
    assert filter(regexp.match, caplog.record_tuples)


def test_handle_timedout_limit_buy(mocker, default_conf) -> None:
    """
    Test handle_timedout_limit_buy() method
    """
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
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
    """
    Test handle_timedout_limit_sell() method
    """
    patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
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


def test_execute_sell_up(default_conf, ticker, fee, ticker_sell_up, mocker) -> None:
    """
    Test execute_sell() method with a ticker going UP
    """
    patch_get_signal(mocker)
    rpc_mock = patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee
    )
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    freqtrade = FreqtradeBot(default_conf)

    # Create some test data
    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_up
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'])

    assert rpc_mock.call_count == 2
    assert 'Selling' in rpc_mock.call_args_list[-1][0][0]
    assert '[ETH/BTC]' in rpc_mock.call_args_list[-1][0][0]
    assert 'Amount' in rpc_mock.call_args_list[-1][0][0]
    assert 'Profit' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001172' in rpc_mock.call_args_list[-1][0][0]
    assert 'profit: 6.11%, 0.00006126' in rpc_mock.call_args_list[-1][0][0]
    assert '0.919 USD' in rpc_mock.call_args_list[-1][0][0]


def test_execute_sell_down(default_conf, ticker, fee, ticker_sell_down, mocker) -> None:
    """
    Test execute_sell() method with a ticker going DOWN
    """
    patch_get_signal(mocker)
    rpc_mock = patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    # Create some test data
    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_down
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_down()['bid'])

    assert rpc_mock.call_count == 2
    assert 'Selling' in rpc_mock.call_args_list[-1][0][0]
    assert '[ETH/BTC]' in rpc_mock.call_args_list[-1][0][0]
    assert 'Amount' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001044' in rpc_mock.call_args_list[-1][0][0]
    assert 'loss: -5.48%, -0.00005492' in rpc_mock.call_args_list[-1][0][0]
    assert '-0.824 USD' in rpc_mock.call_args_list[-1][0][0]


def test_execute_sell_without_conf_sell_up(default_conf, ticker, fee,
                                           ticker_sell_up, mocker) -> None:
    """
    Test execute_sell() method with a ticker going DOWN and with a bot config empty
    """
    patch_get_signal(mocker)
    rpc_mock = patch_RPCManager(mocker)
    patch_coinmarketcap(mocker, value={'price_usd': 12345.0})
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    # Create some test data
    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_up
    )
    freqtrade.config = {}

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'])

    assert rpc_mock.call_count == 2
    assert 'Selling' in rpc_mock.call_args_list[-1][0][0]
    assert '[ETH/BTC]' in rpc_mock.call_args_list[-1][0][0]
    assert 'Amount' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001172' in rpc_mock.call_args_list[-1][0][0]
    assert '(profit: 6.11%, 0.00006126)' in rpc_mock.call_args_list[-1][0][0]
    assert 'USD' not in rpc_mock.call_args_list[-1][0][0]


def test_execute_sell_without_conf_sell_down(default_conf, ticker, fee,
                                             ticker_sell_down, mocker) -> None:
    """
    Test execute_sell() method with a ticker going DOWN and with a bot config empty
    """
    patch_get_signal(mocker)
    rpc_mock = patch_RPCManager(mocker)
    patch_coinmarketcap(mocker, value={'price_usd': 12345.0})
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    # Create some test data
    freqtrade.create_trade()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_down
    )

    freqtrade.config = {}
    freqtrade.execute_sell(trade=trade, limit=ticker_sell_down()['bid'])

    assert rpc_mock.call_count == 2
    assert 'Selling' in rpc_mock.call_args_list[-1][0][0]
    assert '[ETH/BTC]' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001044' in rpc_mock.call_args_list[-1][0][0]
    assert 'loss: -5.48%, -0.00005492' in rpc_mock.call_args_list[-1][0][0]


def test_sell_profit_only_enable_profit(default_conf, limit_buy_order, fee, mocker) -> None:
    """
    Test sell_profit_only feature when enabled
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.Analyze.min_roi_reached', return_value=False)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock(return_value={
            'bid': 0.00002172,
            'ask': 0.00002173,
            'last': 0.00002172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )
    conf = deepcopy(default_conf)
    conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
    }
    freqtrade = FreqtradeBot(conf)
    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(mocker, value=(False, True))
    assert freqtrade.handle_trade(trade) is True


def test_sell_profit_only_disable_profit(default_conf, limit_buy_order, fee, mocker) -> None:
    """
    Test sell_profit_only feature when disabled
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.Analyze.min_roi_reached', return_value=False)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock(return_value={
            'bid': 0.00002172,
            'ask': 0.00002173,
            'last': 0.00002172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )
    conf = deepcopy(default_conf)
    conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }
    freqtrade = FreqtradeBot(conf)
    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(mocker, value=(False, True))
    assert freqtrade.handle_trade(trade) is True


def test_sell_profit_only_enable_loss(default_conf, limit_buy_order, fee, mocker) -> None:
    """
    Test sell_profit_only feature when enabled and we have a loss
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.Analyze.min_roi_reached', return_value=False)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock(return_value={
            'bid': 0.00000172,
            'ask': 0.00000173,
            'last': 0.00000172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )
    conf = deepcopy(default_conf)
    conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
    }
    freqtrade = FreqtradeBot(conf)
    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(mocker, value=(False, True))
    assert freqtrade.handle_trade(trade) is False


def test_sell_profit_only_disable_loss(default_conf, limit_buy_order, fee, mocker) -> None:
    """
    Test sell_profit_only feature when enabled and we have a loss
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.freqtradebot.Analyze.min_roi_reached', return_value=False)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock(return_value={
            'bid': 0.00000172,
            'ask': 0.00000173,
            'last': 0.00000172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )

    conf = deepcopy(default_conf)
    conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }

    freqtrade = FreqtradeBot(conf)
    freqtrade.create_trade()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(mocker, value=(False, True))
    assert freqtrade.handle_trade(trade) is True


def test_get_real_amount_quote(default_conf, trades_for_order, buy_order_fee, caplog, mocker):
    """
    Test get_real_amount - fee in quote currency
    """

    mocker.patch('freqtrade.exchange.get_trades_for_order', return_value=trades_for_order)

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
        )
    freqtrade = FreqtradeBot(default_conf)
    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount - (amount * 0.001)
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.992) from Trades',
                   caplog.record_tuples)


def test_get_real_amount_no_trade(default_conf, buy_order_fee, caplog, mocker):
    """
    Test get_real_amount - fee in quote currency
    """

    mocker.patch('freqtrade.exchange.get_trades_for_order', return_value=[])

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
    amount = buy_order_fee['amount']
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) failed: myTrade-Dict empty found',
                   caplog.record_tuples)


def test_get_real_amount_stake(default_conf, trades_for_order, buy_order_fee, mocker):
    """
    Test get_real_amount - fees in Stake currency
    """
    trades_for_order[0]['fee']['currency'] = 'ETH'

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_BNB(default_conf, trades_for_order, buy_order_fee, mocker):
    """
    Test get_real_amount - Fees in BNB
    """

    trades_for_order[0]['fee']['currency'] = 'BNB'
    trades_for_order[0]['fee']['cost'] = 0.00094518

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_multi(default_conf, trades_for_order2, buy_order_fee, caplog, mocker):
    """
    Test get_real_amount with split trades (multiple trades for this order)
    """

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.get_trades_for_order', return_value=trades_for_order2)
    amount = float(sum(x['amount'] for x in trades_for_order2))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount - (amount * 0.001)
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.992) from Trades',
                   caplog.record_tuples)


def test_get_real_amount_fromorder(default_conf, trades_for_order, buy_order_fee, caplog, mocker):
    """
    Test get_real_amount with split trades (multiple trades for this order)
    """
    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['fee'] = {'cost': 0.004, 'currency': 'LTC'}

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.get_trades_for_order', return_value=[trades_for_order])
    amount = float(sum(x['amount'] for x in trades_for_order))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, limit_buy_order) == amount - 0.004
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.996) from Order',
                   caplog.record_tuples)


def test_get_real_amount_invalid_order(default_conf, trades_for_order, buy_order_fee, mocker):
    """
    Test get_real_amount with split trades (multiple trades for this order)
    """
    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['fee'] = {'cost': 0.004}

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.get_trades_for_order', return_value=[])
    amount = float(sum(x['amount'] for x in trades_for_order))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    # Amount does not change
    assert freqtrade.get_real_amount(trade, limit_buy_order) == amount


def test_get_real_amount_invalid(default_conf, trades_for_order, buy_order_fee, mocker):
    """
    Test get_real_amount - fees in Stake currency
    """
    # Remove "Currency" from fee dict
    trades_for_order[0]['fee'] = {'cost': 0.008}

    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = FreqtradeBot(default_conf)
    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_open_trade(default_conf, mocker):
    """
    Test get_real_amount condition trade.fee_open == 0 or order['status'] == 'open'
    """
    patch_get_signal(mocker)
    patch_RPCManager(mocker)
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
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
    assert freqtrade.get_real_amount(trade, order) == amount
