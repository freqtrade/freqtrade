# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

import logging
import time
from copy import deepcopy
from math import isclose
from unittest.mock import ANY, MagicMock, PropertyMock

import arrow
import pytest

from freqtrade.constants import CANCEL_REASON, MATH_CLOSE_PREC, UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import (DependencyException, ExchangeError, InsufficientFundsError,
                                  InvalidOrderException, OperationalException, PricingError,
                                  TemporaryError)
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, PairLocks, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.rpc import RPCMessageType
from freqtrade.state import RunMode, State
from freqtrade.strategy.interface import SellCheckTuple, SellType
from freqtrade.worker import Worker
from tests.conftest import (create_mock_trades, get_patched_freqtradebot, get_patched_worker,
                            log_has, log_has_re, patch_edge, patch_exchange, patch_get_signal,
                            patch_wallet, patch_whitelist)
from tests.conftest_trades import (MOCK_TRADE_COUNT, mock_order_1, mock_order_2, mock_order_2_sell,
                                   mock_order_3, mock_order_3_sell, mock_order_4,
                                   mock_order_5_stoploss, mock_order_6_sell)


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


def test_process_stopped(mocker, default_conf) -> None:

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    coo_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders')
    freqtrade.process_stopped()
    assert coo_mock.call_count == 0

    default_conf['cancel_open_orders_on_exit'] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.process_stopped()
    assert coo_mock.call_count == 1


def test_bot_cleanup(mocker, default_conf, caplog) -> None:
    mock_cleanup = mocker.patch('freqtrade.freqtradebot.cleanup_db')
    coo_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders')
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.cleanup()
    assert log_has('Cleaning up modules ...', caplog)
    assert mock_cleanup.call_count == 1
    assert coo_mock.call_count == 0

    freqtrade.config['cancel_open_orders_on_exit'] = True
    freqtrade.cleanup()
    assert coo_mock.call_count == 1


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
    conf['bid_strategy']['price_side'] = 'ask'

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
    conf['bid_strategy']['price_side'] = 'ask'

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

    result = freqtrade.wallets.get_trade_stake_amount('ETH/BTC')
    assert result == default_conf['stake_amount']


@pytest.mark.parametrize("amend_last,wallet,max_open,lsamr,expected", [
                        (False, 0.002, 2, 0.5, [0.001, None]),
                        (True, 0.002, 2, 0.5, [0.001, 0.00098]),
                        (False, 0.003, 3, 0.5, [0.001, 0.001, None]),
                        (True, 0.003, 3, 0.5, [0.001, 0.001, 0.00097]),
                        (False, 0.0022, 3, 0.5, [0.001, 0.001, None]),
                        (True, 0.0022, 3, 0.5, [0.001, 0.001, 0.0]),
                        (True, 0.0027, 3, 0.5, [0.001, 0.001, 0.000673]),
                        (True, 0.0022, 3, 1, [0.001, 0.001, 0.0]),
                        ])
def test_check_available_stake_amount(default_conf, ticker, mocker, fee, limit_buy_order_open,
                                      amend_last, wallet, max_open, lsamr, expected) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_balance=MagicMock(return_value=default_conf['stake_amount'] * 2),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee
    )
    default_conf['dry_run_wallet'] = wallet

    default_conf['amend_last_stake_amount'] = amend_last
    default_conf['last_stake_amount_min_ratio'] = lsamr

    freqtrade = FreqtradeBot(default_conf)

    for i in range(0, max_open):

        if expected[i] is not None:
            limit_buy_order_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/BTC')
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_buy('ETH/BTC', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/BTC')


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
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)

    assert freqtrade.wallets.get_trade_stake_amount(
        'NEO/BTC', freqtrade.edge) == (999.9 * 0.5 * 0.01) / 0.20
    assert freqtrade.wallets.get_trade_stake_amount(
        'LTC/BTC', freqtrade.edge) == (999.9 * 0.5 * 0.01) / 0.21


def test_edge_overrides_stoploss(limit_buy_order, fee, caplog, mocker, edge_conf) -> None:

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
        fetch_ticker=MagicMock(return_value={
            'bid': buy_price * 0.79,
            'ask': buy_price * 0.79,
            'last': buy_price * 0.79
        }),
        get_fee=fee,
    )
    #############################################

    # Create a trade with "limit_buy_order" price
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.update(limit_buy_order)
    #############################################

    # stoploss shoud be hit
    assert freqtrade.handle_trade(trade) is True
    assert log_has('Executing Sell for NEO/BTC. Reason: stop_loss', caplog)
    assert trade.sell_reason == SellType.STOP_LOSS.value


def test_edge_should_ignore_strategy_stoploss(limit_buy_order, fee,
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
        fetch_ticker=MagicMock(return_value={
            'bid': buy_price * 0.85,
            'ask': buy_price * 0.85,
            'last': buy_price * 0.85
        }),
        get_fee=fee,
    )
    #############################################

    # Create a trade with "limit_buy_order" price
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.update(limit_buy_order)
    #############################################

    # stoploss shoud not be hit
    assert freqtrade.handle_trade(trade) is False


def test_total_open_trades_stakes(mocker, default_conf, ticker, fee) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf['stake_amount'] = 0.00098751
    default_conf['max_open_trades'] = 2
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()
    trade = Trade.query.first()

    assert trade is not None
    assert trade.stake_amount == 0.00098751
    assert trade.is_open
    assert trade.open_date is not None

    freqtrade.enter_positions()
    trade = Trade.query.order_by(Trade.id.desc()).first()

    assert trade is not None
    assert trade.stake_amount == 0.00098751
    assert trade.is_open
    assert trade.open_date is not None

    assert Trade.total_open_trades_stakes() == 1.97502e-03


def test_create_trade(default_conf, ticker, limit_buy_order, fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.create_trade('ETH/BTC')

    trade = Trade.query.first()
    assert trade is not None
    assert trade.stake_amount == 0.001
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    assert trade.open_rate == 0.00001099
    assert trade.amount == 90.99181073

    assert whitelist == default_conf['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(default_conf, ticker, limit_buy_order,
                                      fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf['stake_amount'] * 0.5)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        freqtrade.create_trade('ETH/BTC')


def test_create_trade_minimal_amount(default_conf, ticker, limit_buy_order_open,
                                     fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    buy_mock = MagicMock(return_value=limit_buy_order_open)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=buy_mock,
        get_fee=fee,
    )
    default_conf['stake_amount'] = 0.0005
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.create_trade('ETH/BTC')
    rate, amount = buy_mock.call_args[1]['rate'], buy_mock.call_args[1]['amount']
    assert rate * amount <= default_conf['stake_amount']


def test_create_trade_too_small_stake_amount(default_conf, ticker, limit_buy_order_open,
                                             fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    buy_mock = MagicMock(return_value=limit_buy_order_open)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=buy_mock,
        get_fee=fee,
    )

    freqtrade = FreqtradeBot(default_conf)
    freqtrade.config['stake_amount'] = 0.000000005

    patch_get_signal(freqtrade)

    assert not freqtrade.create_trade('ETH/BTC')


def test_create_trade_limit_reached(default_conf, ticker, limit_buy_order_open,
                                    fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_balance=MagicMock(return_value=default_conf['stake_amount']),
        get_fee=fee,
    )
    default_conf['max_open_trades'] = 0
    default_conf['stake_amount'] = UNLIMITED_STAKE_AMOUNT

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    assert not freqtrade.create_trade('ETH/BTC')
    assert freqtrade.wallets.get_trade_stake_amount('ETH/BTC', freqtrade.edge) == 0


def test_enter_positions_no_pairs_left(default_conf, ticker, limit_buy_order_open, fee,
                                       mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )

    default_conf['exchange']['pair_whitelist'] = ["ETH/BTC"]
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    n = freqtrade.enter_positions()
    assert n == 1
    assert not log_has_re(r"No currency pair in active pair whitelist.*", caplog)
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(r"No currency pair in active pair whitelist.*", caplog)


def test_enter_positions_no_pairs_in_whitelist(default_conf, ticker, limit_buy_order, fee,
                                               mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )
    default_conf['exchange']['pair_whitelist'] = []
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has("Active pair whitelist is empty.", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_enter_positions_global_pairlock(default_conf, ticker, limit_buy_order, fee,
                                         mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = r"Global pairlock active until.* Not creating new trades."
    n = freqtrade.enter_positions()
    # 0 trades, but it's not because of pairlock.
    assert n == 0
    assert not log_has_re(message, caplog)

    PairLocks.lock_pair('*', arrow.utcnow().shift(minutes=20).datetime, 'Just because')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


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
    assert not freqtrade.create_trade('ETH/BTC')


@pytest.mark.parametrize("max_open", range(0, 5))
@pytest.mark.parametrize("tradable_balance_ratio,modifier", [(1.0, 1), (0.99, 0.8), (0.5, 0.5)])
def test_create_trades_multiple_trades(default_conf, ticker, fee, mocker, limit_buy_order_open,
                                       max_open, tradable_balance_ratio, modifier) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf['max_open_trades'] = max_open
    default_conf['tradable_balance_ratio'] = tradable_balance_ratio
    default_conf['dry_run_wallet'] = 0.001 * max_open

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    n = freqtrade.enter_positions()
    trades = Trade.get_open_trades()
    # Expected trades should be max_open * a modified value
    # depending on the configured tradable_balance
    assert n == max(int(max_open * modifier), 0)
    assert len(trades) == max(int(max_open * modifier), 0)


def test_create_trades_preopen(default_conf, ticker, fee, mocker, limit_buy_order_open) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf['max_open_trades'] = 4
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create 2 existing trades
    freqtrade.execute_buy('ETH/BTC', default_conf['stake_amount'])
    freqtrade.execute_buy('NEO/BTC', default_conf['stake_amount'])

    assert len(Trade.get_open_trades()) == 2
    # Change order_id for new orders
    limit_buy_order_open['id'] = '123444'

    # Create 2 new trades using create_trades
    assert freqtrade.create_trade('ETH/BTC')
    assert freqtrade.create_trade('NEO/BTC')

    trades = Trade.get_open_trades()
    assert len(trades) == 4


def test_process_trade_creation(default_conf, ticker, limit_buy_order, limit_buy_order_open,
                                fee, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        fetch_order=MagicMock(return_value=limit_buy_order),
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
    assert trade.exchange == 'binance'
    assert trade.open_rate == 0.00001098
    assert trade.amount == 91.07468123

    assert log_has(
        'Buy signal found: about create a new trade for ETH/BTC with stake_amount: 0.001 ...',
        caplog
    )


def test_process_exchange_failures(default_conf, ticker, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(side_effect=TemporaryError)
    )
    sleep_mock = mocker.patch('time.sleep', side_effect=lambda _: None)

    worker = Worker(args=None, config=default_conf)
    patch_get_signal(worker.freqtrade)

    worker._process_running()
    assert sleep_mock.has_calls()


def test_process_operational_exception(default_conf, ticker, mocker) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(side_effect=OperationalException)
    )
    worker = Worker(args=None, config=default_conf)
    patch_get_signal(worker.freqtrade)

    assert worker.freqtrade.state == State.RUNNING

    worker._process_running()
    assert worker.freqtrade.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]['status']


def test_process_trade_handling(default_conf, ticker, limit_buy_order_open, fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        fetch_order=MagicMock(return_value=limit_buy_order_open),
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


def test_process_trade_no_whitelist_pair(default_conf, ticker, limit_buy_order,
                                         fee, mocker) -> None:
    """ Test process with trade not in pair list """
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        fetch_order=MagicMock(return_value=limit_buy_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    pair = 'BLK/BTC'
    # Ensure the pair is not in the whitelist!
    assert pair not in default_conf['exchange']['pair_whitelist']

    # create open trade not in whitelist
    Trade.query.session.add(Trade(
        pair=pair,
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        amount=20,
        open_rate=0.01,
        exchange='binance',
    ))
    Trade.query.session.add(Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        amount=12,
        open_rate=0.001,
        exchange='binance',
    ))

    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    # Make sure each pair is only in the list once
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))


def test_process_informative_pairs_added(default_conf, ticker, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)

    def _refresh_whitelist(list):
        return ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC']

    refresh_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(side_effect=TemporaryError),
        refresh_latest_ohlcv=refresh_mock,
    )
    inf_pairs = MagicMock(return_value=[("BTC/ETH", '1m'), ("ETH/USDT", "1h")])
    mocker.patch('freqtrade.strategy.interface.IStrategy.get_signal', return_value=(False, False))
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
    assert ("ETH/BTC", default_conf["timeframe"]) in refresh_mock.call_args[0][0]


@pytest.mark.parametrize("side,ask,bid,last,last_ab,expected", [
    ('ask', 20, 19, 10, 0.0, 20),  # Full ask side
    ('ask', 20, 19, 10, 1.0, 10),  # Full last side
    ('ask', 20, 19, 10, 0.5, 15),  # Between ask and last
    ('ask', 20, 19, 10, 0.7, 13),  # Between ask and last
    ('ask', 20, 19, 10, 0.3, 17),  # Between ask and last
    ('ask', 5, 6, 10, 1.0, 5),  # last bigger than ask
    ('ask', 5, 6, 10, 0.5, 5),  # last bigger than ask
    ('ask', 10, 20, None, 0.5, 10),  # last not available - uses ask
    ('ask', 4, 5, None, 0.5, 4),  # last not available - uses ask
    ('ask', 4, 5, None, 1, 4),  # last not available - uses ask
    ('ask', 4, 5, None, 0, 4),  # last not available - uses ask
    ('bid', 21, 20, 10, 0.0, 20),  # Full bid side
    ('bid', 21, 20, 10, 1.0, 10),  # Full last side
    ('bid', 21, 20, 10, 0.5, 15),  # Between bid and last
    ('bid', 21, 20, 10, 0.7, 13),  # Between bid and last
    ('bid', 21, 20, 10, 0.3, 17),  # Between bid and last
    ('bid', 6, 5, 10, 1.0, 5),  # last bigger than bid
    ('bid', 6, 5, 10, 0.5, 5),  # last bigger than bid
    ('bid', 21, 20, None, 0.5, 20),  # last not available - uses bid
    ('bid', 6, 5, None, 0.5, 5),  # last not available - uses bid
    ('bid', 6, 5, None, 1, 5),  # last not available - uses bid
    ('bid', 6, 5, None, 0, 5),  # last not available - uses bid
])
def test_get_buy_rate(mocker, default_conf, caplog, side, ask, bid,
                      last, last_ab, expected) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf['bid_strategy']['ask_last_balance'] = last_ab
    default_conf['bid_strategy']['price_side'] = side
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 return_value={'ask': ask, 'last': last, 'bid': bid})

    assert freqtrade.get_buy_rate('ETH/BTC', True) == expected
    assert not log_has("Using cached buy rate for ETH/BTC.", caplog)

    assert freqtrade.get_buy_rate('ETH/BTC', False) == expected
    assert log_has("Using cached buy rate for ETH/BTC.", caplog)
    # Running a 2nd time with Refresh on!
    caplog.clear()
    assert freqtrade.get_buy_rate('ETH/BTC', True) == expected
    assert not log_has("Using cached buy rate for ETH/BTC.", caplog)


def test_execute_buy(mocker, default_conf, fee, limit_buy_order, limit_buy_order_open) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    stake_amount = 2
    bid = 0.11
    buy_rate_mock = MagicMock(return_value=bid)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        get_buy_rate=buy_rate_mock,
    )
    buy_mm = MagicMock(return_value=limit_buy_order_open)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=buy_mm,
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    pair = 'ETH/BTC'

    assert not freqtrade.execute_buy(pair, stake_amount)
    assert buy_rate_mock.call_count == 1
    assert buy_mm.call_count == 0
    assert freqtrade.strategy.confirm_trade_entry.call_count == 1
    buy_rate_mock.reset_mock()

    limit_buy_order_open['id'] = '22'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_buy(pair, stake_amount)
    assert buy_rate_mock.call_count == 1
    assert buy_mm.call_count == 1
    call_args = buy_mm.call_args_list[0][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == bid
    assert call_args['amount'] == round(stake_amount / bid, 8)
    buy_rate_mock.reset_mock()

    # Should create an open trade with an open order id
    # As the order is not fulfilled yet
    trade = Trade.query.first()
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id == '22'

    # Test calling with price
    limit_buy_order_open['id'] = '33'
    fix_price = 0.06
    assert freqtrade.execute_buy(pair, stake_amount, fix_price)
    # Make sure get_buy_rate wasn't called again
    assert buy_rate_mock.call_count == 0

    assert buy_mm.call_count == 2
    call_args = buy_mm.call_args_list[1][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == fix_price
    assert call_args['amount'] == round(stake_amount / fix_price, 8)

    # In case of closed order
    limit_buy_order['status'] = 'closed'
    limit_buy_order['price'] = 10
    limit_buy_order['cost'] = 100
    limit_buy_order['id'] = '444'

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
    limit_buy_order['id'] = '555'
    mocker.patch('freqtrade.exchange.Exchange.buy', MagicMock(return_value=limit_buy_order))
    assert freqtrade.execute_buy(pair, stake_amount)
    trade = Trade.query.all()[3]
    assert trade
    assert trade.open_order_id == '555'
    assert trade.open_rate == 0.5
    assert trade.stake_amount == 40.495905365

    # In case of the order is rejected and not filled at all
    limit_buy_order['status'] = 'rejected'
    limit_buy_order['amount'] = 90.99181073
    limit_buy_order['filled'] = 0.0
    limit_buy_order['remaining'] = 90.99181073
    limit_buy_order['price'] = 0.5
    limit_buy_order['cost'] = 0.0
    limit_buy_order['id'] = '66'
    mocker.patch('freqtrade.exchange.Exchange.buy', MagicMock(return_value=limit_buy_order))
    assert not freqtrade.execute_buy(pair, stake_amount)

    # Fail to get price...
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_buy_rate', MagicMock(return_value=0.0))

    with pytest.raises(PricingError, match="Could not determine buy price."):
        freqtrade.execute_buy(pair, stake_amount)


def test_execute_buy_confirm_error(mocker, default_conf, fee, limit_buy_order) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        get_buy_rate=MagicMock(return_value=0.11),
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value=limit_buy_order),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    stake_amount = 2
    pair = 'ETH/BTC'

    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=ValueError)
    assert freqtrade.execute_buy(pair, stake_amount)

    limit_buy_order['id'] = '222'
    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=Exception)
    assert freqtrade.execute_buy(pair, stake_amount)

    limit_buy_order['id'] = '2223'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_buy(pair, stake_amount)

    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    assert not freqtrade.execute_buy(pair, stake_amount)


def test_add_stoploss_on_exchange(mocker, default_conf, limit_buy_order) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 return_value=limit_buy_order['amount'])

    stoploss = MagicMock(return_value={'id': 13434334})
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss)

    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    trade = MagicMock()
    trade.open_order_id = None
    trade.stoploss_order_id = None
    trade.is_open = True
    trades = [trade]

    freqtrade.exit_positions(trades)
    assert trade.stoploss_order_id == '13434334'
    assert stoploss.call_count == 1
    assert trade.is_open is True


def test_handle_stoploss_on_exchange(mocker, default_conf, fee, caplog,
                                     limit_buy_order, limit_sell_order) -> None:
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss
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
    assert stoploss.call_count == 1
    assert trade.stoploss_order_id == "13434334"

    # Second case: when stoploss is set but it is not yet hit
    # should do nothing and return false
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100

    hanging_stoploss_order = MagicMock(return_value={'status': 'open'})
    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', hanging_stoploss_order)

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.stoploss_order_id == 100

    # Third case: when stoploss was set but it was canceled for some reason
    # should set a stoploss immediately and return False
    caplog.clear()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100

    canceled_stoploss_order = MagicMock(return_value={'status': 'canceled'})
    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', canceled_stoploss_order)
    stoploss.reset_mock()

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 1
    assert trade.stoploss_order_id == "13434334"

    # Fourth case: when stoploss is set and it is hit
    # should unset stoploss_order_id and return true
    # as a trade actually happened
    caplog.clear()
    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100
    assert trade

    stoploss_order_hit = MagicMock(return_value={
        'id': 100,
        'status': 'closed',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'amount': limit_buy_order['amount'],
    })
    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', stoploss_order_hit)
    assert freqtrade.handle_stoploss_on_exchange(trade) is True
    assert log_has_re(r'STOP_LOSS_LIMIT is hit for Trade\(id=1, .*\)\.', caplog)
    assert trade.stoploss_order_id is None
    assert trade.is_open is False

    mocker.patch(
        'freqtrade.exchange.Binance.stoploss',
        side_effect=ExchangeError()
    )
    trade.is_open = True
    freqtrade.handle_stoploss_on_exchange(trade)
    assert log_has('Unable to place a stoploss order on exchange.', caplog)
    assert trade.stoploss_order_id is None

    # Fifth case: fetch_order returns InvalidOrder
    # It should try to add stoploss order
    trade.stoploss_order_id = 100
    stoploss.reset_mock()
    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order',
                 side_effect=InvalidOrderException())
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss)
    freqtrade.handle_stoploss_on_exchange(trade)
    assert stoploss.call_count == 1

    # Sixth case: Closed Trade
    # Should not create new order
    trade.stoploss_order_id = None
    trade.is_open = False
    stoploss.reset_mock()
    mocker.patch('freqtrade.exchange.Exchange.fetch_order')
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 0


def test_handle_sle_cancel_cant_recreate(mocker, default_conf, fee, caplog,
                                         limit_buy_order, limit_sell_order) -> None:
    # Sixth case: stoploss order was cancelled but couldn't create new one
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        fetch_stoploss_order=MagicMock(return_value={'status': 'canceled', 'id': 100}),
        stoploss=MagicMock(side_effect=ExchangeError()),
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100
    assert trade

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert log_has_re(r'Stoploss order was cancelled, but unable to recreate one.*', caplog)
    assert trade.stoploss_order_id is None
    assert trade.is_open is True


def test_create_stoploss_order_invalid_order(mocker, default_conf, caplog, fee,
                                             limit_buy_order_open, limit_sell_order):
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    sell_mock = MagicMock(return_value={'id': limit_sell_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        sell=sell_mock,
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        fetch_order=MagicMock(return_value={'status': 'canceled'}),
        stoploss=MagicMock(side_effect=InvalidOrderException()),
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    freqtrade.enter_positions()
    trade = Trade.query.first()
    caplog.clear()
    freqtrade.create_stoploss_order(trade, 200)
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


def test_create_stoploss_order_insufficient_funds(mocker, default_conf, caplog, fee,
                                                  limit_buy_order_open, limit_sell_order):
    sell_mock = MagicMock(return_value={'id': limit_sell_order['id']})
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mock_insuf = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_insufficient_funds')
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        sell=sell_mock,
        get_fee=fee,
        fetch_order=MagicMock(return_value={'status': 'canceled'}),
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=MagicMock(side_effect=InsufficientFundsError()),
    )
    patch_get_signal(freqtrade)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    freqtrade.enter_positions()
    trade = Trade.query.first()
    caplog.clear()
    freqtrade.create_stoploss_order(trade, 200)
    # stoploss_orderid was empty before
    assert trade.stoploss_order_id is None
    assert mock_insuf.call_count == 1
    mock_insuf.reset_mock()

    trade.stoploss_order_id = 'stoploss_orderid'
    freqtrade.create_stoploss_order(trade, 200)
    # No change to stoploss-orderid
    assert trade.stoploss_order_id == 'stoploss_orderid'
    assert mock_insuf.call_count == 1


@pytest.mark.usefixtures("init_persistence")
def test_handle_stoploss_on_exchange_trailing(mocker, default_conf, fee,
                                              limit_buy_order, limit_sell_order) -> None:
    # When trailing stoploss is set
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss,
        stoploss_adjust=MagicMock(return_value=True),
    )

    # enabling TSL
    default_conf['trailing_stop'] = True

    # disabling ROI
    default_conf['minimal_roi']['0'] = 999999999

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # enabling stoploss on exchange
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # setting stoploss
    freqtrade.strategy.stoploss = -0.05

    # setting stoploss_on_exchange_interval to 60 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60

    patch_get_signal(freqtrade)

    freqtrade.enter_positions()
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

    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', stoploss_order_hanging)

    # stoploss initially at 5%
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # price jumped 2x
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', MagicMock(return_value={
        'bid': 0.00002344,
        'ask': 0.00002346,
        'last': 0.00002344
    }))

    cancel_order_mock = MagicMock()
    stoploss_order_mock = MagicMock(return_value={'id': 13434334})
    mocker.patch('freqtrade.exchange.Binance.cancel_stoploss_order', cancel_order_mock)
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss_order_mock)

    # stoploss should not be updated as the interval is 60 seconds
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    cancel_order_mock.assert_not_called()
    stoploss_order_mock.assert_not_called()

    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == 0.00002346 * 0.95

    # setting stoploss_on_exchange_interval to 0 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    cancel_order_mock.assert_called_once_with(100, 'ETH/BTC')
    stoploss_order_mock.assert_called_once_with(amount=85.32423208,
                                                pair='ETH/BTC',
                                                order_types=freqtrade.strategy.order_types,
                                                stop_price=0.00002346 * 0.95)

    # price fell below stoploss, so dry-run sells trade.
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', MagicMock(return_value={
        'bid': 0.00002144,
        'ask': 0.00002146,
        'last': 0.00002144
    }))
    assert freqtrade.handle_trade(trade) is True


def test_handle_stoploss_on_exchange_trailing_error(mocker, default_conf, fee, caplog,
                                                    limit_buy_order, limit_sell_order) -> None:
    # When trailing stoploss is set
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_exchange(mocker)

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss,
        stoploss_adjust=MagicMock(return_value=True),
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
    freqtrade.enter_positions()
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
    mocker.patch('freqtrade.exchange.Binance.cancel_stoploss_order',
                 side_effect=InvalidOrderException())
    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', stoploss_order_hanging)
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert log_has_re(r"Could not cancel stoploss order abcd for pair ETH/BTC.*", caplog)

    # Still try to create order
    assert stoploss.call_count == 1

    # Fail creating stoploss order
    caplog.clear()
    cancel_mock = mocker.patch("freqtrade.exchange.Binance.cancel_stoploss_order", MagicMock())
    mocker.patch("freqtrade.exchange.Binance.stoploss", side_effect=ExchangeError())
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert cancel_mock.call_count == 1
    assert log_has_re(r"Could not create trailing stoploss order for pair ETH/BTC\..*", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_handle_stoploss_on_exchange_custom_stop(mocker, default_conf, fee,
                                                 limit_buy_order, limit_sell_order) -> None:
    # When trailing stoploss is set
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss,
        stoploss_adjust=MagicMock(return_value=True),
    )

    # enabling TSL
    default_conf['use_custom_stoploss'] = True

    # disabling ROI
    default_conf['minimal_roi']['0'] = 999999999

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # enabling stoploss on exchange
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # setting stoploss
    freqtrade.strategy.custom_stoploss = lambda *args, **kwargs: -0.04

    # setting stoploss_on_exchange_interval to 60 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60

    patch_get_signal(freqtrade)

    freqtrade.enter_positions()
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

    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', stoploss_order_hanging)

    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # price jumped 2x
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', MagicMock(return_value={
        'bid': 0.00002344,
        'ask': 0.00002346,
        'last': 0.00002344
    }))

    cancel_order_mock = MagicMock()
    stoploss_order_mock = MagicMock(return_value={'id': 13434334})
    mocker.patch('freqtrade.exchange.Binance.cancel_stoploss_order', cancel_order_mock)
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss_order_mock)

    # stoploss should not be updated as the interval is 60 seconds
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    cancel_order_mock.assert_not_called()
    stoploss_order_mock.assert_not_called()

    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == 0.00002346 * 0.96
    assert trade.stop_loss_pct == -0.04

    # setting stoploss_on_exchange_interval to 0 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    cancel_order_mock.assert_called_once_with(100, 'ETH/BTC')
    stoploss_order_mock.assert_called_once_with(amount=85.32423208,
                                                pair='ETH/BTC',
                                                order_types=freqtrade.strategy.order_types,
                                                stop_price=0.00002346 * 0.96)

    # price fell below stoploss, so dry-run sells trade.
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', MagicMock(return_value={
        'bid': 0.00002144,
        'ask': 0.00002146,
        'last': 0.00002144
    }))
    assert freqtrade.handle_trade(trade) is True


def test_tsl_on_exchange_compatible_with_edge(mocker, edge_conf, fee, caplog,
                                              limit_buy_order, limit_sell_order) -> None:

    # When trailing stoploss is set
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    edge_conf['dry_run_wallet'] = 999.9
    edge_conf['exchange']['name'] = 'binance'
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        sell=MagicMock(return_value={'id': limit_sell_order['id']}),
        get_fee=fee,
        stoploss=stoploss,
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

    # setting stoploss_on_exchange_interval to 0 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    patch_get_signal(freqtrade)

    freqtrade.active_pair_whitelist = freqtrade.edge.adjust(freqtrade.active_pair_whitelist)

    freqtrade.enter_positions()
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

    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order', stoploss_order_hanging)

    # stoploss initially at 20% as edge dictated it.
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.stop_loss == 0.000009384

    cancel_order_mock = MagicMock()
    stoploss_order_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.cancel_stoploss_order', cancel_order_mock)
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss_order_mock)

    # price goes down 5%
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', MagicMock(return_value={
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
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', MagicMock(return_value={
        'bid': 0.00002344,
        'ask': 0.00002346,
        'last': 0.00002344
    }))

    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # stoploss should be set to 1% as trailing is on
    assert trade.stop_loss == 0.00002346 * 0.99
    cancel_order_mock.assert_called_once_with(100, 'NEO/BTC')
    stoploss_order_mock.assert_called_once_with(amount=2132892.49146757,
                                                pair='NEO/BTC',
                                                order_types=freqtrade.strategy.order_types,
                                                stop_price=0.00002346 * 0.99)


def test_enter_positions(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mock_ct = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade',
                           MagicMock(return_value=False))
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has('Found no buy signals for whitelisted currencies. Trying again...', caplog)
    # create_trade should be called once for every pair in the whitelist.
    assert mock_ct.call_count == len(default_conf['exchange']['pair_whitelist'])


def test_enter_positions_exception(mocker, default_conf, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mock_ct = mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.create_trade',
        MagicMock(side_effect=DependencyException)
    )
    n = freqtrade.enter_positions()
    assert n == 0
    assert mock_ct.call_count == len(default_conf['exchange']['pair_whitelist'])
    assert log_has('Unable to create trade for ETH/BTC: ', caplog)


def test_exit_positions(mocker, default_conf, limit_buy_order, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 return_value=limit_buy_order['amount'])

    trade = MagicMock()
    trade.open_order_id = '123'
    trade.open_fee = 0.001
    trades = [trade]
    n = freqtrade.exit_positions(trades)
    assert n == 0
    # Test amount not modified by fee-logic
    assert not log_has(
        'Applying fee to amount for Trade {} from 90.99181073 to 90.81'.format(trade), caplog
    )

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=90.81)
    # test amount modified by fee-logic
    n = freqtrade.exit_positions(trades)
    assert n == 0


def test_exit_positions_exception(mocker, default_conf, limit_buy_order, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order)

    trade = MagicMock()
    trade.open_order_id = None
    trade.open_fee = 0.001
    trade.pair = 'ETH/BTC'
    trades = [trade]

    # Test raise of DependencyException exception
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.handle_trade',
        side_effect=DependencyException()
    )
    n = freqtrade.exit_positions(trades)
    assert n == 0
    assert log_has('Unable to sell trade ETH/BTC: ', caplog)


def test_update_trade_state(mocker, default_conf, limit_buy_order, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 return_value=limit_buy_order['amount'])

    trade = Trade(
        open_order_id=123,
        fee_open=0.001,
        fee_close=0.001,
        open_rate=0.01,
        open_date=arrow.utcnow().datetime,
        amount=11,
        exchange="binance",
    )
    assert not freqtrade.update_trade_state(trade, None)
    assert log_has_re(r'Orderid for trade .* is empty.', caplog)
    # Add datetime explicitly since sqlalchemy defaults apply only once written to database
    freqtrade.update_trade_state(trade, '123')
    # Test amount not modified by fee-logic
    assert not log_has_re(r'Applying fee to .*', caplog)
    assert trade.open_order_id is None
    assert trade.amount == limit_buy_order['amount']

    trade.open_order_id = '123'
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=90.81)
    assert trade.amount != 90.81
    # test amount modified by fee-logic
    freqtrade.update_trade_state(trade, '123')
    assert trade.amount == 90.81
    assert trade.open_order_id is None

    trade.is_open = True
    trade.open_order_id = None
    # Assert we call handle_trade() if trade is feasible for execution
    freqtrade.update_trade_state(trade, '123')

    assert log_has_re('Found open order for.*', caplog)


def test_update_trade_state_withorderdict(default_conf, trades_for_order, limit_buy_order, fee,
                                          mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    # fetch_order should not be called!!
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', MagicMock(side_effect=ValueError))
    patch_exchange(mocker)
    amount = sum(x['amount'] for x in trades_for_order)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_date=arrow.utcnow().datetime,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id="123456",
        is_open=True,
    )
    freqtrade.update_trade_state(trade, '123456', limit_buy_order)
    assert trade.amount != amount
    assert trade.amount == limit_buy_order['amount']


def test_update_trade_state_withorderdict_rounding_fee(default_conf, trades_for_order, fee,
                                                       limit_buy_order, mocker, caplog):
    trades_for_order[0]['amount'] = limit_buy_order['amount'] + 1e-14
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    # fetch_order should not be called!!
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', MagicMock(side_effect=ValueError))
    patch_exchange(mocker)
    amount = sum(x['amount'] for x in trades_for_order)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id='123456',
        is_open=True,
        open_date=arrow.utcnow().datetime,
    )
    freqtrade.update_trade_state(trade, '123456', limit_buy_order)
    assert trade.amount != amount
    assert trade.amount == limit_buy_order['amount']
    assert log_has_re(r'Applying fee on amount for .*', caplog)


def test_update_trade_state_exception(mocker, default_conf,
                                      limit_buy_order, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order)

    trade = MagicMock()
    trade.open_order_id = '123'
    trade.open_fee = 0.001

    # Test raise of OperationalException exception
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
        side_effect=DependencyException()
    )
    freqtrade.update_trade_state(trade, trade.open_order_id)
    assert log_has('Could not update trade amount: ', caplog)


def test_update_trade_state_orderexception(mocker, default_conf, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 MagicMock(side_effect=InvalidOrderException))

    trade = MagicMock()
    trade.open_order_id = '123'
    trade.open_fee = 0.001

    # Test raise of OperationalException exception
    grm_mock = mocker.patch("freqtrade.freqtradebot.FreqtradeBot.get_real_amount", MagicMock())
    freqtrade.update_trade_state(trade, trade.open_order_id)
    assert grm_mock.call_count == 0
    assert log_has(f'Unable to fetch order {trade.open_order_id}: ', caplog)


def test_update_trade_state_sell(default_conf, trades_for_order, limit_sell_order_open,
                                 limit_sell_order, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    # fetch_order should not be called!!
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', MagicMock(side_effect=ValueError))
    wallet_mock = MagicMock()
    mocker.patch('freqtrade.wallets.Wallets.update', wallet_mock)

    patch_exchange(mocker)
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
        open_date=arrow.utcnow().datetime,
        open_order_id="123456",
        is_open=True,
    )
    order = Order.parse_from_ccxt_object(limit_sell_order_open, 'LTC/ETH', 'sell')
    trade.orders.append(order)
    assert order.status == 'open'
    freqtrade.update_trade_state(trade, trade.open_order_id, limit_sell_order)
    assert trade.amount == limit_sell_order['amount']
    # Wallet needs to be updated after closing a limit-sell order to reenable buying
    assert wallet_mock.call_count == 1
    assert not trade.is_open
    # Order is updated by update_trade_state
    assert order.status == 'closed'


def test_handle_trade(default_conf, limit_buy_order, limit_sell_order_open, limit_sell_order,
                      fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value=limit_buy_order),
        sell=MagicMock(return_value=limit_sell_order_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    time.sleep(0.01)  # Race condition fix
    trade.update(limit_buy_order)
    assert trade.is_open is True
    freqtrade.wallets.update()

    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.open_order_id == limit_sell_order['id']

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    assert trade.close_rate == 0.00001173
    assert trade.close_profit == 0.06201058
    assert trade.calc_profit() == 0.00006217
    assert trade.close_date is not None


def test_handle_overlapping_signals(default_conf, ticker, limit_buy_order_open,
                                    fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade, value=(True, True))
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()

    # Buy and Sell triggering, so doing nothing ...
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 0

    # Buy is triggering, so buying ...
    patch_get_signal(freqtrade, value=(True, False))
    freqtrade.enter_positions()
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


def test_handle_trade_roi(default_conf, ticker, limit_buy_order_open,
                          fee, mocker, caplog) -> None:
    caplog.set_level(logging.DEBUG)

    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtrade, value=(True, False))
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_open = True

    # FIX: sniffing logs, suggest handle_trade should not execute_sell
    #      instead that responsibility should be moved out of handle_trade(),
    #      we might just want to check if we are in a sell condition without
    #      executing
    # if ROI is reached we must sell
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade)
    assert log_has("ETH/BTC - Required profit reached. sell_flag=True, sell_type=SellType.ROI",
                   caplog)


def test_handle_trade_use_sell_signal(
        default_conf, ticker, limit_buy_order_open, fee, mocker, caplog) -> None:
    # use_sell_signal is True buy default
    caplog.set_level(logging.DEBUG)
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_open = True

    patch_get_signal(freqtrade, value=(False, False))
    assert not freqtrade.handle_trade(trade)

    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade)
    assert log_has("ETH/BTC - Sell signal received. sell_flag=True, sell_type=SellType.SELL_SIGNAL",
                   caplog)


def test_close_trade(default_conf, ticker, limit_buy_order, limit_buy_order_open, limit_sell_order,
                     fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create trade and sell it
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    trade.update(limit_buy_order)
    trade.update(limit_sell_order)
    assert trade.is_open is False

    with pytest.raises(DependencyException, match=r'.*closed trade.*'):
        freqtrade.handle_trade(trade)


def test_bot_loop_start_called_once(mocker, default_conf, caplog):
    ftbot = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade')
    patch_get_signal(ftbot)
    ftbot.strategy.bot_loop_start = MagicMock(side_effect=ValueError)
    ftbot.strategy.analyze = MagicMock()

    ftbot.process()
    assert log_has_re(r'Strategy caused the following exception.*', caplog)
    assert ftbot.strategy.bot_loop_start.call_count == 1
    assert ftbot.strategy.analyze.call_count == 1


def test_check_handle_timedout_buy_usercustom(default_conf, ticker, limit_buy_order_old, open_trade,
                                              fee, mocker) -> None:
    default_conf["unfilledtimeout"] = {"buy": 1400, "sell": 30}

    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock(return_value=limit_buy_order_old)
    cancel_buy_order = deepcopy(limit_buy_order_old)
    cancel_buy_order['status'] = 'canceled'
    cancel_order_wr_mock = MagicMock(return_value=cancel_buy_order)

    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_buy_order_old),
        cancel_order_with_result=cancel_order_wr_mock,
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    Trade.query.session.add(open_trade)

    # Ensure default is to return empty (so not mocked yet)
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0

    # Return false - trade remains open
    freqtrade.strategy.check_buy_timeout = MagicMock(return_value=False)
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert freqtrade.strategy.check_buy_timeout.call_count == 1

    # Raise Keyerror ... (no impact on trade)
    freqtrade.strategy.check_buy_timeout = MagicMock(side_effect=KeyError)
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert freqtrade.strategy.check_buy_timeout.call_count == 1

    freqtrade.strategy.check_buy_timeout = MagicMock(return_value=True)
    # Trade should be closed since the function returns true
    freqtrade.check_handle_timedout()
    assert cancel_order_wr_mock.call_count == 1
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    assert freqtrade.strategy.check_buy_timeout.call_count == 1


def test_check_handle_timedout_buy(default_conf, ticker, limit_buy_order_old, open_trade,
                                   fee, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    limit_buy_cancel = deepcopy(limit_buy_order_old)
    limit_buy_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_buy_cancel)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_buy_order_old),
        cancel_order_with_result=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    Trade.query.session.add(open_trade)

    freqtrade.strategy.check_buy_timeout = MagicMock(return_value=False)
    # check it does cancel buy orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    # Custom user buy-timeout is never called
    assert freqtrade.strategy.check_buy_timeout.call_count == 0


def test_check_handle_cancelled_buy(default_conf, ticker, limit_buy_order_old, open_trade,
                                    fee, mocker, caplog) -> None:
    """ Handle Buy order cancelled on exchange"""
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    limit_buy_order_old.update({"status": "canceled", 'filled': 0.0})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_buy_order_old),
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    Trade.query.session.add(open_trade)

    # check it does cancel buy orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    assert log_has_re("Buy order cancelled on exchange for Trade.*", caplog)


def test_check_handle_timedout_buy_exception(default_conf, ticker, limit_buy_order_old, open_trade,
                                             fee, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        fetch_ticker=ticker,
        fetch_order=MagicMock(side_effect=ExchangeError),
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf)

    Trade.query.session.add(open_trade)

    # check it does cancel buy orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 0
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1


def test_check_handle_timedout_sell_usercustom(default_conf, ticker, limit_sell_order_old, mocker,
                                               open_trade) -> None:
    default_conf["unfilledtimeout"] = {"buy": 1440, "sell": 1440}
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    open_trade.open_date = arrow.utcnow().shift(hours=-5).datetime
    open_trade.close_date = arrow.utcnow().shift(minutes=-601).datetime
    open_trade.close_profit_abs = 0.001
    open_trade.is_open = False

    Trade.query.session.add(open_trade)
    # Ensure default is false
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0

    freqtrade.strategy.check_sell_timeout = MagicMock(return_value=False)
    # Return false - No impact
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 0
    assert open_trade.is_open is False
    assert freqtrade.strategy.check_sell_timeout.call_count == 1

    freqtrade.strategy.check_sell_timeout = MagicMock(side_effect=KeyError)
    # Return Error - No impact
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 0
    assert open_trade.is_open is False
    assert freqtrade.strategy.check_sell_timeout.call_count == 1

    # Return True - sells!
    freqtrade.strategy.check_sell_timeout = MagicMock(return_value=True)
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    assert open_trade.is_open is True
    assert freqtrade.strategy.check_sell_timeout.call_count == 1


def test_check_handle_timedout_sell(default_conf, ticker, limit_sell_order_old, mocker,
                                    open_trade) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    open_trade.open_date = arrow.utcnow().shift(hours=-5).datetime
    open_trade.close_date = arrow.utcnow().shift(minutes=-601).datetime
    open_trade.close_profit_abs = 0.001
    open_trade.is_open = False

    Trade.query.session.add(open_trade)

    freqtrade.strategy.check_sell_timeout = MagicMock(return_value=False)
    # check it does cancel sell orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 1
    assert open_trade.is_open is True
    # Custom user sell-timeout is never called
    assert freqtrade.strategy.check_sell_timeout.call_count == 0


def test_check_handle_cancelled_sell(default_conf, ticker, limit_sell_order_old, open_trade,
                                     mocker, caplog) -> None:
    """ Handle sell order cancelled on exchange"""
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    limit_sell_order_old.update({"status": "canceled", 'filled': 0.0})
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order_with_result=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    open_trade.open_date = arrow.utcnow().shift(hours=-5).datetime
    open_trade.close_date = arrow.utcnow().shift(minutes=-601).datetime
    open_trade.is_open = False

    Trade.query.session.add(open_trade)

    # check it does cancel sell orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert open_trade.is_open is True
    assert log_has_re("Sell order cancelled on exchange for Trade.*", caplog)


def test_check_handle_timedout_partial(default_conf, ticker, limit_buy_order_old_partial,
                                       open_trade, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    limit_buy_canceled = deepcopy(limit_buy_order_old_partial)
    limit_buy_canceled['status'] = 'canceled'

    cancel_order_mock = MagicMock(return_value=limit_buy_canceled)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_buy_order_old_partial),
        cancel_order_with_result=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    Trade.query.session.add(open_trade)

    # check it does cancel buy orders over the time limit
    # note this is for a partially-complete buy order
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 1
    assert trades[0].amount == 23.0
    assert trades[0].stake_amount == open_trade.open_rate * trades[0].amount


def test_check_handle_timedout_partial_fee(default_conf, ticker, open_trade, caplog, fee,
                                           limit_buy_order_old_partial, trades_for_order,
                                           limit_buy_order_old_partial_canceled, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock(return_value=limit_buy_order_old_partial_canceled)
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=0))
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_buy_order_old_partial),
        cancel_order_with_result=cancel_order_mock,
        get_trades_for_order=MagicMock(return_value=trades_for_order),
    )
    freqtrade = FreqtradeBot(default_conf)

    assert open_trade.amount == limit_buy_order_old_partial['amount']

    open_trade.fee_open = fee()
    open_trade.fee_close = fee()
    Trade.query.session.add(open_trade)
    # cancelling a half-filled order should update the amount to the bought amount
    # and apply fees if necessary.
    freqtrade.check_handle_timedout()

    assert log_has_re(r"Applying fee on amount for Trade.*", caplog)

    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 1
    # Verify that trade has been updated
    assert trades[0].amount == (limit_buy_order_old_partial['amount'] -
                                limit_buy_order_old_partial['remaining']) - 0.023
    assert trades[0].open_order_id is None
    assert trades[0].fee_updated('buy')
    assert pytest.approx(trades[0].fee_open) == 0.001


def test_check_handle_timedout_partial_except(default_conf, ticker, open_trade, caplog, fee,
                                              limit_buy_order_old_partial, trades_for_order,
                                              limit_buy_order_old_partial_canceled, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock(return_value=limit_buy_order_old_partial_canceled)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(return_value=limit_buy_order_old_partial),
        cancel_order_with_result=cancel_order_mock,
        get_trades_for_order=MagicMock(return_value=trades_for_order),
    )
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 MagicMock(side_effect=DependencyException))
    freqtrade = FreqtradeBot(default_conf)

    assert open_trade.amount == limit_buy_order_old_partial['amount']

    open_trade.fee_open = fee()
    open_trade.fee_close = fee()
    Trade.query.session.add(open_trade)
    # cancelling a half-filled order should update the amount to the bought amount
    # and apply fees if necessary.
    freqtrade.check_handle_timedout()

    assert log_has_re(r"Could not update trade amount: .*", caplog)

    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 1
    # Verify that trade has been updated

    assert trades[0].amount == (limit_buy_order_old_partial['amount'] -
                                limit_buy_order_old_partial['remaining'])
    assert trades[0].open_order_id is None
    assert trades[0].fee_open == fee()


def test_check_handle_timedout_exception(default_conf, ticker, open_trade, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()

    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        handle_cancel_buy=MagicMock(),
        handle_cancel_sell=MagicMock(),
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        fetch_order=MagicMock(side_effect=ExchangeError('Oh snap')),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf)

    Trade.query.session.add(open_trade)

    freqtrade.check_handle_timedout()
    assert log_has_re(r"Cannot query order for Trade\(id=1, pair=ETH/BTC, amount=90.99181073, "
                      r"open_rate=0.00001099, open_since="
                      f"{open_trade.open_date.strftime('%Y-%m-%d %H:%M:%S')}"
                      r"\) due to Traceback \(most recent call last\):\n*",
                      caplog)


def test_handle_cancel_buy(mocker, caplog, default_conf, limit_buy_order) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_buy_order = deepcopy(limit_buy_order)
    cancel_buy_order['status'] = 'canceled'
    del cancel_buy_order['filled']

    cancel_order_mock = MagicMock(return_value=cancel_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order_with_result', cancel_order_mock)

    freqtrade = FreqtradeBot(default_conf)
    freqtrade._notify_buy_cancel = MagicMock()

    trade = MagicMock()
    trade.pair = 'LTC/ETH'
    limit_buy_order['filled'] = 0.0
    limit_buy_order['status'] = 'open'
    reason = CANCEL_REASON['TIMEOUT']
    assert freqtrade.handle_cancel_buy(trade, limit_buy_order, reason)
    assert cancel_order_mock.call_count == 1

    cancel_order_mock.reset_mock()
    limit_buy_order['filled'] = 2
    assert not freqtrade.handle_cancel_buy(trade, limit_buy_order, reason)
    assert cancel_order_mock.call_count == 1

    # Order remained open for some reason (cancel failed)
    cancel_buy_order['status'] = 'open'
    cancel_order_mock = MagicMock(return_value=cancel_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order_with_result', cancel_order_mock)
    assert not freqtrade.handle_cancel_buy(trade, limit_buy_order, reason)
    assert log_has_re(r"Order .* for .* not cancelled.", caplog)


@pytest.mark.parametrize("limit_buy_order_canceled_empty", ['binance', 'ftx', 'kraken', 'bittrex'],
                         indirect=['limit_buy_order_canceled_empty'])
def test_handle_cancel_buy_exchanges(mocker, caplog, default_conf,
                                     limit_buy_order_canceled_empty) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = mocker.patch(
        'freqtrade.exchange.Exchange.cancel_order_with_result',
        return_value=limit_buy_order_canceled_empty)
    nofiy_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot._notify_buy_cancel')
    freqtrade = FreqtradeBot(default_conf)

    reason = CANCEL_REASON['TIMEOUT']
    trade = MagicMock()
    trade.pair = 'LTC/ETH'
    assert freqtrade.handle_cancel_buy(trade, limit_buy_order_canceled_empty, reason)
    assert cancel_order_mock.call_count == 0
    assert log_has_re(r'Buy order fully cancelled. Removing .* from database\.', caplog)
    assert nofiy_mock.call_count == 1


@pytest.mark.parametrize('cancelorder', [
    {},
    {'remaining': None},
    'String Return value',
    123
])
def test_handle_cancel_buy_corder_empty(mocker, default_conf, limit_buy_order,
                                        cancelorder) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock(return_value=cancelorder)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        cancel_order=cancel_order_mock
    )

    freqtrade = FreqtradeBot(default_conf)
    freqtrade._notify_buy_cancel = MagicMock()

    trade = MagicMock()
    trade.pair = 'LTC/ETH'
    limit_buy_order['filled'] = 0.0
    limit_buy_order['status'] = 'open'
    reason = CANCEL_REASON['TIMEOUT']
    assert freqtrade.handle_cancel_buy(trade, limit_buy_order, reason)
    assert cancel_order_mock.call_count == 1

    cancel_order_mock.reset_mock()
    limit_buy_order['filled'] = 1.0
    assert not freqtrade.handle_cancel_buy(trade, limit_buy_order, reason)
    assert cancel_order_mock.call_count == 1


def test_handle_cancel_sell_limit(mocker, default_conf, fee) -> None:
    send_msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        cancel_order=cancel_order_mock,
    )
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_sell_rate', return_value=0.245441)

    freqtrade = FreqtradeBot(default_conf)

    trade = Trade(
        pair='LTC/ETH',
        amount=2,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456",
        open_date=arrow.utcnow().datetime,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
    )
    order = {'remaining': 1,
             'amount': 1,
             'status': "open"}
    reason = CANCEL_REASON['TIMEOUT']
    assert freqtrade.handle_cancel_sell(trade, order, reason)
    assert cancel_order_mock.call_count == 1
    assert send_msg_mock.call_count == 1

    send_msg_mock.reset_mock()

    order['amount'] = 2
    assert freqtrade.handle_cancel_sell(trade, order, reason
                                        ) == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    # Assert cancel_order was not called (callcount remains unchanged)
    assert cancel_order_mock.call_count == 1
    assert send_msg_mock.call_count == 1
    assert freqtrade.handle_cancel_sell(trade, order, reason
                                        ) == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    # Message should not be iterated again
    assert trade.sell_order_status == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    assert send_msg_mock.call_count == 1


def test_handle_cancel_sell_cancel_exception(mocker, default_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.cancel_order_with_result', side_effect=InvalidOrderException())

    freqtrade = FreqtradeBot(default_conf)

    trade = MagicMock()
    reason = CANCEL_REASON['TIMEOUT']
    order = {'remaining': 1,
             'amount': 1,
             'status': "open"}
    assert freqtrade.handle_cancel_sell(trade, order, reason) == 'error cancelling order'


def test_execute_sell_up(default_conf, ticker, fee, ticker_sell_up, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    patch_whitelist(mocker, default_conf)
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=False)

    # Create some test data
    freqtrade.enter_positions()
    rpc_mock.reset_mock()

    trade = Trade.query.first()
    assert trade
    assert freqtrade.strategy.confirm_trade_exit.call_count == 0

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_up
    )
    # Prevented sell ...
    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'], sell_reason=SellType.ROI)
    assert rpc_mock.call_count == 0
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1

    # Repatch with true
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'], sell_reason=SellType.ROI)
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1

    assert rpc_mock.call_count == 1
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'trade_id': 1,
        'type': RPCMessageType.SELL,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'limit': 1.172e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.173e-05,
        'profit_amount': 6.223e-05,
        'profit_ratio': 0.0620716,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.ROI.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == last_msg


def test_execute_sell_down(default_conf, ticker, fee, ticker_sell_down, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    patch_whitelist(mocker, default_conf)
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_down
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_down()['bid'],
                           sell_reason=SellType.STOP_LOSS)

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.044e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.043e-05,
        'profit_amount': -5.406e-05,
        'profit_ratio': -0.05392257,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == last_msg


def test_execute_sell_down_stoploss_on_exchange_dry_run(default_conf, ticker, fee,
                                                        ticker_sell_down, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    patch_whitelist(mocker, default_conf)
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_down
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
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'loss',
        'limit': 1.08801e-05,
        'amount': 91.07468123,
        'order_type': 'limit',
        'open_rate': 1.098e-05,
        'current_rate': 1.043e-05,
        'profit_amount': -1.408e-05,
        'profit_ratio': -0.01404051,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == last_msg


def test_execute_sell_sloe_cancel_exception(mocker, default_conf, ticker, fee, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.cancel_stoploss_order',
                 side_effect=InvalidOrderException())
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=300))
    sellmock = MagicMock(return_value={'id': '12345555'})
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        sell=sellmock
    )

    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    PairLock.session = MagicMock()

    freqtrade.config['dry_run'] = False
    trade.stoploss_order_id = "abcd"

    freqtrade.execute_sell(trade=trade, limit=1234,
                           sell_reason=SellType.STOP_LOSS)
    assert sellmock.call_count == 1
    assert log_has('Could not cancel stoploss order abcd', caplog)


def test_execute_sell_with_stoploss_on_exchange(default_conf, ticker, fee, ticker_sell_up,
                                                mocker) -> None:

    default_conf['exchange']['name'] = 'binance'
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    stoploss = MagicMock(return_value={
        'id': 123,
        'info': {
            'foo': 'bar'
        }
    })

    cancel_order = MagicMock(return_value=True)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
        stoploss=stoploss,
        cancel_stoploss_order=cancel_order,
    )

    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade
    trades = [trade]

    freqtrade.check_handle_timedout()
    freqtrade.exit_positions(trades)

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_up
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'],
                           sell_reason=SellType.SELL_SIGNAL)

    trade = Trade.query.first()
    assert trade
    assert cancel_order.call_count == 1
    assert rpc_mock.call_count == 3


def test_may_execute_sell_after_stoploss_on_exchange_hit(default_conf, ticker, fee,
                                                         mocker) -> None:
    default_conf['exchange']['name'] = 'binance'
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
    )

    stoploss = MagicMock(return_value={
        'id': 123,
        'info': {
            'foo': 'bar'
        }
    })

    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss)

    freqtrade = FreqtradeBot(default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()
    freqtrade.check_handle_timedout()
    trade = Trade.query.first()
    trades = [trade]
    assert trade.stoploss_order_id is None

    freqtrade.exit_positions(trades)
    assert trade
    assert trade.stoploss_order_id == '123'
    assert trade.open_order_id is None

    # Assuming stoploss on exchnage is hit
    # stoploss_order_id should become None
    # and trade should be sold at the price of stoploss
    stoploss_executed = MagicMock(return_value={
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
    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order', stoploss_executed)

    freqtrade.exit_positions(trades)
    assert trade.stoploss_order_id is None
    assert trade.is_open is False
    assert trade.sell_reason == SellType.STOPLOSS_ON_EXCHANGE.value
    assert rpc_mock.call_count == 3
    assert rpc_mock.call_args_list[0][0][0]['type'] == RPCMessageType.BUY
    assert rpc_mock.call_args_list[1][0][0]['type'] == RPCMessageType.BUY_FILL
    assert rpc_mock.call_args_list[2][0][0]['type'] == RPCMessageType.SELL


def test_execute_sell_market_order(default_conf, ticker, fee,
                                   ticker_sell_up, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    patch_whitelist(mocker, default_conf)
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_up
    )
    freqtrade.config['order_types']['sell'] = 'market'

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'], sell_reason=SellType.ROI)

    assert not trade.is_open
    assert trade.close_profit == 0.0620716

    assert rpc_mock.call_count == 3
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/BTC',
        'gain': 'profit',
        'limit': 1.172e-05,
        'amount': 91.07468123,
        'order_type': 'market',
        'open_rate': 1.098e-05,
        'current_rate': 1.173e-05,
        'profit_amount': 6.223e-05,
        'profit_ratio': 0.0620716,
        'stake_currency': 'BTC',
        'fiat_currency': 'USD',
        'sell_reason': SellType.ROI.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,

    } == last_msg


def test_execute_sell_insufficient_funds_error(default_conf, ticker, fee,
                                               ticker_sell_up, mocker) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mock_insuf = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_insufficient_funds')
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        sell=MagicMock(side_effect=InsufficientFundsError())
    )
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_up
    )

    assert not freqtrade.execute_sell(trade=trade, limit=ticker_sell_up()['bid'],
                                      sell_reason=SellType.ROI)
    assert mock_insuf.call_count == 1


def test_sell_profit_only_enable_profit(default_conf, limit_buy_order, limit_buy_order_open,
                                        fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    default_conf['ask_strategy'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
        'sell_profit_offset': 0.1,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    freqtrade.wallets.update()
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is False

    freqtrade.config['ask_strategy']['sell_profit_offset'] = 0.0
    assert freqtrade.handle_trade(trade) is True

    assert trade.sell_reason == SellType.SELL_SIGNAL.value


def test_sell_profit_only_disable_profit(default_conf, limit_buy_order, limit_buy_order_open,
                                         fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00002172,
            'ask': 0.00002173,
            'last': 0.00002172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    default_conf['ask_strategy'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    freqtrade.wallets.update()
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.SELL_SIGNAL.value


def test_sell_profit_only_enable_loss(default_conf, limit_buy_order, limit_buy_order_open,
                                      fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00000172,
            'ask': 0.00000173,
            'last': 0.00000172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    default_conf['ask_strategy'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.stop_loss_reached = MagicMock(return_value=SellCheckTuple(
        sell_flag=False, sell_type=SellType.NONE))
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is False


def test_sell_profit_only_disable_loss(default_conf, limit_buy_order, limit_buy_order_open,
                                       fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.0000172,
            'ask': 0.0000173,
            'last': 0.0000172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    default_conf['ask_strategy'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    freqtrade.wallets.update()
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.SELL_SIGNAL.value


def test_sell_not_enough_balance(default_conf, limit_buy_order, limit_buy_order_open,
                                 fee, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00002172,
            'ask': 0.00002173,
            'last': 0.00002172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    amnt = trade.amount
    trade.update(limit_buy_order)
    patch_get_signal(freqtrade, value=(False, True))
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=trade.amount * 0.985))

    assert freqtrade.handle_trade(trade) is True
    assert log_has_re(r'.*Falling back to wallet-amount.', caplog)
    assert trade.amount != amnt


def test__safe_sell_amount(default_conf, fee, caplog, mocker):
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    amount = 95.33
    amount_wallet = 95.29
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=amount_wallet))
    wallet_update = mocker.patch('freqtrade.wallets.Wallets.update')
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456",
        fee_open=fee.return_value,
        fee_close=fee.return_value,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    wallet_update.reset_mock()
    assert freqtrade._safe_sell_amount(trade.pair, trade.amount) == amount_wallet
    assert log_has_re(r'.*Falling back to wallet-amount.', caplog)
    assert wallet_update.call_count == 1
    caplog.clear()
    wallet_update.reset_mock()
    assert freqtrade._safe_sell_amount(trade.pair, amount_wallet) == amount_wallet
    assert not log_has_re(r'.*Falling back to wallet-amount.', caplog)
    assert wallet_update.call_count == 1


def test__safe_sell_amount_error(default_conf, fee, caplog, mocker):
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    amount = 95.33
    amount_wallet = 91.29
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=amount_wallet))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="123456",
        fee_open=fee.return_value,
        fee_close=fee.return_value,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match=r"Not enough amount to sell."):
        assert freqtrade._safe_sell_amount(trade.pair, trade.amount)


def test_locked_pairs(default_conf, ticker, fee, ticker_sell_down, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_down
    )

    freqtrade.execute_sell(trade=trade, limit=ticker_sell_down()['bid'],
                           sell_reason=SellType.STOP_LOSS)
    trade.close(ticker_sell_down()['bid'])
    assert freqtrade.strategy.is_pair_locked(trade.pair)

    # reinit - should buy other pair.
    caplog.clear()
    freqtrade.enter_positions()

    assert log_has_re(f"Pair {trade.pair} is still locked.*", caplog)


def test_ignore_roi_if_buy_signal(default_conf, limit_buy_order, limit_buy_order_open,
                                  fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.0000172,
            'ask': 0.0000173,
            'last': 0.0000172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    default_conf['ask_strategy'] = {
        'ignore_roi_if_buy_signal': True
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    freqtrade.wallets.update()
    patch_get_signal(freqtrade, value=(True, True))
    assert freqtrade.handle_trade(trade) is False

    # Test if buy-signal is absent (should sell due to roi = true)
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.ROI.value


def test_trailing_stop_loss(default_conf, limit_buy_order_open, limit_buy_order,
                            fee, caplog, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001099,
            'ask': 0.00001099,
            'last': 0.00001099
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    default_conf['trailing_stop'] = True
    patch_whitelist(mocker, default_conf)
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()
    trade = Trade.query.first()
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker above buy price
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': 0.00001099 * 1.5,
                     'ask': 0.00001099 * 1.5,
                     'last': 0.00001099 * 1.5
                 }))

    # Stoploss should be adjusted
    assert freqtrade.handle_trade(trade) is False

    # Price fell
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': 0.00001099 * 1.1,
                     'ask': 0.00001099 * 1.1,
                     'last': 0.00001099 * 1.1
                 }))

    caplog.set_level(logging.DEBUG)
    # Sell as trailing-stop is reached
    assert freqtrade.handle_trade(trade) is True
    assert log_has("ETH/BTC - HIT STOP: current price at 0.000012, stoploss is 0.000015, "
                   "initial stoploss was at 0.000010, trade opened at 0.000011", caplog)
    assert trade.sell_reason == SellType.TRAILING_STOP_LOSS.value


def test_trailing_stop_loss_positive(default_conf, limit_buy_order, limit_buy_order_open, fee,
                                     caplog, mocker) -> None:
    buy_price = limit_buy_order['price']
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': buy_price - 0.000001,
            'ask': buy_price - 0.000001,
            'last': buy_price - 0.000001
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0.01
    patch_whitelist(mocker, default_conf)

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    caplog.set_level(logging.DEBUG)
    # stop-loss not reached
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker above buy price
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.000003,
                     'ask': buy_price + 0.000003,
                     'last': buy_price + 0.000003
                 }))
    # stop-loss not reached, adjusted stoploss
    assert freqtrade.handle_trade(trade) is False
    assert log_has("ETH/BTC - Using positive stoploss: 0.01 offset: 0 profit: 0.2666%", caplog)
    assert log_has("ETH/BTC - Adjusting stoploss...", caplog)
    assert trade.stop_loss == 0.0000138501

    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.000002,
                     'ask': buy_price + 0.000002,
                     'last': buy_price + 0.000002
                 }))
    # Lower price again (but still positive)
    assert freqtrade.handle_trade(trade) is True
    assert log_has(
        f"ETH/BTC - HIT STOP: current price at {buy_price + 0.000002:.6f}, "
        f"stoploss is {trade.stop_loss:.6f}, "
        f"initial stoploss was at 0.000010, trade opened at 0.000011", caplog)


def test_trailing_stop_loss_offset(default_conf, limit_buy_order, limit_buy_order_open, fee,
                                   caplog, mocker) -> None:
    buy_price = limit_buy_order['price']
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': buy_price - 0.000001,
            'ask': buy_price - 0.000001,
            'last': buy_price - 0.000001
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    patch_whitelist(mocker, default_conf)
    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0.01
    default_conf['trailing_stop_positive_offset'] = 0.011
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    caplog.set_level(logging.DEBUG)
    # stop-loss not reached
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker above buy price
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.000003,
                     'ask': buy_price + 0.000003,
                     'last': buy_price + 0.000003
                 }))
    # stop-loss not reached, adjusted stoploss
    assert freqtrade.handle_trade(trade) is False
    assert log_has("ETH/BTC - Using positive stoploss: 0.01 offset: 0.011 profit: 0.2666%", caplog)
    assert log_has("ETH/BTC - Adjusting stoploss...", caplog)
    assert trade.stop_loss == 0.0000138501

    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.000002,
                     'ask': buy_price + 0.000002,
                     'last': buy_price + 0.000002
                 }))
    # Lower price again (but still positive)
    assert freqtrade.handle_trade(trade) is True
    assert log_has(
        f"ETH/BTC - HIT STOP: current price at {buy_price + 0.000002:.6f}, "
        f"stoploss is {trade.stop_loss:.6f}, "
        f"initial stoploss was at 0.000010, trade opened at 0.000011", caplog)
    assert trade.sell_reason == SellType.TRAILING_STOP_LOSS.value


def test_tsl_only_offset_reached(default_conf, limit_buy_order, limit_buy_order_open, fee,
                                 caplog, mocker) -> None:
    buy_price = limit_buy_order['price']
    # buy_price: 0.00001099

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': buy_price,
            'ask': buy_price,
            'last': buy_price
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    patch_whitelist(mocker, default_conf)
    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0.05
    default_conf['trailing_stop_positive_offset'] = 0.055
    default_conf['trailing_only_offset_is_reached'] = True

    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    caplog.set_level(logging.DEBUG)
    # stop-loss not reached
    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == 0.0000098910

    # Raise ticker above buy price
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.0000004,
                     'ask': buy_price + 0.0000004,
                     'last': buy_price + 0.0000004
                 }))

    # stop-loss should not be adjusted as offset is not reached yet
    assert freqtrade.handle_trade(trade) is False

    assert not log_has("ETH/BTC - Adjusting stoploss...", caplog)
    assert trade.stop_loss == 0.0000098910

    # price rises above the offset (rises 12% when the offset is 5.5%)
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': buy_price + 0.0000014,
                     'ask': buy_price + 0.0000014,
                     'last': buy_price + 0.0000014
                 }))

    assert freqtrade.handle_trade(trade) is False
    assert log_has("ETH/BTC - Using positive stoploss: 0.05 offset: 0.055 profit: 0.1218%", caplog)
    assert log_has("ETH/BTC - Adjusting stoploss...", caplog)
    assert trade.stop_loss == 0.0000117705


def test_disable_ignore_roi_if_buy_signal(default_conf, limit_buy_order, limit_buy_order_open,
                                          fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00000172,
            'ask': 0.00000173,
            'last': 0.00000172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )
    default_conf['ask_strategy'] = {
        'ignore_roi_if_buy_signal': False
    }
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    # Sell due to min_roi_reached
    patch_get_signal(freqtrade, value=(True, True))
    assert freqtrade.handle_trade(trade) is True

    # Test if buy-signal is absent
    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.SELL_SIGNAL.value


def test_get_real_amount_quote(default_conf, trades_for_order, buy_order_fee, fee, caplog, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount - (amount * 0.001)
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.992).',
                   caplog)


def test_get_real_amount_quote_dust(default_conf, trades_for_order, buy_order_fee, fee,
                                    caplog, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    walletmock = mocker.patch('freqtrade.wallets.Wallets.update')
    mocker.patch('freqtrade.wallets.Wallets.get_free', return_value=8.1122)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    walletmock.reset_mock()
    # Amount is kept as is
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount
    assert walletmock.call_count == 1
    assert log_has_re(r'Fee amount for Trade.* was in base currency '
                      '- Eating Fee 0.008 into dust', caplog)


def test_get_real_amount_no_trade(default_conf, buy_order_fee, caplog, mocker, fee):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])

    amount = buy_order_fee['amount']
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) failed: myTrade-Dict empty found',
                   caplog)


def test_get_real_amount_stake(default_conf, trades_for_order, buy_order_fee, fee, mocker):
    trades_for_order[0]['fee']['currency'] = 'ETH'

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_no_currency_in_fee(default_conf, trades_for_order, buy_order_fee,
                                            fee, mocker):

    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['fee'] = {'cost': 0.004, 'currency': None}
    trades_for_order[0]['fee']['currency'] = None

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount does not change
    assert freqtrade.get_real_amount(trade, limit_buy_order) == amount


def test_get_real_amount_BNB(default_conf, trades_for_order, buy_order_fee, fee, mocker):
    trades_for_order[0]['fee']['currency'] = 'BNB'
    trades_for_order[0]['fee']['cost'] = 0.00094518

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_multi(default_conf, trades_for_order2, buy_order_fee, caplog, fee, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order2)
    amount = float(sum(x['amount'] for x in trades_for_order2))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount - (amount * 0.001)
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.992).',
                   caplog)

    assert trade.fee_open == 0.001
    assert trade.fee_close == 0.001
    assert trade.fee_open_cost is not None
    assert trade.fee_open_currency is not None
    assert trade.fee_close_cost is None
    assert trade.fee_close_currency is None


def test_get_real_amount_multi2(default_conf, trades_for_order3, buy_order_fee, caplog, fee,
                                mocker, markets):
    # Different fee currency on both trades
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order3)
    amount = float(sum(x['amount'] for x in trades_for_order3))
    default_conf['stake_currency'] = 'ETH'
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    # Fake markets entry to enable fee parsing
    markets['BNB/ETH'] = markets['ETH/BTC']
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 return_value={'ask': 0.19, 'last': 0.2})

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount - (amount * 0.0005)
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.996).',
                   caplog)
    # Overall fee is average of both trade's fee
    assert trade.fee_open == 0.001518575
    assert trade.fee_open_cost is not None
    assert trade.fee_open_currency is not None
    assert trade.fee_close_cost is None
    assert trade.fee_close_currency is None


def test_get_real_amount_fromorder(default_conf, trades_for_order, buy_order_fee, fee,
                                   caplog, mocker):
    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['fee'] = {'cost': 0.004, 'currency': 'LTC'}

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order',
                 return_value=[trades_for_order])
    amount = float(sum(x['amount'] for x in trades_for_order))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    # Ticker rate cannot be found for this to work.
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', side_effect=ExchangeError)

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, limit_buy_order) == amount - 0.004
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.996).',
                   caplog)


def test_get_real_amount_invalid_order(default_conf, trades_for_order, buy_order_fee, fee, mocker):
    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['fee'] = {'cost': 0.004}

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    amount = float(sum(x['amount'] for x in trades_for_order))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount does not change
    assert freqtrade.get_real_amount(trade, limit_buy_order) == amount


def test_get_real_amount_wrong_amount(default_conf, trades_for_order, buy_order_fee, fee, mocker):
    limit_buy_order = deepcopy(buy_order_fee)
    limit_buy_order['amount'] = limit_buy_order['amount'] - 0.001

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = float(sum(x['amount'] for x in trades_for_order))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount does not change
    with pytest.raises(DependencyException, match=r"Half bought\? Amounts don't match"):
        freqtrade.get_real_amount(trade, limit_buy_order)


def test_get_real_amount_wrong_amount_rounding(default_conf, trades_for_order, buy_order_fee, fee,
                                               mocker):
    # Floats should not be compared directly.
    limit_buy_order = deepcopy(buy_order_fee)
    trades_for_order[0]['amount'] = trades_for_order[0]['amount'] + 1e-15

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = float(sum(x['amount'] for x in trades_for_order))
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # Amount changes by fee amount.
    assert isclose(freqtrade.get_real_amount(trade, limit_buy_order), amount - (amount * 0.001),
                   abs_tol=MATH_CLOSE_PREC,)


def test_get_real_amount_invalid(default_conf, trades_for_order, buy_order_fee, fee, mocker):
    # Remove "Currency" from fee dict
    trades_for_order[0]['fee'] = {'cost': 0.008}

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = sum(x['amount'] for x in trades_for_order)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,

        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    # Amount does not change
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount


def test_get_real_amount_open_trade(default_conf, fee, mocker):
    amount = 12345
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id="123456"
    )
    order = {
        'id': 'mocked_order',
        'amount': amount,
        'status': 'open',
        'side': 'buy',
    }
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert freqtrade.get_real_amount(trade, order) == amount


@pytest.mark.parametrize('amount,fee_abs,wallet,amount_exp', [
    (8.0, 0.0, 10, 8),
    (8.0, 0.0, 0, 8),
    (8.0, 0.1, 0, 7.9),
    (8.0, 0.1, 10, 8),
    (8.0, 0.1, 8.0, 8.0),
    (8.0, 0.1, 7.9, 7.9),
])
def test_apply_fee_conditional(default_conf, fee, caplog, mocker,
                               amount, fee_abs, wallet, amount_exp):
    walletmock = mocker.patch('freqtrade.wallets.Wallets.update')
    mocker.patch('freqtrade.wallets.Wallets.get_free', return_value=wallet)
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    walletmock.reset_mock()
    # Amount is kept as is
    assert freqtrade.apply_fee_conditional(trade, 'LTC', amount, fee_abs) == amount_exp
    assert walletmock.call_count == 1


def test_order_book_depth_of_market(default_conf, ticker, limit_buy_order_open, limit_buy_order,
                                    fee, mocker, order_book_l2):
    default_conf['bid_strategy']['check_depth_of_market']['enabled'] = True
    default_conf['bid_strategy']['check_depth_of_market']['bids_to_ask_delta'] = 0.1
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book', order_book_l2)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade is not None
    assert trade.stake_amount == 0.001
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'

    assert len(Trade.query.all()) == 1

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    assert trade.open_rate == 0.00001099
    assert whitelist == default_conf['exchange']['pair_whitelist']


def test_order_book_depth_of_market_high_delta(default_conf, ticker, limit_buy_order,
                                               fee, mocker, order_book_l2):
    default_conf['bid_strategy']['check_depth_of_market']['enabled'] = True
    # delta is 100 which is impossible to reach. hence check_depth_of_market will return false
    default_conf['bid_strategy']['check_depth_of_market']['bids_to_ask_delta'] = 100
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book', order_book_l2)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value={'id': limit_buy_order['id']}),
        get_fee=fee,
    )
    # Save state of current whitelist
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade is None


def test_order_book_bid_strategy1(mocker, default_conf, order_book_l2) -> None:
    """
    test if function get_buy_rate will return the order book price
    instead of the ask rate
    """
    patch_exchange(mocker)
    ticker_mock = MagicMock(return_value={'ask': 0.045, 'last': 0.046})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_l2_order_book=order_book_l2,
        fetch_ticker=ticker_mock,

    )
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['use_order_book'] = True
    default_conf['bid_strategy']['order_book_top'] = 2
    default_conf['bid_strategy']['ask_last_balance'] = 0
    default_conf['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf)
    assert freqtrade.get_buy_rate('ETH/BTC', True) == 0.043935
    assert ticker_mock.call_count == 0


def test_order_book_bid_strategy_exception(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    ticker_mock = MagicMock(return_value={'ask': 0.042, 'last': 0.046})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_l2_order_book=MagicMock(return_value={'bids': [[]], 'asks': [[]]}),
        fetch_ticker=ticker_mock,

    )
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['use_order_book'] = True
    default_conf['bid_strategy']['order_book_top'] = 1
    default_conf['bid_strategy']['ask_last_balance'] = 0
    default_conf['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf)
    # orderbook shall be used even if tickers would be lower.
    with pytest.raises(PricingError):
        freqtrade.get_buy_rate('ETH/BTC', refresh=True)
    assert log_has_re(r'Buy Price from orderbook could not be determined.', caplog)


def test_check_depth_of_market_buy(default_conf, mocker, order_book_l2) -> None:
    """
    test check depth of market
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_l2_order_book=order_book_l2
    )
    default_conf['telegram']['enabled'] = False
    default_conf['exchange']['name'] = 'binance'
    default_conf['bid_strategy']['check_depth_of_market']['enabled'] = True
    # delta is 100 which is impossible to reach. hence function will return false
    default_conf['bid_strategy']['check_depth_of_market']['bids_to_ask_delta'] = 100
    freqtrade = FreqtradeBot(default_conf)

    conf = default_conf['bid_strategy']['check_depth_of_market']
    assert freqtrade._check_depth_of_market_buy('ETH/BTC', conf) is False


def test_order_book_ask_strategy(default_conf, limit_buy_order_open, limit_buy_order, fee,
                                 limit_sell_order_open, mocker, order_book_l2, caplog) -> None:
    """
    test order book ask strategy
    """
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book', order_book_l2)
    default_conf['exchange']['name'] = 'binance'
    default_conf['ask_strategy']['use_order_book'] = True
    default_conf['ask_strategy']['order_book_min'] = 1
    default_conf['ask_strategy']['order_book_max'] = 2
    default_conf['telegram']['enabled'] = False
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 0.00001172,
            'ask': 0.00001173,
            'last': 0.00001172
        }),
        buy=MagicMock(return_value=limit_buy_order_open),
        sell=MagicMock(return_value=limit_sell_order_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf)
    patch_get_signal(freqtrade)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    time.sleep(0.01)  # Race condition fix
    trade.update(limit_buy_order)
    freqtrade.wallets.update()
    assert trade.is_open is True

    patch_get_signal(freqtrade, value=(False, True))
    assert freqtrade.handle_trade(trade) is True
    assert trade.close_rate_requested == order_book_l2.return_value['asks'][0][0]

    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book',
                 return_value={'bids': [[]], 'asks': [[]]})
    with pytest.raises(PricingError):
        freqtrade.handle_trade(trade)
    assert log_has('Sell Price at location 1 from orderbook could not be determined.', caplog)


@pytest.mark.parametrize('side,ask,bid,last,last_ab,expected', [
    ('bid', 12.0, 11.0, 11.5, 0.0, 11.0),  # full bid side
    ('bid', 12.0, 11.0, 11.5, 1.0, 11.5),  # full last side
    ('bid', 12.0, 11.0, 11.5, 0.5, 11.25),  # between bid and lat
    ('bid', 12.0, 11.2, 10.5, 0.0, 11.2),  # Last smaller than bid
    ('bid', 12.0, 11.2, 10.5, 1.0, 11.2),  # Last smaller than bid - uses bid
    ('bid', 12.0, 11.2, 10.5, 0.5, 11.2),  # Last smaller than bid - uses bid
    ('bid', 0.003, 0.002, 0.005, 0.0, 0.002),
    ('ask', 12.0, 11.0, 12.5, 0.0, 12.0),  # full ask side
    ('ask', 12.0, 11.0, 12.5, 1.0, 12.5),  # full last side
    ('ask', 12.0, 11.0, 12.5, 0.5, 12.25),  # between bid and lat
    ('ask', 12.2, 11.2, 10.5, 0.0, 12.2),  # Last smaller than ask
    ('ask', 12.0, 11.0, 10.5, 1.0, 12.0),  # Last smaller than ask - uses ask
    ('ask', 12.0, 11.2, 10.5, 0.5, 12.0),  # Last smaller than ask - uses ask
    ('ask', 10.0, 11.0, 11.0, 0.0, 10.0),
    ('ask', 10.11, 11.2, 11.0, 0.0, 10.11),
    ('ask', 0.001, 0.002, 11.0, 0.0, 0.001),
    ('ask', 0.006, 1.0, 11.0, 0.0, 0.006),
])
def test_get_sell_rate(default_conf, mocker, caplog, side, bid, ask,
                       last, last_ab, expected) -> None:
    caplog.set_level(logging.DEBUG)

    default_conf['ask_strategy']['price_side'] = side
    default_conf['ask_strategy']['bid_last_balance'] = last_ab
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 return_value={'ask': ask, 'bid': bid, 'last': last})
    pair = "ETH/BTC"

    # Test regular mode
    ft = get_patched_freqtradebot(mocker, default_conf)
    rate = ft.get_sell_rate(pair, True)
    assert not log_has("Using cached sell rate for ETH/BTC.", caplog)
    assert isinstance(rate, float)
    assert rate == expected
    # Use caching
    rate = ft.get_sell_rate(pair, False)
    assert rate == expected
    assert log_has("Using cached sell rate for ETH/BTC.", caplog)


@pytest.mark.parametrize('side,expected', [
    ('bid', 0.043936),  # Value from order_book_l2 fiture - bids side
    ('ask', 0.043949),  # Value from order_book_l2 fiture - asks side
])
def test_get_sell_rate_orderbook(default_conf, mocker, caplog, side, expected, order_book_l2):
    caplog.set_level(logging.DEBUG)
    # Test orderbook mode
    default_conf['ask_strategy']['price_side'] = side
    default_conf['ask_strategy']['use_order_book'] = True
    default_conf['ask_strategy']['order_book_min'] = 1
    default_conf['ask_strategy']['order_book_max'] = 2
    pair = "ETH/BTC"
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book', order_book_l2)
    ft = get_patched_freqtradebot(mocker, default_conf)
    rate = ft.get_sell_rate(pair, True)
    assert not log_has("Using cached sell rate for ETH/BTC.", caplog)
    assert isinstance(rate, float)
    assert rate == expected
    rate = ft.get_sell_rate(pair, False)
    assert rate == expected
    assert log_has("Using cached sell rate for ETH/BTC.", caplog)


def test_get_sell_rate_orderbook_exception(default_conf, mocker, caplog):
    # Test orderbook mode
    default_conf['ask_strategy']['price_side'] = 'ask'
    default_conf['ask_strategy']['use_order_book'] = True
    default_conf['ask_strategy']['order_book_min'] = 1
    default_conf['ask_strategy']['order_book_max'] = 2
    pair = "ETH/BTC"
    # Test What happens if the exchange returns an empty orderbook.
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book',
                 return_value={'bids': [[]], 'asks': [[]]})
    ft = get_patched_freqtradebot(mocker, default_conf)
    with pytest.raises(PricingError):
        ft.get_sell_rate(pair, True)
    assert log_has("Sell Price at location from orderbook could not be determined.", caplog)


def test_get_sell_rate_exception(default_conf, mocker, caplog):
    # Ticker on one side can be empty in certain circumstances.
    default_conf['ask_strategy']['price_side'] = 'ask'
    pair = "ETH/BTC"
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 return_value={'ask': None, 'bid': 0.12, 'last': None})
    ft = get_patched_freqtradebot(mocker, default_conf)
    with pytest.raises(PricingError, match=r"Sell-Rate for ETH/BTC was empty."):
        ft.get_sell_rate(pair, True)

    ft.config['ask_strategy']['price_side'] = 'bid'
    assert ft.get_sell_rate(pair, True) == 0.12
    # Reverse sides
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 return_value={'ask': 0.13, 'bid': None, 'last': None})
    with pytest.raises(PricingError, match=r"Sell-Rate for ETH/BTC was empty."):
        ft.get_sell_rate(pair, True)

    ft.config['ask_strategy']['price_side'] = 'ask'
    assert ft.get_sell_rate(pair, True) == 0.13


def test_startup_state(default_conf, mocker):
    default_conf['pairlist'] = {'method': 'VolumePairList',
                                'config': {'number_assets': 20}
                                }
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    worker = get_patched_worker(mocker, default_conf)
    assert worker.freqtrade.state is State.RUNNING


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


@pytest.mark.usefixtures("init_persistence")
def test_sync_wallet_dry_run(mocker, default_conf, ticker, fee, limit_buy_order_open, caplog):
    default_conf['dry_run'] = True
    # Initialize to 2 times stake amount
    default_conf['dry_run_wallet'] = 0.002
    default_conf['max_open_trades'] = 2
    default_conf['tradable_balance_ratio'] = 1.0
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        buy=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )

    bot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(bot)
    assert bot.wallets.get_free('BTC') == 0.002

    n = bot.enter_positions()
    assert n == 2
    trades = Trade.query.all()
    assert len(trades) == 2

    bot.config['max_open_trades'] = 3
    n = bot.enter_positions()
    assert n == 0
    assert log_has_re(r"Unable to create trade for XRP/BTC: "
                      r"Available balance \(0.0 BTC\) is lower than stake amount \(0.001 BTC\)",
                      caplog)


@pytest.mark.usefixtures("init_persistence")
def test_cancel_all_open_orders(mocker, default_conf, fee, limit_buy_order, limit_sell_order):
    default_conf['cancel_open_orders_on_exit'] = True
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 side_effect=[
                     ExchangeError(), limit_sell_order, limit_buy_order, limit_sell_order])
    buy_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_buy')
    sell_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_sell')

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    create_mock_trades(fee)
    trades = Trade.query.all()
    assert len(trades) == MOCK_TRADE_COUNT
    freqtrade.cancel_all_open_orders()
    assert buy_mock.call_count == 1
    assert sell_mock.call_count == 2


@pytest.mark.usefixtures("init_persistence")
def test_check_for_open_trades(mocker, default_conf, fee):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    freqtrade.check_for_open_trades()
    assert freqtrade.rpc.send_msg.call_count == 0

    create_mock_trades(fee)
    trade = Trade.query.first()
    trade.is_open = True

    freqtrade.check_for_open_trades()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert 'Handle these trades manually' in freqtrade.rpc.send_msg.call_args[0][0]['status']


@pytest.mark.usefixtures("init_persistence")
def test_update_open_orders(mocker, default_conf, fee, caplog):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    create_mock_trades(fee)

    freqtrade.update_open_orders()
    assert not log_has_re(r"Error updating Order .*", caplog)

    freqtrade.config['dry_run'] = False
    freqtrade.update_open_orders()

    assert log_has_re(r"Error updating Order .*", caplog)
    caplog.clear()

    assert len(Order.get_open_orders()) == 3
    matching_buy_order = mock_order_4()
    matching_buy_order.update({
        'status': 'closed',
    })
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=matching_buy_order)
    freqtrade.update_open_orders()
    # Only stoploss and sell orders are kept open
    assert len(Order.get_open_orders()) == 2


@pytest.mark.usefixtures("init_persistence")
def test_update_closed_trades_without_assigned_fees(mocker, default_conf, fee):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    def patch_with_fee(order):
        order.update({'fee': {'cost': 0.1, 'rate': 0.01,
                      'currency': order['symbol'].split('/')[0]}})
        return order

    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 side_effect=[
                     patch_with_fee(mock_order_2_sell()),
                     patch_with_fee(mock_order_3_sell()),
                     patch_with_fee(mock_order_1()),
                     patch_with_fee(mock_order_2()),
                     patch_with_fee(mock_order_3()),
                     patch_with_fee(mock_order_4()),
                 ]
                 )

    create_mock_trades(fee)
    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT
    for trade in trades:
        assert trade.fee_open_cost is None
        assert trade.fee_open_currency is None
        assert trade.fee_close_cost is None
        assert trade.fee_close_currency is None

    freqtrade.update_closed_trades_without_assigned_fees()

    # Does nothing for dry-run
    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT
    for trade in trades:
        assert trade.fee_open_cost is None
        assert trade.fee_open_currency is None
        assert trade.fee_close_cost is None
        assert trade.fee_close_currency is None

    freqtrade.config['dry_run'] = False

    freqtrade.update_closed_trades_without_assigned_fees()

    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT

    for trade in trades:
        if trade.is_open:
            # Exclude Trade 4 - as the order is still open.
            if trade.select_order('buy', False):
                assert trade.fee_open_cost is not None
                assert trade.fee_open_currency is not None
            else:
                assert trade.fee_open_cost is None
                assert trade.fee_open_currency is None

        else:
            assert trade.fee_close_cost is not None
            assert trade.fee_close_currency is not None


@pytest.mark.usefixtures("init_persistence")
def test_reupdate_buy_order_fees(mocker, default_conf, fee, caplog):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mock_uts = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.update_trade_state')

    create_mock_trades(fee)
    trades = Trade.get_trades().all()

    freqtrade.reupdate_buy_order_fees(trades[0])
    assert log_has_re(r"Trying to reupdate buy fees for .*", caplog)
    assert mock_uts.call_count == 1
    assert mock_uts.call_args_list[0][0][0] == trades[0]
    assert mock_uts.call_args_list[0][0][1] == mock_order_1()['id']
    assert log_has_re(r"Updating buy-fee on trade .* for order .*\.", caplog)
    mock_uts.reset_mock()
    caplog.clear()

    # Test with trade without orders
    trade = Trade(
        pair='XRP/ETH',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=arrow.utcnow().datetime,
        is_open=True,
        amount=20,
        open_rate=0.01,
        exchange='binance',
    )
    Trade.query.session.add(trade)

    freqtrade.reupdate_buy_order_fees(trade)
    assert log_has_re(r"Trying to reupdate buy fees for .*", caplog)
    assert mock_uts.call_count == 0
    assert not log_has_re(r"Updating buy-fee on trade .* for order .*\.", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_handle_insufficient_funds(mocker, default_conf, fee):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mock_rlo = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.refind_lost_order')
    mock_bof = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.reupdate_buy_order_fees')
    create_mock_trades(fee)
    trades = Trade.get_trades().all()

    # Trade 0 has only a open buy order, no closed order
    freqtrade.handle_insufficient_funds(trades[0])
    assert mock_rlo.call_count == 0
    assert mock_bof.call_count == 1

    mock_rlo.reset_mock()
    mock_bof.reset_mock()

    # Trade 1 has closed buy and sell orders
    freqtrade.handle_insufficient_funds(trades[1])
    assert mock_rlo.call_count == 1
    assert mock_bof.call_count == 0

    mock_rlo.reset_mock()
    mock_bof.reset_mock()

    # Trade 2 has closed buy and sell orders
    freqtrade.handle_insufficient_funds(trades[2])
    assert mock_rlo.call_count == 1
    assert mock_bof.call_count == 0

    mock_rlo.reset_mock()
    mock_bof.reset_mock()

    # Trade 3 has an opne buy order
    freqtrade.handle_insufficient_funds(trades[3])
    assert mock_rlo.call_count == 0
    assert mock_bof.call_count == 1


@pytest.mark.usefixtures("init_persistence")
def test_refind_lost_order(mocker, default_conf, fee, caplog):
    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mock_uts = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.update_trade_state')

    mock_fo = mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                           return_value={'status': 'open'})

    def reset_open_orders(trade):
        trade.open_order_id = None
        trade.stoploss_order_id = None

    create_mock_trades(fee)
    trades = Trade.get_trades().all()

    caplog.clear()

    # No open order
    trade = trades[0]
    reset_open_orders(trade)
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is None

    freqtrade.refind_lost_order(trade)
    order = mock_order_1()
    assert log_has_re(r"Order Order(.*order_id=" + order['id'] + ".*) is no longer open.", caplog)
    assert mock_fo.call_count == 0
    assert mock_uts.call_count == 0
    # No change to orderid - as update_trade_state is mocked
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is None

    caplog.clear()
    mock_fo.reset_mock()

    # Open buy order
    trade = trades[3]
    reset_open_orders(trade)
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is None

    freqtrade.refind_lost_order(trade)
    order = mock_order_4()
    assert log_has_re(r"Trying to refind Order\(.*", caplog)
    assert mock_fo.call_count == 0
    assert mock_uts.call_count == 0
    # No change to orderid - as update_trade_state is mocked
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is None

    caplog.clear()
    mock_fo.reset_mock()

    # Open stoploss order
    trade = trades[4]
    reset_open_orders(trade)
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is None

    freqtrade.refind_lost_order(trade)
    order = mock_order_5_stoploss()
    assert log_has_re(r"Trying to refind Order\(.*", caplog)
    assert mock_fo.call_count == 1
    assert mock_uts.call_count == 1
    # stoploss_order_id is "refound" and added to the trade
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is not None

    caplog.clear()
    mock_fo.reset_mock()
    mock_uts.reset_mock()

    # Open sell order
    trade = trades[5]
    reset_open_orders(trade)
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is None

    freqtrade.refind_lost_order(trade)
    order = mock_order_6_sell()
    assert log_has_re(r"Trying to refind Order\(.*", caplog)
    assert mock_fo.call_count == 1
    assert mock_uts.call_count == 1
    # sell-orderid is "refound" and added to the trade
    assert trade.open_order_id == order['id']
    assert trade.stoploss_order_id is None

    caplog.clear()

    # Test error case
    mock_fo = mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                           side_effect=ExchangeError())
    order = mock_order_5_stoploss()

    freqtrade.refind_lost_order(trades[4])
    assert log_has(f"Error updating {order['id']}.", caplog)
