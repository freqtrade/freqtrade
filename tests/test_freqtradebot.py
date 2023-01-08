# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

import logging
import time
from copy import deepcopy
from typing import List
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import arrow
import pytest
from pandas import DataFrame

from freqtrade.constants import CANCEL_REASON, UNLIMITED_STAKE_AMOUNT
from freqtrade.enums import (CandleType, ExitCheckTuple, ExitType, RPCMessageType, RunMode,
                             SignalDirection, State)
from freqtrade.exceptions import (DependencyException, ExchangeError, InsufficientFundsError,
                                  InvalidOrderException, OperationalException, PricingError,
                                  TemporaryError)
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, PairLocks, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.plugins.protections.iprotection import ProtectionReturn
from freqtrade.worker import Worker
from tests.conftest import (create_mock_trades, create_mock_trades_usdt, get_patched_freqtradebot,
                            get_patched_worker, log_has, log_has_re, patch_edge, patch_exchange,
                            patch_get_signal, patch_wallet, patch_whitelist)
from tests.conftest_trades import (MOCK_TRADE_COUNT, entry_side, exit_side, mock_order_1,
                                   mock_order_2, mock_order_2_sell, mock_order_3, mock_order_3_sell,
                                   mock_order_4, mock_order_5_stoploss, mock_order_6_sell)
from tests.conftest_trades_usdt import mock_trade_usdt_4


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


def test_freqtradebot_state(mocker, default_conf_usdt, markets) -> None:
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    assert freqtrade.state is State.RUNNING

    default_conf_usdt.pop('initial_state')
    freqtrade = FreqtradeBot(default_conf_usdt)
    assert freqtrade.state is State.STOPPED


def test_process_stopped(mocker, default_conf_usdt) -> None:

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    coo_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders')
    freqtrade.process_stopped()
    assert coo_mock.call_count == 0

    default_conf_usdt['cancel_open_orders_on_exit'] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.process_stopped()
    assert coo_mock.call_count == 1


def test_process_calls_sendmsg(mocker, default_conf_usdt) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.process()
    assert freqtrade.rpc.process_msg_queue.call_count == 1


def test_bot_cleanup(mocker, default_conf_usdt, caplog) -> None:
    mock_cleanup = mocker.patch('freqtrade.freqtradebot.Trade.commit')
    coo_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders')
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.cleanup()
    assert log_has('Cleaning up modules ...', caplog)
    assert mock_cleanup.call_count == 1
    assert coo_mock.call_count == 0

    freqtrade.config['cancel_open_orders_on_exit'] = True
    freqtrade.cleanup()
    assert coo_mock.call_count == 1


def test_bot_cleanup_db_errors(mocker, default_conf_usdt, caplog) -> None:
    mocker.patch('freqtrade.freqtradebot.Trade.commit',
                 side_effect=OperationalException())
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.check_for_open_trades',
                 side_effect=OperationalException())
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.emc = MagicMock()
    freqtrade.emc.shutdown = MagicMock()
    freqtrade.cleanup()
    assert freqtrade.emc.shutdown.call_count == 1


@pytest.mark.parametrize('runmode', [
    RunMode.DRY_RUN,
    RunMode.LIVE
])
def test_order_dict(default_conf_usdt, mocker, runmode, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['entry_pricing']['price_side'] = 'ask'

    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re(".*stoploss_on_exchange .* dry-run", caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']

    caplog.clear()
    # is left untouched
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re(".*stoploss_on_exchange .* dry-run", caplog)


def test_get_trade_stake_amount(default_conf_usdt, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)

    freqtrade = FreqtradeBot(default_conf_usdt)

    result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT')
    assert result == default_conf_usdt['stake_amount']


@pytest.mark.parametrize("amend_last,wallet,max_open,lsamr,expected", [
                        (False, 120, 2, 0.5, [60, None]),
                        (True, 120, 2, 0.5, [60, 58.8]),
                        (False, 180, 3, 0.5, [60, 60, None]),
                        (True, 180, 3, 0.5, [60, 60, 58.2]),
                        (False, 122, 3, 0.5, [60, 60, None]),
                        (True, 122, 3, 0.5, [60, 60, 0.0]),
                        (True, 167, 3, 0.5, [60, 60, 45.33]),
                        (True, 122, 3, 1, [60, 60, 0.0]),
])
def test_check_available_stake_amount(
    default_conf_usdt, ticker_usdt, mocker, fee, limit_buy_order_usdt_open,
    amend_last, wallet, max_open, lsamr, expected
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee
    )
    default_conf_usdt['dry_run_wallet'] = wallet

    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr

    freqtrade = FreqtradeBot(default_conf_usdt)

    for i in range(0, max_open):

        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT')
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT')


def test_edge_called_in_process(mocker, edge_conf) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)

    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
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


@pytest.mark.parametrize('buy_price_mult,ignore_strat_sl', [
    (0.79, False),   # Override stoploss
    (0.85, True),    # Override strategy stoploss
])
def test_edge_overrides_stoploss(limit_order, fee, caplog, mocker,
                                 buy_price_mult, ignore_strat_sl, edge_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')

    # Strategy stoploss is -0.1 but Edge imposes a stoploss at -0.2
    # Thus, if price falls 21%, stoploss should be triggered
    #
    # mocking the ticker: price is falling ...
    enter_price = limit_order['buy']['price']
    ticker_val = {
            'bid': enter_price,
            'ask': enter_price,
            'last': enter_price,
        }
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value=ticker_val),
        get_fee=fee,
    )
    #############################################

    # Create a trade with "limit_buy_order_usdt" price
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.query.first()
    caplog.clear()
    #############################################
    ticker_val.update({
            'bid': enter_price * buy_price_mult,
            'ask': enter_price * buy_price_mult,
            'last': enter_price * buy_price_mult,
    })

    # stoploss shoud be hit
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re('Exit for NEO/BTC detected. Reason: stop_loss.*', caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        # Test compatibility ...
        assert trade.sell_reason == ExitType.STOP_LOSS.value


def test_total_open_trades_stakes(mocker, default_conf_usdt, ticker_usdt, fee) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()
    trade = Trade.query.first()

    assert trade is not None
    assert trade.stake_amount == 60.0
    assert trade.is_open
    assert trade.open_date is not None

    freqtrade.enter_positions()
    trade = Trade.query.order_by(Trade.id.desc()).first()

    assert trade is not None
    assert trade.stake_amount == 60.0
    assert trade.is_open
    assert trade.open_date is not None

    assert Trade.total_open_trades_stakes() == 120.0


@pytest.mark.parametrize("is_short,open_rate", [
    (False, 2.0),
    (True, 2.2)
])
def test_create_trade(default_conf_usdt, ticker_usdt, limit_order,
                      fee, mocker, is_short, open_rate) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.create_trade('ETH/USDT')

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'

    # Simulate fulfilled LIMIT_BUY order for trade
    oobj = Order.parse_from_ccxt_object(
        limit_order[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)

    assert trade.open_rate == open_rate
    assert trade.amount == 30.0

    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(default_conf_usdt, ticker_usdt, fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize('stake_amount,create,amount_enough,max_open_trades', [
    (5.0, True, True, 99),
    (0.049, True, False, 99),  # Amount will be adjusted to min - which is 0.051
    (0, False, True, 99),
    (UNLIMITED_STAKE_AMOUNT, False, True, 0),
])
def test_create_trade_minimal_amount(
    default_conf_usdt, ticker_usdt, limit_order_open, fee, mocker,
    stake_amount, create, amount_enough, max_open_trades, caplog, is_short
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    enter_mock = MagicMock(return_value=limit_order_open[entry_side(is_short)])
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=enter_mock,
        get_fee=fee,
    )
    default_conf_usdt['max_open_trades'] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['stake_amount'] = stake_amount
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    if create:
        assert freqtrade.create_trade('ETH/USDT')
        if amount_enough:
            rate, amount = enter_mock.call_args[1]['rate'], enter_mock.call_args[1]['amount']
            assert rate * amount <= default_conf_usdt['stake_amount']
        else:
            assert log_has_re(
                r"Stake amount for pair .* is too small.*",
                caplog
            )
    else:
        assert not freqtrade.create_trade('ETH/USDT')
        if not max_open_trades:
            assert freqtrade.wallets.get_trade_stake_amount('ETH/USDT', freqtrade.edge) == 0


@pytest.mark.parametrize('whitelist,positions', [
    (["ETH/USDT"], 1),  # No pairs left
    ([], 0),  # No pairs in whitelist
])
def test_enter_positions_no_pairs_left(default_conf_usdt, ticker_usdt, limit_buy_order_usdt_open,
                                       fee, whitelist, positions, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    default_conf_usdt['exchange']['pair_whitelist'] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re(r"No currency pair in active pair whitelist.*", caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re(r"No currency pair in active pair whitelist.*", caplog)
    else:
        assert n == 0
        assert log_has("Active pair whitelist is empty.", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_enter_positions_global_pairlock(default_conf_usdt, ticker_usdt, limit_buy_order_usdt, fee,
                                         mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = r"Global pairlock active until.* Not creating new trades."
    n = freqtrade.enter_positions()
    # 0 trades, but it's not because of pairlock.
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()

    PairLocks.lock_pair('*', arrow.utcnow().shift(minutes=20).datetime, 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize('is_short', [False, True])
def test_handle_protections(mocker, default_conf_usdt, fee, is_short):
    default_conf_usdt['protections'] = [
        {"method": "CooldownPeriod", "stop_duration": 60},
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,
            "trade_limit": 4,
            "stop_duration_candles": 4,
            "only_per_pair": False
        }
    ]

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.protections._protection_handlers[1].global_stop = MagicMock(
        return_value=ProtectionReturn(True, arrow.utcnow().shift(hours=1).datetime, "asdf"))
    create_mock_trades(fee, is_short)
    freqtrade.handle_protections('ETC/BTC', '*')
    send_msg_mock = freqtrade.rpc.send_msg
    assert send_msg_mock.call_count == 2
    assert send_msg_mock.call_args_list[0][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER
    assert send_msg_mock.call_args_list[1][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL


def test_create_trade_no_signal(default_conf_usdt, fee, mocker) -> None:
    default_conf_usdt['dry_run'] = True

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_fee=fee,
    )
    default_conf_usdt['stake_amount'] = 10
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)

    assert not freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize("max_open", range(0, 5))
@pytest.mark.parametrize("tradable_balance_ratio,modifier", [(1.0, 1), (0.99, 0.8), (0.5, 0.5)])
def test_create_trades_multiple_trades(
    default_conf_usdt, ticker_usdt, fee, mocker, limit_buy_order_usdt_open,
    max_open, tradable_balance_ratio, modifier
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = max_open
    default_conf_usdt['tradable_balance_ratio'] = tradable_balance_ratio
    default_conf_usdt['dry_run_wallet'] = 60.0 * max_open

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    n = freqtrade.enter_positions()
    trades = Trade.get_open_trades()
    # Expected trades should be max_open * a modified value
    # depending on the configured tradable_balance
    assert n == max(int(max_open * modifier), 0)
    assert len(trades) == max(int(max_open * modifier), 0)


def test_create_trades_preopen(default_conf_usdt, ticker_usdt, fee, mocker,
                               limit_buy_order_usdt_open, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 4
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
        get_funding_fees=MagicMock(side_effect=ExchangeError()),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    # Create 2 existing trades
    freqtrade.execute_entry('ETH/USDT', default_conf_usdt['stake_amount'])
    freqtrade.execute_entry('NEO/BTC', default_conf_usdt['stake_amount'])
    assert log_has("Could not find funding fee.", caplog)

    assert len(Trade.get_open_trades()) == 2
    # Change order_id for new orders
    limit_buy_order_usdt_open['id'] = '123444'

    # Create 2 new trades using create_trades
    assert freqtrade.create_trade('ETH/USDT')
    assert freqtrade.create_trade('NEO/BTC')

    trades = Trade.get_open_trades()
    assert len(trades) == 4


@pytest.mark.parametrize('is_short', [False, True])
def test_process_trade_creation(default_conf_usdt, ticker_usdt, limit_order, limit_order_open,
                                is_short, fee, mocker, caplog
                                ) -> None:
    ticker_side = 'ask' if is_short else 'bid'
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        fetch_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades

    freqtrade.process()

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == default_conf_usdt['stake_amount']
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    assert trade.open_rate == ticker_usdt.return_value[ticker_side]
    assert pytest.approx(trade.amount) == 60 / ticker_usdt.return_value[ticker_side]

    assert log_has(
        f'{"Short" if is_short else "Long"} signal found: about create a new trade for ETH/USDT '
        'with stake_amount: 60.0 ...',
        caplog
    )


def test_process_exchange_failures(default_conf_usdt, ticker_usdt, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=TemporaryError)
    )
    sleep_mock = mocker.patch('time.sleep', side_effect=lambda _: None)

    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)

    worker._process_running()
    assert sleep_mock.has_calls()


def test_process_operational_exception(default_conf_usdt, ticker_usdt, mocker) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=OperationalException)
    )
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)

    assert worker.freqtrade.state == State.RUNNING

    worker._process_running()
    assert worker.freqtrade.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]['status']


def test_process_trade_handling(default_conf_usdt, ticker_usdt, limit_buy_order_usdt_open, fee,
                                mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        fetch_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades
    freqtrade.process()

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1

    # Nothing happened ...
    freqtrade.process()
    assert len(trades) == 1


def test_process_trade_no_whitelist_pair(default_conf_usdt, ticker_usdt, limit_buy_order_usdt,
                                         fee, mocker) -> None:
    """ Test process with trade not in pair list """
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        fetch_order=MagicMock(return_value=limit_buy_order_usdt),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    pair = 'BLK/BTC'
    # Ensure the pair is not in the whitelist!
    assert pair not in default_conf_usdt['exchange']['pair_whitelist']

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
        pair='ETH/USDT',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        amount=12,
        open_rate=0.001,
        exchange='binance',
    ))
    Trade.commit()

    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    # Make sure each pair is only in the list once
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))


def test_process_informative_pairs_added(default_conf_usdt, ticker_usdt, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)

    refresh_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=TemporaryError),
        refresh_latest_ohlcv=refresh_mock,
    )
    inf_pairs = MagicMock(return_value=[
        ("BTC/ETH", '1m', CandleType.SPOT),
        ("ETH/USDT", "1h", CandleType.SPOT)
    ])
    mocker.patch.multiple(
        'freqtrade.strategy.interface.IStrategy',
        get_exit_signal=MagicMock(return_value=(False, False)),
        get_entry_signal=MagicMock(return_value=(None, None))
    )
    mocker.patch('time.sleep', return_value=None)

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.informative_pairs = inf_pairs
    # patch_get_signal(freqtrade)

    freqtrade.process()
    assert inf_pairs.call_count == 1
    assert refresh_mock.call_count == 1
    assert ("BTC/ETH", "1m", CandleType.SPOT) in refresh_mock.call_args[0][0]
    assert ("ETH/USDT", "1h", CandleType.SPOT) in refresh_mock.call_args[0][0]
    assert ("ETH/USDT", default_conf_usdt["timeframe"],
            CandleType.SPOT) in refresh_mock.call_args[0][0]


@pytest.mark.parametrize("is_short,trading_mode,exchange_name,margin_mode,liq_buffer,liq_price", [
    (False, 'spot', 'binance', None, 0.0, None),
    (True, 'spot', 'binance', None, 0.0, None),
    (False, 'spot', 'gateio', None, 0.0, None),
    (True, 'spot', 'gateio', None, 0.0, None),
    (False, 'spot', 'okx', None, 0.0, None),
    (True, 'spot', 'okx', None, 0.0, None),
    (True, 'futures', 'binance', 'isolated', 0.0, 11.88151815181518),
    (False, 'futures', 'binance', 'isolated', 0.0, 8.080471380471382),
    (True, 'futures', 'gateio', 'isolated', 0.0, 11.87413417771621),
    (False, 'futures', 'gateio', 'isolated', 0.0, 8.085708510208207),
    (True, 'futures', 'binance', 'isolated', 0.05, 11.7874422442244),
    (False, 'futures', 'binance', 'isolated', 0.05, 8.17644781144781),
    (True, 'futures', 'gateio', 'isolated', 0.05, 11.7804274688304),
    (False, 'futures', 'gateio', 'isolated', 0.05, 8.181423084697796),
    (True, 'futures', 'okx', 'isolated', 0.0, 11.87413417771621),
    (False, 'futures', 'okx', 'isolated', 0.0, 8.085708510208207),
])
def test_execute_entry(mocker, default_conf_usdt, fee, limit_order,
                       limit_order_open, is_short, trading_mode,
                       exchange_name, margin_mode, liq_buffer, liq_price) -> None:
    """
    exchange_name = binance, is_short = true
        leverage = 5
        position = 0.2 * 5
        ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
        ((2 + 0.01) - ((-1) * 1 * 10)) / ((1 * 0.01) - ((-1) * 1)) = 11.89108910891089

    exchange_name = binance, is_short = false
        ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
        ((2 + 0.01) - (1 * 1 * 10)) / ((1 * 0.01) - (1 * 1)) = 8.070707070707071

    exchange_name = gateio/okx, is_short = true
        (open_rate + (wallet_balance / position)) / (1 + (mm_ratio + taker_fee_rate))
        (10 + (2 / 1)) / (1 + (0.01 + 0.0006)) = 11.87413417771621

    exchange_name = gateio/okx, is_short = false
        (open_rate - (wallet_balance / position)) / (1 - (mm_ratio + taker_fee_rate))
        (10 - (2 / 1)) / (1 - (0.01 + 0.0006)) = 8.085708510208207
    """
    # TODO: Split this test into multiple tests to improve readability
    open_order = limit_order_open[entry_side(is_short)]
    order = limit_order[entry_side(is_short)]
    default_conf_usdt['trading_mode'] = trading_mode
    default_conf_usdt['liquidation_buffer'] = liq_buffer
    leverage = 1.0 if trading_mode == 'spot' else 5.0
    default_conf_usdt['exchange']['name'] = exchange_name
    if margin_mode:
        default_conf_usdt['margin_mode'] = margin_mode
    mocker.patch('freqtrade.exchange.Gateio.validate_ordertypes')
    patch_RPCManager(mocker)
    patch_exchange(mocker, id=exchange_name)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    freqtrade.strategy.leverage = MagicMock(return_value=leverage)
    stake_amount = 2
    bid = 0.11
    enter_rate_mock = MagicMock(return_value=bid)
    enter_mm = MagicMock(return_value=open_order)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_rate=enter_rate_mock,
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=enter_mm,
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_max_pair_stake_amount=MagicMock(return_value=500000),
        get_fee=fee,
        get_funding_fees=MagicMock(return_value=0),
        name=exchange_name,
        get_maintenance_ratio_and_amt=MagicMock(return_value=(0.01, 0.01)),
        get_max_leverage=MagicMock(return_value=10),
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Okx',
        get_max_pair_stake_amount=MagicMock(return_value=500000),
    )
    pair = 'ETH/USDT'

    assert not freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    assert enter_rate_mock.call_count == 1
    assert enter_mm.call_count == 0
    assert freqtrade.strategy.confirm_trade_entry.call_count == 1
    enter_rate_mock.reset_mock()

    open_order['id'] = '22'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_entry(pair, stake_amount)
    assert enter_rate_mock.call_count == 1
    assert enter_mm.call_count == 1
    call_args = enter_mm.call_args_list[0][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == bid
    assert pytest.approx(call_args['amount']) == round(stake_amount / bid * leverage, 8)
    enter_rate_mock.reset_mock()

    # Should create an open trade with an open order id
    # As the order is not fulfilled yet
    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id == '22'

    # Test calling with price
    open_order['id'] = '33'
    fix_price = 0.06
    assert freqtrade.execute_entry(pair, stake_amount, fix_price, is_short=is_short)
    # Make sure get_rate wasn't called again
    assert enter_rate_mock.call_count == 0

    assert enter_mm.call_count == 2
    call_args = enter_mm.call_args_list[1][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == fix_price
    assert pytest.approx(call_args['amount']) == round(stake_amount / fix_price * leverage, 8)

    # In case of closed order
    order['status'] = 'closed'
    order['average'] = 10
    order['cost'] = 300
    order['id'] = '444'

    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=order))
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.query.all()[2]
    trade.is_short = is_short
    assert trade
    assert trade.open_order_id is None
    assert trade.open_rate == 10
    assert trade.stake_amount == round(order['average'] * order['filled'] / leverage, 8)
    assert pytest.approx(trade.liquidation_price) == liq_price

    # In case of rejected or expired order and partially filled
    order['status'] = 'expired'
    order['amount'] = 30.0
    order['filled'] = 20.0
    order['remaining'] = 10.00
    order['average'] = 0.5
    order['cost'] = 10.0
    order['id'] = '555'
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=order))
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.query.all()[3]
    trade.is_short = is_short
    assert trade
    assert trade.open_order_id is None
    assert trade.open_rate == 0.5
    assert trade.stake_amount == round(order['average'] * order['filled'] / leverage, 8)

    # Test with custom stake
    order['status'] = 'open'
    order['id'] = '556'

    freqtrade.strategy.custom_stake_amount = lambda **kwargs: 150.0
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.query.all()[4]
    trade.is_short = is_short
    assert trade
    assert pytest.approx(trade.stake_amount) == 150

    # Exception case
    order['id'] = '557'
    freqtrade.strategy.custom_stake_amount = lambda **kwargs: 20 / 0
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.query.all()[5]
    trade.is_short = is_short
    assert trade
    assert pytest.approx(trade.stake_amount) == 2.0

    # In case of the order is rejected and not filled at all
    order['status'] = 'rejected'
    order['amount'] = 30.0 * leverage
    order['filled'] = 0.0
    order['remaining'] = 30.0
    order['average'] = 0.5
    order['cost'] = 0.0
    order['id'] = '66'
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=order))
    assert not freqtrade.execute_entry(pair, stake_amount)
    assert freqtrade.strategy.leverage.call_count == 0 if trading_mode == 'spot' else 2

    # Fail to get price...
    mocker.patch('freqtrade.exchange.Exchange.get_rate', MagicMock(return_value=0.0))

    with pytest.raises(PricingError, match="Could not determine entry price."):
        freqtrade.execute_entry(pair, stake_amount, is_short=is_short)

    # In case of custom entry price
    mocker.patch('freqtrade.exchange.Exchange.get_rate', return_value=0.50)
    order['status'] = 'open'
    order['id'] = '5566'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: 0.508
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.query.all()[6]
    trade.is_short = is_short
    assert trade
    assert trade.open_rate_requested == 0.508

    # In case of custom entry price set to None

    order['status'] = 'open'
    order['id'] = '5567'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: None

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_rate=MagicMock(return_value=10),
    )

    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.query.all()[7]
    trade.is_short = is_short
    assert trade
    assert trade.open_rate_requested == 10

    # In case of custom entry price not float type
    order['status'] = 'open'
    order['id'] = '5568'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: "string price"
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.query.all()[8]
    # Trade(id=9, pair=ETH/USDT, amount=0.20000000, is_short=False,
    #   leverage=1.0, open_rate=10.00000000, open_since=...)
    # Trade(id=9, pair=ETH/USDT, amount=0.60000000, is_short=True,
    #   leverage=3.0, open_rate=10.00000000, open_since=...)
    trade.is_short = is_short
    assert trade
    assert trade.open_rate_requested == 10

    # In case of too high stake amount

    order['status'] = 'open'
    order['id'] = '55672'

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_max_pair_stake_amount=MagicMock(return_value=500),
    )
    freqtrade.exchange.get_max_pair_stake_amount = MagicMock(return_value=500)

    assert freqtrade.execute_entry(pair, 2000, is_short=is_short)
    trade = Trade.query.all()[9]
    trade.is_short = is_short
    assert pytest.approx(trade.stake_amount) == 500

    order['id'] = '55673'

    freqtrade.strategy.leverage.reset_mock()
    assert freqtrade.execute_entry(pair, 200, leverage_=3)
    assert freqtrade.strategy.leverage.call_count == 0
    trade = Trade.query.all()[10]
    assert trade.leverage == 1 if trading_mode == 'spot' else 3


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_entry_confirm_error(mocker, default_conf_usdt, fee, limit_order, is_short) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_rate=MagicMock(return_value=0.11),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    stake_amount = 2
    pair = 'ETH/USDT'

    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=ValueError)
    assert freqtrade.execute_entry(pair, stake_amount)

    limit_order[entry_side(is_short)]['id'] = '222'
    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=Exception)
    assert freqtrade.execute_entry(pair, stake_amount)

    limit_order[entry_side(is_short)]['id'] = '2223'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_entry(pair, stake_amount)

    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    assert not freqtrade.execute_entry(pair, stake_amount)


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_entry_min_leverage(mocker, default_conf_usdt, fee, limit_order, is_short) -> None:
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_rate=MagicMock(return_value=0.11),
        # Minimum stake-amount is ~5$
        get_maintenance_ratio_and_amt=MagicMock(return_value=(0.0, 0.0)),
        _fetch_and_calculate_funding_fees=MagicMock(return_value=0),
        get_fee=fee,
        get_max_leverage=MagicMock(return_value=5.0),
    )
    stake_amount = 2
    pair = 'SOL/BUSD:BUSD'
    freqtrade.strategy.leverage = MagicMock(return_value=5.0)

    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.query.first()
    assert trade.leverage == 5.0
    # assert trade.stake_amount == 2


@pytest.mark.parametrize("is_short", [False, True])
def test_add_stoploss_on_exchange(mocker, default_conf_usdt, limit_order, is_short) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    order = limit_order[entry_side(is_short)]
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=order)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])

    stoploss = MagicMock(return_value={'id': 13434334})
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss)

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # TODO: should not be magicmock
    trade = MagicMock()
    trade.is_short = is_short
    trade.open_order_id = None
    trade.stoploss_order_id = None
    trade.is_open = True
    trades = [trade]

    freqtrade.exit_positions(trades)
    assert trade.stoploss_order_id == '13434334'
    assert stoploss.call_count == 1
    assert trade.is_open is True


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_stoploss_on_exchange(mocker, default_conf_usdt, fee, caplog, is_short,
                                     limit_order) -> None:
    stoploss = MagicMock(return_value={'id': 13434334})
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(side_effect=[
            enter_order,
            exit_order,
        ]),
        get_fee=fee,
        stoploss=stoploss
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # First case: when stoploss is not yet set but the order is open
    # should get the stoploss order id immediately
    # and should return false as no trade actually happened
    # TODO: should not be magicmock
    trade = MagicMock()
    trade.is_short = is_short
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
    trade.stoploss_order_id = "100"

    hanging_stoploss_order = MagicMock(return_value={'status': 'open'})
    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order', hanging_stoploss_order)

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.stoploss_order_id == "100"

    # Third case: when stoploss was set but it was canceled for some reason
    # should set a stoploss immediately and return False
    caplog.clear()
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = "100"

    canceled_stoploss_order = MagicMock(return_value={'status': 'canceled'})
    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order', canceled_stoploss_order)
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
    trade.is_short = is_short
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = "100"
    trade.orders.append(Order(
        ft_order_side='stoploss',
        order_id='100',
        ft_pair=trade.pair,
        ft_is_open=True,
        ft_amount=trade.amount,
        ft_price=0.0,
    ))
    assert trade

    stoploss_order_hit = MagicMock(return_value={
        'id': "100",
        'status': 'closed',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'amount': enter_order['amount'],
    })
    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order', stoploss_order_hit)
    assert freqtrade.handle_stoploss_on_exchange(trade) is True
    assert log_has_re(r'STOP_LOSS_LIMIT is hit for Trade\(id=1, .*\)\.', caplog)
    assert trade.stoploss_order_id is None
    assert trade.is_open is False
    caplog.clear()

    mocker.patch(
        'freqtrade.exchange.Exchange.stoploss',
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
    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order',
                 side_effect=InvalidOrderException())
    mocker.patch('freqtrade.exchange.Exchange.stoploss', stoploss)
    freqtrade.handle_stoploss_on_exchange(trade)
    assert stoploss.call_count == 1

    # Sixth case: Closed Trade
    # Should not create new order
    trade.stoploss_order_id = None
    trade.is_open = False
    stoploss.reset_mock()
    mocker.patch('freqtrade.exchange.Exchange.fetch_order')
    mocker.patch('freqtrade.exchange.Exchange.stoploss', stoploss)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 0

    # Seventh case: emergency exit triggered
    # Trailing stop should not act anymore
    stoploss_order_cancelled = MagicMock(side_effect=[{
        'id': "100",
        'status': 'canceled',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'amount': enter_order['amount'],
        'info': {'stopPrice': 22},
    }])
    trade.stoploss_order_id = 100
    trade.is_open = True
    trade.stoploss_last_update = arrow.utcnow().shift(hours=-1).datetime
    trade.stop_loss = 24
    freqtrade.config['trailing_stop'] = True
    stoploss = MagicMock(side_effect=InvalidOrderException())

    mocker.patch('freqtrade.exchange.Exchange.cancel_stoploss_order_with_result',
                 side_effect=InvalidOrderException())
    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order', stoploss_order_cancelled)
    mocker.patch('freqtrade.exchange.Exchange.stoploss', stoploss)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.stoploss_order_id is None
    assert trade.is_open is False
    assert trade.exit_reason == str(ExitType.EMERGENCY_EXIT)


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_sle_cancel_cant_recreate(mocker, default_conf_usdt, fee, caplog, is_short,
                                         limit_order) -> None:
    # Sixth case: stoploss order was cancelled but couldn't create new one
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(side_effect=[
            {'id': enter_order['id']},
            {'id': exit_order['id']},
        ]),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        fetch_stoploss_order=MagicMock(return_value={'status': 'canceled', 'id': 100}),
        stoploss=MagicMock(side_effect=ExchangeError()),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.is_short = is_short
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100
    assert trade

    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert log_has_re(r'Stoploss order was cancelled, but unable to recreate one.*', caplog)
    assert trade.stoploss_order_id is None
    assert trade.is_open is True


@pytest.mark.parametrize("is_short", [False, True])
def test_create_stoploss_order_invalid_order(
    mocker, default_conf_usdt, caplog, fee, is_short, limit_order, limit_order_open
):
    open_order = limit_order_open[entry_side(is_short)]
    order = limit_order[exit_side(is_short)]
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    create_order_mock = MagicMock(side_effect=[
        open_order,
        {'id': order['id']}
    ])
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=create_order_mock,
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        fetch_order=MagicMock(return_value={'status': 'canceled'}),
        stoploss=MagicMock(side_effect=InvalidOrderException()),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.is_short = is_short
    caplog.clear()
    freqtrade.create_stoploss_order(trade, 200)
    assert trade.stoploss_order_id is None
    assert trade.exit_reason == ExitType.EMERGENCY_EXIT.value
    assert log_has("Unable to place a stoploss order on exchange. ", caplog)
    assert log_has("Exiting the trade forcefully", caplog)

    # Should call a market sell
    assert create_order_mock.call_count == 2
    assert create_order_mock.call_args[1]['ordertype'] == 'market'
    assert create_order_mock.call_args[1]['pair'] == trade.pair
    assert create_order_mock.call_args[1]['amount'] == trade.amount

    # Rpc is sending first buy, then sell
    assert rpc_mock.call_count == 3
    assert rpc_mock.call_args_list[2][0][0]['sell_reason'] == ExitType.EMERGENCY_EXIT.value
    assert rpc_mock.call_args_list[2][0][0]['order_type'] == 'market'


@pytest.mark.parametrize("is_short", [False, True])
def test_create_stoploss_order_insufficient_funds(
    mocker, default_conf_usdt, caplog, fee, limit_order, is_short
):
    exit_order = limit_order[exit_side(is_short)]['id']
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    mock_insuf = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_insufficient_funds')
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(side_effect=[
            limit_order[entry_side(is_short)],
            exit_order,
        ]),
        get_fee=fee,
        fetch_order=MagicMock(return_value={'status': 'canceled'}),
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=MagicMock(side_effect=InsufficientFundsError()),
    )
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.is_short = is_short
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


@pytest.mark.parametrize("is_short,bid,ask,stop_price,amt,hang_price", [
    (False, [4.38, 4.16], [4.4, 4.17], ['2.0805', 4.4 * 0.95], 27.39726027, 3),
    (True, [1.09, 1.21], [1.1, 1.22], ['2.321', 1.09 * 1.05], 27.27272727, 1.5),
])
@pytest.mark.usefixtures("init_persistence")
def test_handle_stoploss_on_exchange_trailing(
    mocker, default_conf_usdt, fee, is_short, bid, ask, limit_order, stop_price, amt, hang_price
) -> None:
    # When trailing stoploss is set
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 2.19,
            'ask': 2.2,
            'last': 2.19,
        }),
        create_order=MagicMock(side_effect=[
            {'id': enter_order['id']},
            {'id': exit_order['id']},
        ]),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss,
        stoploss_adjust=MagicMock(return_value=True),
    )

    # enabling TSL
    default_conf_usdt['trailing_stop'] = True

    # disabling ROI
    default_conf_usdt['minimal_roi']['0'] = 999999999

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    # enabling stoploss on exchange
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # setting stoploss
    freqtrade.strategy.stoploss = 0.05 if is_short else -0.05

    # setting stoploss_on_exchange_interval to 60 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60

    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.is_short = is_short
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100
    trade.stoploss_last_update = arrow.utcnow().shift(minutes=-20).datetime

    stoploss_order_hanging = MagicMock(return_value={
        'id': 100,
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': hang_price,
        'average': 2,
        'info': {
            'stopPrice': stop_price[0]
        }
    })

    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', stoploss_order_hanging)

    # stoploss initially at 5%
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # price jumped 2x
    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': bid[0],
            'ask': ask[0],
            'last': bid[0],
        })
    )

    cancel_order_mock = MagicMock()
    stoploss_order_mock = MagicMock(return_value={'id': 'so1'})
    mocker.patch('freqtrade.exchange.Binance.cancel_stoploss_order', cancel_order_mock)
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss_order_mock)

    # stoploss should not be updated as the interval is 60 seconds
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    cancel_order_mock.assert_not_called()
    stoploss_order_mock.assert_not_called()

    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == stop_price[1]

    # setting stoploss_on_exchange_interval to 0 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    cancel_order_mock.assert_called_once_with(100, 'ETH/USDT')
    stoploss_order_mock.assert_called_once_with(
        amount=pytest.approx(amt),
        pair='ETH/USDT',
        order_types=freqtrade.strategy.order_types,
        stop_price=stop_price[1],
        side=exit_side(is_short),
        leverage=1.0
    )

    # price fell below stoploss, so dry-run sells trade.
    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': bid[1],
            'ask': ask[1],
            'last': bid[1],
        })
    )
    assert freqtrade.handle_trade(trade) is True
    assert trade.stoploss_order_id is None


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_stoploss_on_exchange_trailing_error(
    mocker, default_conf_usdt, fee, caplog, limit_order, is_short
) -> None:
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    # When trailing stoploss is set
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_exchange(mocker)

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(side_effect=[
            {'id': enter_order['id']},
            {'id': exit_order['id']},
        ]),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss,
        stoploss_adjust=MagicMock(return_value=True),
    )

    # enabling TSL
    default_conf_usdt['trailing_stop'] = True

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    # enabling stoploss on exchange
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # setting stoploss
    freqtrade.strategy.stoploss = 0.05 if is_short else -0.05

    # setting stoploss_on_exchange_interval to 60 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.is_short = is_short
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = "abcd"
    trade.stop_loss = 0.2
    trade.stoploss_last_update = arrow.utcnow().shift(minutes=-601).datetime.replace(tzinfo=None)
    trade.is_short = is_short

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
    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order',
                 return_value=stoploss_order_hanging)
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert log_has_re(r"Could not cancel stoploss order abcd for pair ETH/USDT.*", caplog)

    # Still try to create order
    assert stoploss.call_count == 1

    # Fail creating stoploss order
    trade.stoploss_last_update = arrow.utcnow().shift(minutes=-601).datetime
    caplog.clear()
    cancel_mock = mocker.patch("freqtrade.exchange.Binance.cancel_stoploss_order", MagicMock())
    mocker.patch("freqtrade.exchange.Binance.stoploss", side_effect=ExchangeError())
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert cancel_mock.call_count == 1
    assert log_has_re(r"Could not create trailing stoploss order for pair ETH/USDT\..*", caplog)


def test_stoploss_on_exchange_price_rounding(
        mocker, default_conf_usdt, fee, open_trade_usdt) -> None:
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_fee=fee,
    )
    price_mock = MagicMock(side_effect=lambda p, s: int(s))
    stoploss_mock = MagicMock(return_value={'id': '13434334'})
    adjust_mock = MagicMock(return_value=False)
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss_mock,
        stoploss_adjust=adjust_mock,
        price_to_precision=price_mock,
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    open_trade_usdt.stoploss_order_id = '13434334'
    open_trade_usdt.stop_loss = 222.55

    freqtrade.handle_trailing_stoploss_on_exchange(open_trade_usdt, {})
    assert price_mock.call_count == 1
    assert adjust_mock.call_count == 1
    assert adjust_mock.call_args_list[0][0][0] == 222


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.usefixtures("init_persistence")
def test_handle_stoploss_on_exchange_custom_stop(
    mocker, default_conf_usdt, fee, is_short, limit_order
) -> None:
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    # When trailing stoploss is set
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(side_effect=[
            {'id': enter_order['id']},
            {'id': exit_order['id']},
        ]),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss,
        stoploss_adjust=MagicMock(return_value=True),
    )

    # enabling TSL
    default_conf_usdt['use_custom_stoploss'] = True

    # disabling ROI
    default_conf_usdt['minimal_roi']['0'] = 999999999

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    # enabling stoploss on exchange
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    # setting stoploss
    freqtrade.strategy.custom_stoploss = lambda *args, **kwargs: -0.04

    # setting stoploss_on_exchange_interval to 60 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60

    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    freqtrade.enter_positions()
    trade = Trade.query.first()
    trade.is_short = is_short
    trade.is_open = True
    trade.open_order_id = None
    trade.stoploss_order_id = 100
    trade.stoploss_last_update = arrow.utcnow().shift(minutes=-601).datetime

    stoploss_order_hanging = MagicMock(return_value={
        'id': 100,
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'info': {
            'stopPrice': '2.0805'
        }
    })

    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', stoploss_order_hanging)

    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # price jumped 2x
    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': 4.38 if not is_short else 1.9 / 2,
            'ask': 4.4 if not is_short else 2.2 / 2,
            'last': 4.38 if not is_short else 1.9 / 2,
        })
    )

    cancel_order_mock = MagicMock()
    stoploss_order_mock = MagicMock(return_value={'id': 'so1'})
    mocker.patch('freqtrade.exchange.Binance.cancel_stoploss_order', cancel_order_mock)
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss_order_mock)

    # stoploss should not be updated as the interval is 60 seconds
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    cancel_order_mock.assert_not_called()
    stoploss_order_mock.assert_not_called()

    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == 4.4 * 0.96 if not is_short else 1.1
    assert trade.stop_loss_pct == -0.04 if not is_short else 0.04

    # setting stoploss_on_exchange_interval to 0 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    cancel_order_mock.assert_called_once_with(100, 'ETH/USDT')
    # Long uses modified ask - offset, short modified bid + offset
    stoploss_order_mock.assert_called_once_with(
        amount=pytest.approx(trade.amount),
        pair='ETH/USDT',
        order_types=freqtrade.strategy.order_types,
        stop_price=4.4 * 0.96 if not is_short else 0.95 * 1.04,
        side=exit_side(is_short),
        leverage=1.0
    )

    # price fell below stoploss, so dry-run sells trade.
    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': 4.17,
            'ask': 4.19,
            'last': 4.17
        })
    )
    assert freqtrade.handle_trade(trade) is True


def test_tsl_on_exchange_compatible_with_edge(mocker, edge_conf, fee, limit_order) -> None:

    enter_order = limit_order['buy']
    exit_order = limit_order['sell']

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
            'bid': 2.19,
            'ask': 2.2,
            'last': 2.19
        }),
        create_order=MagicMock(side_effect=[
            {'id': enter_order['id']},
            {'id': exit_order['id']},
        ]),
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
    trade.stoploss_last_update = arrow.utcnow()

    stoploss_order_hanging = MagicMock(return_value={
        'id': 100,
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'stopPrice': '2.178'
    })

    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order', stoploss_order_hanging)

    # stoploss initially at 20% as edge dictated it.
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert pytest.approx(trade.stop_loss) == 1.76

    cancel_order_mock = MagicMock()
    stoploss_order_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.cancel_stoploss_order', cancel_order_mock)
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss_order_mock)

    # price goes down 5%
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', MagicMock(return_value={
        'bid': 2.19 * 0.95,
        'ask': 2.2 * 0.95,
        'last': 2.19 * 0.95
    }))
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # stoploss should remain the same
    assert pytest.approx(trade.stop_loss) == 1.76

    # stoploss on exchange should not be canceled
    cancel_order_mock.assert_not_called()

    # price jumped 2x
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', MagicMock(return_value={
        'bid': 4.38,
        'ask': 4.4,
        'last': 4.38
    }))

    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    # stoploss should be set to 1% as trailing is on
    assert trade.stop_loss == 4.4 * 0.99
    cancel_order_mock.assert_called_once_with(100, 'NEO/BTC')
    stoploss_order_mock.assert_called_once_with(
        amount=pytest.approx(11.41438356),
        pair='NEO/BTC',
        order_types=freqtrade.strategy.order_types,
        stop_price=4.4 * 0.99,
        side='sell',
        leverage=1.0
    )


@pytest.mark.parametrize('return_value,side_effect,log_message', [
    (False, None, 'Found no enter signals for whitelisted currencies. Trying again...'),
    (None, DependencyException, 'Unable to create trade for ETH/USDT: ')
])
def test_enter_positions(mocker, default_conf_usdt, return_value, side_effect,
                         log_message, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    mock_ct = mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.create_trade',
        MagicMock(
            return_value=return_value,
            side_effect=side_effect
        )
    )
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has(log_message, caplog)
    # create_trade should be called once for every pair in the whitelist.
    assert mock_ct.call_count == len(default_conf_usdt['exchange']['pair_whitelist'])


@pytest.mark.parametrize("is_short", [False, True])
def test_exit_positions(mocker, default_conf_usdt, limit_order, is_short, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 return_value=limit_order[entry_side(is_short)])
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])

    # TODO: should not be magicmock
    trade = MagicMock()
    trade.is_short = is_short
    trade.open_order_id = '123'
    trade.open_fee = 0.001
    trades = [trade]
    n = freqtrade.exit_positions(trades)
    assert n == 0
    # Test amount not modified by fee-logic
    assert not log_has_re(r'Applying fee to amount for Trade .*', caplog)

    gra = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=0.0)
    # test amount modified by fee-logic
    n = freqtrade.exit_positions(trades)
    assert n == 0
    assert gra.call_count == 0


@pytest.mark.parametrize("is_short", [False, True])
def test_exit_positions_exception(mocker, default_conf_usdt, limit_order, caplog, is_short) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    order = limit_order[entry_side(is_short)]
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=order)

    # TODO: should not be magicmock
    trade = MagicMock()
    trade.is_short = is_short
    trade.open_order_id = None
    trade.pair = 'ETH/USDT'
    trades = [trade]

    # Test raise of DependencyException exception
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.handle_trade',
        side_effect=DependencyException()
    )
    caplog.clear()
    n = freqtrade.exit_positions(trades)
    assert n == 0
    assert log_has('Unable to exit trade ETH/USDT: ', caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_update_trade_state(mocker, default_conf_usdt, limit_order, is_short, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    order = limit_order[entry_side(is_short)]

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=order)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=0.0)
    order_id = order['id']

    trade = Trade(
        open_order_id=order_id,
        fee_open=0.001,
        fee_close=0.001,
        open_rate=0.01,
        open_date=arrow.utcnow().datetime,
        amount=11,
        exchange="binance",
        is_short=is_short,
        leverage=1,
    )
    trade.orders.append(Order(
        ft_order_side=entry_side(is_short),
        price=0.01,
        order_id=order_id,

    ))
    assert not freqtrade.update_trade_state(trade, None)
    assert log_has_re(r'Orderid for trade .* is empty.', caplog)
    caplog.clear()
    # Add datetime explicitly since sqlalchemy defaults apply only once written to database
    freqtrade.update_trade_state(trade, order_id)
    # Test amount not modified by fee-logic
    assert not log_has_re(r'Applying fee to .*', caplog)
    caplog.clear()
    assert trade.open_order_id is None
    assert trade.amount == order['amount']

    trade.open_order_id = order_id
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=0.01)
    assert trade.amount == 30.0
    # test amount modified by fee-logic
    freqtrade.update_trade_state(trade, order_id)
    assert trade.amount == 29.99
    assert trade.open_order_id is None

    trade.is_open = True
    trade.open_order_id = None
    # Assert we call handle_trade() if trade is feasible for execution
    freqtrade.update_trade_state(trade, order_id)

    assert log_has_re('Found open order for.*', caplog)
    limit_buy_order_usdt_new = deepcopy(limit_order)
    limit_buy_order_usdt_new['filled'] = 0.0
    limit_buy_order_usdt_new['status'] = 'canceled'

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', side_effect=ValueError)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order_usdt_new)
    res = freqtrade.update_trade_state(trade, order_id)
    # Cancelled empty
    assert res is True


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize('initial_amount,has_rounding_fee', [
    (30.0 + 1e-14, True),
    (8.0, False)
])
def test_update_trade_state_withorderdict(
    default_conf_usdt, trades_for_order, limit_order, fee, mocker, initial_amount,
    has_rounding_fee, is_short, caplog
):
    order = limit_order[entry_side(is_short)]
    trades_for_order[0]['amount'] = initial_amount
    order_id = "oid_123456"
    order['id'] = order_id
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    # fetch_order should not be called!!
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', MagicMock(side_effect=ValueError))
    patch_exchange(mocker)
    amount = sum(x['amount'] for x in trades_for_order)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    caplog.clear()
    trade = Trade(
        pair='LTC/USDT',
        amount=amount,
        exchange='binance',
        open_rate=2.0,
        open_date=arrow.utcnow().datetime,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_order_id=order_id,
        is_open=True,
        leverage=1,
        is_short=is_short,
    )
    trade.orders.append(
        Order(
            ft_order_side=entry_side(is_short),
            ft_pair=trade.pair,
            ft_is_open=True,
            order_id=order_id,
        )
    )
    log_text = r'Applying fee on amount for .*'
    freqtrade.update_trade_state(trade, order_id, order)
    assert trade.amount != amount
    if has_rounding_fee:
        assert pytest.approx(trade.amount) == 29.992
        assert log_has_re(log_text, caplog)
    else:
        assert pytest.approx(trade.amount) == order['amount']
        assert not log_has_re(log_text, caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_update_trade_state_exception(mocker, default_conf_usdt, is_short, limit_order,
                                      caplog) -> None:
    order = limit_order[entry_side(is_short)]
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=order)

    # TODO: should not be magicmock
    trade = MagicMock()
    trade.open_order_id = '123'
    trade.amount = 123

    # Test raise of OperationalException exception
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
        side_effect=DependencyException()
    )
    freqtrade.update_trade_state(trade, trade.open_order_id)
    assert log_has('Could not update trade amount: ', caplog)


def test_update_trade_state_orderexception(mocker, default_conf_usdt, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 MagicMock(side_effect=InvalidOrderException))

    # TODO: should not be magicmock
    trade = MagicMock()
    trade.open_order_id = '123'

    # Test raise of OperationalException exception
    grm_mock = mocker.patch("freqtrade.freqtradebot.FreqtradeBot.get_real_amount", MagicMock())
    freqtrade.update_trade_state(trade, trade.open_order_id)
    assert grm_mock.call_count == 0
    assert log_has(f'Unable to fetch order {trade.open_order_id}: ', caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_update_trade_state_sell(
    default_conf_usdt, trades_for_order, limit_order_open, limit_order, is_short, mocker
):
    buy_order = limit_order[entry_side(is_short)]
    open_order = limit_order_open[exit_side(is_short)]
    l_order = limit_order[exit_side(is_short)]
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    # fetch_order should not be called!!
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', MagicMock(side_effect=ValueError))
    wallet_mock = MagicMock()
    mocker.patch('freqtrade.wallets.Wallets.update', wallet_mock)

    patch_exchange(mocker)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    amount = l_order["amount"]
    wallet_mock.reset_mock()
    trade = Trade(
        pair='LTC/ETH',
        amount=amount,
        exchange='binance',
        open_rate=0.245441,
        fee_open=0.0025,
        fee_close=0.0025,
        open_date=arrow.utcnow().datetime,
        open_order_id=open_order['id'],
        is_open=True,
        interest_rate=0.0005,
        leverage=1,
        is_short=is_short,
    )
    order = Order.parse_from_ccxt_object(buy_order, 'LTC/ETH', entry_side(is_short))
    trade.orders.append(order)

    order = Order.parse_from_ccxt_object(open_order, 'LTC/ETH', exit_side(is_short))
    trade.orders.append(order)
    assert order.status == 'open'
    freqtrade.update_trade_state(trade, trade.open_order_id, l_order)
    assert trade.amount == l_order['amount']
    # Wallet needs to be updated after closing a limit-sell order to reenable buying
    assert wallet_mock.call_count == 1
    assert not trade.is_open
    # Order is updated by update_trade_state
    assert order.status == 'closed'


@pytest.mark.parametrize('is_short,close_profit', [
    (False, 0.09451372),
    (True, 0.08635224),
])
def test_handle_trade(
    default_conf_usdt, limit_order_open, limit_order, fee, mocker, is_short, close_profit
) -> None:
    open_order = limit_order_open[exit_side(is_short)]
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 2.19,
            'ask': 2.2,
            'last': 2.19
        }),
        create_order=MagicMock(side_effect=[
            enter_order,
            open_order,
        ]),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade

    time.sleep(0.01)  # Race condition fix
    assert trade.is_open is True
    freqtrade.wallets.update()

    patch_get_signal(freqtrade, enter_long=False, exit_short=is_short,
                     exit_long=not is_short, exit_tag='sell_signal1')
    assert freqtrade.handle_trade(trade) is True
    assert trade.open_order_id == exit_order['id']

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.orders[-1].ft_is_open = False
    trade.orders[-1].status = 'closed'
    trade.orders[-1].filled = trade.orders[-1].remaining
    trade.orders[-1].remaining = 0.0

    trade.update_trade(trade.orders[-1])

    assert trade.close_rate == (2.0 if is_short else 2.2)
    assert pytest.approx(trade.close_profit) == close_profit
    assert pytest.approx(trade.calc_profit(trade.close_rate)) == 5.685
    assert trade.close_date is not None
    assert trade.exit_reason == 'sell_signal1'


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_overlapping_signals(
    default_conf_usdt, ticker_usdt, limit_order_open, fee, mocker, is_short
) -> None:
    open_order = limit_order_open[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[
            open_order,
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )

    freqtrade = FreqtradeBot(default_conf_usdt)
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, enter_short=True, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=True, exit_long=True)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()

    # Buy and Sell triggering, so doing nothing ...
    trades = Trade.query.all()

    nb_trades = len(trades)
    assert nb_trades == 0

    # Buy is triggering, so buying ...
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trades = Trade.query.all()
    for trade in trades:
        trade.is_short = is_short
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Buy and Sell are not triggering, so doing nothing ...
    patch_get_signal(freqtrade, enter_long=False)
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.query.all()
    for trade in trades:
        trade.is_short = is_short
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Buy and Sell are triggering, so doing nothing ...
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, enter_short=True, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=True, exit_long=True)
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.query.all()
    for trade in trades:
        trade.is_short = is_short
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Sell is triggering, guess what : we are Selling!
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    trades = Trade.query.all()
    for trade in trades:
        trade.is_short = is_short
    assert freqtrade.handle_trade(trades[0]) is True


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_trade_roi(default_conf_usdt, ticker_usdt, limit_order_open, fee, mocker, caplog,
                          is_short) -> None:

    open_order = limit_order_open[entry_side(is_short)]

    caplog.set_level(logging.DEBUG)

    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[
            open_order,
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    trade.is_open = True

    # FIX: sniffing logs, suggest handle_trade should not execute_trade_exit
    #      instead that responsibility should be moved out of handle_trade(),
    #      we might just want to check if we are in a sell condition without
    #      executing
    # if ROI is reached we must sell
    caplog.clear()
    patch_get_signal(freqtrade)
    assert freqtrade.handle_trade(trade)
    assert log_has("ETH/USDT - Required profit reached. exit_type=ExitType.ROI",
                   caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_trade_use_exit_signal(
    default_conf_usdt, ticker_usdt, limit_order_open, fee, mocker, caplog, is_short
) -> None:

    enter_open_order = limit_order_open[exit_side(is_short)]
    exit_open_order = limit_order_open[entry_side(is_short)]

    # use_exit_signal is True buy default
    caplog.set_level(logging.DEBUG)
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[
            enter_open_order,
            exit_open_order,
        ]),
        get_fee=fee,
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    trade.is_open = True

    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert not freqtrade.handle_trade(trade)

    if is_short:
        patch_get_signal(freqtrade, enter_long=False, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    assert freqtrade.handle_trade(trade)
    assert log_has("ETH/USDT - Sell signal received. exit_type=ExitType.EXIT_SIGNAL",
                   caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_close_trade(
    default_conf_usdt, ticker_usdt, limit_order_open, limit_order, fee, mocker, is_short
) -> None:
    open_order = limit_order_open[exit_side(is_short)]
    enter_order = limit_order[exit_side(is_short)]
    exit_order = limit_order[entry_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=open_order),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # Create trade and sell it
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade

    oobj = Order.parse_from_ccxt_object(enter_order, enter_order['symbol'], trade.entry_side)
    trade.update_trade(oobj)
    oobj = Order.parse_from_ccxt_object(exit_order, exit_order['symbol'], trade.exit_side)
    trade.update_trade(oobj)
    assert trade.is_open is False

    with pytest.raises(DependencyException, match=r'.*closed trade.*'):
        freqtrade.handle_trade(trade)


def test_bot_loop_start_called_once(mocker, default_conf_usdt, caplog):
    ftbot = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade')
    patch_get_signal(ftbot)
    ftbot.strategy.bot_loop_start = MagicMock(side_effect=ValueError)
    ftbot.strategy.analyze = MagicMock()

    ftbot.process()
    assert log_has_re(r'Strategy caused the following exception.*', caplog)
    assert ftbot.strategy.bot_loop_start.call_count == 1
    assert ftbot.strategy.analyze.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_entry_usercustom(
    default_conf_usdt, ticker_usdt, limit_buy_order_old, open_trade,
    limit_sell_order_old, fee, mocker, is_short
) -> None:

    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    old_order['id'] = open_trade.open_order_id

    default_conf_usdt["unfilledtimeout"] = {"entry": 1400, "exit": 30}

    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock(return_value=old_order)
    cancel_enter_order = deepcopy(old_order)
    cancel_enter_order['status'] = 'canceled'
    cancel_order_wr_mock = MagicMock(return_value=cancel_enter_order)

    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=old_order),
        cancel_order=cancel_order_mock,
        cancel_order_with_result=cancel_order_wr_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade.is_short = is_short
    open_trade.orders[0].side = 'sell' if is_short else 'buy'
    open_trade.orders[0].ft_order_side = 'sell' if is_short else 'buy'
    Trade.query.session.add(open_trade)
    Trade.commit()

    # Ensure default is to return empty (so not mocked yet)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0

    # Return false - trade remains open
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 1
    freqtrade.strategy.check_entry_timeout = MagicMock(side_effect=KeyError)

    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 1
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=True)

    # Trade should be closed since the function returns true
    freqtrade.manage_open_orders()
    assert cancel_order_wr_mock.call_count == 1
    assert rpc_mock.call_count == 2
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    assert freqtrade.strategy.check_entry_timeout.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_entry(
    default_conf_usdt, ticker_usdt, limit_buy_order_old, open_trade,
    limit_sell_order_old, fee, mocker, is_short
) -> None:
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    rpc_mock = patch_RPCManager(mocker)
    open_trade.open_order_id = old_order['id']
    order = Order.parse_from_ccxt_object(old_order, 'mocked', 'buy')
    open_trade.orders[0] = order
    limit_buy_cancel = deepcopy(old_order)
    limit_buy_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_buy_cancel)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=old_order),
        cancel_order_with_result=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

    open_trade.is_short = is_short
    Trade.query.session.add(open_trade)
    Trade.commit()

    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=1234)
    # check it does cancel buy orders over the time limit
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    # Custom user buy-timeout is never called
    assert freqtrade.strategy.check_entry_timeout.call_count == 0
    # Entry adjustment is never called
    assert freqtrade.strategy.adjust_entry_price.call_count == 0


@pytest.mark.parametrize("is_short", [False, True])
def test_adjust_entry_cancel(
    default_conf_usdt, ticker_usdt, limit_buy_order_old, open_trade,
    limit_sell_order_old, fee, mocker, caplog, is_short
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    old_order['id'] = open_trade.open_order_id
    limit_buy_cancel = deepcopy(old_order)
    limit_buy_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_buy_cancel)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=old_order),
        cancel_order_with_result=cancel_order_mock,
        get_fee=fee
    )

    open_trade.is_short = is_short
    Trade.query.session.add(open_trade)
    Trade.commit()

    # Timeout to not interfere
    freqtrade.strategy.ft_check_timed_out = MagicMock(return_value=False)

    # check that order is cancelled
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=None)
    freqtrade.manage_open_orders()
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 0
    assert len(Order.query.all()) == 0
    assert log_has_re(
        f"{'Sell' if is_short else 'Buy'} order user requested order cancel*", caplog)
    assert log_has_re(
        f"{'Sell' if is_short else 'Buy'} order fully cancelled.*", caplog)

    # Entry adjustment is called
    assert freqtrade.strategy.adjust_entry_price.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_adjust_entry_maintain_replace(
    default_conf_usdt, ticker_usdt, limit_buy_order_old, open_trade,
    limit_sell_order_old, fee, mocker, caplog, is_short
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    old_order['id'] = open_trade.open_order_id
    limit_buy_cancel = deepcopy(old_order)
    limit_buy_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_buy_cancel)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=old_order),
        cancel_order_with_result=cancel_order_mock,
        get_fee=fee
    )

    open_trade.is_short = is_short
    Trade.query.session.add(open_trade)
    Trade.commit()

    # Timeout to not interfere
    freqtrade.strategy.ft_check_timed_out = MagicMock(return_value=False)

    # Check that order is maintained
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=old_order['price'])
    freqtrade.manage_open_orders()
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 1
    assert len(Order.get_open_orders()) == 1
    # Entry adjustment is called
    assert freqtrade.strategy.adjust_entry_price.call_count == 1

    # Check that order is replaced
    freqtrade.get_valid_enter_price_and_stake = MagicMock(return_value={100, 10, 1})
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=1234)
    freqtrade.manage_open_orders()
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 1
    nb_all_orders = len(Order.query.all())
    assert nb_all_orders == 2
    # New order seems to be in closed status?
    # nb_open_orders = len(Order.get_open_orders())
    # assert nb_open_orders == 1
    assert log_has_re(
        f"{'Sell' if is_short else 'Buy'} order cancelled to be replaced*", caplog)
    # Entry adjustment is called
    assert freqtrade.strategy.adjust_entry_price.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_check_handle_cancelled_buy(
    default_conf_usdt, ticker_usdt, limit_buy_order_old, open_trade,
    limit_sell_order_old, fee, mocker, caplog, is_short
) -> None:
    """ Handle Buy order cancelled on exchange"""
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    old_order.update({"status": "canceled", 'filled': 0.0})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=old_order),
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade.orders = []
    open_trade.is_short = is_short
    Trade.query.session.add(open_trade)
    Trade.commit()

    # check it does cancel buy orders over the time limit
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 2
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 0
    assert log_has_re(
        f"{'Sell' if is_short else 'Buy'} order cancelled on exchange for Trade.*", caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_buy_exception(
    default_conf_usdt, ticker_usdt, open_trade, is_short, fee, mocker
) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(side_effect=ExchangeError),
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

    open_trade.is_short = is_short
    Trade.query.session.add(open_trade)
    Trade.commit()

    # check it does cancel buy orders over the time limit
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_exit_usercustom(
    default_conf_usdt, ticker_usdt, limit_sell_order_old, mocker,
    is_short, open_trade_usdt, caplog
) -> None:
    default_conf_usdt["unfilledtimeout"] = {"entry": 1440, "exit": 1440, "exit_timeout_count": 1}
    open_trade_usdt.open_order_id = limit_sell_order_old['id']
    order = Order.parse_from_ccxt_object(limit_sell_order_old, 'mocked', 'sell')
    open_trade_usdt.orders[0] = order
    if is_short:
        limit_sell_order_old['side'] = 'buy'
        open_trade_usdt.is_short = is_short

    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_min_pair_stake_amount', return_value=0.0)
    et_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.execute_trade_exit')
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

    open_trade_usdt.open_date = arrow.utcnow().shift(hours=-5).datetime
    open_trade_usdt.close_date = arrow.utcnow().shift(minutes=-601).datetime
    open_trade_usdt.close_profit_abs = 0.001

    Trade.query.session.add(open_trade_usdt)
    Trade.commit()
    # Ensure default is false
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0

    freqtrade.strategy.check_exit_timeout = MagicMock(return_value=False)
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    # Return false - No impact
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert freqtrade.strategy.check_exit_timeout.call_count == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 0

    freqtrade.strategy.check_exit_timeout = MagicMock(side_effect=KeyError)
    freqtrade.strategy.check_entry_timeout = MagicMock(side_effect=KeyError)
    # Return Error - No impact
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert freqtrade.strategy.check_exit_timeout.call_count == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 0

    # Return True - sells!
    freqtrade.strategy.check_exit_timeout = MagicMock(return_value=True)
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=True)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    assert freqtrade.strategy.check_exit_timeout.call_count == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 0

    # 2nd canceled trade - Fail execute sell
    caplog.clear()
    open_trade_usdt.open_order_id = limit_sell_order_old['id']
    mocker.patch('freqtrade.persistence.Trade.get_exit_order_count', return_value=1)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.execute_trade_exit',
                 side_effect=DependencyException)
    freqtrade.manage_open_orders()
    assert log_has_re('Unable to emergency sell .*', caplog)

    et_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.execute_trade_exit')
    caplog.clear()
    # 2nd canceled trade ...
    open_trade_usdt.open_order_id = limit_sell_order_old['id']

    # If cancelling fails - no emergency sell!
    with patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_exit', return_value=False):
        freqtrade.manage_open_orders()
        assert et_mock.call_count == 0

    freqtrade.manage_open_orders()
    assert log_has_re('Emergency exiting trade.*', caplog)
    assert et_mock.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_exit(
    default_conf_usdt, ticker_usdt, limit_sell_order_old, mocker, is_short, open_trade_usdt
) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    limit_sell_order_old['id'] = open_trade_usdt.open_order_id
    limit_sell_order_old['side'] = 'buy' if is_short else 'sell'
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=cancel_order_mock,
        get_min_pair_stake_amount=MagicMock(return_value=0),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

    open_trade_usdt.open_date = arrow.utcnow().shift(hours=-5).datetime
    open_trade_usdt.close_date = arrow.utcnow().shift(minutes=-601).datetime
    open_trade_usdt.close_profit_abs = 0.001
    open_trade_usdt.is_short = is_short

    Trade.query.session.add(open_trade_usdt)
    Trade.commit()

    freqtrade.strategy.check_exit_timeout = MagicMock(return_value=False)
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    # check it does cancel sell orders over the time limit
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    assert open_trade_usdt.is_open is True
    # Custom user sell-timeout is never called
    assert freqtrade.strategy.check_exit_timeout.call_count == 0
    assert freqtrade.strategy.check_entry_timeout.call_count == 0


@pytest.mark.parametrize("is_short", [False, True])
def test_check_handle_cancelled_exit(
    default_conf_usdt, ticker_usdt, limit_sell_order_old, open_trade_usdt,
    is_short, mocker, caplog
) -> None:
    """ Handle sell order cancelled on exchange"""
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    limit_sell_order_old.update({"status": "canceled", 'filled': 0.0})
    limit_sell_order_old['side'] = 'buy' if is_short else 'sell'
    limit_sell_order_old['id'] = open_trade_usdt.open_order_id

    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order_with_result=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

    open_trade_usdt.open_date = arrow.utcnow().shift(hours=-5).datetime
    open_trade_usdt.close_date = arrow.utcnow().shift(minutes=-601).datetime
    open_trade_usdt.is_short = is_short

    Trade.query.session.add(open_trade_usdt)
    Trade.commit()

    # check it does cancel sell orders over the time limit
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 2
    assert open_trade_usdt.is_open is True
    exit_name = 'Buy' if is_short else 'Sell'
    assert log_has_re(f"{exit_name} order cancelled on exchange for Trade.*", caplog)


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("leverage", [1, 3, 5, 10])
def test_manage_open_orders_partial(
    default_conf_usdt, ticker_usdt, limit_buy_order_old_partial, is_short, leverage,
    open_trade, mocker
) -> None:
    rpc_mock = patch_RPCManager(mocker)
    open_trade.is_short = is_short
    open_trade.leverage = leverage
    open_trade.orders[0].ft_order_side = 'sell' if is_short else 'buy'
    limit_buy_order_old_partial['id'] = open_trade.open_order_id
    limit_buy_order_old_partial['side'] = 'sell' if is_short else 'buy'
    limit_buy_canceled = deepcopy(limit_buy_order_old_partial)
    limit_buy_canceled['status'] = 'canceled'

    cancel_order_mock = MagicMock(return_value=limit_buy_canceled)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_buy_order_old_partial),
        cancel_order_with_result=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    prior_stake = open_trade.stake_amount
    Trade.query.session.add(open_trade)
    Trade.commit()

    # check it does cancel buy orders over the time limit
    # note this is for a partially-complete buy order
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 3
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 1
    assert trades[0].amount == 23.0
    assert trades[0].stake_amount == open_trade.open_rate * trades[0].amount / leverage
    assert trades[0].stake_amount != prior_stake


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_partial_fee(
    default_conf_usdt, ticker_usdt, open_trade, caplog, fee, is_short,
    limit_buy_order_old_partial, trades_for_order,
    limit_buy_order_old_partial_canceled, mocker
) -> None:
    open_trade.is_short = is_short
    open_trade.orders[0].ft_order_side = 'sell' if is_short else 'buy'
    rpc_mock = patch_RPCManager(mocker)
    limit_buy_order_old_partial['id'] = open_trade.open_order_id
    limit_buy_order_old_partial_canceled['id'] = open_trade.open_order_id
    limit_buy_order_old_partial['side'] = 'sell' if is_short else 'buy'
    limit_buy_order_old_partial_canceled['side'] = 'sell' if is_short else 'buy'

    cancel_order_mock = MagicMock(return_value=limit_buy_order_old_partial_canceled)
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=0))
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_buy_order_old_partial),
        cancel_order_with_result=cancel_order_mock,
        get_trades_for_order=MagicMock(return_value=trades_for_order),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

    assert open_trade.amount == limit_buy_order_old_partial['amount']

    open_trade.fee_open = fee()
    open_trade.fee_close = fee()
    Trade.query.session.add(open_trade)
    Trade.commit()
    # cancelling a half-filled order should update the amount to the bought amount
    # and apply fees if necessary.
    freqtrade.manage_open_orders()

    assert log_has_re(r"Applying fee on amount for Trade.*", caplog)

    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 3
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 1
    # Verify that trade has been updated
    assert trades[0].amount == (limit_buy_order_old_partial['amount'] -
                                limit_buy_order_old_partial['remaining']) - 0.023
    assert trades[0].open_order_id is None
    assert trades[0].fee_updated(open_trade.entry_side)
    assert pytest.approx(trades[0].fee_open) == 0.001


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_partial_except(
    default_conf_usdt, ticker_usdt, open_trade, caplog, fee, is_short,
    limit_buy_order_old_partial, trades_for_order,
    limit_buy_order_old_partial_canceled, mocker
) -> None:
    open_trade.is_short = is_short
    open_trade.orders[0].ft_order_side = 'sell' if is_short else 'buy'
    rpc_mock = patch_RPCManager(mocker)
    limit_buy_order_old_partial_canceled['id'] = open_trade.open_order_id
    limit_buy_order_old_partial['id'] = open_trade.open_order_id
    if is_short:
        limit_buy_order_old_partial['side'] = 'sell'
    cancel_order_mock = MagicMock(return_value=limit_buy_order_old_partial_canceled)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_buy_order_old_partial),
        cancel_order_with_result=cancel_order_mock,
        get_trades_for_order=MagicMock(return_value=trades_for_order),
    )
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 MagicMock(side_effect=DependencyException))
    freqtrade = FreqtradeBot(default_conf_usdt)

    assert open_trade.amount == limit_buy_order_old_partial['amount']

    open_trade.fee_open = fee()
    open_trade.fee_close = fee()
    Trade.query.session.add(open_trade)
    Trade.commit()
    # cancelling a half-filled order should update the amount to the bought amount
    # and apply fees if necessary.
    freqtrade.manage_open_orders()

    assert log_has_re(r"Could not update trade amount: .*", caplog)

    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 3
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    assert len(trades) == 1
    # Verify that trade has been updated

    assert trades[0].amount == (limit_buy_order_old_partial['amount'] -
                                limit_buy_order_old_partial['remaining'])
    assert trades[0].open_order_id is None
    assert trades[0].fee_open == fee()


def test_manage_open_orders_exception(default_conf_usdt, ticker_usdt, open_trade_usdt, mocker,
                                      caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()

    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        handle_cancel_enter=MagicMock(),
        handle_cancel_exit=MagicMock(),
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(side_effect=ExchangeError('Oh snap')),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

    Trade.query.session.add(open_trade_usdt)
    Trade.commit()

    caplog.clear()
    freqtrade.manage_open_orders()
    assert log_has_re(r"Cannot query order for Trade\(id=1, pair=ADA/USDT, amount=30.00000000, "
                      r"is_short=False, leverage=1.0, "
                      r"open_rate=2.00000000, open_since="
                      f"{open_trade_usdt.open_date.strftime('%Y-%m-%d %H:%M:%S')}"
                      r"\) due to Traceback \(most recent call last\):\n*",
                      caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_cancel_enter(mocker, caplog, default_conf_usdt, limit_order, is_short, fee) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    l_order = limit_order[entry_side(is_short)]
    cancel_buy_order = deepcopy(limit_order[entry_side(is_short)])
    cancel_buy_order['status'] = 'canceled'
    del cancel_buy_order['filled']

    cancel_order_mock = MagicMock(return_value=cancel_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order_with_result', cancel_order_mock)

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade._notify_enter_cancel = MagicMock()

    trade = mock_trade_usdt_4(fee, is_short)
    Trade.query.session.add(trade)
    Trade.commit()

    l_order['filled'] = 0.0
    l_order['status'] = 'open'
    reason = CANCEL_REASON['TIMEOUT']
    assert freqtrade.handle_cancel_enter(trade, l_order, reason)
    assert cancel_order_mock.call_count == 1

    cancel_order_mock.reset_mock()
    caplog.clear()
    l_order['filled'] = 0.01
    assert not freqtrade.handle_cancel_enter(trade, l_order, reason)
    assert cancel_order_mock.call_count == 0
    assert log_has_re("Order .* for .* not cancelled, as the filled amount.* unexitable.*", caplog)

    caplog.clear()
    cancel_order_mock.reset_mock()
    l_order['filled'] = 2
    assert not freqtrade.handle_cancel_enter(trade, l_order, reason)
    assert cancel_order_mock.call_count == 1

    # Order remained open for some reason (cancel failed)
    cancel_buy_order['status'] = 'open'
    cancel_order_mock = MagicMock(return_value=cancel_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order_with_result', cancel_order_mock)
    assert not freqtrade.handle_cancel_enter(trade, l_order, reason)
    assert log_has_re(r"Order .* for .* not cancelled.", caplog)
    # min_pair_stake empty should not crash
    mocker.patch('freqtrade.exchange.Exchange.get_min_pair_stake_amount', return_value=None)
    assert not freqtrade.handle_cancel_enter(trade, limit_order[entry_side(is_short)], reason)


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("limit_buy_order_canceled_empty", ['binance', 'kraken', 'bittrex'],
                         indirect=['limit_buy_order_canceled_empty'])
def test_handle_cancel_enter_exchanges(mocker, caplog, default_conf_usdt, is_short, fee,
                                       limit_buy_order_canceled_empty) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = mocker.patch(
        'freqtrade.exchange.Exchange.cancel_order_with_result',
        return_value=limit_buy_order_canceled_empty)
    nofiy_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot._notify_enter_cancel')
    freqtrade = FreqtradeBot(default_conf_usdt)

    reason = CANCEL_REASON['TIMEOUT']

    trade = mock_trade_usdt_4(fee, is_short)
    Trade.query.session.add(trade)
    Trade.commit()
    assert freqtrade.handle_cancel_enter(trade, limit_buy_order_canceled_empty, reason)
    assert cancel_order_mock.call_count == 0
    assert log_has_re(
        f'{trade.entry_side.capitalize()} order fully cancelled. '
        r'Removing .* from database\.',
        caplog
    )
    assert nofiy_mock.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize('cancelorder', [
    {},
    {'remaining': None},
    'String Return value',
    123
])
def test_handle_cancel_enter_corder_empty(mocker, default_conf_usdt, limit_order, is_short, fee,
                                          cancelorder) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    l_order = limit_order[entry_side(is_short)]
    cancel_order_mock = MagicMock(return_value=cancelorder)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        cancel_order=cancel_order_mock,
        fetch_order=MagicMock(side_effect=InvalidOrderException)
    )

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade._notify_enter_cancel = MagicMock()
    trade = mock_trade_usdt_4(fee, is_short)
    Trade.query.session.add(trade)
    Trade.commit()
    l_order['filled'] = 0.0
    l_order['status'] = 'open'
    reason = CANCEL_REASON['TIMEOUT']
    assert freqtrade.handle_cancel_enter(trade, l_order, reason)
    assert cancel_order_mock.call_count == 1

    cancel_order_mock.reset_mock()
    l_order['filled'] = 1.0
    order = deepcopy(l_order)
    order['status'] = 'canceled'
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=order)
    assert not freqtrade.handle_cancel_enter(trade, l_order, reason)
    assert cancel_order_mock.call_count == 1


def test_handle_cancel_exit_limit(mocker, default_conf_usdt, fee) -> None:
    send_msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        cancel_order=cancel_order_mock,
    )
    mocker.patch('freqtrade.exchange.Exchange.get_rate', return_value=0.245441)
    mocker.patch('freqtrade.exchange.Exchange.get_min_pair_stake_amount', return_value=0.2)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_order_fee')

    freqtrade = FreqtradeBot(default_conf_usdt)

    trade = Trade(
        pair='LTC/ETH',
        amount=2,
        exchange='binance',
        open_rate=0.245441,
        open_order_id="sell_123456",
        open_date=arrow.utcnow().shift(days=-2).datetime,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        close_rate=0.555,
        close_date=arrow.utcnow().datetime,
        exit_reason="sell_reason_whatever",
        stake_amount=0.245441 * 2,
    )
    trade.orders = [
        Order(
            ft_order_side='buy',
            ft_pair=trade.pair,
            ft_is_open=False,
            order_id='buy_123456',
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
             ),
        Order(
            ft_order_side='sell',
            ft_pair=trade.pair,
            ft_is_open=True,
            order_id='sell_123456',
            status="open",
            symbol=trade.pair,
            order_type="limit",
            side="sell",
            price=trade.open_rate,
            average=trade.open_rate,
            filled=0.0,
            remaining=trade.amount,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
             ),
    ]
    order = {'id': "sell_123456",
             'remaining': 1,
             'amount': 1,
             'status': "open"}
    reason = CANCEL_REASON['TIMEOUT']
    send_msg_mock.reset_mock()
    assert freqtrade.handle_cancel_exit(trade, order, reason)
    assert cancel_order_mock.call_count == 1
    assert send_msg_mock.call_count == 1
    assert trade.close_rate is None
    assert trade.exit_reason is None
    assert trade.open_order_id is None

    send_msg_mock.reset_mock()

    # Partial exit - below exit threshold
    order['amount'] = 2
    order['filled'] = 1.9
    assert not freqtrade.handle_cancel_exit(trade, order, reason)
    # Assert cancel_order was not called (callcount remains unchanged)
    assert cancel_order_mock.call_count == 1
    assert send_msg_mock.call_count == 1
    assert (send_msg_mock.call_args_list[0][0][0]['reason']
            == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN'])

    assert not freqtrade.handle_cancel_exit(trade, order, reason)

    assert (send_msg_mock.call_args_list[0][0][0]['reason']
            == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN'])

    # Message should not be iterated again
    assert trade.exit_order_status == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    assert send_msg_mock.call_count == 1

    send_msg_mock.reset_mock()

    order['filled'] = 1
    assert freqtrade.handle_cancel_exit(trade, order, reason)
    assert send_msg_mock.call_count == 1
    assert (send_msg_mock.call_args_list[0][0][0]['reason']
            == CANCEL_REASON['PARTIALLY_FILLED'])


def test_handle_cancel_exit_cancel_exception(mocker, default_conf_usdt) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_min_pair_stake_amount', return_value=0.0)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order_with_result',
                 side_effect=InvalidOrderException())

    freqtrade = FreqtradeBot(default_conf_usdt)

    # TODO: should not be magicmock
    trade = MagicMock()
    reason = CANCEL_REASON['TIMEOUT']
    order = {'remaining': 1,
             'amount': 1,
             'status': "open"}
    assert not freqtrade.handle_cancel_exit(trade, order, reason)


@pytest.mark.parametrize("is_short, open_rate, amt", [
    (False, 2.0, 30.0),
    (True, 2.02, 29.70297029),
])
def test_execute_trade_exit_up(default_conf_usdt, ticker_usdt, fee, ticker_usdt_sell_up, mocker,
                               ticker_usdt_sell_down, is_short, open_rate, amt) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=False)

    # Create some test data
    freqtrade.enter_positions()
    rpc_mock.reset_mock()

    trade = Trade.query.first()
    assert trade.is_short == is_short
    assert trade
    assert freqtrade.strategy.confirm_trade_exit.call_count == 0

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_down if is_short else ticker_usdt_sell_up
    )
    # Prevented sell ...
    freqtrade.execute_trade_exit(
        trade=trade,
        limit=(ticker_usdt_sell_down()['ask'] if is_short else ticker_usdt_sell_up()['bid']),
        exit_check=ExitCheckTuple(exit_type=ExitType.ROI)
    )
    assert rpc_mock.call_count == 0
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1
    assert id(freqtrade.strategy.confirm_trade_exit.call_args_list[0][1]['trade']) != id(trade)
    assert freqtrade.strategy.confirm_trade_exit.call_args_list[0][1]['trade'].id == trade.id

    # Repatch with true
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)
    freqtrade.execute_trade_exit(
        trade=trade,
        limit=(ticker_usdt_sell_down()['ask'] if is_short else ticker_usdt_sell_up()['bid']),
        exit_check=ExitCheckTuple(exit_type=ExitType.ROI)
    )
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1

    assert rpc_mock.call_count == 1
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'trade_id': 1,
        'type': RPCMessageType.EXIT,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'gain': 'profit',
        'limit': 2.0 if is_short else 2.2,
        'order_rate': 2.0 if is_short else 2.2,
        'amount': pytest.approx(amt),
        'order_type': 'limit',
        'buy_tag': None,
        'direction': 'Short' if trade.is_short else 'Long',
        'leverage': 1.0,
        'enter_tag': None,
        'open_rate': open_rate,
        'current_rate': 2.01 if is_short else 2.3,
        'profit_amount': 0.29554455 if is_short else 5.685,
        'profit_ratio': 0.00493809 if is_short else 0.09451372,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': ExitType.ROI.value,
        'exit_reason': ExitType.ROI.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'sub_trade': False,
        'cumulative_profit': 0.0,
        'stake_amount': pytest.approx(60),
    } == last_msg


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_trade_exit_down(default_conf_usdt, ticker_usdt, fee, ticker_usdt_sell_down,
                                 ticker_usdt_sell_up, mocker, is_short) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_up if is_short else ticker_usdt_sell_down
    )
    freqtrade.execute_trade_exit(
        trade=trade, limit=(ticker_usdt_sell_up if is_short else ticker_usdt_sell_down)()['bid'],
        exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS))

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.EXIT,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'direction': 'Short' if trade.is_short else 'Long',
        'leverage': 1.0,
        'gain': 'loss',
        'limit': 2.2 if is_short else 2.01,
        'order_rate': 2.2 if is_short else 2.01,
        'amount': pytest.approx(29.70297029) if is_short else 30.0,
        'order_type': 'limit',
        'buy_tag': None,
        'enter_tag': None,
        'open_rate': 2.02 if is_short else 2.0,
        'current_rate': 2.2 if is_short else 2.0,
        'profit_amount': -5.65990099 if is_short else -0.00075,
        'profit_ratio': -0.0945681 if is_short else -1.247e-05,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': ExitType.STOP_LOSS.value,
        'exit_reason': ExitType.STOP_LOSS.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'sub_trade': False,
        'cumulative_profit': 0.0,
        'stake_amount': pytest.approx(60),
    } == last_msg


@pytest.mark.parametrize(
    "is_short,amount,open_rate,current_rate,limit,profit_amount,profit_ratio,profit_or_loss", [
        (False, 30, 2.0, 2.3, 2.25, 7.18125, 0.11938903, 'profit'),
        (True, 29.70297029, 2.02, 2.2, 2.25, -7.14876237, -0.11944465, 'loss'),
    ])
def test_execute_trade_exit_custom_exit_price(
        default_conf_usdt, ticker_usdt, fee, ticker_usdt_sell_up, is_short, amount, open_rate,
        current_rate, limit, profit_amount, profit_ratio, profit_or_loss, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )
    config = deepcopy(default_conf_usdt)
    config['custom_price_max_distance_ratio'] = 0.1
    patch_whitelist(mocker, config)
    freqtrade = FreqtradeBot(config)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=False)

    # Create some test data
    freqtrade.enter_positions()
    rpc_mock.reset_mock()

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade
    assert freqtrade.strategy.confirm_trade_exit.call_count == 0

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_up
    )

    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)

    # Set a custom exit price
    freqtrade.strategy.custom_exit_price = lambda **kwargs: 2.25
    freqtrade.execute_trade_exit(
        trade=trade,
        limit=ticker_usdt_sell_up()['ask' if is_short else 'bid'],
        exit_check=ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL, exit_reason='foo')
    )

    # Sell price must be different to default bid price

    assert freqtrade.strategy.confirm_trade_exit.call_count == 1

    assert rpc_mock.call_count == 1
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'trade_id': 1,
        'type': RPCMessageType.EXIT,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'direction': 'Short' if trade.is_short else 'Long',
        'leverage': 1.0,
        'gain': profit_or_loss,
        'limit': limit,
        'order_rate': limit,
        'amount': pytest.approx(amount),
        'order_type': 'limit',
        'buy_tag': None,
        'enter_tag': None,
        'open_rate': open_rate,
        'current_rate': current_rate,
        'profit_amount': pytest.approx(profit_amount),
        'profit_ratio': profit_ratio,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': 'foo',
        'exit_reason': 'foo',
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'sub_trade': False,
        'cumulative_profit': 0.0,
        'stake_amount': pytest.approx(60),
    } == last_msg


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_trade_exit_down_stoploss_on_exchange_dry_run(
        default_conf_usdt, ticker_usdt, fee, is_short, ticker_usdt_sell_down,
        ticker_usdt_sell_up, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade.is_short == is_short
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_up if is_short else ticker_usdt_sell_down
    )

    default_conf_usdt['dry_run'] = True
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    # Setting trade stoploss to 0.01

    trade.stop_loss = 2.0 * 1.01 if is_short else 2.0 * 0.99
    freqtrade.execute_trade_exit(
        trade=trade, limit=trade.stop_loss,
        exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS))

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]

    assert {
        'type': RPCMessageType.EXIT,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'direction': 'Short' if trade.is_short else 'Long',
        'leverage': 1.0,
        'gain': 'loss',
        'limit': 2.02 if is_short else 1.98,
        'order_rate': 2.02 if is_short else 1.98,
        'amount': pytest.approx(29.70297029 if is_short else 30.0),
        'order_type': 'limit',
        'buy_tag': None,
        'enter_tag': None,
        'open_rate': 2.02 if is_short else 2.0,
        'current_rate': 2.2 if is_short else 2.0,
        'profit_amount': -0.3 if is_short else -0.8985,
        'profit_ratio': -0.00501253 if is_short else -0.01493766,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': ExitType.STOP_LOSS.value,
        'exit_reason': ExitType.STOP_LOSS.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'sub_trade': False,
        'cumulative_profit': 0.0,
        'stake_amount': pytest.approx(60),
    } == last_msg


def test_execute_trade_exit_sloe_cancel_exception(
        mocker, default_conf_usdt, ticker_usdt, fee, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.exchange.Exchange.cancel_stoploss_order',
                 side_effect=InvalidOrderException())
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=300))
    create_order_mock = MagicMock(side_effect=[
        {'id': '12345554'},
        {'id': '12345555'},
    ])
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        create_order=create_order_mock,
    )

    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    PairLock.session = MagicMock()

    freqtrade.config['dry_run'] = False
    trade.stoploss_order_id = "abcd"

    freqtrade.execute_trade_exit(trade=trade, limit=1234,
                                 exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS))
    assert create_order_mock.call_count == 2
    assert log_has('Could not cancel stoploss order abcd', caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_trade_exit_with_stoploss_on_exchange(
        default_conf_usdt, ticker_usdt, fee, ticker_usdt_sell_up, is_short, mocker) -> None:

    default_conf_usdt['exchange']['name'] = 'binance'
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
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
        stoploss=stoploss,
        cancel_stoploss_order=cancel_order,
        _is_dry_limit_order_filled=MagicMock(side_effect=[True, False]),
    )

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade
    trades = [trade]

    freqtrade.manage_open_orders()
    freqtrade.exit_positions(trades)

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_up
    )

    freqtrade.execute_trade_exit(
        trade=trade,
        limit=ticker_usdt_sell_up()['ask' if is_short else 'bid'],
        exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS)
    )

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade
    assert cancel_order.call_count == 1
    assert rpc_mock.call_count == 4


@pytest.mark.parametrize("is_short", [False, True])
def test_may_execute_trade_exit_after_stoploss_on_exchange_hit(
        default_conf_usdt, ticker_usdt, fee, mocker, is_short) -> None:
    default_conf_usdt['exchange']['name'] = 'binance'
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
        _is_dry_limit_order_filled=MagicMock(side_effect=[False, True]),
    )

    stoploss = MagicMock(return_value={
        'id': 123,
        'info': {
            'foo': 'bar'
        }
    })

    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss)

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade, enter_long=not is_short, enter_short=is_short)

    # Create some test data
    freqtrade.enter_positions()
    freqtrade.manage_open_orders()
    trade = Trade.query.first()
    trades = [trade]
    assert trade.stoploss_order_id is None

    freqtrade.exit_positions(trades)
    assert trade
    assert trade.stoploss_order_id == '123'
    assert trade.open_order_id is None

    # Assuming stoploss on exchange is hit
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
    assert trade.exit_reason == ExitType.STOPLOSS_ON_EXCHANGE.value
    assert rpc_mock.call_count == 4
    assert rpc_mock.call_args_list[1][0][0]['type'] == RPCMessageType.ENTRY
    assert rpc_mock.call_args_list[1][0][0]['amount'] > 20
    assert rpc_mock.call_args_list[2][0][0]['type'] == RPCMessageType.ENTRY_FILL
    assert rpc_mock.call_args_list[3][0][0]['type'] == RPCMessageType.EXIT_FILL


@pytest.mark.parametrize(
    "is_short,amount,current_rate,limit,profit_amount,profit_ratio,profit_or_loss", [
        (False, 30, 2.3, 2.2, 5.685, 0.09451372, 'profit'),
        (True, 29.70297029, 2.2, 2.3, -8.63762376, -0.1443212, 'loss'),
    ])
def test_execute_trade_exit_market_order(
    default_conf_usdt, ticker_usdt, fee, is_short, current_rate, amount, caplog,
    limit, profit_amount, profit_ratio, profit_or_loss, ticker_usdt_sell_up, mocker
) -> None:
    """
    amount
        long: 60 / 2.0 = 30
        short: 60 / 2.02 = 29.70297029
    open_value
        long: (30 * 2.0) + (30 * 2.0 * 0.0025) = 60.15
        short: (29.702970297029704 * 2.02) - (29.702970297029704 * 2.02 * 0.0025) = 59.85
    close_value
        long: (30 * 2.2) - (30 * 2.2 * 0.0025) = 65.835
        short: (29.702970297029704 * 2.3) + (29.702970297029704 * 2.3 * 0.0025) = 68.48762376237624
    profit
        long: 65.835 - 60.15 = 5.684999999999995
        short: 59.85 - 68.48762376237624 = -8.637623762376244
    profit_ratio
        long: (65.835/60.15) - 1 = 0.0945137157107232
        short: 1 - (68.48762376237624/59.85) = -0.1443211990371971
    """
    open_rate = ticker_usdt.return_value['ask' if is_short else 'bid']
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=True),
        get_funding_fees=MagicMock(side_effect=ExchangeError()),
    )
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_up,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )
    freqtrade.config['order_types']['exit'] = 'market'

    freqtrade.execute_trade_exit(
        trade=trade,
        limit=ticker_usdt_sell_up()['ask' if is_short else 'bid'],
        exit_check=ExitCheckTuple(exit_type=ExitType.ROI)
    )
    assert log_has("Could not update funding fee.", caplog)

    assert not trade.is_open
    assert pytest.approx(trade.close_profit) == profit_ratio

    assert rpc_mock.call_count == 4
    last_msg = rpc_mock.call_args_list[-2][0][0]
    assert {
        'type': RPCMessageType.EXIT,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'direction': 'Short' if trade.is_short else 'Long',
        'leverage': 1.0,
        'gain': profit_or_loss,
        'limit': limit,
        'order_rate': limit,
        'amount': pytest.approx(amount),
        'order_type': 'market',
        'buy_tag': None,
        'enter_tag': None,
        'open_rate': open_rate,
        'current_rate': current_rate,
        'profit_amount': pytest.approx(profit_amount),
        'profit_ratio': profit_ratio,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': ExitType.ROI.value,
        'exit_reason': ExitType.ROI.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'sub_trade': False,
        'cumulative_profit': 0.0,
        'stake_amount': pytest.approx(60),

    } == last_msg


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_trade_exit_insufficient_funds_error(default_conf_usdt, ticker_usdt, fee, is_short,
                                                     ticker_usdt_sell_up, mocker) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_insuf = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_insufficient_funds')
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        create_order=MagicMock(side_effect=[
            {'id': 1234553382},
            InsufficientFundsError(),
        ]),
    )
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_up
    )

    sell_reason = ExitCheckTuple(exit_type=ExitType.ROI)
    assert not freqtrade.execute_trade_exit(
        trade=trade,
        limit=ticker_usdt_sell_up()['ask' if is_short else 'bid'],
        exit_check=sell_reason
    )
    assert mock_insuf.call_count == 1


@pytest.mark.parametrize('profit_only,bid,ask,handle_first,handle_second,exit_type,is_short', [
    # Enable profit
    (True, 2.18, 2.2, False, True, ExitType.EXIT_SIGNAL.value, False),
    (True, 2.18, 2.2, False, True, ExitType.EXIT_SIGNAL.value, True),
    # # Disable profit
    (False, 3.19, 3.2, True,  False, ExitType.EXIT_SIGNAL.value, False),
    (False, 3.19, 3.2, True,  False, ExitType.EXIT_SIGNAL.value, True),
    # # Enable loss
    # # * Shouldn't this be ExitType.STOP_LOSS.value
    (True, 0.21, 0.22, False, False, None, False),
    (True, 2.41, 2.42, False, False, None, True),
    # Disable loss
    (False, 0.10, 0.22, True, False, ExitType.EXIT_SIGNAL.value, False),
    (False, 0.10, 0.22, True, False, ExitType.EXIT_SIGNAL.value, True),
])
def test_exit_profit_only(
        default_conf_usdt, limit_order, limit_order_open, is_short,
        fee, mocker, profit_only, bid, ask, handle_first, handle_second, exit_type) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    eside = entry_side(is_short)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': bid,
            'ask': ask,
            'last': bid
        }),
        create_order=MagicMock(side_effect=[
            limit_order[eside],
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )
    default_conf_usdt.update({
        'use_exit_signal': True,
        'exit_profit_only': profit_only,
        'exit_profit_offset': 0.1,
    })
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.custom_exit = MagicMock(return_value=None)
    if exit_type == ExitType.EXIT_SIGNAL.value:
        freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    else:
        freqtrade.strategy.stop_loss_reached = MagicMock(return_value=ExitCheckTuple(
            exit_type=ExitType.NONE))
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade.is_short == is_short
    oobj = Order.parse_from_ccxt_object(limit_order[eside], limit_order[eside]['symbol'], eside)
    trade.update_order(limit_order[eside])
    trade.update_trade(oobj)
    freqtrade.wallets.update()
    if profit_only:
        assert freqtrade.handle_trade(trade) is False
        # Custom-exit is called
        freqtrade.strategy.custom_exit.call_count == 1

    patch_get_signal(freqtrade, enter_long=False, exit_short=is_short, exit_long=not is_short)
    assert freqtrade.handle_trade(trade) is handle_first

    if handle_second:
        freqtrade.strategy.exit_profit_offset = 0.0
        assert freqtrade.handle_trade(trade) is True


def test_sell_not_enough_balance(default_conf_usdt, limit_order, limit_order_open,
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
        create_order=MagicMock(side_effect=[
            limit_order_open['buy'],
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )

    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    amnt = trade.amount

    oobj = Order.parse_from_ccxt_object(limit_order['buy'], limit_order['buy']['symbol'], 'buy')
    trade.update_trade(oobj)
    patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=trade.amount * 0.985))

    assert freqtrade.handle_trade(trade) is True
    assert log_has_re(r'.*Falling back to wallet-amount.', caplog)
    assert trade.amount != amnt


@pytest.mark.parametrize('amount_wallet,has_err', [
    (95.29, False),
    (91.29, True)
])
def test__safe_exit_amount(default_conf_usdt, fee, caplog, mocker, amount_wallet, has_err):
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    amount = 95.33
    amount_wallet = amount_wallet
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
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    if has_err:
        with pytest.raises(DependencyException, match=r"Not enough amount to exit trade."):
            assert freqtrade._safe_exit_amount(trade, trade.pair, trade.amount)
    else:
        wallet_update.reset_mock()
        assert trade.amount != amount_wallet
        assert freqtrade._safe_exit_amount(trade, trade.pair, trade.amount) == amount_wallet
        assert log_has_re(r'.*Falling back to wallet-amount.', caplog)
        assert trade.amount == amount_wallet
        assert wallet_update.call_count == 1
        caplog.clear()
        wallet_update.reset_mock()
        assert freqtrade._safe_exit_amount(trade, trade.pair, amount_wallet) == amount_wallet
        assert not log_has_re(r'.*Falling back to wallet-amount.', caplog)
        assert wallet_update.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_locked_pairs(default_conf_usdt, ticker_usdt, fee,
                      ticker_usdt_sell_down, mocker, caplog, is_short) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_down
    )

    freqtrade.execute_trade_exit(
        trade=trade,
        limit=ticker_usdt_sell_down()['ask' if is_short else 'bid'],
        exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS)
    )
    trade.close(ticker_usdt_sell_down()['bid'])
    assert freqtrade.strategy.is_pair_locked(trade.pair, side='*')
    # Boths sides are locked
    assert freqtrade.strategy.is_pair_locked(trade.pair, side='long')
    assert freqtrade.strategy.is_pair_locked(trade.pair, side='short')

    # reinit - should buy other pair.
    caplog.clear()
    freqtrade.enter_positions()

    assert log_has_re(fr"Pair {trade.pair} \* is locked.*", caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_ignore_roi_if_entry_signal(default_conf_usdt, limit_order, limit_order_open, is_short,
                                    fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    eside = entry_side(is_short)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 2.19,
            'ask': 2.2,
            'last': 2.19
        }),
        create_order=MagicMock(side_effect=[
            limit_order_open[eside],
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )
    default_conf_usdt['ignore_roi_if_entry_signal'] = True

    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short
    oobj = Order.parse_from_ccxt_object(
        limit_order[eside], limit_order[eside]['symbol'], eside)
    trade.update_trade(oobj)
    freqtrade.wallets.update()
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, enter_short=True, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=True, exit_long=True)

    assert freqtrade.handle_trade(trade) is False

    # Test if entry-signal is absent (should sell due to roi = true)
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, exit_short=False, exit_tag='something')
    else:
        patch_get_signal(freqtrade, enter_long=False, exit_long=False, exit_tag='something')
    assert freqtrade.handle_trade(trade) is True
    assert trade.exit_reason == ExitType.ROI.value


@pytest.mark.parametrize("is_short,val1,val2", [
    (False, 1.5, 1.1),
    (True, 0.5, 0.9)
])
def test_trailing_stop_loss(default_conf_usdt, limit_order_open,
                            is_short, val1, val2, fee, caplog, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 2.0,
            'ask': 2.0,
            'last': 2.0
        }),
        create_order=MagicMock(side_effect=[
            limit_order_open[entry_side(is_short)],
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )
    default_conf_usdt['trailing_stop'] = True
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()
    trade = Trade.query.first()
    assert trade.is_short == is_short
    assert freqtrade.handle_trade(trade) is False

    # Raise praise into profits
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': 2.0 * val1,
                     'ask': 2.0 * val1,
                     'last': 2.0 * val1
                 }))

    # Stoploss should be adjusted
    assert freqtrade.handle_trade(trade) is False
    caplog.clear()
    # Price fell
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': 2.0 * val2,
                     'ask': 2.0 * val2,
                     'last': 2.0 * val2
                 }))

    caplog.set_level(logging.DEBUG)
    # Sell as trailing-stop is reached
    assert freqtrade.handle_trade(trade) is True
    stop_multi = 1.1 if is_short else 0.9
    assert log_has(f"ETH/USDT - HIT STOP: current price at {(2.0 * val2):6f}, "
                   f"stoploss is {(2.0 * val1 * stop_multi):6f}, "
                   f"initial stoploss was at {(2.0 * stop_multi):6f}, trade opened at 2.000000",
                   caplog)
    assert trade.exit_reason == ExitType.TRAILING_STOP_LOSS.value


@pytest.mark.parametrize('offset,trail_if_reached,second_sl,is_short', [
    (0, False, 2.0394, False),
    (0.011, False, 2.0394, False),
    (0.055, True, 1.8, False),
    (0, False, 2.1614, True),
    (0.011, False, 2.1614, True),
    (0.055, True, 2.42, True),
])
def test_trailing_stop_loss_positive(
    default_conf_usdt, limit_order, limit_order_open,
    offset, fee, caplog, mocker, trail_if_reached, second_sl, is_short
) -> None:
    enter_price = limit_order[entry_side(is_short)]['price']
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    eside = entry_side(is_short)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': enter_price - (-0.01 if is_short else 0.01),
            'ask': enter_price - (-0.01 if is_short else 0.01),
            'last': enter_price - (-0.01 if is_short else 0.01),
        }),
        create_order=MagicMock(side_effect=[
            limit_order[eside],
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )
    default_conf_usdt['trailing_stop'] = True
    default_conf_usdt['trailing_stop_positive'] = 0.01
    if offset:
        default_conf_usdt['trailing_stop_positive_offset'] = offset
        default_conf_usdt['trailing_only_offset_is_reached'] = trail_if_reached
    patch_whitelist(mocker, default_conf_usdt)

    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade.is_short == is_short
    oobj = Order.parse_from_ccxt_object(limit_order[eside], limit_order[eside]['symbol'], eside)
    trade.update_order(limit_order[eside])
    trade.update_trade(oobj)
    caplog.set_level(logging.DEBUG)
    # stop-loss not reached
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker_usdt above buy price
    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': enter_price + (-0.06 if is_short else 0.06),
            'ask': enter_price + (-0.06 if is_short else 0.06),
            'last': enter_price + (-0.06 if is_short else 0.06),
        })
    )
    caplog.clear()
    # stop-loss not reached, adjusted stoploss
    assert freqtrade.handle_trade(trade) is False
    caplog_text = (f"ETH/USDT - Using positive stoploss: 0.01 offset: {offset} profit: "
                   f"{'2.49' if not is_short else '2.24'}%")
    if trail_if_reached:
        assert not log_has(caplog_text, caplog)
        assert not log_has("ETH/USDT - Adjusting stoploss...", caplog)
    else:
        assert log_has(caplog_text, caplog)
        assert log_has("ETH/USDT - Adjusting stoploss...", caplog)
    assert pytest.approx(trade.stop_loss) == second_sl
    caplog.clear()

    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': enter_price + (-0.135 if is_short else 0.125),
            'ask': enter_price + (-0.135 if is_short else 0.125),
            'last': enter_price + (-0.135 if is_short else 0.125),
        })
    )
    assert freqtrade.handle_trade(trade) is False
    assert log_has(
        f"ETH/USDT - Using positive stoploss: 0.01 offset: {offset} profit: "
        f"{'5.72' if not is_short else '5.67'}%",
        caplog
    )
    assert log_has("ETH/USDT - Adjusting stoploss...", caplog)

    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': enter_price + (-0.02 if is_short else 0.02),
            'ask': enter_price + (-0.02 if is_short else 0.02),
            'last': enter_price + (-0.02 if is_short else 0.02),
        })
    )
    # Lower price again (but still positive)
    assert freqtrade.handle_trade(trade) is True
    assert log_has(
        f"ETH/USDT - HIT STOP: current price at {enter_price + (-0.02 if is_short else 0.02):.6f}, "
        f"stoploss is {trade.stop_loss:.6f}, "
        f"initial stoploss was at {'2.42' if is_short else '1.80'}0000, "
        f"trade opened at {2.2 if is_short else 2.0}00000",
        caplog)
    assert trade.exit_reason == ExitType.TRAILING_STOP_LOSS.value


@pytest.mark.parametrize("is_short", [False, True])
def test_disable_ignore_roi_if_entry_signal(default_conf_usdt, limit_order, limit_order_open,
                                            is_short, fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    eside = entry_side(is_short)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 2.0,
            'ask': 2.0,
            'last': 2.0
        }),
        create_order=MagicMock(side_effect=[
            limit_order_open[eside],
            {'id': 1234553382},
            {'id': 1234553383}
        ]),
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )
    default_conf_usdt['exit_pricing'] = {
        'ignore_roi_if_entry_signal': False
    }
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_short = is_short

    oobj = Order.parse_from_ccxt_object(
        limit_order[eside], limit_order[eside]['symbol'], eside)
    trade.update_trade(oobj)
    # Sell due to min_roi_reached
    patch_get_signal(freqtrade, enter_long=not is_short, enter_short=is_short, exit_short=is_short)
    assert freqtrade.handle_trade(trade) is True

    # Test if entry-signal is absent
    patch_get_signal(freqtrade)
    assert freqtrade.handle_trade(trade) is True
    assert trade.exit_reason == ExitType.ROI.value


def test_get_real_amount_quote(default_conf_usdt, trades_for_order, buy_order_fee, fee, caplog,
                               mocker):
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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    caplog.clear()
    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee, order_obj) == (amount * 0.001)
    assert log_has(
        'Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, is_short=False,'
        ' leverage=1.0, open_rate=0.24544100, open_since=closed), fee=0.008.',
        caplog
    )


def test_get_real_amount_quote_dust(default_conf_usdt, trades_for_order, buy_order_fee, fee,
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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    walletmock.reset_mock()
    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    # Amount is kept as is
    assert freqtrade.get_real_amount(trade, buy_order_fee, order_obj) is None
    assert walletmock.call_count == 1
    assert log_has_re(r'Fee amount for Trade.* was in base currency '
                      '- Eating Fee 0.008 into dust', caplog)


def test_get_real_amount_no_trade(default_conf_usdt, buy_order_fee, caplog, mocker, fee):
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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee, order_obj) is None
    assert log_has(
        'Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
        'is_short=False, leverage=1.0, open_rate=0.24544100, open_since=closed) failed: '
        'myTrade-Dict empty found',
        caplog
    )


@pytest.mark.parametrize(
    'fee_par,fee_reduction_amount,use_ticker_usdt_rate,expected_log', [
        # basic, amount does not change
        ({'cost': 0.008, 'currency': 'ETH'}, 0, False, None),
        # no currency in fee
        ({'cost': 0.004, 'currency': None}, 0, True, None),
        # BNB no rate
        ({'cost': 0.00094518, 'currency': 'BNB'}, 0, True, (
            'Fee for Trade Trade(id=None, pair=LTC/ETH, amount=8.00000000, is_short=False, '
            'leverage=1.0, open_rate=0.24544100, open_since=closed) [buy]: 0.00094518 BNB -'
            ' rate: None'
        )),
        # from order
        ({'cost': 0.004, 'currency': 'LTC'}, 0.004, False, (
            'Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
            'is_short=False, leverage=1.0, open_rate=0.24544100, open_since=closed), fee=0.004.'
        )),
        # invalid, no currency in from fee dict
        ({'cost': 0.008, 'currency': None}, 0, True, None),
    ])
def test_get_real_amount(
    default_conf_usdt, trades_for_order, buy_order_fee, fee, mocker, caplog,
    fee_par, fee_reduction_amount, use_ticker_usdt_rate, expected_log
):

    buy_order = deepcopy(buy_order_fee)
    buy_order['fee'] = fee_par
    trades_for_order[0]['fee'] = fee_par

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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    if not use_ticker_usdt_rate:
        mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', side_effect=ExchangeError)

    caplog.clear()
    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    res = freqtrade.get_real_amount(trade, buy_order, order_obj)
    if fee_reduction_amount == 0:
        assert res is None
    else:
        assert res == fee_reduction_amount

    if expected_log:
        assert log_has(expected_log, caplog)


@pytest.mark.parametrize(
    'fee_cost, fee_currency, fee_reduction_amount, expected_fee, expected_log_amount', [
        # basic, amount is reduced by fee
        (None, None, 0.001, 0.001, 7.992),
        # different fee currency on both trades, fee is average of both trade's fee
        (0.02, 'BNB', 0.0005, 0.001518575, 7.996),
    ])
def test_get_real_amount_multi(
    default_conf_usdt, trades_for_order2, buy_order_fee, caplog, fee, mocker, markets,
    fee_cost, fee_currency, fee_reduction_amount, expected_fee, expected_log_amount,
):

    trades_for_order = deepcopy(trades_for_order2)
    if fee_cost:
        trades_for_order[0]['fee']['cost'] = fee_cost
    if fee_currency:
        trades_for_order[0]['fee']['currency'] = fee_currency

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    amount = float(sum(x['amount'] for x in trades_for_order))
    default_conf_usdt['stake_currency'] = "ETH"

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
    markets['BNB/ETH'] = markets['ETH/USDT']
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 return_value={'ask': 0.19, 'last': 0.2})

    # Amount is reduced by "fee"
    expected_amount = amount * fee_reduction_amount
    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    assert freqtrade.get_real_amount(trade, buy_order_fee, order_obj) == expected_amount
    assert log_has(
        (
            'Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
            'is_short=False, leverage=1.0, open_rate=0.24544100, open_since=closed), '
            f'fee={expected_amount}.'
        ),
        caplog
    )

    assert trade.fee_open == expected_fee
    assert trade.fee_close == expected_fee
    assert trade.fee_open_cost is not None
    assert trade.fee_open_currency is not None
    assert trade.fee_close_cost is None
    assert trade.fee_close_currency is None


def test_get_real_amount_invalid_order(default_conf_usdt, trades_for_order, buy_order_fee, fee,
                                       mocker):
    limit_buy_order_usdt = deepcopy(buy_order_fee)
    limit_buy_order_usdt['fee'] = {'cost': 0.004}

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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    # Amount does not change
    assert freqtrade.get_real_amount(trade, limit_buy_order_usdt, order_obj) is None


def test_get_real_amount_fees_order(default_conf_usdt, market_buy_order_usdt_doublefee,
                                    fee, mocker):

    tfo_mock = mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.exchange.Exchange.get_valid_pair_combination', return_value='BNB/USDT')
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', return_value={'last': 200})
    trade = Trade(
        pair='LTC/USDT',
        amount=30.0,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    # Amount does not change
    assert trade.fee_open == 0.0025
    order_obj = Order.parse_from_ccxt_object(market_buy_order_usdt_doublefee, 'LTC/ETH', 'buy')
    assert freqtrade.get_real_amount(trade, market_buy_order_usdt_doublefee, order_obj) is None
    assert tfo_mock.call_count == 0
    # Fetch fees from trades dict if available to get "proper" values
    assert round(trade.fee_open, 4) == 0.001


def test_get_real_amount_wrong_amount(default_conf_usdt, trades_for_order, buy_order_fee, fee,
                                      mocker):
    limit_buy_order_usdt = deepcopy(buy_order_fee)
    limit_buy_order_usdt['amount'] = limit_buy_order_usdt['amount'] - 0.001

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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    # Amount does not change
    with pytest.raises(DependencyException, match=r"Half bought\? Amounts don't match"):
        freqtrade.get_real_amount(trade, limit_buy_order_usdt, order_obj)


def test_get_real_amount_wrong_amount_rounding(default_conf_usdt, trades_for_order, buy_order_fee,
                                               fee, mocker):
    # Floats should not be compared directly.
    limit_buy_order_usdt = deepcopy(buy_order_fee)
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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    # Amount changes by fee amount.
    assert pytest.approx(freqtrade.get_real_amount(
        trade, limit_buy_order_usdt, order_obj)) == (amount * 0.001)


def test_get_real_amount_open_trade_usdt(default_conf_usdt, fee, mocker):
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
        'price': 0.245441,
    }
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    order_obj = Order.parse_from_ccxt_object(order, 'LTC/ETH', 'buy')
    assert freqtrade.get_real_amount(trade, order, order_obj) is None


def test_get_real_amount_in_point(default_conf_usdt, buy_order_fee, fee, mocker, caplog):
    limit_buy_order_usdt = deepcopy(buy_order_fee)

    # Fees amount in "POINT"
    trades = [{
        "info": {
        },
        "id": "some_trade_id",
        "timestamp": 1660092505903,
        "datetime": "2022-08-10T00:48:25.903Z",
        "symbol": "CEL/USDT",
        "order": "some_order_id",
        "type": None,
        "side": "sell",
        "takerOrMaker": "taker",
        "price": 1.83255,
        "amount": 83.126,
        "cost": 152.3325513,
        "fee": {
            "currency": "POINT",
            "cost": 0.3046651026
        },
        "fees": [
            {
                "cost": "0",
                "currency": "USDT"
            },
            {
                "cost": "0",
                "currency": "GT"
            },
            {
                "cost": "0.3046651026",
                "currency": "POINT"
            }
        ]
    }]

    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades)
    amount = float(sum(x['amount'] for x in trades))
    trade = Trade(
        pair='CEL/USDT',
        amount=amount,
        exchange='binance',
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
        open_order_id="123456"
    )
    limit_buy_order_usdt['amount'] = amount
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    res = freqtrade.get_real_amount(trade, limit_buy_order_usdt, order_obj)
    assert res is None
    assert trade.fee_open_currency is None
    assert trade.fee_open_cost is None
    message = "Not updating buy-fee - rate: None, POINT."
    assert log_has(message, caplog)
    caplog.clear()
    freqtrade.config['exchange']['unknown_fee_rate'] = 1
    res = freqtrade.get_real_amount(trade, limit_buy_order_usdt, order_obj)
    assert res is None
    assert trade.fee_open_currency == 'POINT'
    assert pytest.approx(trade.fee_open_cost) == 0.3046651026
    assert trade.fee_open == 0.002
    assert trade.fee_open != fee.return_value
    assert not log_has(message, caplog)


@pytest.mark.parametrize('amount,fee_abs,wallet,amount_exp', [
    (8.0, 0.0, 10, None),
    (8.0, 0.0, 0, None),
    (8.0, 0.1, 0, 0.1),
    (8.0, 0.1, 10, None),
    (8.0, 0.1, 8.0, None),
    (8.0, 0.1, 7.9, 0.1),
])
def test_apply_fee_conditional(default_conf_usdt, fee, mocker,
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
    order = Order(
        ft_order_side='buy',
        order_id='100',
        ft_pair=trade.pair,
        ft_is_open=True,
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    walletmock.reset_mock()
    # Amount is kept as is
    assert freqtrade.apply_fee_conditional(trade, 'LTC', amount, fee_abs, order) == amount_exp
    assert walletmock.call_count == 1


@pytest.mark.parametrize("delta, is_high_delta", [
    (0.1, False),
    (100, True),
])
@pytest.mark.parametrize('is_short', [False, True])
def test_order_book_depth_of_market(
    default_conf_usdt, ticker_usdt, limit_order_open,
    fee, mocker, order_book_l2, delta, is_high_delta, is_short
):
    ticker_side = 'ask' if is_short else 'bid'

    default_conf_usdt['entry_pricing']['check_depth_of_market']['enabled'] = True
    default_conf_usdt['entry_pricing']['check_depth_of_market']['bids_to_ask_delta'] = delta
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book', order_book_l2)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    if is_high_delta:
        assert trade is None
    else:
        trade.is_short = is_short
        assert trade is not None
        assert pytest.approx(trade.stake_amount) == 60.0
        assert trade.is_open
        assert trade.open_date is not None
        assert trade.exchange == 'binance'

        assert len(Trade.query.all()) == 1

        # Simulate fulfilled LIMIT_BUY order for trade
        oobj = Order.parse_from_ccxt_object(
            limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
        trade.update_trade(oobj)

        assert trade.open_rate == ticker_usdt.return_value[ticker_side]
        assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


@pytest.mark.parametrize('exception_thrown,ask,last,order_book_top,order_book', [
    (False, 0.045, 0.046, 2, None),
    (True,  0.042, 0.046, 1, {'bids': [[]], 'asks': [[]]})
])
def test_order_book_entry_pricing1(mocker, default_conf_usdt, order_book_l2, exception_thrown,
                                   ask, last, order_book_top, order_book, caplog) -> None:
    """
    test if function get_rate will return the order book price instead of the ask rate
    """
    patch_exchange(mocker)
    ticker_usdt_mock = MagicMock(return_value={'ask': ask, 'last': last})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_l2_order_book=MagicMock(return_value=order_book) if order_book else order_book_l2,
        fetch_ticker=ticker_usdt_mock,
    )
    default_conf_usdt['exchange']['name'] = 'binance'
    default_conf_usdt['entry_pricing']['use_order_book'] = True
    default_conf_usdt['entry_pricing']['order_book_top'] = order_book_top
    default_conf_usdt['entry_pricing']['price_last_balance'] = 0
    default_conf_usdt['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf_usdt)
    if exception_thrown:
        with pytest.raises(PricingError):
            freqtrade.exchange.get_rate('ETH/USDT', side="entry", is_short=False, refresh=True)
        assert log_has_re(
            r'ETH/USDT - Entry Price at location 1 from orderbook could not be determined.', caplog)
    else:
        assert freqtrade.exchange.get_rate(
            'ETH/USDT', side="entry", is_short=False, refresh=True) == 0.043935
        assert ticker_usdt_mock.call_count == 0


def test_check_depth_of_market(default_conf_usdt, mocker, order_book_l2) -> None:
    """
    test check depth of market
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_l2_order_book=order_book_l2
    )
    default_conf_usdt['telegram']['enabled'] = False
    default_conf_usdt['exchange']['name'] = 'binance'
    default_conf_usdt['entry_pricing']['check_depth_of_market']['enabled'] = True
    # delta is 100 which is impossible to reach. hence function will return false
    default_conf_usdt['entry_pricing']['check_depth_of_market']['bids_to_ask_delta'] = 100
    freqtrade = FreqtradeBot(default_conf_usdt)

    conf = default_conf_usdt['entry_pricing']['check_depth_of_market']
    assert freqtrade._check_depth_of_market('ETH/BTC', conf, side=SignalDirection.LONG) is False


@pytest.mark.parametrize('is_short', [False, True])
def test_order_book_exit_pricing(
        default_conf_usdt, limit_buy_order_usdt_open, limit_buy_order_usdt, fee, is_short,
        limit_sell_order_usdt_open, mocker, order_book_l2, caplog) -> None:
    """
    test order book ask strategy
    """
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book', order_book_l2)
    default_conf_usdt['exchange']['name'] = 'binance'
    default_conf_usdt['exit_pricing']['use_order_book'] = True
    default_conf_usdt['exit_pricing']['order_book_top'] = 1
    default_conf_usdt['telegram']['enabled'] = False
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(side_effect=[
            limit_buy_order_usdt_open,
            limit_sell_order_usdt_open,
        ]),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    time.sleep(0.01)  # Race condition fix
    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, limit_buy_order_usdt['symbol'], 'buy')
    trade.update_trade(oobj)
    freqtrade.wallets.update()
    assert trade.is_open is True

    if is_short:
        patch_get_signal(freqtrade, enter_long=False,  exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    assert freqtrade.handle_trade(trade) is True
    assert trade.close_rate_requested == order_book_l2.return_value['asks'][0][0]

    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book',
                 return_value={'bids': [[]], 'asks': [[]]})
    with pytest.raises(PricingError):
        freqtrade.handle_trade(trade)
    assert log_has_re(
        r"ETH/USDT - Exit Price at location 1 from orderbook could not be determined\..*",
        caplog)


def test_startup_state(default_conf_usdt, mocker):
    default_conf_usdt['pairlist'] = {'method': 'VolumePairList',
                                     'config': {'number_assets': 20}
                                     }
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    worker = get_patched_worker(mocker, default_conf_usdt)
    assert worker.freqtrade.state is State.RUNNING


def test_startup_trade_reinit(default_conf_usdt, edge_conf, mocker):

    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    reinit_mock = MagicMock()
    mocker.patch('freqtrade.persistence.Trade.stoploss_reinitialization', reinit_mock)

    ftbot = get_patched_freqtradebot(mocker, default_conf_usdt)
    ftbot.startup()
    assert reinit_mock.call_count == 1

    reinit_mock.reset_mock()

    ftbot = get_patched_freqtradebot(mocker, edge_conf)
    ftbot.startup()
    assert reinit_mock.call_count == 0


@pytest.mark.usefixtures("init_persistence")
def test_sync_wallet_dry_run(mocker, default_conf_usdt, ticker_usdt, fee, limit_buy_order_usdt_open,
                             caplog):
    default_conf_usdt['dry_run'] = True
    # Initialize to 2 times stake amount
    default_conf_usdt['dry_run_wallet'] = 120.0
    default_conf_usdt['max_open_trades'] = 2
    default_conf_usdt['tradable_balance_ratio'] = 1.0
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )

    bot = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(bot)
    assert bot.wallets.get_free('USDT') == 120.0

    n = bot.enter_positions()
    assert n == 2
    trades = Trade.query.all()
    assert len(trades) == 2

    bot.config['max_open_trades'] = 3
    n = bot.enter_positions()
    assert n == 0
    assert log_has_re(r"Unable to create trade for XRP/USDT: "
                      r"Available balance \(0.0 USDT\) is lower than stake amount \(60.0 USDT\)",
                      caplog)


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short,buy_calls,sell_calls", [
    (False, 1, 2),
    (True, 1, 2),
])
def test_cancel_all_open_orders(mocker, default_conf_usdt, fee, limit_order, limit_order_open,
                                is_short, buy_calls, sell_calls):
    default_conf_usdt['cancel_open_orders_on_exit'] = True
    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_order',
        side_effect=[
            ExchangeError(),
            limit_order[exit_side(is_short)],
            limit_order_open[entry_side(is_short)],
            limit_order_open[exit_side(is_short)],
        ]
    )
    buy_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_enter')
    sell_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_exit')

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades(fee, is_short=is_short)
    trades = Trade.query.all()
    assert len(trades) == MOCK_TRADE_COUNT
    freqtrade.cancel_all_open_orders()
    assert buy_mock.call_count == buy_calls
    assert sell_mock.call_count == sell_calls


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_check_for_open_trades(mocker, default_conf_usdt, fee, is_short):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    freqtrade.check_for_open_trades()
    assert freqtrade.rpc.send_msg.call_count == 0

    create_mock_trades(fee, is_short)
    trade = Trade.query.first()
    trade.is_short = is_short
    trade.is_open = True

    freqtrade.check_for_open_trades()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert 'Handle these trades manually' in freqtrade.rpc.send_msg.call_args[0][0]['status']


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.usefixtures("init_persistence")
def test_startup_update_open_orders(mocker, default_conf_usdt, fee, caplog, is_short):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades(fee, is_short=is_short)

    freqtrade.startup_update_open_orders()
    assert not log_has_re(r"Error updating Order .*", caplog)
    caplog.clear()

    freqtrade.config['dry_run'] = False
    freqtrade.startup_update_open_orders()

    assert len(Order.get_open_orders()) == 3
    matching_buy_order = mock_order_4(is_short=is_short)
    matching_buy_order.update({
        'status': 'closed',
    })
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=matching_buy_order)
    freqtrade.startup_update_open_orders()
    # Only stoploss and sell orders are kept open
    assert len(Order.get_open_orders()) == 2

    caplog.clear()
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', side_effect=ExchangeError)
    freqtrade.startup_update_open_orders()
    assert log_has_re(r"Error updating Order .*", caplog)

    mocker.patch('freqtrade.exchange.Exchange.fetch_order', side_effect=InvalidOrderException)
    hto_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_timedout_order')
    # Orders which are no longer found after X days should be assumed as canceled.
    freqtrade.startup_update_open_orders()
    assert log_has_re(r"Order is older than \d days.*", caplog)
    assert hto_mock.call_count == 2
    assert hto_mock.call_args_list[0][0][0]['status'] == 'canceled'
    assert hto_mock.call_args_list[1][0][0]['status'] == 'canceled'


@pytest.mark.usefixtures("init_persistence")
def test_startup_backpopulate_precision(mocker, default_conf_usdt, fee, caplog):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)

    trades = Trade.get_trades().all()
    trades[-1].exchange = 'some_other_exchange'
    for trade in trades:
        assert trade.price_precision is None
        assert trade.amount_precision is None
        assert trade.precision_mode is None

    freqtrade.startup_backpopulate_precision()
    trades = Trade.get_trades().all()
    for trade in trades:
        if trade.exchange == 'some_other_exchange':
            assert trade.price_precision is None
            assert trade.amount_precision is None
            assert trade.precision_mode is None
        else:
            assert trade.price_precision is not None
            assert trade.amount_precision is not None
            assert trade.precision_mode is not None


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_update_trades_without_assigned_fees(mocker, default_conf_usdt, fee, is_short):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    def patch_with_fee(order):
        order.update({'fee': {'cost': 0.1, 'rate': 0.01,
                      'currency': order['symbol'].split('/')[0]}})
        return order

    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 side_effect=[
                     patch_with_fee(mock_order_2_sell(is_short=is_short)),
                     patch_with_fee(mock_order_3_sell(is_short=is_short)),
                     patch_with_fee(mock_order_1(is_short=is_short)),
                     patch_with_fee(mock_order_2(is_short=is_short)),
                     patch_with_fee(mock_order_3(is_short=is_short)),
                     patch_with_fee(mock_order_4(is_short=is_short)),
                 ]
                 )

    create_mock_trades(fee, is_short=is_short)
    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT
    for trade in trades:
        trade.is_short = is_short
        assert trade.fee_open_cost is None
        assert trade.fee_open_currency is None
        assert trade.fee_close_cost is None
        assert trade.fee_close_currency is None

    freqtrade.update_trades_without_assigned_fees()

    # Does nothing for dry-run
    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT
    for trade in trades:
        assert trade.fee_open_cost is None
        assert trade.fee_open_currency is None
        assert trade.fee_close_cost is None
        assert trade.fee_close_currency is None

    freqtrade.config['dry_run'] = False

    freqtrade.update_trades_without_assigned_fees()

    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT

    for trade in trades:
        if trade.is_open:
            # Exclude Trade 4 - as the order is still open.
            if trade.select_order(entry_side(is_short), False):
                assert trade.fee_open_cost is not None
                assert trade.fee_open_currency is not None
            else:
                assert trade.fee_open_cost is None
                assert trade.fee_open_currency is None

        else:
            assert trade.fee_close_cost is not None
            assert trade.fee_close_currency is not None


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_reupdate_enter_order_fees(mocker, default_conf_usdt, fee, caplog, is_short):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_uts = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.update_trade_state')

    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 return_value={'status': 'open'})
    create_mock_trades(fee, is_short)
    trades = Trade.get_trades().all()

    freqtrade.handle_insufficient_funds(trades[3])
    # assert log_has_re(r"Trying to reupdate buy fees for .*", caplog)
    assert mock_uts.call_count == 1
    assert mock_uts.call_args_list[0][0][0] == trades[3]
    assert mock_uts.call_args_list[0][0][1] == mock_order_4(is_short)['id']
    assert log_has_re(r"Trying to refind lost order for .*", caplog)
    mock_uts.reset_mock()
    caplog.clear()

    # Test with trade without orders
    trade = Trade(
        pair='XRP/ETH',
        stake_amount=60.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=arrow.utcnow().datetime,
        is_open=True,
        amount=30,
        open_rate=2.0,
        exchange='binance',
        is_short=is_short
    )
    Trade.query.session.add(trade)

    freqtrade.handle_insufficient_funds(trade)
    # assert log_has_re(r"Trying to reupdate buy fees for .*", caplog)
    assert mock_uts.call_count == 0


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_handle_insufficient_funds(mocker, default_conf_usdt, fee, is_short, caplog):
    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_uts = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.update_trade_state')

    mock_fo = mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                           return_value={'status': 'open'})

    def reset_open_orders(trade):
        trade.open_order_id = None
        trade.stoploss_order_id = None
        trade.is_short = is_short

    create_mock_trades(fee, is_short=is_short)
    trades = Trade.get_trades().all()

    caplog.clear()

    # No open order
    trade = trades[0]
    reset_open_orders(trade)
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is None

    freqtrade.handle_insufficient_funds(trade)
    order = mock_order_1(is_short=is_short)
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

    freqtrade.handle_insufficient_funds(trade)
    order = mock_order_4(is_short=is_short)
    assert log_has_re(r"Trying to refind Order\(.*", caplog)
    assert mock_fo.call_count == 1
    assert mock_uts.call_count == 1
    # Found open buy order
    assert trade.open_order_id is not None
    assert trade.stoploss_order_id is None

    caplog.clear()
    mock_fo.reset_mock()

    # Open stoploss order
    trade = trades[4]
    reset_open_orders(trade)
    assert trade.open_order_id is None
    assert trade.stoploss_order_id is None

    freqtrade.handle_insufficient_funds(trade)
    order = mock_order_5_stoploss(is_short=is_short)
    assert log_has_re(r"Trying to refind Order\(.*", caplog)
    assert mock_fo.call_count == 1
    assert mock_uts.call_count == 2
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

    freqtrade.handle_insufficient_funds(trade)
    order = mock_order_6_sell(is_short=is_short)
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
    order = mock_order_5_stoploss(is_short=is_short)

    freqtrade.handle_insufficient_funds(trades[4])
    assert log_has(f"Error updating {order['id']}.", caplog)


def test_get_valid_price(mocker, default_conf_usdt) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['custom_price_max_distance_ratio'] = 0.02

    custom_price_string = "10"
    custom_price_badstring = "10abc"
    custom_price_float = 10.0
    custom_price_int = 10

    custom_price_over_max_alwd = 11.0
    custom_price_under_min_alwd = 9.0
    proposed_price = 10.1

    valid_price_from_string = freqtrade.get_valid_price(custom_price_string, proposed_price)
    valid_price_from_badstring = freqtrade.get_valid_price(custom_price_badstring, proposed_price)
    valid_price_from_int = freqtrade.get_valid_price(custom_price_int, proposed_price)
    valid_price_from_float = freqtrade.get_valid_price(custom_price_float, proposed_price)

    valid_price_at_max_alwd = freqtrade.get_valid_price(custom_price_over_max_alwd, proposed_price)
    valid_price_at_min_alwd = freqtrade.get_valid_price(custom_price_under_min_alwd, proposed_price)

    assert isinstance(valid_price_from_string, float)
    assert isinstance(valid_price_from_badstring, float)
    assert isinstance(valid_price_from_int, float)
    assert isinstance(valid_price_from_float, float)

    assert valid_price_from_string == custom_price_float
    assert valid_price_from_badstring == proposed_price
    assert valid_price_from_int == custom_price_int
    assert valid_price_from_float == custom_price_float

    assert valid_price_at_max_alwd < custom_price_over_max_alwd
    assert valid_price_at_max_alwd > proposed_price

    assert valid_price_at_min_alwd > custom_price_under_min_alwd
    assert valid_price_at_min_alwd < proposed_price


@pytest.mark.parametrize('trading_mode,calls,t1,t2', [
    ('spot', 0, "2021-09-01 00:00:00", "2021-09-01 08:00:00"),
    ('margin', 0, "2021-09-01 00:00:00", "2021-09-01 08:00:00"),
    ('futures', 31, "2021-09-01 00:00:02", "2021-09-01 08:00:01"),
    ('futures', 32, "2021-08-31 23:59:59", "2021-09-01 08:00:01"),
    ('futures', 32, "2021-09-01 00:00:02", "2021-09-01 08:00:02"),
    ('futures', 33, "2021-08-31 23:59:59", "2021-09-01 08:00:02"),
    ('futures', 33, "2021-08-31 23:59:59", "2021-09-01 08:00:03"),
    ('futures', 33, "2021-08-31 23:59:59", "2021-09-01 08:00:04"),
    ('futures', 33, "2021-08-31 23:59:59", "2021-09-01 08:00:05"),
    ('futures', 33, "2021-08-31 23:59:59", "2021-09-01 08:00:06"),
    ('futures', 33, "2021-08-31 23:59:59", "2021-09-01 08:00:07"),
    ('futures', 33, "2021-08-31 23:59:58", "2021-09-01 08:00:07"),
])
def test_update_funding_fees_schedule(mocker, default_conf, trading_mode, calls, time_machine,
                                      t1, t2):
    time_machine.move_to(f"{t1} +00:00", tick=False)

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.update_funding_fees', return_value=True)
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    time_machine.move_to(f"{t2} +00:00", tick=False)
    # Check schedule jobs in debugging with freqtrade._schedule.jobs
    freqtrade._schedule.run_pending()

    assert freqtrade.update_funding_fees.call_count == calls


@pytest.mark.parametrize('schedule_off', [False, True])
@pytest.mark.parametrize('is_short', [True, False])
def test_update_funding_fees(
    mocker,
    default_conf,
    time_machine,
    fee,
    ticker_usdt_sell_up,
    is_short,
    limit_order_open,
    schedule_off
):
    """
    nominal_value = mark_price * size
    funding_fee = nominal_value * funding_rate
    size = 123
    "LTC/USDT"
        time: 0, mark: 3.3, fundRate: 0.00032583, nominal_value: 405.9, fundFee: 0.132254397
        time: 8, mark: 3.2, fundRate: 0.00024472, nominal_value: 393.6, fundFee: 0.096321792
    "ETH/USDT"
        time: 0, mark: 2.4, fundRate: 0.0001, nominal_value: 295.2, fundFee: 0.02952
        time: 8, mark: 2.5, fundRate: 0.0001, nominal_value: 307.5, fundFee: 0.03075
    "ETC/USDT"
        time: 0, mark: 4.3, fundRate: 0.00031077, nominal_value: 528.9, fundFee: 0.164366253
        time: 8, mark: 4.1, fundRate: 0.00022655, nominal_value: 504.3, fundFee: 0.114249165
    "XRP/USDT"
        time: 0, mark: 1.2, fundRate: 0.00049426, nominal_value: 147.6, fundFee: 0.072952776
        time: 8, mark: 1.2, fundRate: 0.00032715, nominal_value: 147.6, fundFee: 0.04828734
    """
    # SETUP
    time_machine.move_to("2021-09-01 00:00:00 +00:00")

    open_order = limit_order_open[entry_side(is_short)]
    open_exit_order = limit_order_open[exit_side(is_short)]
    bid = 0.11
    enter_rate_mock = MagicMock(return_value=bid)
    enter_mm = MagicMock(return_value=open_order)
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'

    date_midnight = arrow.get('2021-09-01 00:00:00').datetime
    date_eight = arrow.get('2021-09-01 08:00:00').datetime
    date_sixteen = arrow.get('2021-09-01 16:00:00').datetime
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    # 16:00 entry is actually never used
    # But should be kept in the test to ensure we're filtering correctly.
    funding_rates = {
        "LTC/USDT":
            DataFrame([
                [date_midnight, 0.00032583, 0, 0, 0, 0],
                [date_eight, 0.00024472, 0, 0, 0, 0],
                [date_sixteen, 0.00024472, 0, 0, 0, 0],
            ], columns=columns),
        "ETH/USDT":
            DataFrame([
                [date_midnight, 0.0001, 0, 0, 0, 0],
                [date_eight, 0.0001, 0, 0, 0, 0],
                [date_sixteen, 0.0001, 0, 0, 0, 0],
            ], columns=columns),
        "XRP/USDT":
            DataFrame([
                [date_midnight, 0.00049426, 0, 0, 0, 0],
                [date_eight, 0.00032715, 0, 0, 0, 0],
                [date_sixteen, 0.00032715, 0, 0, 0, 0],
            ], columns=columns)
    }

    mark_prices = {
        "LTC/USDT":
            DataFrame([
                [date_midnight, 3.3, 0, 0, 0, 0],
                [date_eight, 3.2, 0, 0, 0, 0],
                [date_sixteen, 3.2, 0, 0, 0, 0],
            ], columns=columns),
        "ETH/USDT":
            DataFrame([
                [date_midnight, 2.4, 0, 0, 0, 0],
                [date_eight, 2.5, 0, 0, 0, 0],
                [date_sixteen, 2.5, 0, 0, 0, 0],
            ], columns=columns),
        "XRP/USDT":
            DataFrame([
                [date_midnight, 1.2, 0, 0, 0, 0],
                [date_eight, 1.2, 0, 0, 0, 0],
                [date_sixteen, 1.2, 0, 0, 0, 0],
            ], columns=columns)
    }

    def refresh_latest_ohlcv_mock(pairlist, **kwargs):
        ret = {}
        for p, tf, ct in pairlist:
            if ct == CandleType.MARK:
                ret[(p, tf, ct)] = mark_prices[p]
            else:
                ret[(p, tf, ct)] = funding_rates[p]

        return ret

    mocker.patch('freqtrade.exchange.Exchange.refresh_latest_ohlcv',
                 side_effect=refresh_latest_ohlcv_mock)

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_rate=enter_rate_mock,
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=enter_mm,
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
        get_maintenance_ratio_and_amt=MagicMock(return_value=(0.01, 0.01)),
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # initial funding fees,
    freqtrade.execute_entry('ETH/USDT', 123, is_short=is_short)
    freqtrade.execute_entry('LTC/USDT', 2.0, is_short=is_short)
    freqtrade.execute_entry('XRP/USDT', 123, is_short=is_short)
    multipl = 1 if is_short else -1
    trades = Trade.get_open_trades()
    assert len(trades) == 3
    for trade in trades:
        assert pytest.approx(trade.funding_fees) == 0
    mocker.patch('freqtrade.exchange.Exchange.create_order', return_value=open_exit_order)
    time_machine.move_to("2021-09-01 08:00:00 +00:00")
    if schedule_off:
        for trade in trades:
            freqtrade.execute_trade_exit(
                trade=trade,
                # The values of the next 2 params are irrelevant for this test
                limit=ticker_usdt_sell_up()['bid'],
                exit_check=ExitCheckTuple(exit_type=ExitType.ROI)
            )
            assert trade.funding_fees == pytest.approx(sum(
                trade.amount *
                mark_prices[trade.pair].iloc[1:2]['open'] *
                funding_rates[trade.pair].iloc[1:2]['open'] * multipl
            ))

    else:
        freqtrade._schedule.run_pending()

    # Funding fees for 00:00 and 08:00
    for trade in trades:
        assert trade.funding_fees == pytest.approx(sum(
            trade.amount *
            mark_prices[trade.pair].iloc[1:2]['open'] *
            funding_rates[trade.pair].iloc[1:2]['open'] *
            multipl
        ))


def test_update_funding_fees_error(mocker, default_conf, caplog):
    mocker.patch('freqtrade.exchange.Exchange.get_funding_fees', side_effect=ExchangeError())
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.update_funding_fees()

    log_has("Could not update funding fees for open trades.", caplog)


def test_position_adjust(mocker, default_conf_usdt, fee) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=10000)
    default_conf_usdt.update({
        "position_adjustment_enable": True,
        "dry_run": False,
        "stake_amount": 10.0,
        "dry_run_wallet": 1000.0,
    })
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    bid = 11
    stake_amount = 10
    buy_rate_mock = MagicMock(return_value=bid)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_rate=buy_rate_mock,
        fetch_ticker=MagicMock(return_value={
            'bid': 10,
            'ask': 12,
            'last': 11
        }),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    pair = 'ETH/USDT'

    # Initial buy
    closed_successful_buy_order = {
        'pair': pair,
        'ft_pair': pair,
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'status': 'closed',
        'price': bid,
        'average': bid,
        'cost': bid * stake_amount,
        'amount': stake_amount,
        'filled': stake_amount,
        'ft_is_open': False,
        'id': '650',
        'order_id': '650'
    }
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=closed_successful_buy_order))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 MagicMock(return_value=closed_successful_buy_order))
    assert freqtrade.execute_entry(pair, stake_amount)
    # Should create an closed trade with an no open order id
    # Order is filled and trade is open
    orders = Order.query.all()
    assert orders
    assert len(orders) == 1
    trade = Trade.query.first()
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id is None
    assert trade.open_rate == 11
    assert trade.stake_amount == 110

    # Assume it does nothing since order is closed and trade is open
    freqtrade.update_trades_without_assigned_fees()

    trade = Trade.query.first()
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id is None
    assert trade.open_rate == 11
    assert trade.stake_amount == 110
    assert not trade.fee_updated('buy')

    freqtrade.manage_open_orders()

    trade = Trade.query.first()
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id is None
    assert trade.open_rate == 11
    assert trade.stake_amount == 110
    assert not trade.fee_updated('buy')

    # First position adjustment buy.
    open_dca_order_1 = {
        'ft_pair': pair,
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'status': None,
        'price': 9,
        'amount': 12,
        'cost': 108,
        'ft_is_open': True,
        'id': '651',
        'order_id': '651'
    }
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=open_dca_order_1))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 MagicMock(return_value=open_dca_order_1))
    assert freqtrade.execute_entry(pair, stake_amount, trade=trade)

    orders = Order.query.all()
    assert orders
    assert len(orders) == 2
    trade = Trade.query.first()
    assert trade
    assert trade.open_order_id == '651'
    assert trade.open_rate == 11
    assert trade.amount == 10
    assert trade.stake_amount == 110
    assert not trade.fee_updated('buy')
    trades: List[Trade] = Trade.get_open_trades_without_assigned_fees()
    assert len(trades) == 1
    assert trade.is_open
    assert not trade.fee_updated('buy')
    order = trade.select_order('buy', False)
    assert order
    assert order.order_id == '650'

    def make_sure_its_651(*args, **kwargs):

        if args[0] == '650':
            return closed_successful_buy_order
        if args[0] == '651':
            return open_dca_order_1
        return None

    # Assume it does nothing since order is still open
    fetch_order_mm = MagicMock(side_effect=make_sure_its_651)
    mocker.patch('freqtrade.exchange.Exchange.create_order', fetch_order_mm)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', fetch_order_mm)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order', fetch_order_mm)
    freqtrade.update_trades_without_assigned_fees()

    orders = Order.query.all()
    assert orders
    assert len(orders) == 2
    # Assert that the trade is found as open and without fees
    trades: List[Trade] = Trade.get_open_trades_without_assigned_fees()
    assert len(trades) == 1
    # Assert trade is as expected
    trade = Trade.query.first()
    assert trade
    assert trade.open_order_id == '651'
    assert trade.open_rate == 11
    assert trade.amount == 10
    assert trade.stake_amount == 110
    assert not trade.fee_updated('buy')

    # Make sure the closed order is found as the first order.
    order = trade.select_order('buy', False)
    assert order.order_id == '650'

    # Now close the order so it should update.
    closed_dca_order_1 = {
        'ft_pair': pair,
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'status': 'closed',
        'price': 9,
        'average': 9,
        'amount': 12,
        'filled': 12,
        'cost': 108,
        'ft_is_open': False,
        'id': '651',
        'order_id': '651',
        'datetime': arrow.utcnow().isoformat(),
    }

    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=closed_dca_order_1))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 MagicMock(return_value=closed_dca_order_1))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 MagicMock(return_value=closed_dca_order_1))
    freqtrade.manage_open_orders()

    # Assert trade is as expected (averaged dca)
    trade = Trade.query.first()
    assert trade
    assert trade.open_order_id is None
    assert pytest.approx(trade.open_rate) == 9.90909090909
    assert trade.amount == 22
    assert pytest.approx(trade.stake_amount) == 218

    orders = Order.query.all()
    assert orders
    assert len(orders) == 2

    # Make sure the closed order is found as the second order.
    order = trade.select_order('buy', False)
    assert order.order_id == '651'

    # Assert that the trade is not found as open and without fees
    trades: List[Trade] = Trade.get_open_trades_without_assigned_fees()
    assert len(trades) == 1

    # Add a second DCA
    closed_dca_order_2 = {
        'ft_pair': pair,
        'status': 'closed',
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'price': 7,
        'average': 7,
        'amount': 15,
        'filled': 15,
        'cost': 105,
        'ft_is_open': False,
        'id': '652',
        'order_id': '652'
    }
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=closed_dca_order_2))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 MagicMock(return_value=closed_dca_order_2))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 MagicMock(return_value=closed_dca_order_2))
    assert freqtrade.execute_entry(pair, stake_amount, trade=trade)

    # Assert trade is as expected (averaged dca)
    trade = Trade.query.first()
    assert trade
    assert trade.open_order_id is None
    assert pytest.approx(trade.open_rate) == 8.729729729729
    assert trade.amount == 37
    assert trade.stake_amount == 323

    orders = Order.query.all()
    assert orders
    assert len(orders) == 3

    # Make sure the closed order is found as the second order.
    order = trade.select_order('buy', False)
    assert order.order_id == '652'
    closed_sell_dca_order_1 = {
        'ft_pair': pair,
        'status': 'closed',
        'ft_order_side': 'sell',
        'side': 'sell',
        'type': 'limit',
        'price': 8,
        'average': 8,
        'amount': 15,
        'filled': 15,
        'cost': 120,
        'ft_is_open': False,
        'id': '653',
        'order_id': '653'
    }
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=closed_sell_dca_order_1))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 MagicMock(return_value=closed_sell_dca_order_1))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 MagicMock(return_value=closed_sell_dca_order_1))
    assert freqtrade.execute_trade_exit(trade=trade, limit=8,
                                        exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT),
                                        sub_trade_amt=15)

    # Assert trade is as expected (averaged dca)
    trade = Trade.query.first()
    assert trade
    assert trade.open_order_id is None
    assert trade.is_open
    assert trade.amount == 22
    assert trade.stake_amount == 192.05405405405406
    assert pytest.approx(trade.open_rate) == 8.729729729729

    orders = Order.query.all()
    assert orders
    assert len(orders) == 4

    # Make sure the closed order is found as the second order.
    order = trade.select_order('sell', False)
    assert order.order_id == '653'


def test_position_adjust2(mocker, default_conf_usdt, fee) -> None:
    """
    TODO: Should be adjusted to test both long and short
    buy 100 @ 11
    sell 50 @ 8
    sell 50 @ 16
    """
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=10000)
    default_conf_usdt.update({
        "position_adjustment_enable": True,
        "dry_run": False,
        "stake_amount": 200.0,
        "dry_run_wallet": 1000.0,
    })
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    bid = 11
    amount = 100
    buy_rate_mock = MagicMock(return_value=bid)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_rate=buy_rate_mock,
        fetch_ticker=MagicMock(return_value={
            'bid': 10,
            'ask': 12,
            'last': 11
        }),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    pair = 'ETH/USDT'
    # Initial buy
    closed_successful_buy_order = {
        'pair': pair,
        'ft_pair': pair,
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'status': 'closed',
        'price': bid,
        'average': bid,
        'cost': bid * amount,
        'amount': amount,
        'filled': amount,
        'ft_is_open': False,
        'id': '600',
        'order_id': '600'
    }
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=closed_successful_buy_order))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 MagicMock(return_value=closed_successful_buy_order))
    assert freqtrade.execute_entry(pair, amount)
    # Should create an closed trade with an no open order id
    # Order is filled and trade is open
    orders = Order.query.all()
    assert orders
    assert len(orders) == 1
    trade = Trade.query.first()
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id is None
    assert trade.open_rate == bid
    assert trade.stake_amount == bid * amount

    # Assume it does nothing since order is closed and trade is open
    freqtrade.update_trades_without_assigned_fees()

    trade = Trade.query.first()
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id is None
    assert trade.open_rate == bid
    assert trade.stake_amount == bid * amount
    assert not trade.fee_updated(trade.entry_side)

    freqtrade.manage_open_orders()

    trade = Trade.query.first()
    assert trade
    assert trade.is_open is True
    assert trade.open_order_id is None
    assert trade.open_rate == bid
    assert trade.stake_amount == bid * amount
    assert not trade.fee_updated(trade.entry_side)

    amount = 50
    ask = 8
    closed_sell_dca_order_1 = {
        'ft_pair': pair,
        'status': 'closed',
        'ft_order_side': 'sell',
        'side': 'sell',
        'type': 'limit',
        'price': ask,
        'average': ask,
        'amount': amount,
        'filled': amount,
        'cost': amount * ask,
        'ft_is_open': False,
        'id': '601',
        'order_id': '601'
    }
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=closed_sell_dca_order_1))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 MagicMock(return_value=closed_sell_dca_order_1))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 MagicMock(return_value=closed_sell_dca_order_1))
    assert freqtrade.execute_trade_exit(trade=trade, limit=ask,
                                        exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT),
                                        sub_trade_amt=amount)
    trades: List[Trade] = trade.get_open_trades_without_assigned_fees()
    assert len(trades) == 1
    # Assert trade is as expected (averaged dca)

    trade = Trade.query.first()
    assert trade
    assert trade.open_order_id is None
    assert trade.amount == 50
    assert trade.open_rate == 11
    assert trade.stake_amount == 550
    assert pytest.approx(trade.realized_profit) == -152.375
    assert pytest.approx(trade.close_profit_abs) == -152.375

    orders = Order.query.all()
    assert orders
    assert len(orders) == 2
    # Make sure the closed order is found as the second order.
    order = trade.select_order('sell', False)
    assert order.order_id == '601'

    amount = 50
    ask = 16
    closed_sell_dca_order_2 = {
        'ft_pair': pair,
        'status': 'closed',
        'ft_order_side': 'sell',
        'side': 'sell',
        'type': 'limit',
        'price': ask,
        'average': ask,
        'amount': amount,
        'filled': amount,
        'cost': amount * ask,
        'ft_is_open': False,
        'id': '602',
        'order_id': '602'
    }
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=closed_sell_dca_order_2))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 MagicMock(return_value=closed_sell_dca_order_2))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                 MagicMock(return_value=closed_sell_dca_order_2))
    assert freqtrade.execute_trade_exit(trade=trade, limit=ask,
                                        exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT),
                                        sub_trade_amt=amount)
    # Assert trade is as expected (averaged dca)

    trade = Trade.query.first()
    assert trade
    assert trade.open_order_id is None
    assert trade.amount == 50
    assert trade.open_rate == 11
    assert trade.stake_amount == 550
    # Trade fully realized
    assert pytest.approx(trade.realized_profit) == 94.25
    assert pytest.approx(trade.close_profit_abs) == 94.25
    orders = Order.query.all()
    assert orders
    assert len(orders) == 3

    # Make sure the closed order is found as the second order.
    order = trade.select_order('sell', False)
    assert order.order_id == '602'
    assert trade.is_open is False


@pytest.mark.parametrize('data', [
    (
        # tuple 1 - side amount, price
        # tuple 2 - amount, open_rate, stake_amount, cumulative_profit, realized_profit, rel_profit
        (('buy', 100, 10), (100.0, 10.0, 1000.0, 0.0, None, None)),
        (('buy', 100, 15), (200.0, 12.5, 2500.0, 0.0, None, None)),
        (('sell', 50, 12), (150.0, 12.5, 1875.0, -28.0625, -28.0625, -0.044788)),
        (('sell', 100, 20), (50.0, 12.5, 625.0, 713.8125, 741.875, 0.59201995)),
        (('sell', 50, 5), (50.0, 12.5, 625.0, 336.625, 336.625, 0.1343142)),  # final profit (sum)
    ),
    (
        (('buy', 100, 3), (100.0, 3.0, 300.0, 0.0, None, None)),
        (('buy', 100, 7), (200.0, 5.0, 1000.0, 0.0, None, None)),
        (('sell', 100, 11), (100.0, 5.0, 500.0, 596.0, 596.0, 1.189027)),
        (('buy', 150, 15), (250.0, 11.0, 2750.0, 596.0, 596.0, 1.189027)),
        (('sell', 100, 19), (150.0, 11.0, 1650.0, 1388.5, 792.5, 0.7186579)),
        (('sell', 150, 23), (150.0, 11.0, 1650.0, 3175.75, 3175.75, 0.9747170)),  # final profit
    )
])
def test_position_adjust3(mocker, default_conf_usdt, fee, data) -> None:
    default_conf_usdt.update({
        "position_adjustment_enable": True,
        "dry_run": False,
        "stake_amount": 200.0,
        "dry_run_wallet": 1000.0,
    })
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=10000)
    freqtrade = FreqtradeBot(default_conf_usdt)
    trade = None
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    for idx, (order, result) in enumerate(data):
        amount = order[1]
        price = order[2]
        price_mock = MagicMock(return_value=price)
        mocker.patch.multiple(
            'freqtrade.exchange.Exchange',
            get_rate=price_mock,
            fetch_ticker=MagicMock(return_value={
                'bid': 10,
                'ask': 12,
                'last': 11
            }),
            get_min_pair_stake_amount=MagicMock(return_value=1),
            get_fee=fee,
        )
        pair = 'ETH/USDT'
        closed_successful_order = {
            'pair': pair,
            'ft_pair': pair,
            'ft_order_side': order[0],
            'side': order[0],
            'type': 'limit',
            'status': 'closed',
            'price': price,
            'average': price,
            'cost': price * amount,
            'amount': amount,
            'filled': amount,
            'ft_is_open': False,
            'id': f'60{idx}',
            'order_id': f'60{idx}'
        }
        mocker.patch('freqtrade.exchange.Exchange.create_order',
                     MagicMock(return_value=closed_successful_order))
        mocker.patch('freqtrade.exchange.Exchange.fetch_order_or_stoploss_order',
                     MagicMock(return_value=closed_successful_order))
        if order[0] == 'buy':
            assert freqtrade.execute_entry(pair, amount, trade=trade)
        else:
            assert freqtrade.execute_trade_exit(
                trade=trade, limit=price,
                exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT),
                sub_trade_amt=amount)

        orders1 = Order.query.all()
        assert orders1
        assert len(orders1) == idx + 1

        trade = Trade.query.first()
        assert trade
        if idx < len(data) - 1:
            assert trade.is_open is True
        assert trade.open_order_id is None
        assert trade.amount == result[0]
        assert trade.open_rate == result[1]
        assert trade.stake_amount == result[2]
        assert pytest.approx(trade.realized_profit) == result[3]
        assert pytest.approx(trade.close_profit_abs) == result[4]
        assert pytest.approx(trade.close_profit) == result[5]

        order_obj = trade.select_order(order[0], False)
        assert order_obj.order_id == f'60{idx}'

    trade = Trade.query.first()
    assert trade
    assert trade.open_order_id is None
    assert trade.is_open is False


def test_process_open_trade_positions_exception(mocker, default_conf_usdt, fee, caplog) -> None:
    default_conf_usdt.update({
        "position_adjustment_enable": True,
    })
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.check_and_call_adjust_trade_position',
                 side_effect=DependencyException())

    create_mock_trades(fee)

    freqtrade.process_open_trade_positions()
    assert log_has_re(r"Unable to adjust position of trade for .*", caplog)


def test_check_and_call_adjust_trade_position(mocker, default_conf_usdt, fee, caplog) -> None:
    default_conf_usdt.update({
        "position_adjustment_enable": True,
        "max_entry_position_adjustment": 0,
    })
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    buy_rate_mock = MagicMock(return_value=10)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_rate=buy_rate_mock,
        fetch_ticker=MagicMock(return_value={
            'bid': 10,
            'ask': 12,
            'last': 11
        }),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    create_mock_trades(fee)
    caplog.set_level(logging.DEBUG)
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=10)
    freqtrade.process_open_trade_positions()
    assert log_has_re(r"Max adjustment entries for .* has been reached\.", caplog)

    caplog.clear()
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=-10)
    freqtrade.process_open_trade_positions()
    assert log_has_re(r"LIMIT_SELL has been fulfilled.*", caplog)
