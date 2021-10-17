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
from freqtrade.enums import RPCMessageType, RunMode, SellType, State
from freqtrade.exceptions import (DependencyException, ExchangeError, InsufficientFundsError,
                                  InvalidOrderException, OperationalException, PricingError,
                                  TemporaryError)
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, PairLocks, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.strategy.interface import SellCheckTuple
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


def test_bot_cleanup(mocker, default_conf_usdt, caplog) -> None:
    mock_cleanup = mocker.patch('freqtrade.freqtradebot.cleanup_db')
    coo_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders')
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.cleanup()
    assert log_has('Cleaning up modules ...', caplog)
    assert mock_cleanup.call_count == 1
    assert coo_mock.call_count == 0

    freqtrade.config['cancel_open_orders_on_exit'] = True
    freqtrade.cleanup()
    assert coo_mock.call_count == 1


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
        'buy': 'market',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['bid_strategy']['price_side'] = 'ask'

    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re(".*stoploss_on_exchange .* dry-run", caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']

    caplog.clear()
    # is left untouched
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'buy': 'market',
        'sell': 'limit',
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

    def _refresh_whitelist(list):
        return ['ETH/USDT', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC']

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


@pytest.mark.parametrize('buy_price_mult,ignore_strat_sl', [
    # Override stoploss
    (0.79, False),
    # Override strategy stoploss
    (0.85, True)
])
def test_edge_overrides_stoploss(limit_buy_order_usdt, fee, caplog, mocker,
                                 buy_price_mult, ignore_strat_sl, edge_conf) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')

    # Strategy stoploss is -0.1 but Edge imposes a stoploss at -0.2
    # Thus, if price falls 21%, stoploss should be triggered
    #
    # mocking the ticker_usdt: price is falling ...
    buy_price = limit_buy_order_usdt['price']
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': buy_price * buy_price_mult,
            'ask': buy_price * buy_price_mult,
            'last': buy_price * buy_price_mult,
        }),
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
    trade.update(limit_buy_order_usdt)
    #############################################

    # stoploss shoud be hit
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has('Executing Sell for NEO/BTC. Reason: stop_loss', caplog)
        assert trade.sell_reason == SellType.STOP_LOSS.value


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


def test_create_trade(default_conf_usdt, ticker_usdt, limit_buy_order_usdt, fee, mocker) -> None:
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
    patch_get_signal(freqtrade)
    freqtrade.create_trade('ETH/USDT')

    trade = Trade.query.first()
    assert trade is not None
    assert trade.stake_amount == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order_usdt)

    assert trade.open_rate == 2.0
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


@pytest.mark.parametrize('stake_amount,create,amount_enough,max_open_trades', [
    (5.0, True, True, 99),
    (0.00005, True, False, 99),
    (0, False, True, 99),
    (UNLIMITED_STAKE_AMOUNT, False, True, 0),
])
def test_create_trade_minimal_amount(
    default_conf_usdt, ticker_usdt, limit_buy_order_usdt_open, fee, mocker,
    stake_amount, create, amount_enough, max_open_trades, caplog
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    buy_mock = MagicMock(return_value=limit_buy_order_usdt_open)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=buy_mock,
        get_fee=fee,
    )
    default_conf_usdt['max_open_trades'] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['stake_amount'] = stake_amount
    patch_get_signal(freqtrade)

    if create:
        assert freqtrade.create_trade('ETH/USDT')
        if amount_enough:
            rate, amount = buy_mock.call_args[1]['rate'], buy_mock.call_args[1]['amount']
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

    PairLocks.lock_pair('*', arrow.utcnow().shift(minutes=20).datetime, 'Just because')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


def test_handle_protections(mocker, default_conf_usdt, fee):
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
        return_value=(True, arrow.utcnow().shift(hours=1).datetime, "asdf"))
    create_mock_trades(fee)
    freqtrade.handle_protections('ETC/BTC')
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
    patch_get_signal(freqtrade, value=(False, False, None))

    Trade.query = MagicMock()
    Trade.query.filter = MagicMock()
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
                               limit_buy_order_usdt_open) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 4
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    # Create 2 existing trades
    freqtrade.execute_entry('ETH/USDT', default_conf_usdt['stake_amount'])
    freqtrade.execute_entry('NEO/BTC', default_conf_usdt['stake_amount'])

    assert len(Trade.get_open_trades()) == 2
    # Change order_id for new orders
    limit_buy_order_usdt_open['id'] = '123444'

    # Create 2 new trades using create_trades
    assert freqtrade.create_trade('ETH/USDT')
    assert freqtrade.create_trade('NEO/BTC')

    trades = Trade.get_open_trades()
    assert len(trades) == 4


def test_process_trade_creation(default_conf_usdt, ticker_usdt, limit_buy_order_usdt,
                                limit_buy_order_usdt_open, fee, mocker, caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        fetch_order=MagicMock(return_value=limit_buy_order_usdt),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades

    freqtrade.process()

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert trade.stake_amount == default_conf_usdt['stake_amount']
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    assert trade.open_rate == 2.0
    assert trade.amount == 30.0

    assert log_has(
        'Buy signal found: about create a new trade for ETH/USDT with stake_amount: 60.0 ...',
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

    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    # Make sure each pair is only in the list once
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))


def test_process_informative_pairs_added(default_conf_usdt, ticker_usdt, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)

    def _refresh_whitelist(list):
        return ['ETH/USDT', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC']

    refresh_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=TemporaryError),
        refresh_latest_ohlcv=refresh_mock,
    )
    inf_pairs = MagicMock(return_value=[("BTC/ETH", '1m'), ("ETH/USDT", "1h")])
    mocker.patch(
        'freqtrade.strategy.interface.IStrategy.get_signal',
        return_value=(False, False, '')
    )
    mocker.patch('time.sleep', return_value=None)

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.pairlists._validate_whitelist = _refresh_whitelist
    freqtrade.strategy.informative_pairs = inf_pairs
    # patch_get_signal(freqtrade)

    freqtrade.process()
    assert inf_pairs.call_count == 1
    assert refresh_mock.call_count == 1
    assert ("BTC/ETH", "1m") in refresh_mock.call_args[0][0]
    assert ("ETH/USDT", "1h") in refresh_mock.call_args[0][0]
    assert ("ETH/USDT", default_conf_usdt["timeframe"]) in refresh_mock.call_args[0][0]


def test_execute_entry(mocker, default_conf_usdt, fee, limit_buy_order_usdt,
                       limit_buy_order_usdt_open) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    stake_amount = 2
    bid = 0.11
    buy_rate_mock = MagicMock(return_value=bid)
    buy_mm = MagicMock(return_value=limit_buy_order_usdt_open)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_rate=buy_rate_mock,
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=buy_mm,
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    pair = 'ETH/USDT'

    assert not freqtrade.execute_entry(pair, stake_amount)
    assert buy_rate_mock.call_count == 1
    assert buy_mm.call_count == 0
    assert freqtrade.strategy.confirm_trade_entry.call_count == 1
    buy_rate_mock.reset_mock()

    limit_buy_order_usdt_open['id'] = '22'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_entry(pair, stake_amount)
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
    limit_buy_order_usdt_open['id'] = '33'
    fix_price = 0.06
    assert freqtrade.execute_entry(pair, stake_amount, fix_price)
    # Make sure get_rate wasn't called again
    assert buy_rate_mock.call_count == 0

    assert buy_mm.call_count == 2
    call_args = buy_mm.call_args_list[1][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == fix_price
    assert call_args['amount'] == round(stake_amount / fix_price, 8)

    # In case of closed order
    limit_buy_order_usdt['status'] = 'closed'
    limit_buy_order_usdt['price'] = 10
    limit_buy_order_usdt['cost'] = 100
    limit_buy_order_usdt['id'] = '444'

    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=limit_buy_order_usdt))
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.query.all()[2]
    assert trade
    assert trade.open_order_id is None
    assert trade.open_rate == 10
    assert trade.stake_amount == 100

    # In case of rejected or expired order and partially filled
    limit_buy_order_usdt['status'] = 'expired'
    limit_buy_order_usdt['amount'] = 30.0
    limit_buy_order_usdt['filled'] = 20.0
    limit_buy_order_usdt['remaining'] = 10.00
    limit_buy_order_usdt['price'] = 0.5
    limit_buy_order_usdt['cost'] = 15.0
    limit_buy_order_usdt['id'] = '555'
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=limit_buy_order_usdt))
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.query.all()[3]
    assert trade
    assert trade.open_order_id == '555'
    assert trade.open_rate == 0.5
    assert trade.stake_amount == 15.0

    # Test with custom stake
    limit_buy_order_usdt['status'] = 'open'
    limit_buy_order_usdt['id'] = '556'

    freqtrade.strategy.custom_stake_amount = lambda **kwargs: 150.0
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.query.all()[4]
    assert trade
    assert trade.stake_amount == 150

    # Exception case
    limit_buy_order_usdt['id'] = '557'
    freqtrade.strategy.custom_stake_amount = lambda **kwargs: 20 / 0
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.query.all()[5]
    assert trade
    assert trade.stake_amount == 2.0

    # In case of the order is rejected and not filled at all
    limit_buy_order_usdt['status'] = 'rejected'
    limit_buy_order_usdt['amount'] = 30.0
    limit_buy_order_usdt['filled'] = 0.0
    limit_buy_order_usdt['remaining'] = 30.0
    limit_buy_order_usdt['price'] = 0.5
    limit_buy_order_usdt['cost'] = 0.0
    limit_buy_order_usdt['id'] = '66'
    mocker.patch('freqtrade.exchange.Exchange.create_order',
                 MagicMock(return_value=limit_buy_order_usdt))
    assert not freqtrade.execute_entry(pair, stake_amount)

    # Fail to get price...
    mocker.patch('freqtrade.exchange.Exchange.get_rate', MagicMock(return_value=0.0))

    with pytest.raises(PricingError, match="Could not determine buy price."):
        freqtrade.execute_entry(pair, stake_amount)

    # In case of custom entry price
    mocker.patch('freqtrade.exchange.Exchange.get_rate', return_value=0.50)
    limit_buy_order_usdt['status'] = 'open'
    limit_buy_order_usdt['id'] = '5566'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: 0.508
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.query.all()[6]
    assert trade
    assert trade.open_rate_requested == 0.508

    # In case of custom entry price set to None
    limit_buy_order_usdt['status'] = 'open'
    limit_buy_order_usdt['id'] = '5567'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: None

    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_rate=MagicMock(return_value=10),
    )

    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.query.all()[7]
    assert trade
    assert trade.open_rate_requested == 10

    # In case of custom entry price not float type
    limit_buy_order_usdt['status'] = 'open'
    limit_buy_order_usdt['id'] = '5568'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: "string price"
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.query.all()[8]
    assert trade
    assert trade.open_rate_requested == 10


def test_execute_entry_confirm_error(mocker, default_conf_usdt, fee, limit_buy_order_usdt) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 1.9,
            'ask': 2.2,
            'last': 1.9
        }),
        create_order=MagicMock(return_value=limit_buy_order_usdt),
        get_rate=MagicMock(return_value=0.11),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    stake_amount = 2
    pair = 'ETH/USDT'

    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=ValueError)
    assert freqtrade.execute_entry(pair, stake_amount)

    limit_buy_order_usdt['id'] = '222'
    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=Exception)
    assert freqtrade.execute_entry(pair, stake_amount)

    limit_buy_order_usdt['id'] = '2223'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_entry(pair, stake_amount)

    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    assert not freqtrade.execute_entry(pair, stake_amount)


def test_add_stoploss_on_exchange(mocker, default_conf_usdt, limit_buy_order_usdt) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order_usdt)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 return_value=limit_buy_order_usdt['amount'])

    stoploss = MagicMock(return_value={'id': 13434334})
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss)

    freqtrade = FreqtradeBot(default_conf_usdt)
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


def test_handle_stoploss_on_exchange(mocker, default_conf_usdt, fee, caplog,
                                     limit_buy_order_usdt, limit_sell_order_usdt) -> None:
    stoploss = MagicMock(return_value={'id': 13434334})
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
            {'id': limit_buy_order_usdt['id']},
            {'id': limit_sell_order_usdt['id']},
        ]),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        stoploss=stoploss
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
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
        'amount': limit_buy_order_usdt['amount'],
    })
    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order', stoploss_order_hit)
    assert freqtrade.handle_stoploss_on_exchange(trade) is True
    assert log_has_re(r'STOP_LOSS_LIMIT is hit for Trade\(id=1, .*\)\.', caplog)
    assert trade.stoploss_order_id is None
    assert trade.is_open is False
    caplog.clear()

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


def test_handle_sle_cancel_cant_recreate(mocker, default_conf_usdt, fee, caplog,
                                         limit_buy_order_usdt, limit_sell_order_usdt) -> None:
    # Sixth case: stoploss order was cancelled but couldn't create new one
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
            {'id': limit_buy_order_usdt['id']},
            {'id': limit_sell_order_usdt['id']},
        ]),
        get_fee=fee,
    )
    mocker.patch.multiple(
        'freqtrade.exchange.Binance',
        fetch_stoploss_order=MagicMock(return_value={'status': 'canceled', 'id': 100}),
        stoploss=MagicMock(side_effect=ExchangeError()),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
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


def test_create_stoploss_order_invalid_order(mocker, default_conf_usdt, caplog, fee,
                                             limit_buy_order_usdt_open, limit_sell_order_usdt):
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    create_order_mock = MagicMock(side_effect=[
        limit_buy_order_usdt_open,
        {'id': limit_sell_order_usdt['id']}
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
    patch_get_signal(freqtrade)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True

    freqtrade.enter_positions()
    trade = Trade.query.first()
    caplog.clear()
    freqtrade.create_stoploss_order(trade, 200)
    assert trade.stoploss_order_id is None
    assert trade.sell_reason == SellType.EMERGENCY_SELL.value
    assert log_has("Unable to place a stoploss order on exchange. ", caplog)
    assert log_has("Exiting the trade forcefully", caplog)

    # Should call a market sell
    assert create_order_mock.call_count == 2
    assert create_order_mock.call_args[1]['ordertype'] == 'market'
    assert create_order_mock.call_args[1]['pair'] == trade.pair
    assert create_order_mock.call_args[1]['amount'] == trade.amount

    # Rpc is sending first buy, then sell
    assert rpc_mock.call_count == 2
    assert rpc_mock.call_args_list[1][0][0]['sell_reason'] == SellType.EMERGENCY_SELL.value
    assert rpc_mock.call_args_list[1][0][0]['order_type'] == 'market'


def test_create_stoploss_order_insufficient_funds(mocker, default_conf_usdt, caplog, fee,
                                                  limit_buy_order_usdt_open, limit_sell_order_usdt):
    sell_mock = MagicMock(return_value={'id': limit_sell_order_usdt['id']})
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
            limit_buy_order_usdt_open,
            sell_mock,
        ]),
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
def test_handle_stoploss_on_exchange_trailing(mocker, default_conf_usdt, fee,
                                              limit_buy_order_usdt, limit_sell_order_usdt) -> None:
    # When trailing stoploss is set
    stoploss = MagicMock(return_value={'id': 13434334})
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': 2.19,
            'ask': 2.2,
            'last': 2.19
        }),
        create_order=MagicMock(side_effect=[
            {'id': limit_buy_order_usdt['id']},
            {'id': limit_sell_order_usdt['id']},
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
            'stopPrice': '2.0805'
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
            'bid': 4.38,
            'ask': 4.4,
            'last': 4.38
        })
    )

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
    assert trade.stop_loss == 4.4 * 0.95

    # setting stoploss_on_exchange_interval to 0 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    cancel_order_mock.assert_called_once_with(100, 'ETH/USDT')
    stoploss_order_mock.assert_called_once_with(
        amount=27.39726027,
        pair='ETH/USDT',
        order_types=freqtrade.strategy.order_types,
        stop_price=4.4 * 0.95
    )

    # price fell below stoploss, so dry-run sells trade.
    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': 4.16,
            'ask': 4.17,
            'last': 4.16
        })
    )
    assert freqtrade.handle_trade(trade) is True


def test_handle_stoploss_on_exchange_trailing_error(
        mocker, default_conf_usdt, fee, caplog, limit_buy_order_usdt, limit_sell_order_usdt
) -> None:
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
            {'id': limit_buy_order_usdt['id']},
            {'id': limit_sell_order_usdt['id']},
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
    mocker.patch('freqtrade.exchange.Binance.fetch_stoploss_order',
                 return_value=stoploss_order_hanging)
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert log_has_re(r"Could not cancel stoploss order abcd for pair ETH/USDT.*", caplog)

    # Still try to create order
    assert stoploss.call_count == 1

    # Fail creating stoploss order
    caplog.clear()
    cancel_mock = mocker.patch("freqtrade.exchange.Binance.cancel_stoploss_order", MagicMock())
    mocker.patch("freqtrade.exchange.Binance.stoploss", side_effect=ExchangeError())
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert cancel_mock.call_count == 1
    assert log_has_re(r"Could not create trailing stoploss order for pair ETH/USDT\..*", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_handle_stoploss_on_exchange_custom_stop(
        mocker, default_conf_usdt, fee, limit_buy_order_usdt, limit_sell_order_usdt) -> None:
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
            {'id': limit_buy_order_usdt['id']},
            {'id': limit_sell_order_usdt['id']},
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
            'bid': 4.38,
            'ask': 4.4,
            'last': 4.38
        })
    )

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
    assert trade.stop_loss == 4.4 * 0.96
    assert trade.stop_loss_pct == -0.04

    # setting stoploss_on_exchange_interval to 0 seconds
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0

    assert freqtrade.handle_stoploss_on_exchange(trade) is False

    cancel_order_mock.assert_called_once_with(100, 'ETH/USDT')
    stoploss_order_mock.assert_called_once_with(
        amount=31.57894736,
        pair='ETH/USDT',
        order_types=freqtrade.strategy.order_types,
        stop_price=4.4 * 0.96
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


def test_tsl_on_exchange_compatible_with_edge(mocker, edge_conf, fee,
                                              limit_buy_order_usdt, limit_sell_order_usdt) -> None:

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
            {'id': limit_buy_order_usdt['id']},
            {'id': limit_sell_order_usdt['id']},
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

    stoploss_order_hanging = MagicMock(return_value={
        'id': 100,
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'info': {
            'stopPrice': '2.178'
        }
    })

    mocker.patch('freqtrade.exchange.Exchange.fetch_stoploss_order', stoploss_order_hanging)

    # stoploss initially at 20% as edge dictated it.
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert isclose(trade.stop_loss, 1.76)

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
    assert isclose(trade.stop_loss, 1.76)

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
        amount=11.41438356,
        pair='NEO/BTC',
        order_types=freqtrade.strategy.order_types,
        stop_price=4.4 * 0.99
    )


@pytest.mark.parametrize('return_value,side_effect,log_message', [
    (False, None, 'Found no buy signals for whitelisted currencies. Trying again...'),
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


def test_exit_positions_exception(mocker, default_conf_usdt, limit_buy_order_usdt, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order_usdt)

    trade = MagicMock()
    trade.open_order_id = None
    trade.open_fee = 0.001
    trade.pair = 'ETH/USDT'
    trades = [trade]

    # Test raise of DependencyException exception
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.handle_trade',
        side_effect=DependencyException()
    )
    n = freqtrade.exit_positions(trades)
    assert n == 0
    assert log_has('Unable to sell trade ETH/USDT: ', caplog)


def test_update_trade_state(mocker, default_conf_usdt, limit_buy_order_usdt, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order_usdt)
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=[])
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount',
                 return_value=limit_buy_order_usdt['amount'])

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
    caplog.clear()
    # Add datetime explicitly since sqlalchemy defaults apply only once written to database
    freqtrade.update_trade_state(trade, '123')
    # Test amount not modified by fee-logic
    assert not log_has_re(r'Applying fee to .*', caplog)
    caplog.clear()
    assert trade.open_order_id is None
    assert trade.amount == limit_buy_order_usdt['amount']

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


@pytest.mark.parametrize('initial_amount,has_rounding_fee', [
    (30.0 + 1e-14, True),
    (8.0, False)
])
def test_update_trade_state_withorderdict(default_conf_usdt, trades_for_order, limit_buy_order_usdt,
                                          fee, mocker, initial_amount, has_rounding_fee, caplog):
    trades_for_order[0]['amount'] = initial_amount
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
        open_order_id="123456",
        is_open=True,
    )
    freqtrade.update_trade_state(trade, '123456', limit_buy_order_usdt)
    assert trade.amount != amount
    assert trade.amount == limit_buy_order_usdt['amount']
    if has_rounding_fee:
        assert log_has_re(r'Applying fee on amount for .*', caplog)


def test_update_trade_state_exception(mocker, default_conf_usdt,
                                      limit_buy_order_usdt, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=limit_buy_order_usdt)

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


def test_update_trade_state_orderexception(mocker, default_conf_usdt, caplog) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
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


def test_update_trade_state_sell(default_conf_usdt, trades_for_order, limit_sell_order_usdt_open,
                                 limit_sell_order_usdt, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_trades_for_order', return_value=trades_for_order)
    # fetch_order should not be called!!
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', MagicMock(side_effect=ValueError))
    wallet_mock = MagicMock()
    mocker.patch('freqtrade.wallets.Wallets.update', wallet_mock)

    patch_exchange(mocker)
    amount = limit_sell_order_usdt["amount"]
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
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
    order = Order.parse_from_ccxt_object(limit_sell_order_usdt_open, 'LTC/ETH', 'sell')
    trade.orders.append(order)
    assert order.status == 'open'
    freqtrade.update_trade_state(trade, trade.open_order_id, limit_sell_order_usdt)
    assert trade.amount == limit_sell_order_usdt['amount']
    # Wallet needs to be updated after closing a limit-sell order to reenable buying
    assert wallet_mock.call_count == 1
    assert not trade.is_open
    # Order is updated by update_trade_state
    assert order.status == 'closed'


def test_handle_trade(default_conf_usdt, limit_buy_order_usdt, limit_sell_order_usdt_open,
                      limit_sell_order_usdt, fee, mocker) -> None:
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
            limit_buy_order_usdt,
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
    trade.update(limit_buy_order_usdt)
    assert trade.is_open is True
    freqtrade.wallets.update()

    patch_get_signal(freqtrade, value=(False, True, None))
    assert freqtrade.handle_trade(trade) is True
    assert trade.open_order_id == limit_sell_order_usdt['id']

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order_usdt)

    assert trade.close_rate == 2.2
    assert trade.close_profit == 0.09451372
    assert trade.calc_profit() == 5.685
    assert trade.close_date is not None


def test_handle_overlapping_signals(default_conf_usdt, ticker_usdt, limit_buy_order_usdt_open,
                                    fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[
            limit_buy_order_usdt_open,
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )

    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, value=(True, True, None))
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()

    # Buy and Sell triggering, so doing nothing ...
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 0

    # Buy is triggering, so buying ...
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Buy and Sell are not triggering, so doing nothing ...
    patch_get_signal(freqtrade, value=(False, False, None))
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Buy and Sell are triggering, so doing nothing ...
    patch_get_signal(freqtrade, value=(True, True, None))
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.query.all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True

    # Sell is triggering, guess what : we are Selling!
    patch_get_signal(freqtrade, value=(False, True, None))
    trades = Trade.query.all()
    assert freqtrade.handle_trade(trades[0]) is True


def test_handle_trade_roi(default_conf_usdt, ticker_usdt, limit_buy_order_usdt_open,
                          fee, mocker, caplog) -> None:
    caplog.set_level(logging.DEBUG)

    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[
            limit_buy_order_usdt_open,
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_open = True

    # FIX: sniffing logs, suggest handle_trade should not execute_trade_exit
    #      instead that responsibility should be moved out of handle_trade(),
    #      we might just want to check if we are in a sell condition without
    #      executing
    # if ROI is reached we must sell
    patch_get_signal(freqtrade, value=(False, True, None))
    assert freqtrade.handle_trade(trade)
    assert log_has("ETH/USDT - Required profit reached. sell_type=SellType.ROI",
                   caplog)


def test_handle_trade_use_sell_signal(default_conf_usdt, ticker_usdt, limit_buy_order_usdt_open,
                                      limit_sell_order_usdt_open, fee, mocker, caplog) -> None:
    # use_sell_signal is True buy default
    caplog.set_level(logging.DEBUG)
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[
            limit_buy_order_usdt_open,
            limit_sell_order_usdt_open,
        ]),
        get_fee=fee,
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.is_open = True

    patch_get_signal(freqtrade, value=(False, False, None))
    assert not freqtrade.handle_trade(trade)

    patch_get_signal(freqtrade, value=(False, True, None))
    assert freqtrade.handle_trade(trade)
    assert log_has("ETH/USDT - Sell signal received. sell_type=SellType.SELL_SIGNAL",
                   caplog)


def test_close_trade(default_conf_usdt, ticker_usdt, limit_buy_order_usdt,
                     limit_buy_order_usdt_open, limit_sell_order_usdt, fee, mocker) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    # Create trade and sell it
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    trade.update(limit_buy_order_usdt)
    trade.update(limit_sell_order_usdt)
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


def test_check_handle_timedout_buy_usercustom(default_conf_usdt, ticker_usdt, limit_buy_order_old,
                                              open_trade, fee, mocker) -> None:
    default_conf_usdt["unfilledtimeout"] = {"buy": 1400, "sell": 30}

    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock(return_value=limit_buy_order_old)
    cancel_buy_order = deepcopy(limit_buy_order_old)
    cancel_buy_order['status'] = 'canceled'
    cancel_order_wr_mock = MagicMock(return_value=cancel_buy_order)

    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_buy_order_old),
        cancel_order_with_result=cancel_order_wr_mock,
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

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


def test_check_handle_timedout_buy(default_conf_usdt, ticker_usdt, limit_buy_order_old, open_trade,
                                   fee, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
    limit_buy_cancel = deepcopy(limit_buy_order_old)
    limit_buy_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_buy_cancel)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_buy_order_old),
        cancel_order_with_result=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

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


def test_check_handle_cancelled_buy(default_conf_usdt, ticker_usdt, limit_buy_order_old, open_trade,
                                    fee, mocker, caplog) -> None:
    """ Handle Buy order cancelled on exchange"""
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    limit_buy_order_old.update({"status": "canceled", 'filled': 0.0})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_buy_order_old),
        cancel_order=cancel_order_mock,
        get_fee=fee
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

    Trade.query.session.add(open_trade)

    # check it does cancel buy orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    assert log_has_re("Buy order cancelled on exchange for Trade.*", caplog)


def test_check_handle_timedout_buy_exception(default_conf_usdt, ticker_usdt,
                                             open_trade, fee, mocker) -> None:
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

    Trade.query.session.add(open_trade)

    # check it does cancel buy orders over the time limit
    freqtrade.check_handle_timedout()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 0
    trades = Trade.query.filter(Trade.open_order_id.is_(open_trade.open_order_id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1


def test_check_handle_timedout_sell_usercustom(default_conf_usdt, ticker_usdt, limit_sell_order_old,
                                               mocker, open_trade) -> None:
    default_conf_usdt["unfilledtimeout"] = {"buy": 1440, "sell": 1440}
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

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


def test_check_handle_timedout_sell(default_conf_usdt, ticker_usdt, limit_sell_order_old, mocker,
                                    open_trade) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

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


def test_check_handle_cancelled_sell(default_conf_usdt, ticker_usdt, limit_sell_order_old,
                                     open_trade, mocker, caplog) -> None:
    """ Handle sell order cancelled on exchange"""
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    limit_sell_order_old.update({"status": "canceled", 'filled': 0.0})
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order_with_result=cancel_order_mock
    )
    freqtrade = FreqtradeBot(default_conf_usdt)

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


def test_check_handle_timedout_partial(default_conf_usdt, ticker_usdt, limit_buy_order_old_partial,
                                       open_trade, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
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


def test_check_handle_timedout_partial_fee(default_conf_usdt, ticker_usdt, open_trade, caplog, fee,
                                           limit_buy_order_old_partial, trades_for_order,
                                           limit_buy_order_old_partial_canceled, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
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


def test_check_handle_timedout_partial_except(default_conf_usdt, ticker_usdt, open_trade, caplog,
                                              fee, limit_buy_order_old_partial, trades_for_order,
                                              limit_buy_order_old_partial_canceled, mocker) -> None:
    rpc_mock = patch_RPCManager(mocker)
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


def test_check_handle_timedout_exception(default_conf_usdt, ticker_usdt, open_trade_usdt, mocker,
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

    freqtrade.check_handle_timedout()
    assert log_has_re(r"Cannot query order for Trade\(id=1, pair=ADA/USDT, amount=30.00000000, "
                      r"open_rate=2.00000000, open_since="
                      f"{open_trade_usdt.open_date.strftime('%Y-%m-%d %H:%M:%S')}"
                      r"\) due to Traceback \(most recent call last\):\n*",
                      caplog)


def test_handle_cancel_enter(mocker, caplog, default_conf_usdt, limit_buy_order_usdt) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_buy_order = deepcopy(limit_buy_order_usdt)
    cancel_buy_order['status'] = 'canceled'
    del cancel_buy_order['filled']

    cancel_order_mock = MagicMock(return_value=cancel_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order_with_result', cancel_order_mock)

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade._notify_enter_cancel = MagicMock()

    trade = MagicMock()
    trade.pair = 'LTC/USDT'
    trade.open_rate = 200
    limit_buy_order_usdt['filled'] = 0.0
    limit_buy_order_usdt['status'] = 'open'
    reason = CANCEL_REASON['TIMEOUT']
    assert freqtrade.handle_cancel_enter(trade, limit_buy_order_usdt, reason)
    assert cancel_order_mock.call_count == 1

    cancel_order_mock.reset_mock()
    caplog.clear()
    limit_buy_order_usdt['filled'] = 0.01
    assert not freqtrade.handle_cancel_enter(trade, limit_buy_order_usdt, reason)
    assert cancel_order_mock.call_count == 0
    assert log_has_re("Order .* for .* not cancelled, as the filled amount.* unsellable.*", caplog)

    caplog.clear()
    cancel_order_mock.reset_mock()
    limit_buy_order_usdt['filled'] = 2
    assert not freqtrade.handle_cancel_enter(trade, limit_buy_order_usdt, reason)
    assert cancel_order_mock.call_count == 1

    # Order remained open for some reason (cancel failed)
    cancel_buy_order['status'] = 'open'
    cancel_order_mock = MagicMock(return_value=cancel_buy_order)
    mocker.patch('freqtrade.exchange.Exchange.cancel_order_with_result', cancel_order_mock)
    assert not freqtrade.handle_cancel_enter(trade, limit_buy_order_usdt, reason)
    assert log_has_re(r"Order .* for .* not cancelled.", caplog)


@pytest.mark.parametrize("limit_buy_order_canceled_empty", ['binance', 'ftx', 'kraken', 'bittrex'],
                         indirect=['limit_buy_order_canceled_empty'])
def test_handle_cancel_enter_exchanges(mocker, caplog, default_conf_usdt,
                                       limit_buy_order_canceled_empty) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = mocker.patch(
        'freqtrade.exchange.Exchange.cancel_order_with_result',
        return_value=limit_buy_order_canceled_empty)
    nofiy_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot._notify_enter_cancel')
    freqtrade = FreqtradeBot(default_conf_usdt)

    reason = CANCEL_REASON['TIMEOUT']
    trade = MagicMock()
    trade.pair = 'LTC/ETH'
    assert freqtrade.handle_cancel_enter(trade, limit_buy_order_canceled_empty, reason)
    assert cancel_order_mock.call_count == 0
    assert log_has_re(r'Buy order fully cancelled. Removing .* from database\.', caplog)
    assert nofiy_mock.call_count == 1


@pytest.mark.parametrize('cancelorder', [
    {},
    {'remaining': None},
    'String Return value',
    123
])
def test_handle_cancel_enter_corder_empty(mocker, default_conf_usdt, limit_buy_order_usdt,
                                          cancelorder) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock(return_value=cancelorder)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        cancel_order=cancel_order_mock
    )

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade._notify_enter_cancel = MagicMock()

    trade = MagicMock()
    trade.pair = 'LTC/USDT'
    trade.open_rate = 200
    limit_buy_order_usdt['filled'] = 0.0
    limit_buy_order_usdt['status'] = 'open'
    reason = CANCEL_REASON['TIMEOUT']
    assert freqtrade.handle_cancel_enter(trade, limit_buy_order_usdt, reason)
    assert cancel_order_mock.call_count == 1

    cancel_order_mock.reset_mock()
    limit_buy_order_usdt['filled'] = 1.0
    assert not freqtrade.handle_cancel_enter(trade, limit_buy_order_usdt, reason)
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

    freqtrade = FreqtradeBot(default_conf_usdt)

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
    assert freqtrade.handle_cancel_exit(trade, order, reason)
    assert cancel_order_mock.call_count == 1
    assert send_msg_mock.call_count == 1

    send_msg_mock.reset_mock()

    order['amount'] = 2
    assert freqtrade.handle_cancel_exit(trade, order, reason
                                        ) == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    # Assert cancel_order was not called (callcount remains unchanged)
    assert cancel_order_mock.call_count == 1
    assert send_msg_mock.call_count == 1
    assert freqtrade.handle_cancel_exit(trade, order, reason
                                        ) == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    # Message should not be iterated again
    assert trade.sell_order_status == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    assert send_msg_mock.call_count == 1


def test_handle_cancel_exit_cancel_exception(mocker, default_conf_usdt) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.cancel_order_with_result', side_effect=InvalidOrderException())

    freqtrade = FreqtradeBot(default_conf_usdt)

    trade = MagicMock()
    reason = CANCEL_REASON['TIMEOUT']
    order = {'remaining': 1,
             'amount': 1,
             'status': "open"}
    assert freqtrade.handle_cancel_exit(trade, order, reason) == 'error cancelling order'


def test_execute_trade_exit_up(default_conf_usdt, ticker_usdt, fee, ticker_usdt_sell_up, mocker
                               ) -> None:
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
        fetch_ticker=ticker_usdt_sell_up
    )
    # Prevented sell ...
    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['bid'],
                                 sell_reason=SellCheckTuple(sell_type=SellType.ROI))
    assert rpc_mock.call_count == 0
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1

    # Repatch with true
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)

    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['bid'],
                                 sell_reason=SellCheckTuple(sell_type=SellType.ROI))
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1

    assert rpc_mock.call_count == 1
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'trade_id': 1,
        'type': RPCMessageType.SELL,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'gain': 'profit',
        'limit': 2.2,
        'amount': 30.0,
        'order_type': 'limit',
        'open_rate': 2.0,
        'current_rate': 2.3,
        'profit_amount': 5.685,
        'profit_ratio': 0.09451372,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': SellType.ROI.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == last_msg


def test_execute_trade_exit_down(default_conf_usdt, ticker_usdt, fee, ticker_usdt_sell_down,
                                 mocker) -> None:
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
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_down
    )

    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_down()['bid'],
                                 sell_reason=SellCheckTuple(sell_type=SellType.STOP_LOSS))

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'gain': 'loss',
        'limit': 2.01,
        'amount': 30.0,
        'order_type': 'limit',
        'open_rate': 2.0,
        'current_rate': 2.0,
        'profit_amount': -0.00075,
        'profit_ratio': -1.247e-05,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == last_msg


def test_execute_trade_exit_custom_exit_price(default_conf_usdt, ticker_usdt, fee,
                                              ticker_usdt_sell_up, mocker) -> None:
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
        fetch_ticker=ticker_usdt_sell_up
    )

    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)

    # Set a custom exit price
    freqtrade.strategy.custom_exit_price = lambda **kwargs: 2.25

    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['bid'],
                                 sell_reason=SellCheckTuple(sell_type=SellType.SELL_SIGNAL))

    # Sell price must be different to default bid price

    assert freqtrade.strategy.confirm_trade_exit.call_count == 1

    assert rpc_mock.call_count == 1
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'trade_id': 1,
        'type': RPCMessageType.SELL,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'gain': 'profit',
        'limit': 2.25,
        'amount': 30.0,
        'order_type': 'limit',
        'open_rate': 2.0,
        'current_rate': 2.3,
        'profit_amount': 7.18125,
        'profit_ratio': 0.11938903,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': SellType.SELL_SIGNAL.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
    } == last_msg


def test_execute_trade_exit_down_stoploss_on_exchange_dry_run(
        default_conf_usdt, ticker_usdt, fee, ticker_usdt_sell_down, mocker) -> None:
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
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_down
    )

    default_conf_usdt['dry_run'] = True
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    # Setting trade stoploss to 0.01

    trade.stop_loss = 2.0 * 0.99
    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_down()['bid'],
                                 sell_reason=SellCheckTuple(sell_type=SellType.STOP_LOSS))

    assert rpc_mock.call_count == 2
    last_msg = rpc_mock.call_args_list[-1][0][0]

    assert {
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'gain': 'loss',
        'limit': 1.98,
        'amount': 30.0,
        'order_type': 'limit',
        'open_rate': 2.0,
        'current_rate': 2.0,
        'profit_amount': -0.8985,
        'profit_ratio': -0.01493766,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': SellType.STOP_LOSS.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
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
                                 sell_reason=SellCheckTuple(sell_type=SellType.STOP_LOSS))
    assert create_order_mock.call_count == 2
    assert log_has('Could not cancel stoploss order abcd', caplog)


def test_execute_trade_exit_with_stoploss_on_exchange(default_conf_usdt, ticker_usdt, fee,
                                                      ticker_usdt_sell_up, mocker) -> None:

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
        fetch_ticker=ticker_usdt_sell_up
    )

    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['bid'],
                                 sell_reason=SellCheckTuple(sell_type=SellType.STOP_LOSS))

    trade = Trade.query.first()
    assert trade
    assert cancel_order.call_count == 1
    assert rpc_mock.call_count == 3


def test_may_execute_trade_exit_after_stoploss_on_exchange_hit(default_conf_usdt, ticker_usdt, fee,
                                                               mocker) -> None:
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


def test_execute_trade_exit_market_order(default_conf_usdt, ticker_usdt, fee,
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
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_up
    )
    freqtrade.config['order_types']['sell'] = 'market'

    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['bid'],
                                 sell_reason=SellCheckTuple(sell_type=SellType.ROI))

    assert not trade.is_open
    assert trade.close_profit == 0.09451372

    assert rpc_mock.call_count == 3
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.SELL,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'gain': 'profit',
        'limit': 2.2,
        'amount': 30.0,
        'order_type': 'market',
        'open_rate': 2.0,
        'current_rate': 2.3,
        'profit_amount': 5.685,
        'profit_ratio': 0.09451372,
        'stake_currency': 'USDT',
        'fiat_currency': 'USD',
        'sell_reason': SellType.ROI.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,

    } == last_msg


def test_execute_trade_exit_insufficient_funds_error(default_conf_usdt, ticker_usdt, fee,
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
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_up
    )

    sell_reason = SellCheckTuple(sell_type=SellType.ROI)
    assert not freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['bid'],
                                            sell_reason=sell_reason)
    assert mock_insuf.call_count == 1


@pytest.mark.parametrize('profit_only,bid,ask,handle_first,handle_second,sell_type', [
    # Enable profit
    (True, 1.9, 2.2, False, True, SellType.SELL_SIGNAL.value),
    # Disable profit
    (False, 2.9, 3.2, True,  False, SellType.SELL_SIGNAL.value),
    # Enable loss
    # * Shouldn't this be SellType.STOP_LOSS.value
    (True, 0.19, 0.22, False, False, None),
    # Disable loss
    (False, 0.10, 0.22, True, False, SellType.SELL_SIGNAL.value),
])
def test_sell_profit_only(
        default_conf_usdt, limit_buy_order_usdt, limit_buy_order_usdt_open,
        fee, mocker, profit_only, bid, ask, handle_first, handle_second, sell_type) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': bid,
            'ask': ask,
            'last': bid
        }),
        create_order=MagicMock(side_effect=[
            limit_buy_order_usdt_open,
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )
    default_conf_usdt.update({
        'use_sell_signal': True,
        'sell_profit_only': profit_only,
        'sell_profit_offset': 0.1,
    })
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    if sell_type == SellType.SELL_SIGNAL.value:
        freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    else:
        freqtrade.strategy.stop_loss_reached = MagicMock(return_value=SellCheckTuple(
            sell_type=SellType.NONE))
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order_usdt)
    freqtrade.wallets.update()
    patch_get_signal(freqtrade, value=(False, True, None))
    assert freqtrade.handle_trade(trade) is handle_first

    if handle_second:
        freqtrade.strategy.sell_profit_offset = 0.0
        assert freqtrade.handle_trade(trade) is True


def test_sell_not_enough_balance(default_conf_usdt, limit_buy_order_usdt, limit_buy_order_usdt_open,
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
            limit_buy_order_usdt_open,
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
    trade.update(limit_buy_order_usdt)
    patch_get_signal(freqtrade, value=(False, True, None))
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
        with pytest.raises(DependencyException, match=r"Not enough amount to sell."):
            assert freqtrade._safe_exit_amount(trade.pair, trade.amount)
    else:
        wallet_update.reset_mock()
        assert freqtrade._safe_exit_amount(trade.pair, trade.amount) == amount_wallet
        assert log_has_re(r'.*Falling back to wallet-amount.', caplog)
        assert wallet_update.call_count == 1
        caplog.clear()
        wallet_update.reset_mock()
        assert freqtrade._safe_exit_amount(trade.pair, amount_wallet) == amount_wallet
        assert not log_has_re(r'.*Falling back to wallet-amount.', caplog)
        assert wallet_update.call_count == 1


def test_locked_pairs(default_conf_usdt, ticker_usdt, fee, ticker_usdt_sell_down, mocker,
                      caplog) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt_sell_down
    )

    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_down()['bid'],
                                 sell_reason=SellCheckTuple(sell_type=SellType.STOP_LOSS))
    trade.close(ticker_usdt_sell_down()['bid'])
    assert freqtrade.strategy.is_pair_locked(trade.pair)

    # reinit - should buy other pair.
    caplog.clear()
    freqtrade.enter_positions()

    assert log_has_re(f"Pair {trade.pair} is still locked.*", caplog)


def test_ignore_roi_if_buy_signal(default_conf_usdt, limit_buy_order_usdt,
                                  limit_buy_order_usdt_open, fee, mocker) -> None:
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
            limit_buy_order_usdt_open,
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )
    default_conf_usdt['ignore_roi_if_buy_signal'] = True

    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order_usdt)
    freqtrade.wallets.update()
    patch_get_signal(freqtrade, value=(True, True, None))
    assert freqtrade.handle_trade(trade) is False

    # Test if buy-signal is absent (should sell due to roi = true)
    patch_get_signal(freqtrade, value=(False, True, None))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.ROI.value


def test_trailing_stop_loss(default_conf_usdt, limit_buy_order_usdt_open,
                            fee, caplog, mocker) -> None:
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
            limit_buy_order_usdt_open,
            {'id': 1234553382},
        ]),
        get_fee=fee,
    )
    default_conf_usdt['trailing_stop'] = True
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()
    trade = Trade.query.first()
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker_usdt above buy price
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': 2.0 * 1.5,
                     'ask': 2.0 * 1.5,
                     'last': 2.0 * 1.5
                 }))

    # Stoploss should be adjusted
    assert freqtrade.handle_trade(trade) is False
    caplog.clear()
    # Price fell
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(return_value={
                     'bid': 2.0 * 1.1,
                     'ask': 2.0 * 1.1,
                     'last': 2.0 * 1.1
                 }))

    caplog.set_level(logging.DEBUG)
    # Sell as trailing-stop is reached
    assert freqtrade.handle_trade(trade) is True
    assert log_has("ETH/USDT - HIT STOP: current price at 2.200000, stoploss is 2.700000, "
                   "initial stoploss was at 1.800000, trade opened at 2.000000", caplog)
    assert trade.sell_reason == SellType.TRAILING_STOP_LOSS.value


@pytest.mark.parametrize('offset,trail_if_reached,second_sl', [
    (0, False, 2.0394),
    (0.011, False, 2.0394),
    (0.055, True, 1.8),
])
def test_trailing_stop_loss_positive(
    default_conf_usdt, limit_buy_order_usdt, limit_buy_order_usdt_open,
    offset, fee, caplog, mocker, trail_if_reached, second_sl
) -> None:
    buy_price = limit_buy_order_usdt['price']
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock(return_value={
            'bid': buy_price - 0.01,
            'ask': buy_price - 0.01,
            'last': buy_price - 0.01
        }),
        create_order=MagicMock(side_effect=[
            limit_buy_order_usdt_open,
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
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order_usdt)
    caplog.set_level(logging.DEBUG)
    # stop-loss not reached
    assert freqtrade.handle_trade(trade) is False

    # Raise ticker_usdt above buy price
    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': buy_price + 0.06,
            'ask': buy_price + 0.06,
            'last': buy_price + 0.06
        })
    )
    # stop-loss not reached, adjusted stoploss
    assert freqtrade.handle_trade(trade) is False
    caplog_text = f"ETH/USDT - Using positive stoploss: 0.01 offset: {offset} profit: 0.0249%"
    if trail_if_reached:
        assert not log_has(caplog_text, caplog)
        assert not log_has("ETH/USDT - Adjusting stoploss...", caplog)
    else:
        assert log_has(caplog_text, caplog)
        assert log_has("ETH/USDT - Adjusting stoploss...", caplog)
    assert trade.stop_loss == second_sl
    caplog.clear()

    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': buy_price + 0.125,
            'ask': buy_price + 0.125,
            'last': buy_price + 0.125,
        })
    )
    assert freqtrade.handle_trade(trade) is False
    assert log_has(
        f"ETH/USDT - Using positive stoploss: 0.01 offset: {offset} profit: 0.0572%",
        caplog
    )
    assert log_has("ETH/USDT - Adjusting stoploss...", caplog)

    mocker.patch(
        'freqtrade.exchange.Exchange.fetch_ticker',
        MagicMock(return_value={
            'bid': buy_price + 0.02,
            'ask': buy_price + 0.02,
            'last': buy_price + 0.02
        })
    )
    # Lower price again (but still positive)
    assert freqtrade.handle_trade(trade) is True
    assert log_has(
        f"ETH/USDT - HIT STOP: current price at {buy_price + 0.02:.6f}, "
        f"stoploss is {trade.stop_loss:.6f}, "
        f"initial stoploss was at 1.800000, trade opened at 2.000000", caplog)
    assert trade.sell_reason == SellType.TRAILING_STOP_LOSS.value


def test_disable_ignore_roi_if_buy_signal(default_conf_usdt, limit_buy_order_usdt,
                                          limit_buy_order_usdt_open, fee, mocker) -> None:
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
            limit_buy_order_usdt_open,
            {'id': 1234553382},
            {'id': 1234553383}
        ]),
        get_fee=fee,
        _is_dry_limit_order_filled=MagicMock(return_value=False),
    )
    default_conf_usdt['ask_strategy'] = {
        'ignore_roi_if_buy_signal': False
    }
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)

    freqtrade.enter_positions()

    trade = Trade.query.first()
    trade.update(limit_buy_order_usdt)
    # Sell due to min_roi_reached
    patch_get_signal(freqtrade, value=(True, True, None))
    assert freqtrade.handle_trade(trade) is True

    # Test if buy-signal is absent
    patch_get_signal(freqtrade, value=(False, True, None))
    assert freqtrade.handle_trade(trade) is True
    assert trade.sell_reason == SellType.ROI.value


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

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount - (amount * 0.001)
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.992).',
                   caplog)


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
    # Amount is kept as is
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount
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

    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, buy_order_fee) == amount
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
                   'open_rate=0.24544100, open_since=closed) failed: myTrade-Dict empty found',
                   caplog)


@pytest.mark.parametrize(
    'fee_par,fee_reduction_amount,use_ticker_usdt_rate,expected_log', [
        # basic, amount does not change
        ({'cost': 0.008, 'currency': 'ETH'}, 0, False, None),
        # no currency in fee
        ({'cost': 0.004, 'currency': None}, 0, True, None),
        # BNB no rate
        ({'cost': 0.00094518, 'currency': 'BNB'}, 0, True, (
            'Fee for Trade Trade(id=None, pair=LTC/ETH, amount=8.00000000, open_rate=0.24544100,'
            ' open_since=closed) [buy]: 0.00094518 BNB - rate: None'
        )),
        # from order
        ({'cost': 0.004, 'currency': 'LTC'}, 0.004, False, (
            'Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
            'open_rate=0.24544100, open_since=closed) (from 8.0 to 7.996).'
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
    assert freqtrade.get_real_amount(trade, buy_order) == amount - fee_reduction_amount

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
    expected_amount = amount - (amount * fee_reduction_amount)
    assert freqtrade.get_real_amount(trade, buy_order_fee) == expected_amount
    assert log_has(
        (
            'Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, '
            f'open_rate=0.24544100, open_since=closed) (from 8.0 to {expected_log_amount}).'
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

    # Amount does not change
    assert freqtrade.get_real_amount(trade, limit_buy_order_usdt) == amount


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

    # Amount does not change
    with pytest.raises(DependencyException, match=r"Half bought\? Amounts don't match"):
        freqtrade.get_real_amount(trade, limit_buy_order_usdt)


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

    # Amount changes by fee amount.
    assert isclose(
        freqtrade.get_real_amount(trade, limit_buy_order_usdt),
        amount - (amount * 0.001),
        abs_tol=MATH_CLOSE_PREC,
    )


def test_get_real_amount_open_trade(default_conf_usdt, fee, mocker):
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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    assert freqtrade.get_real_amount(trade, order) == amount


@pytest.mark.parametrize('amount,fee_abs,wallet,amount_exp', [
    (8.0, 0.0, 10, 8),
    (8.0, 0.0, 0, 8),
    (8.0, 0.1, 0, 7.9),
    (8.0, 0.1, 10, 8),
    (8.0, 0.1, 8.0, 8.0),
    (8.0, 0.1, 7.9, 7.9),
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
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    walletmock.reset_mock()
    # Amount is kept as is
    assert freqtrade.apply_fee_conditional(trade, 'LTC', amount, fee_abs) == amount_exp
    assert walletmock.call_count == 1


@pytest.mark.parametrize("delta, is_high_delta", [
    (0.1, False),
    (100, True),
])
def test_order_book_depth_of_market(
    default_conf_usdt, ticker_usdt, limit_buy_order_usdt_open, limit_buy_order_usdt,
    fee, mocker, order_book_l2, delta, is_high_delta
):
    default_conf_usdt['bid_strategy']['check_depth_of_market']['enabled'] = True
    default_conf_usdt['bid_strategy']['check_depth_of_market']['bids_to_ask_delta'] = delta
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book', order_book_l2)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )

    # Save state of current whitelist
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()

    trade = Trade.query.first()
    if is_high_delta:
        assert trade is None
    else:
        assert trade is not None
        assert trade.stake_amount == 60.0
        assert trade.is_open
        assert trade.open_date is not None
        assert trade.exchange == 'binance'

        assert len(Trade.query.all()) == 1

        # Simulate fulfilled LIMIT_BUY order for trade
        trade.update(limit_buy_order_usdt)

        assert trade.open_rate == 2.0
        assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


@pytest.mark.parametrize('exception_thrown,ask,last,order_book_top,order_book', [
    (False, 0.045, 0.046, 2, None),
    (True,  0.042, 0.046, 1, {'bids': [[]], 'asks': [[]]})
])
def test_order_book_bid_strategy1(mocker, default_conf_usdt, order_book_l2, exception_thrown,
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
    default_conf_usdt['bid_strategy']['use_order_book'] = True
    default_conf_usdt['bid_strategy']['order_book_top'] = order_book_top
    default_conf_usdt['bid_strategy']['ask_last_balance'] = 0
    default_conf_usdt['telegram']['enabled'] = False

    freqtrade = FreqtradeBot(default_conf_usdt)
    if exception_thrown:
        with pytest.raises(PricingError):
            freqtrade.exchange.get_rate('ETH/USDT', refresh=True, side="buy")
        assert log_has_re(
            r'Buy Price at location 1 from orderbook could not be determined.', caplog)
    else:
        assert freqtrade.exchange.get_rate('ETH/USDT', refresh=True, side="buy") == 0.043935
        assert ticker_usdt_mock.call_count == 0


def test_check_depth_of_market_buy(default_conf_usdt, mocker, order_book_l2) -> None:
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
    default_conf_usdt['bid_strategy']['check_depth_of_market']['enabled'] = True
    # delta is 100 which is impossible to reach. hence function will return false
    default_conf_usdt['bid_strategy']['check_depth_of_market']['bids_to_ask_delta'] = 100
    freqtrade = FreqtradeBot(default_conf_usdt)

    conf = default_conf_usdt['bid_strategy']['check_depth_of_market']
    assert freqtrade._check_depth_of_market_buy('ETH/USDT', conf) is False


def test_order_book_ask_strategy(
        default_conf_usdt, limit_buy_order_usdt_open, limit_buy_order_usdt, fee,
        limit_sell_order_usdt_open, mocker, order_book_l2, caplog) -> None:
    """
    test order book ask strategy
    """
    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book', order_book_l2)
    default_conf_usdt['exchange']['name'] = 'binance'
    default_conf_usdt['ask_strategy']['use_order_book'] = True
    default_conf_usdt['ask_strategy']['order_book_top'] = 1
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
    trade.update(limit_buy_order_usdt)
    freqtrade.wallets.update()
    assert trade.is_open is True

    patch_get_signal(freqtrade, value=(False, True, None))
    assert freqtrade.handle_trade(trade) is True
    assert trade.close_rate_requested == order_book_l2.return_value['asks'][0][0]

    mocker.patch('freqtrade.exchange.Exchange.fetch_l2_order_book',
                 return_value={'bids': [[]], 'asks': [[]]})
    with pytest.raises(PricingError):
        freqtrade.handle_trade(trade)
    assert log_has_re(r'Sell Price at location 1 from orderbook could not be determined\..*',
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
def test_cancel_all_open_orders(mocker, default_conf_usdt, fee, limit_buy_order_usdt,
                                limit_sell_order_usdt):
    default_conf_usdt['cancel_open_orders_on_exit'] = True
    mocker.patch('freqtrade.exchange.Exchange.fetch_order',
                 side_effect=[
                     ExchangeError(),
                     limit_sell_order_usdt,
                     limit_buy_order_usdt,
                     limit_sell_order_usdt
                 ])
    buy_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_enter')
    sell_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_exit')

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades(fee)
    trades = Trade.query.all()
    assert len(trades) == MOCK_TRADE_COUNT
    freqtrade.cancel_all_open_orders()
    assert buy_mock.call_count == 1
    assert sell_mock.call_count == 2


@pytest.mark.usefixtures("init_persistence")
def test_check_for_open_trades(mocker, default_conf_usdt, fee):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    freqtrade.check_for_open_trades()
    assert freqtrade.rpc.send_msg.call_count == 0

    create_mock_trades(fee)
    trade = Trade.query.first()
    trade.is_open = True

    freqtrade.check_for_open_trades()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert 'Handle these trades manually' in freqtrade.rpc.send_msg.call_args[0][0]['status']


@pytest.mark.usefixtures("init_persistence")
def test_startup_update_open_orders(mocker, default_conf_usdt, fee, caplog):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades(fee)

    freqtrade.startup_update_open_orders()
    assert not log_has_re(r"Error updating Order .*", caplog)
    caplog.clear()

    freqtrade.config['dry_run'] = False
    freqtrade.startup_update_open_orders()

    assert log_has_re(r"Error updating Order .*", caplog)
    caplog.clear()

    assert len(Order.get_open_orders()) == 3
    matching_buy_order = mock_order_4()
    matching_buy_order.update({
        'status': 'closed',
    })
    mocker.patch('freqtrade.exchange.Exchange.fetch_order', return_value=matching_buy_order)
    freqtrade.startup_update_open_orders()
    # Only stoploss and sell orders are kept open
    assert len(Order.get_open_orders()) == 2


@pytest.mark.usefixtures("init_persistence")
def test_update_closed_trades_without_assigned_fees(mocker, default_conf_usdt, fee):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

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
def test_reupdate_enter_order_fees(mocker, default_conf_usdt, fee, caplog):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_uts = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.update_trade_state')

    create_mock_trades(fee)
    trades = Trade.get_trades().all()

    freqtrade.reupdate_enter_order_fees(trades[0])
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
        stake_amount=60.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=arrow.utcnow().datetime,
        is_open=True,
        amount=30,
        open_rate=2.0,
        exchange='binance',
    )
    Trade.query.session.add(trade)

    freqtrade.reupdate_enter_order_fees(trade)
    assert log_has_re(r"Trying to reupdate buy fees for .*", caplog)
    assert mock_uts.call_count == 0
    assert not log_has_re(r"Updating buy-fee on trade .* for order .*\.", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_handle_insufficient_funds(mocker, default_conf_usdt, fee):
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_rlo = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.refind_lost_order')
    mock_bof = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.reupdate_enter_order_fees')
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
def test_refind_lost_order(mocker, default_conf_usdt, fee, caplog):
    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
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
