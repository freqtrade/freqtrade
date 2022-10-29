from unittest.mock import MagicMock

import pytest

from freqtrade.enums import ExitCheckTuple, ExitType, TradingMode
from freqtrade.persistence import Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc.rpc import RPC
from tests.conftest import get_patched_freqtradebot, log_has_re, patch_get_signal


def test_may_execute_exit_stoploss_on_exchange_multi(default_conf, ticker, fee,
                                                     limit_buy_order, mocker) -> None:
    """
    Tests workflow of selling stoploss_on_exchange.
    Sells
    * first trade as stoploss
    * 2nd trade is kept
    * 3rd trade is sold via sell-signal
    """
    default_conf['max_open_trades'] = 3
    default_conf['exchange']['name'] = 'binance'

    stoploss = {
        'id': 123,
        'info': {}
    }
    stoploss_order_open = {
        "id": "123",
        "timestamp": 1542707426845,
        "datetime": "2018-11-20T09:50:26.845Z",
        "lastTradeTimestamp": None,
        "symbol": "BTC/USDT",
        "type": "stop_loss_limit",
        "side": "sell",
        "price": 1.08801,
        "amount": 90.99181074,
        "cost": 0.0,
        "average": 0.0,
        "filled": 0.0,
        "remaining": 0.0,
        "status": "open",
        "fee": None,
        "trades": None
    }
    stoploss_order_closed = stoploss_order_open.copy()
    stoploss_order_closed['status'] = 'closed'
    stoploss_order_closed['filled'] = stoploss_order_closed['amount']

    # Sell first trade based on stoploss, keep 2nd and 3rd trade open
    stoploss_order_mock = MagicMock(
        side_effect=[stoploss_order_closed, stoploss_order_open, stoploss_order_open])
    # Sell 3rd trade (not called for the first trade)
    should_sell_mock = MagicMock(side_effect=[
        [],
        [ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL)]]
    )
    cancel_order_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Binance.stoploss', stoploss)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
        fetch_stoploss_order=stoploss_order_mock,
        cancel_stoploss_order_with_result=cancel_order_mock,
    )

    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        create_stoploss_order=MagicMock(return_value=True),
        _notify_exit=MagicMock(),
    )
    mocker.patch("freqtrade.strategy.interface.IStrategy.should_exit", should_sell_mock)
    wallets_mock = mocker.patch("freqtrade.wallets.Wallets.update", MagicMock())
    mocker.patch("freqtrade.wallets.Wallets.get_free", MagicMock(return_value=1000))

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    # Switch ordertype to market to close trade immediately
    freqtrade.strategy.order_types['exit'] = 'market'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()
    assert freqtrade.strategy.confirm_trade_entry.call_count == 3
    freqtrade.strategy.confirm_trade_entry.reset_mock()
    assert freqtrade.strategy.confirm_trade_exit.call_count == 0
    wallets_mock.reset_mock()

    trades = Trade.query.all()
    # Make sure stoploss-order is open and trade is bought (since we mock update_trade_state)
    for trade in trades:
        stoploss_order_closed['id'] = '3'
        oobj = Order.parse_from_ccxt_object(stoploss_order_closed, trade.pair, 'stoploss')

        trade.orders.append(oobj)
        trade.stoploss_order_id = '3'
        trade.open_order_id = None

    n = freqtrade.exit_positions(trades)
    assert n == 2
    assert should_sell_mock.call_count == 2
    assert freqtrade.strategy.confirm_trade_entry.call_count == 0
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1
    freqtrade.strategy.confirm_trade_exit.reset_mock()

    # Only order for 3rd trade needs to be cancelled
    assert cancel_order_mock.call_count == 1
    # Wallets must be updated between stoploss cancellation and selling, and will be updated again
    # during update_trade_state
    assert wallets_mock.call_count == 4

    trade = trades[0]
    assert trade.exit_reason == ExitType.STOPLOSS_ON_EXCHANGE.value
    assert not trade.is_open

    trade = trades[1]
    assert not trade.exit_reason
    assert trade.is_open

    trade = trades[2]
    assert trade.exit_reason == ExitType.EXIT_SIGNAL.value
    assert not trade.is_open


@pytest.mark.parametrize("balance_ratio,result1", [
                        (1, 200),
                        (0.99, 198),
])
def test_forcebuy_last_unlimited(default_conf, ticker, fee, mocker, balance_ratio, result1) -> None:
    """
    Tests workflow unlimited stake-amount
    Buy 4 trades, forcebuy a 5th trade
    Sell one trade, calculated stake amount should now be lower than before since
    one trade was sold at a loss.
    """
    default_conf['max_open_trades'] = 5
    default_conf['force_entry_enable'] = True
    default_conf['stake_amount'] = 'unlimited'
    default_conf['tradable_balance_ratio'] = balance_ratio
    default_conf['dry_run_wallet'] = 1000
    default_conf['exchange']['name'] = 'binance'
    default_conf['telegram']['enabled'] = True
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
    )

    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        create_stoploss_order=MagicMock(return_value=True),
        _notify_exit=MagicMock(),
    )
    should_sell_mock = MagicMock(side_effect=[
        [],
        [ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL)],
        [],
        [],
        []]
    )
    mocker.patch("freqtrade.strategy.interface.IStrategy.should_exit", should_sell_mock)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtrade)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    # Switch ordertype to market to close trade immediately
    freqtrade.strategy.order_types['exit'] = 'market'
    patch_get_signal(freqtrade)

    # Create 4 trades
    n = freqtrade.enter_positions()
    assert n == 4

    trades = Trade.query.all()
    assert len(trades) == 4
    assert freqtrade.wallets.get_trade_stake_amount('XRP/BTC') == result1

    rpc._rpc_force_entry('TKN/BTC', None)

    trades = Trade.query.all()
    assert len(trades) == 5

    for trade in trades:
        assert pytest.approx(trade.stake_amount) == result1
        # Reset trade open order id's
        trade.open_order_id = None
    trades = Trade.get_open_trades()
    assert len(trades) == 5
    bals = freqtrade.wallets.get_all_balances()

    n = freqtrade.exit_positions(trades)
    assert n == 1
    trades = Trade.get_open_trades()
    # One trade sold
    assert len(trades) == 4
    # stake-amount should now be reduced, since one trade was sold at a loss.
    assert freqtrade.wallets.get_trade_stake_amount('XRP/BTC') < result1
    # Validate that balance of sold trade is not in dry-run balances anymore.
    bals2 = freqtrade.wallets.get_all_balances()
    assert bals != bals2
    assert len(bals) == 6
    assert len(bals2) == 5
    assert 'LTC' in bals
    assert 'LTC' not in bals2


def test_dca_buying(default_conf_usdt, ticker_usdt, fee, mocker) -> None:
    default_conf_usdt['position_adjustment_enable'] = True

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )

    patch_get_signal(freqtrade)
    freqtrade.enter_positions()

    assert len(Trade.get_trades().all()) == 1
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60
    assert trade.open_rate == 2.0
    # No adjustment
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60

    # Reduce bid amount
    ticker_usdt_modif = ticker_usdt.return_value
    ticker_usdt_modif['bid'] = ticker_usdt_modif['bid'] * 0.995
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', return_value=ticker_usdt_modif)

    # additional buy order
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    for o in trade.orders:
        assert o.status == "closed"
    assert pytest.approx(trade.stake_amount) == 120

    # Open-rate averaged between 2.0 and 2.0 * 0.995
    assert trade.open_rate < 2.0
    assert trade.open_rate > 2.0 * 0.995

    # No action - profit raised above 1% (the bar set in the strategy).
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert pytest.approx(trade.stake_amount) == 120
    assert trade.orders[0].amount == 30
    assert pytest.approx(trade.orders[1].amount) == 60 / ticker_usdt_modif['bid']

    assert pytest.approx(trade.amount) == trade.orders[0].amount + trade.orders[1].amount
    assert trade.nr_of_successful_buys == 2
    assert trade.nr_of_successful_entries == 2

    # Sell
    patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert trade.is_open is False
    assert trade.orders[0].amount == 30
    assert trade.orders[0].side == 'buy'
    assert pytest.approx(trade.orders[1].amount) == 60 / ticker_usdt_modif['bid']
    # Sold everything
    assert trade.orders[-1].side == 'sell'
    assert trade.orders[2].amount == trade.amount

    assert trade.nr_of_successful_buys == 2
    assert trade.nr_of_successful_entries == 2


def test_dca_short(default_conf_usdt, ticker_usdt, fee, mocker) -> None:
    default_conf_usdt['position_adjustment_enable'] = True

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: round(y, 4),
        price_to_precision=lambda s, x, y: y,
    )

    patch_get_signal(freqtrade, enter_long=False, enter_short=True)
    freqtrade.enter_positions()

    assert len(Trade.get_trades().all()) == 1
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60
    assert trade.open_rate == 2.02
    assert trade.orders[0].amount == trade.amount
    # No adjustment
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60

    # Reduce bid amount
    ticker_usdt_modif = ticker_usdt.return_value
    ticker_usdt_modif['ask'] = ticker_usdt_modif['ask'] * 1.004
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker', return_value=ticker_usdt_modif)

    # additional buy order
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    for o in trade.orders:
        assert o.status == "closed"
    assert pytest.approx(trade.stake_amount) == 120

    # Open-rate averaged between 2.0 and 2.0 * 1.015
    assert trade.open_rate >= 2.02
    assert trade.open_rate < 2.02 * 1.015

    # No action - profit raised above 1% (the bar set in the strategy).
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert pytest.approx(trade.stake_amount) == 120
    assert trade.orders[1].amount == round(60 / ticker_usdt_modif['ask'], 4)

    assert trade.amount == trade.orders[0].amount + trade.orders[1].amount
    assert trade.nr_of_successful_entries == 2

    # Buy
    patch_get_signal(freqtrade, enter_long=False, exit_short=True)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert trade.is_open is False
    # assert trade.orders[0].amount == 30
    assert trade.orders[0].side == 'sell'
    assert trade.orders[1].amount == round(60 / ticker_usdt_modif['ask'], 4)
    # Sold everything
    assert trade.orders[-1].side == 'buy'
    assert trade.orders[2].amount == trade.amount

    assert trade.nr_of_successful_entries == 2
    assert trade.nr_of_successful_exits == 1


@pytest.mark.parametrize('leverage', [
    1, 2
])
def test_dca_order_adjust(default_conf_usdt, ticker_usdt, leverage, fee, mocker) -> None:
    default_conf_usdt['position_adjustment_enable'] = True
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
    )
    mocker.patch('freqtrade.exchange.Exchange._is_dry_limit_order_filled', return_value=False)
    mocker.patch("freqtrade.exchange.Exchange.get_max_leverage", return_value=10)
    mocker.patch("freqtrade.exchange.Exchange.get_funding_fees", return_value=0)
    mocker.patch("freqtrade.exchange.Exchange.get_maintenance_ratio_and_amt", return_value=(0, 0))

    patch_get_signal(freqtrade)
    freqtrade.strategy.custom_entry_price = lambda **kwargs: ticker_usdt['ask'] * 0.96
    freqtrade.strategy.leverage = MagicMock(return_value=leverage)
    freqtrade.strategy.minimal_roi = {0: 0.2}

    freqtrade.enter_positions()

    assert len(Trade.get_trades().all()) == 1
    trade: Trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert trade.open_order_id is not None
    assert pytest.approx(trade.stake_amount) == 60
    assert trade.open_rate == 1.96
    assert trade.stop_loss_pct is None
    assert trade.stop_loss == 0.0
    assert trade.leverage == leverage
    assert trade.stake_amount == 60
    assert trade.initial_stop_loss == 0.0
    assert trade.initial_stop_loss_pct is None
    # No adjustment
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert trade.open_order_id is not None
    assert pytest.approx(trade.stake_amount) == 60

    # Cancel order and place new one
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=1.99)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert trade.open_order_id is not None
    # Open rate is not adjusted yet
    assert trade.open_rate == 1.96
    assert trade.stop_loss_pct is None
    assert trade.stop_loss == 0.0
    assert trade.stake_amount == 60
    assert trade.initial_stop_loss == 0.0
    assert trade.initial_stop_loss_pct is None

    # Fill order
    mocker.patch('freqtrade.exchange.Exchange._is_dry_limit_order_filled', return_value=True)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert trade.open_order_id is None
    # Open rate is not adjusted yet
    assert trade.open_rate == 1.99
    assert pytest.approx(trade.stake_amount) == 60
    assert trade.stop_loss_pct == -0.1
    assert pytest.approx(trade.stop_loss) == 1.99 * (1 - 0.1 / leverage)
    assert pytest.approx(trade.initial_stop_loss) == 1.99 * (1 - 0.1 / leverage)
    assert trade.initial_stop_loss_pct == -0.1

    # 2nd order - not filling
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=120)
    mocker.patch('freqtrade.exchange.Exchange._is_dry_limit_order_filled', return_value=False)

    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 3
    assert trade.open_order_id is not None
    assert trade.open_rate == 1.99
    assert trade.orders[-1].price == 1.96
    assert trade.orders[-1].cost == 120 * leverage

    # Replace new order with diff. order at a lower price
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=1.95)

    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 4
    assert trade.open_order_id is not None
    assert trade.open_rate == 1.99
    assert pytest.approx(trade.stake_amount) == 60
    assert trade.orders[-1].price == 1.95
    assert pytest.approx(trade.orders[-1].cost) == 120 * leverage

    # Fill DCA order
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=None)
    mocker.patch('freqtrade.exchange.Exchange._is_dry_limit_order_filled', return_value=True)
    freqtrade.strategy.adjust_entry_price = MagicMock(side_effect=ValueError)

    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 4
    assert trade.open_order_id is None
    assert pytest.approx(trade.open_rate) == 1.963153456
    assert trade.orders[-1].price == 1.95
    assert pytest.approx(trade.orders[-1].cost) == 120 * leverage
    assert trade.orders[-1].status == 'closed'

    assert pytest.approx(trade.amount) == 91.689215 * leverage
    # Check the 2 filled orders equal the above amount
    assert pytest.approx(trade.orders[1].amount) == 30.150753768 * leverage
    assert pytest.approx(trade.orders[-1].amount) == 61.538461232 * leverage


@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_exiting(default_conf_usdt, ticker_usdt, fee, mocker, caplog, leverage) -> None:
    default_conf_usdt['position_adjustment_enable'] = True

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.trading_mode = TradingMode.FUTURES
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
        get_min_pair_stake_amount=MagicMock(return_value=10),
    )
    mocker.patch("freqtrade.exchange.Exchange.get_max_leverage", return_value=10)

    patch_get_signal(freqtrade)
    freqtrade.strategy.leverage = MagicMock(return_value=leverage)
    freqtrade.enter_positions()

    assert len(Trade.get_trades().all()) == 1
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60
    assert pytest.approx(trade.amount) == 30.0 * leverage
    assert trade.open_rate == 2.0

    # Too small size
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=-59)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60
    assert pytest.approx(trade.amount) == 30.0 * leverage
    assert log_has_re(
        r"Remaining amount of \d\.\d+.* would be smaller than the minimum of 10.", caplog)

    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=-20)

    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert trade.orders[-1].ft_order_side == 'sell'
    assert pytest.approx(trade.stake_amount) == 40.198
    assert pytest.approx(trade.amount) == 20.099 * leverage
    assert trade.open_rate == 2.0
    assert trade.is_open
    caplog.clear()

    # Sell more than what we got (we got ~20 coins left)
    # First adjusts the amount to 20 - then rejects.
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=-50)
    freqtrade.process()
    assert log_has_re("Adjusting amount to trade.amount as it is higher.*", caplog)
    assert log_has_re("Remaining amount of 0.0 would be smaller than the minimum of 10.", caplog)
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert trade.orders[-1].ft_order_side == 'sell'
    assert pytest.approx(trade.stake_amount) == 40.198
    assert trade.is_open

    # use amount that would trunc to 0.0 once selling
    mocker.patch("freqtrade.exchange.Exchange.amount_to_contract_precision",
                 lambda s, p, v: round(v, 1))
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=-0.01)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert trade.orders[-1].ft_order_side == 'sell'
    assert pytest.approx(trade.stake_amount) == 40.198
    assert trade.is_open
    assert log_has_re('Amount to exit is 0.0 due to exchange limits - not exiting.', caplog)
