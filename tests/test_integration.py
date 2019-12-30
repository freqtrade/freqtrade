
from unittest.mock import MagicMock

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import SellCheckTuple, SellType
from tests.conftest import get_patched_freqtradebot, patch_get_signal
from freqtrade.rpc.rpc import RPC


def test_may_execute_sell_stoploss_on_exchange_multi(default_conf, ticker, fee,
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

    stoploss_limit = {
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
    # Sell first trade based on stoploss, keep 2nd and 3rd trade open
    stoploss_order_mock = MagicMock(
        side_effect=[stoploss_order_closed, stoploss_order_open, stoploss_order_open])
    # Sell 3rd trade (not called for the first trade)
    should_sell_mock = MagicMock(side_effect=[
        SellCheckTuple(sell_flag=False, sell_type=SellType.NONE),
        SellCheckTuple(sell_flag=True, sell_type=SellType.SELL_SIGNAL)]
    )
    cancel_order_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Binance.stoploss_limit', stoploss_limit)
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        symbol_amount_prec=lambda s, x, y: y,
        symbol_price_prec=lambda s, x, y: y,
        get_order=stoploss_order_mock,
        cancel_order=cancel_order_mock,
    )

    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        create_stoploss_order=MagicMock(return_value=True),
        update_trade_state=MagicMock(),
        _notify_sell=MagicMock(),
    )
    mocker.patch("freqtrade.strategy.interface.IStrategy.should_sell", should_sell_mock)
    wallets_mock = mocker.patch("freqtrade.wallets.Wallets.update", MagicMock())
    mocker.patch("freqtrade.wallets.Wallets.get_free", MagicMock(return_value=1000))

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    # Switch ordertype to market to close trade immediately
    freqtrade.strategy.order_types['sell'] = 'market'
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.enter_positions()
    wallets_mock.reset_mock()
    Trade.session = MagicMock()

    trades = Trade.query.all()
    # Make sure stoploss-order is open and trade is bought (since we mock update_trade_state)
    for trade in trades:
        trade.stoploss_order_id = 3
        trade.open_order_id = None

    n = freqtrade.exit_positions(trades)
    assert n == 2
    assert should_sell_mock.call_count == 2

    # Only order for 3rd trade needs to be cancelled
    assert cancel_order_mock.call_count == 1
    # Wallets should only be called once per sell cycle
    assert wallets_mock.call_count == 1

    trade = trades[0]
    assert trade.sell_reason == SellType.STOPLOSS_ON_EXCHANGE.value
    assert not trade.is_open

    trade = trades[1]
    assert not trade.sell_reason
    assert trade.is_open

    trade = trades[2]
    assert trade.sell_reason == SellType.SELL_SIGNAL.value
    assert not trade.is_open


def test_forcebuy_last_unlimited(default_conf, ticker, fee, limit_buy_order, mocker) -> None:
    """
    Tests workflow
    """
    default_conf['max_open_trades'] = 5
    default_conf['forcebuy_enable'] = True
    default_conf['stake_amount'] = 'unlimited'
    default_conf['dry_run_wallet'] = 1000
    default_conf['exchange']['name'] = 'binance'
    default_conf['telegram']['enabled'] = True
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        symbol_amount_prec=lambda s, x, y: y,
        symbol_price_prec=lambda s, x, y: y,
    )

    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        create_stoploss_order=MagicMock(return_value=True),
        update_trade_state=MagicMock(),
        _notify_sell=MagicMock(),
    )
    should_sell_mock = MagicMock(side_effect=[
        SellCheckTuple(sell_flag=False, sell_type=SellType.NONE),
        SellCheckTuple(sell_flag=True, sell_type=SellType.SELL_SIGNAL),
        SellCheckTuple(sell_flag=False, sell_type=SellType.NONE),
        SellCheckTuple(sell_flag=False, sell_type=SellType.NONE),
        SellCheckTuple(sell_flag=None, sell_type=SellType.NONE)]
    )
    mocker.patch("freqtrade.strategy.interface.IStrategy.should_sell", should_sell_mock)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtrade)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    # Switch ordertype to market to close trade immediately
    freqtrade.strategy.order_types['sell'] = 'market'
    patch_get_signal(freqtrade)

    # Create 4 trades
    n = freqtrade.enter_positions()
    assert n == 4

    trades = Trade.query.all()
    assert len(trades) == 4
    rpc._rpc_forcebuy('TKN/BTC', None)

    trades = Trade.query.all()
    assert len(trades) == 5

    for trade in trades:
        assert trade.stake_amount == 200
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
    # Validate that balance of sold trade is not in dry-run balances anymore.
    bals2 = freqtrade.wallets.get_all_balances()
    assert bals != bals2
    assert len(bals) == 6
    assert len(bals2) == 5
    assert 'LTC' in bals
    assert 'LTC' not in bals2
