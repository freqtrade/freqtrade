
from unittest.mock import MagicMock, PropertyMock


from freqtrade.persistence import Trade
from freqtrade.strategy.interface import SellCheckTuple, SellType
from tests.conftest import (patch_exchange, get_patched_freqtradebot,
                            patch_get_signal)


def test_may_execute_sell_stoploss_on_exchange_multi(default_conf,
                                                     ticker, fee,
                                                     limit_buy_order,
                                                     markets, mocker) -> None:
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
        get_ticker=ticker,
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

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    # Switch ordertype to market to close trade immediately
    freqtrade.strategy.order_types['sell'] = 'market'
    patch_get_signal(freqtrade)

    # Create some test data
    freqtrade.create_trades()
    wallets_mock.reset_mock()
    Trade.session = MagicMock()

    trades = Trade.query.all()
    # Make sure stoploss-order is open and trade is bought (since we mock update_trade_state)
    for trade in trades:
        trade.stoploss_order_id = 3
        trade.open_order_id = None

    freqtrade.process_maybe_execute_sells(trades)
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
