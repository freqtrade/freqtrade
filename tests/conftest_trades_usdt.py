from datetime import datetime, timedelta, timezone

from freqtrade.persistence.models import Order, Trade


MOCK_TRADE_COUNT = 6


def mock_order_usdt_1():
    return {
        'id': '1234',
        'symbol': 'ADA/USDT',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 2.0,
        'amount': 10.0,
        'filled': 10.0,
        'remaining': 0.0,
    }


def mock_trade_usdt_1(fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=20.0,
        amount=10.0,
        amount_requested=10.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=17),
        open_rate=2.0,
        exchange='binance',
        open_order_id='dry_run_buy_12345',
        strategy='StrategyTestV2',
        timeframe=5,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_1(), 'ADA/USDT', 'buy')
    trade.orders.append(o)
    return trade


def mock_order_usdt_2():
    return {
        'id': '1235',
        'symbol': 'ETC/USDT',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 2.0,
        'amount': 100.0,
        'filled': 100.0,
        'remaining': 0.0,
    }


def mock_order_usdt_2_sell():
    return {
        'id': '12366',
        'symbol': 'ETC/USDT',
        'status': 'closed',
        'side': 'sell',
        'type': 'limit',
        'price': 2.05,
        'amount': 100.0,
        'filled': 100.0,
        'remaining': 0.0,
    }


def mock_trade_usdt_2(fee):
    """
    Closed trade...
    """
    trade = Trade(
        pair='ETC/USDT',
        stake_amount=200.0,
        amount=100.0,
        amount_requested=100.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=2.0,
        close_rate=2.05,
        close_profit=5.0,
        close_profit_abs=3.9875,
        exchange='binance',
        is_open=False,
        open_order_id='dry_run_sell_12345',
        strategy='StrategyTestV2',
        timeframe=5,
        sell_reason='sell_signal',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc) - timedelta(minutes=2),
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_2(), 'ETC/USDT', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_usdt_2_sell(), 'ETC/USDT', 'sell')
    trade.orders.append(o)
    return trade


def mock_order_usdt_3():
    return {
        'id': '41231a12a',
        'symbol': 'XRP/USDT',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 1.0,
        'amount': 30.0,
        'filled': 30.0,
        'remaining': 0.0,
    }


def mock_order_usdt_3_sell():
    return {
        'id': '41231a666a',
        'symbol': 'XRP/USDT',
        'status': 'closed',
        'side': 'sell',
        'type': 'stop_loss_limit',
        'price': 1.1,
        'average': 1.1,
        'amount': 30.0,
        'filled': 30.0,
        'remaining': 0.0,
    }


def mock_trade_usdt_3(fee):
    """
    Closed trade
    """
    trade = Trade(
        pair='XRP/USDT',
        stake_amount=30.0,
        amount=30.0,
        amount_requested=30.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=1.0,
        close_rate=1.1,
        close_profit=10.0,
        close_profit_abs=9.8425,
        exchange='binance',
        is_open=False,
        strategy='StrategyTestV2',
        timeframe=5,
        sell_reason='roi',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc),
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_3(), 'XRP/USDT', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_usdt_3_sell(), 'XRP/USDT', 'sell')
    trade.orders.append(o)
    return trade


def mock_order_usdt_4():
    return {
        'id': 'prod_buy_12345',
        'symbol': 'ETC/USDT',
        'status': 'open',
        'side': 'buy',
        'type': 'limit',
        'price': 2.0,
        'amount': 10.0,
        'filled': 0.0,
        'remaining': 30.0,
    }


def mock_trade_usdt_4(fee):
    """
    Simulate prod entry
    """
    trade = Trade(
        pair='ETC/USDT',
        stake_amount=20.0,
        amount=10.0,
        amount_requested=10.01,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=14),
        is_open=True,
        open_rate=2.0,
        exchange='binance',
        open_order_id='prod_buy_12345',
        strategy='StrategyTestV2',
        timeframe=5,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_4(), 'ETC/USDT', 'buy')
    trade.orders.append(o)
    return trade


def mock_order_usdt_5():
    return {
        'id': 'prod_buy_3455',
        'symbol': 'XRP/USDT',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 2.0,
        'amount': 10.0,
        'filled': 10.0,
        'remaining': 0.0,
    }


def mock_order_usdt_5_stoploss():
    return {
        'id': 'prod_stoploss_3455',
        'symbol': 'XRP/USDT',
        'status': 'open',
        'side': 'sell',
        'type': 'stop_loss_limit',
        'price': 2.0,
        'amount': 10.0,
        'filled': 0.0,
        'remaining': 30.0,
    }


def mock_trade_usdt_5(fee):
    """
    Simulate prod entry with stoploss
    """
    trade = Trade(
        pair='XRP/USDT',
        stake_amount=20.0,
        amount=10.0,
        amount_requested=10.01,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=12),
        is_open=True,
        open_rate=2.0,
        exchange='binance',
        strategy='SampleStrategy',
        stoploss_order_id='prod_stoploss_3455',
        timeframe=5,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_5(), 'XRP/USDT', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_usdt_5_stoploss(), 'XRP/USDT', 'stoploss')
    trade.orders.append(o)
    return trade


def mock_order_usdt_6():
    return {
        'id': 'prod_buy_6',
        'symbol': 'LTC/USDT',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 10.0,
        'amount': 2.0,
        'filled': 2.0,
        'remaining': 0.0,
    }


def mock_order_usdt_6_sell():
    return {
        'id': 'prod_sell_6',
        'symbol': 'LTC/USDT',
        'status': 'open',
        'side': 'sell',
        'type': 'limit',
        'price': 12.0,
        'amount': 2.0,
        'filled': 0.0,
        'remaining': 2.0,
    }


def mock_trade_usdt_6(fee):
    """
    Simulate prod entry with open sell order
    """
    trade = Trade(
        pair='LTC/USDT',
        stake_amount=20.0,
        amount=2.0,
        amount_requested=2.0,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=5),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_rate=10.0,
        exchange='binance',
        strategy='SampleStrategy',
        open_order_id="prod_sell_6",
        timeframe=5,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_6(), 'LTC/USDT', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_usdt_6_sell(), 'LTC/USDT', 'sell')
    trade.orders.append(o)
    return trade
