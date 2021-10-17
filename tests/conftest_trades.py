from datetime import datetime, timedelta, timezone

from freqtrade.persistence.models import Order, Trade


MOCK_TRADE_COUNT = 6


def mock_order_1():
    return {
        'id': '1234',
        'symbol': 'ETH/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
    }


def mock_trade_1(fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=17),
        open_rate=0.123,
        exchange='binance',
        open_order_id='dry_run_buy_12345',
        strategy='StrategyTestV2',
        timeframe=5,
    )
    o = Order.parse_from_ccxt_object(mock_order_1(), 'ETH/BTC', 'buy')
    trade.orders.append(o)
    return trade


def mock_order_2():
    return {
        'id': '1235',
        'symbol': 'ETC/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
    }


def mock_order_2_sell():
    return {
        'id': '12366',
        'symbol': 'ETC/BTC',
        'status': 'closed',
        'side': 'sell',
        'type': 'limit',
        'price': 0.128,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
    }


def mock_trade_2(fee):
    """
    Closed trade...
    """
    trade = Trade(
        pair='ETC/BTC',
        stake_amount=0.001,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        close_rate=0.128,
        close_profit=0.005,
        close_profit_abs=0.000584127,
        exchange='binance',
        is_open=False,
        open_order_id='dry_run_sell_12345',
        strategy='StrategyTestV2',
        timeframe=5,
        sell_reason='sell_signal',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc) - timedelta(minutes=2),
    )
    o = Order.parse_from_ccxt_object(mock_order_2(), 'ETC/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_2_sell(), 'ETC/BTC', 'sell')
    trade.orders.append(o)
    return trade


def mock_order_3():
    return {
        'id': '41231a12a',
        'symbol': 'XRP/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.05,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
    }


def mock_order_3_sell():
    return {
        'id': '41231a666a',
        'symbol': 'XRP/BTC',
        'status': 'closed',
        'side': 'sell',
        'type': 'stop_loss_limit',
        'price': 0.06,
        'average': 0.06,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
    }


def mock_trade_3(fee):
    """
    Closed trade
    """
    trade = Trade(
        pair='XRP/BTC',
        stake_amount=0.001,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.05,
        close_rate=0.06,
        close_profit=0.01,
        close_profit_abs=0.000155,
        exchange='binance',
        is_open=False,
        strategy='StrategyTestV2',
        timeframe=5,
        sell_reason='roi',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc),
    )
    o = Order.parse_from_ccxt_object(mock_order_3(), 'XRP/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_3_sell(), 'XRP/BTC', 'sell')
    trade.orders.append(o)
    return trade


def mock_order_4():
    return {
        'id': 'prod_buy_12345',
        'symbol': 'ETC/BTC',
        'status': 'open',
        'side': 'buy',
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 0.0,
        'remaining': 123.0,
    }


def mock_trade_4(fee):
    """
    Simulate prod entry
    """
    trade = Trade(
        pair='ETC/BTC',
        stake_amount=0.001,
        amount=123.0,
        amount_requested=124.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=14),
        is_open=True,
        open_rate=0.123,
        exchange='binance',
        open_order_id='prod_buy_12345',
        strategy='StrategyTestV2',
        timeframe=5,
    )
    o = Order.parse_from_ccxt_object(mock_order_4(), 'ETC/BTC', 'buy')
    trade.orders.append(o)
    return trade


def mock_order_5():
    return {
        'id': 'prod_buy_3455',
        'symbol': 'XRP/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
    }


def mock_order_5_stoploss():
    return {
        'id': 'prod_stoploss_3455',
        'symbol': 'XRP/BTC',
        'status': 'open',
        'side': 'sell',
        'type': 'stop_loss_limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 0.0,
        'remaining': 123.0,
    }


def mock_trade_5(fee):
    """
    Simulate prod entry with stoploss
    """
    trade = Trade(
        pair='XRP/BTC',
        stake_amount=0.001,
        amount=123.0,
        amount_requested=124.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=12),
        is_open=True,
        open_rate=0.123,
        exchange='binance',
        strategy='SampleStrategy',
        stoploss_order_id='prod_stoploss_3455',
        timeframe=5,
    )
    o = Order.parse_from_ccxt_object(mock_order_5(), 'XRP/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_5_stoploss(), 'XRP/BTC', 'stoploss')
    trade.orders.append(o)
    return trade


def mock_order_6():
    return {
        'id': 'prod_buy_6',
        'symbol': 'LTC/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.15,
        'amount': 2.0,
        'filled': 2.0,
        'remaining': 0.0,
    }


def mock_order_6_sell():
    return {
        'id': 'prod_sell_6',
        'symbol': 'LTC/BTC',
        'status': 'open',
        'side': 'sell',
        'type': 'limit',
        'price': 0.20,
        'amount': 2.0,
        'filled': 0.0,
        'remaining': 2.0,
    }


def mock_trade_6(fee):
    """
    Simulate prod entry with open sell order
    """
    trade = Trade(
        pair='LTC/BTC',
        stake_amount=0.001,
        amount=2.0,
        amount_requested=2.0,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=5),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_rate=0.15,
        exchange='binance',
        strategy='SampleStrategy',
        open_order_id="prod_sell_6",
        timeframe=5,
    )
    o = Order.parse_from_ccxt_object(mock_order_6(), 'LTC/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_6_sell(), 'LTC/BTC', 'sell')
    trade.orders.append(o)
    return trade
