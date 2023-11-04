from datetime import datetime, timedelta, timezone

from freqtrade.persistence.models import Order, Trade


MOCK_TRADE_COUNT = 6


def entry_side(is_short: bool):
    return "sell" if is_short else "buy"


def exit_side(is_short: bool):
    return "buy" if is_short else "sell"


def direc(is_short: bool):
    return "short" if is_short else "long"


def mock_order_usdt_1(is_short: bool):
    return {
        'id': f'prod_entry_1_{direc(is_short)}',
        'symbol': 'LTC/USDT',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 10.0,
        'amount': 2.0,
        'filled': 2.0,
        'remaining': 0.0,
    }


def mock_order_usdt_1_exit(is_short: bool):
    return {
        'id': f'prod_exit_1_{direc(is_short)}',
        'symbol': 'LTC/USDT',
        'status': 'open',
        'side': exit_side(is_short),
        'type': 'limit',
        'price': 8.0,
        'amount': 2.0,
        'filled': 0.0,
        'remaining': 2.0,
    }


def mock_trade_usdt_1(fee, is_short: bool):
    """
    Simulate prod entry with open sell order
    """
    trade = Trade(
        pair='LTC/USDT',
        stake_amount=20.0,
        amount=2.0,
        amount_requested=2.0,
        open_date=datetime.now(tz=timezone.utc) - timedelta(days=2, minutes=20),
        close_date=datetime.now(tz=timezone.utc) - timedelta(days=2, minutes=5),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=False,
        open_rate=10.0,
        close_rate=8.0,
        close_profit=-0.2,
        close_profit_abs=-4.09,
        exchange='binance',
        strategy='SampleStrategy',
        timeframe=5,
        is_short=is_short,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_1(is_short), 'LTC/USDT', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_usdt_1_exit(is_short),
                                     'LTC/USDT', exit_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_usdt_2(is_short: bool):
    return {
        'id': f'1235_{direc(is_short)}',
        'symbol': 'NEO/USDT',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 2.0,
        'amount': 100.0,
        'filled': 100.0,
        'remaining': 0.0,
    }


def mock_order_usdt_2_exit(is_short: bool):
    return {
        'id': f'12366_{direc(is_short)}',
        'symbol': 'NEO/USDT',
        'status': 'open',
        'side': exit_side(is_short),
        'type': 'limit',
        'price': 2.05,
        'amount': 100.0,
        'filled': 0.0,
        'remaining': 100.0,
    }


def mock_trade_usdt_2(fee, is_short: bool):
    """
    Closed trade...
    """
    trade = Trade(
        pair='NEO/USDT',
        stake_amount=200.0,
        amount=100.0,
        amount_requested=100.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=2.0,
        close_rate=2.05,
        close_profit=0.05,
        close_profit_abs=3.9875,
        exchange='binance',
        is_open=False,
        strategy='StrategyTestV2',
        timeframe=5,
        enter_tag='TEST1',
        exit_reason='exit_signal',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc) - timedelta(minutes=2),
        is_short=is_short,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_2(is_short), 'NEO/USDT', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(
        mock_order_usdt_2_exit(is_short), 'NEO/USDT', exit_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_usdt_3(is_short: bool):
    return {
        'id': f'41231a12a_{direc(is_short)}',
        'symbol': 'XRP/USDT',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 1.0,
        'amount': 30.0,
        'filled': 30.0,
        'remaining': 0.0,
    }


def mock_order_usdt_3_exit(is_short: bool):
    return {
        'id': f'41231a666a_{direc(is_short)}',
        'symbol': 'XRP/USDT',
        'status': 'closed',
        'side': exit_side(is_short),
        'type': 'stop_loss_limit',
        'price': 1.1,
        'average': 1.1,
        'amount': 30.0,
        'filled': 30.0,
        'remaining': 0.0,
    }


def mock_trade_usdt_3(fee, is_short: bool):
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
        close_profit=0.1,
        close_profit_abs=2.8425,
        exchange='binance',
        is_open=False,
        strategy='StrategyTestV2',
        timeframe=5,
        enter_tag='TEST3',
        exit_reason='roi',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc),
        is_short=is_short,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_3(is_short), 'XRP/USDT', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_usdt_3_exit(is_short),
                                     'XRP/USDT', exit_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_usdt_4(is_short: bool):
    return {
        'id': f'prod_buy_12345_{direc(is_short)}',
        'symbol': 'NEO/USDT',
        'status': 'open',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 2.0,
        'amount': 10.0,
        'filled': 0.0,
        'remaining': 30.0,
    }


def mock_trade_usdt_4(fee, is_short: bool):
    """
    Simulate prod entry
    """
    trade = Trade(
        pair='NEO/USDT',
        stake_amount=20.0,
        amount=10.0,
        amount_requested=10.01,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=14),
        is_open=True,
        open_rate=2.0,
        exchange='binance',
        strategy='StrategyTestV2',
        timeframe=5,
        is_short=is_short,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_4(is_short), 'NEO/USDT', entry_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_usdt_5(is_short: bool):
    return {
        'id': f'prod_buy_3455_{direc(is_short)}',
        'symbol': 'XRP/USDT',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 2.0,
        'amount': 10.0,
        'filled': 10.0,
        'remaining': 0.0,
    }


def mock_order_usdt_5_stoploss(is_short: bool):
    return {
        'id': f'prod_stoploss_3455_{direc(is_short)}',
        'symbol': 'XRP/USDT',
        'status': 'open',
        'side': exit_side(is_short),
        'type': 'stop_loss_limit',
        'price': 2.0,
        'amount': 10.0,
        'filled': 0.0,
        'remaining': 30.0,
    }


def mock_trade_usdt_5(fee, is_short: bool):
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
        stoploss_order_id=f'prod_stoploss_3455_{direc(is_short)}',
        timeframe=5,
        is_short=is_short,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_5(is_short), 'XRP/USDT', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_usdt_5_stoploss(is_short), 'XRP/USDT', 'stoploss')
    trade.orders.append(o)
    return trade


def mock_order_usdt_6(is_short: bool):
    return {
        'id': f'prod_entry_6_{direc(is_short)}',
        'symbol': 'LTC/USDT',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 10.0,
        'amount': 2.0,
        'filled': 2.0,
        'remaining': 0.0,
    }


def mock_order_usdt_6_exit(is_short: bool):
    return {
        'id': f'prod_exit_6_{direc(is_short)}',
        'symbol': 'LTC/USDT',
        'status': 'open',
        'side': exit_side(is_short),
        'type': 'limit',
        'price': 12.0,
        'amount': 2.0,
        'filled': 0.0,
        'remaining': 2.0,
    }


def mock_trade_usdt_6(fee, is_short: bool):
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
        timeframe=5,
        is_short=is_short,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_6(is_short), 'LTC/USDT', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_usdt_6_exit(is_short),
                                     'LTC/USDT', exit_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_usdt_7(is_short: bool):
    return {
        'id': f'1234_{direc(is_short)}',
        'symbol': 'ADA/USDT',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 2.0,
        'amount': 10.0,
        'filled': 10.0,
        'remaining': 0.0,
    }


def mock_trade_usdt_7(fee, is_short: bool):
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
        strategy='StrategyTestV2',
        timeframe=5,
        is_short=is_short,
    )
    o = Order.parse_from_ccxt_object(mock_order_usdt_7(is_short), 'ADA/USDT', entry_side(is_short))
    trade.orders.append(o)
    return trade
