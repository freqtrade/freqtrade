from freqtrade.persistence.models import Order, Trade


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
        open_rate=0.123,
        exchange='bittrex',
        open_order_id='dry_run_buy_12345',
        strategy='DefaultStrategy',
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
        exchange='bittrex',
        is_open=False,
        open_order_id='dry_run_sell_12345',
        strategy='DefaultStrategy',
    )
    o = Order.parse_from_ccxt_object(mock_order_2(), 'ETH/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_2_sell(), 'ETH/BTC', 'sell')
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
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.05,
        close_rate=0.06,
        close_profit=0.01,
        exchange='bittrex',
        is_open=False,
    )
    o = Order.parse_from_ccxt_object(mock_order_3(), 'ETH/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_3_sell(), 'ETH/BTC', 'sell')
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
        open_rate=0.123,
        exchange='bittrex',
        open_order_id='prod_buy_12345',
        strategy='DefaultStrategy',
    )
    o = Order.parse_from_ccxt_object(mock_order_4(), 'ETH/BTC', 'buy')
    trade.orders.append(o)
    return trade
