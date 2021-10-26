from datetime import datetime, timedelta, timezone

from freqtrade.persistence.models import Order, Trade


MOCK_TRADE_COUNT = 3


def mock_order_1():
    return {
        'id': 'prod_buy_1',
        'symbol': 'LTC/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.15,
        'amount': 2.0,
        'filled': 2.0,
        'remaining': 0.0,
    }


def mock_order_1_sell():
    return {
        'id': 'prod_sell_1',
        'symbol': 'LTC/BTC',
        'status': 'open',
        'side': 'sell',
        'type': 'limit',
        'price': 0.20,
        'amount': 2.0,
        'filled': 0.0,
        'remaining': 2.0,
    }


def mock_trade_tags_1(fee):
    trade = Trade(
        pair='LTC/BTC',
        stake_amount=0.001,
        amount=2.0,
        amount_requested=2.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=15),
        open_rate=0.15,
        exchange='binance',
        open_order_id='dry_run_buy_123455',
        strategy='StrategyTestV2',
        timeframe=5,
        buy_tag="BUY_TAG1",
        sell_reason="SELL_REASON2"
    )
    o = Order.parse_from_ccxt_object(mock_order_1(), 'LTC/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_1_sell(), 'LTC/BTC', 'sell')
    trade.orders.append(o)
    return trade


def mock_order_2():
    return {
        'id': '1239',
        'symbol': 'LTC/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.120,
        'amount': 100.0,
        'filled': 100.0,
        'remaining': 0.0,
    }


def mock_order_2_sell():
    return {
        'id': '12392',
        'symbol': 'LTC/BTC',
        'status': 'closed',
        'side': 'sell',
        'type': 'limit',
        'price': 0.138,
        'amount': 100.0,
        'filled': 100.0,
        'remaining': 0.0,
    }


def mock_trade_tags_2(fee):
    trade = Trade(
        pair='LTC/BTC',
        stake_amount=0.001,
        amount=100.0,
        amount_requested=100.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=13),
        open_rate=0.120,
        exchange='binance',
        open_order_id='dry_run_buy_123456',
        strategy='StrategyTestV2',
        timeframe=5,
        buy_tag="BUY_TAG2",
        sell_reason="SELL_REASON1"
    )
    o = Order.parse_from_ccxt_object(mock_order_2(), 'LTC/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_2_sell(), 'LTC/BTC', 'sell')
    trade.orders.append(o)
    return trade


def mock_order_3():
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


def mock_order_3_sell():
    return {
        'id': '12352',
        'symbol': 'ETC/BTC',
        'status': 'closed',
        'side': 'sell',
        'type': 'limit',
        'price': 0.128,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
    }


def mock_trade_tags_3(fee):
    trade = Trade(
        pair='ETC/BTC',
        stake_amount=0.001,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=12),
        open_rate=0.123,
        exchange='binance',
        open_order_id='dry_run_buy_123457',
        strategy='StrategyTestV2',
        timeframe=5,
        buy_tag="BUY_TAG1",
        sell_reason="SELL_REASON2"
    )
    o = Order.parse_from_ccxt_object(mock_order_3(), 'ETC/BTC', 'buy')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_3_sell(), 'ETC/BTC', 'sell')
    trade.orders.append(o)
    return trade
