from datetime import datetime, timedelta, timezone

from freqtrade.persistence.models import Order, Trade


MOCK_TRADE_COUNT = 6


def entry_side(is_short: bool):
    return "sell" if is_short else "buy"


def exit_side(is_short: bool):
    return "buy" if is_short else "sell"


def direc(is_short: bool):
    return "short" if is_short else "long"


def mock_order_1(is_short: bool):
    return {
        'id': f'1234_{direc(is_short)}',
        'symbol': 'ETH/BTC',
        'status': 'open',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 0.123,
        'average': 0.123,
        'amount': 123.0,
        'filled': 50.0,
        'cost': 15.129,
        'remaining': 123.0 - 50.0,
    }


def mock_trade_1(fee, is_short: bool):
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
        open_order_id=f'dry_run_buy_{direc(is_short)}_12345',
        strategy='StrategyTestV3',
        timeframe=5,
        is_short=is_short
    )
    o = Order.parse_from_ccxt_object(mock_order_1(is_short), 'ETH/BTC', entry_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_2(is_short: bool):
    return {
        'id': f'1235_{direc(is_short)}',
        'symbol': 'ETC/BTC',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 123.0,
        'cost': 15.129,
        'remaining': 0.0,
    }


def mock_order_2_sell(is_short: bool):
    return {
        'id': f'12366_{direc(is_short)}',
        'symbol': 'ETC/BTC',
        'status': 'closed',
        'side': exit_side(is_short),
        'type': 'limit',
        'price': 0.128,
        'amount': 123.0,
        'filled': 123.0,
        'cost': 15.129,
        'remaining': 0.0,
    }


def mock_trade_2(fee, is_short: bool):
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
        close_profit=-0.005 if is_short else 0.005,
        close_profit_abs=-0.005584127 if is_short else 0.000584127,
        exchange='binance',
        is_open=False,
        strategy='StrategyTestV3',
        timeframe=5,
        enter_tag='TEST1',
        exit_reason='sell_signal',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc) - timedelta(minutes=2),
        is_short=is_short
    )
    o = Order.parse_from_ccxt_object(mock_order_2(is_short), 'ETC/BTC', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_2_sell(is_short), 'ETC/BTC', exit_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_3(is_short: bool):
    return {
        'id': f'41231a12a_{direc(is_short)}',
        'symbol': 'XRP/BTC',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 0.05,
        'amount': 123.0,
        'filled': 123.0,
        'cost': 15.129,
        'remaining': 0.0,
    }


def mock_order_3_sell(is_short: bool):
    return {
        'id': f'41231a666a_{direc(is_short)}',
        'symbol': 'XRP/BTC',
        'status': 'closed',
        'side': exit_side(is_short),
        'type': 'stop_loss_limit',
        'price': 0.06,
        'average': 0.06,
        'amount': 123.0,
        'filled': 123.0,
        'cost': 15.129,
        'remaining': 0.0,
    }


def mock_trade_3(fee, is_short: bool):
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
        close_profit=-0.01 if is_short else 0.01,
        close_profit_abs=-0.001155 if is_short else 0.000155,
        exchange='binance',
        is_open=False,
        strategy='StrategyTestV3',
        timeframe=5,
        exit_reason='roi',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc),
        is_short=is_short
    )
    o = Order.parse_from_ccxt_object(mock_order_3(is_short), 'XRP/BTC', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_3_sell(is_short), 'XRP/BTC', exit_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_4(is_short: bool):
    return {
        'id': f'prod_buy_{direc(is_short)}_12345',
        'symbol': 'ETC/BTC',
        'status': 'open',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 0.0,
        'cost': 15.129,
        'remaining': 123.0,
    }


def mock_trade_4(fee, is_short: bool):
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
        open_order_id=f'prod_buy_{direc(is_short)}_12345',
        strategy='StrategyTestV3',
        timeframe=5,
        is_short=is_short,
        stop_loss_pct=0.10
    )
    o = Order.parse_from_ccxt_object(mock_order_4(is_short), 'ETC/BTC', entry_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_5(is_short: bool):
    return {
        'id': f'prod_buy_{direc(is_short)}_3455',
        'symbol': 'XRP/BTC',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 123.0,
        'cost': 15.129,
        'remaining': 0.0,
    }


def mock_order_5_stoploss(is_short: bool):
    return {
        'id': f'prod_stoploss_{direc(is_short)}_3455',
        'symbol': 'XRP/BTC',
        'status': 'open',
        'side': exit_side(is_short),
        'type': 'stop_loss_limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 0.0,
        'cost': 0.0,
        'remaining': 123.0,
    }


def mock_trade_5(fee, is_short: bool):
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
        enter_tag='TEST1',
        stoploss_order_id=f'prod_stoploss_{direc(is_short)}_3455',
        timeframe=5,
        is_short=is_short,
        stop_loss_pct=0.10,
    )
    o = Order.parse_from_ccxt_object(mock_order_5(is_short), 'XRP/BTC', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_5_stoploss(is_short), 'XRP/BTC', 'stoploss')
    trade.orders.append(o)
    return trade


def mock_order_6(is_short: bool):
    return {
        'id': f'prod_buy_{direc(is_short)}_6',
        'symbol': 'LTC/BTC',
        'status': 'closed',
        'side': entry_side(is_short),
        'type': 'limit',
        'price': 0.15,
        'amount': 2.0,
        'filled': 2.0,
        'cost': 0.3,
        'remaining': 0.0,
    }


def mock_order_6_sell(is_short: bool):
    return {
        'id': f'prod_sell_{direc(is_short)}_6',
        'symbol': 'LTC/BTC',
        'status': 'open',
        'side': exit_side(is_short),
        'type': 'limit',
        'price': 0.15 if is_short else 0.20,
        'amount': 2.0,
        'filled': 0.0,
        'cost': 0.0,
        'remaining': 2.0,
    }


def mock_trade_6(fee, is_short: bool):
    """
    Simulate prod entry with open exit order
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
        enter_tag='TEST2',
        open_order_id=f"prod_sell_{direc(is_short)}_6",
        timeframe=5,
        is_short=is_short
    )
    o = Order.parse_from_ccxt_object(mock_order_6(is_short), 'LTC/BTC', entry_side(is_short))
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(mock_order_6_sell(is_short), 'LTC/BTC', exit_side(is_short))
    trade.orders.append(o)
    return trade


def short_order():
    return {
        'id': '1236',
        'symbol': 'ETC/BTC',
        'status': 'closed',
        'side': 'sell',
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 123.0,
        'cost': 15.129,
        'remaining': 0.0,
    }


def exit_short_order():
    return {
        'id': '12367',
        'symbol': 'ETC/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.128,
        'amount': 123.0,
        'filled': 123.0,
        'cost': 15.744,
        'remaining': 0.0,
    }


def short_trade(fee):
    """
        10 minute short limit trade on binance

        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per day
        open_rate: 0.123 base
        close_rate: 0.128 base
        amount: 123.0 crypto
        stake_amount: 15.129 base
        borrowed: 123.0  crypto
        time-periods: 10 minutes(rounds up to 1/24 time-period of 1 day)
        interest: borrowed * interest_rate * time-periods
                    = 123.0 * 0.0005 * 1/24 = 0.0025625 crypto
        open_value: (amount * open_rate) - (amount * open_rate * fee)
            = (123 * 0.123) - (123 * 0.123 * 0.0025)
            = 15.091177499999999
        amount_closed: amount + interest = 123 + 0.0025625 = 123.0025625
        close_value: (amount_closed * close_rate) + (amount_closed * close_rate * fee)
            = (123.0025625 * 0.128) + (123.0025625 * 0.128 * 0.0025)
            = 15.78368882
        total_profit = open_value - close_value
            = 15.091177499999999 - 15.78368882
            = -0.6925113200000013
        total_profit_percentage = total_profit / stake_amount
            = -0.6925113200000013 / 15.129
            = -0.04577376693766946

    """
    trade = Trade(
        pair='ETC/BTC',
        stake_amount=15.129,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        # close_rate=0.128,
        # close_profit=-0.04577376693766946,
        # close_profit_abs=-0.6925113200000013,
        exchange='binance',
        is_open=True,
        open_order_id=None,
        strategy='DefaultStrategy',
        timeframe=5,
        exit_reason='sell_signal',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        # close_date=datetime.now(tz=timezone.utc) - timedelta(minutes=2),
        is_short=True
    )
    o = Order.parse_from_ccxt_object(short_order(), 'ETC/BTC', 'sell')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(exit_short_order(), 'ETC/BTC', 'sell')
    trade.orders.append(o)
    return trade


def leverage_order():
    return {
        'id': '1237',
        'symbol': 'DOGE/BTC',
        'status': 'closed',
        'side': 'buy',
        'type': 'limit',
        'price': 0.123,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
        'cost': 15.129,
        'leverage': 5.0
    }


def leverage_order_sell():
    return {
        'id': '12368',
        'symbol': 'DOGE/BTC',
        'status': 'closed',
        'side': 'sell',
        'type': 'limit',
        'price': 0.128,
        'amount': 123.0,
        'filled': 123.0,
        'remaining': 0.0,
        'cost': 15.744,
        'leverage': 5.0
    }


def leverage_trade(fee):
    """
    5 hour short limit trade on kraken

        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per day
        open_rate: 0.123 base
        close_rate: 0.128 base
        amount: 615 crypto
        stake_amount: 15.129 base
        borrowed: 60.516  base
        leverage: 5
        hours: 5
        interest: borrowed * interest_rate * ceil(1 + hours/4)
                    = 60.516 * 0.0005 * ceil(1 + 5/4) = 0.090774 base
        open_value: (amount * open_rate) + (amount * open_rate * fee)
            = (615.0 * 0.123) + (615.0 * 0.123 * 0.0025)
            = 75.83411249999999

        close_value: (amount_closed * close_rate) - (amount_closed * close_rate * fee) - interest
            = (615.0 * 0.128) - (615.0 * 0.128 * 0.0025) - 0.090774
            = 78.432426
        total_profit = close_value - open_value
            = 78.432426 - 75.83411249999999
            = 2.5983135000000175
        total_profit_percentage = ((close_value/open_value)-1) * leverage
            = ((78.432426/75.83411249999999)-1) * 5
            = 0.1713156134055116
    """
    trade = Trade(
        pair='DOGE/BTC',
        stake_amount=15.129,
        amount=615.0,
        leverage=5.0,
        amount_requested=615.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        close_rate=0.128,
        close_profit=0.1713156134055116,
        close_profit_abs=2.5983135000000175,
        exchange='kraken',
        is_open=False,
        open_order_id='dry_run_leverage_buy_12368',
        strategy='DefaultStrategy',
        timeframe=5,
        exit_reason='sell_signal',
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=300),
        close_date=datetime.now(tz=timezone.utc),
        interest_rate=0.0005
    )
    o = Order.parse_from_ccxt_object(leverage_order(), 'DOGE/BTC', 'sell')
    trade.orders.append(o)
    o = Order.parse_from_ccxt_object(leverage_order_sell(), 'DOGE/BTC', 'sell')
    trade.orders.append(o)
    return trade
