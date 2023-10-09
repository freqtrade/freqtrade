import json
from datetime import datetime, timezone

import pytest

from freqtrade.persistence.trade_model import Trade
from tests.conftest import create_mock_trades_usdt


@pytest.mark.usefixtures("init_persistence")
def test_trade_fromjson():
    """Test the Trade.from_json() method."""
    trade_string = """{
        "trade_id": 25,
        "pair": "ETH/USDT",
        "base_currency": "ETH",
        "quote_currency": "USDT",
        "is_open": false,
        "exchange": "binance",
        "amount": 407.0,
        "amount_requested": 102.92547026,
        "stake_amount": 102.7494348,
        "strategy": "SampleStrategy55",
        "buy_tag": "Strategy2",
        "enter_tag": "Strategy2",
        "timeframe": 5,
        "fee_open": 0.001,
        "fee_open_cost": 0.1027494,
        "fee_open_currency": "ETH",
        "fee_close": 0.001,
        "fee_close_cost": 0.1054944,
        "fee_close_currency": "USDT",
        "open_date": "2022-10-18 09:12:42",
        "open_timestamp": 1666084362912,
        "open_rate": 0.2518998249562391,
        "open_rate_requested": 0.2516,
        "open_trade_value": 102.62575199,
        "close_date": "2022-10-18 09:45:22",
        "close_timestamp": 1666086322208,
        "realized_profit": 2.76315361,
        "close_rate": 0.2592,
        "close_rate_requested": 0.2592,
        "close_profit": 0.026865,
        "close_profit_pct": 2.69,
        "close_profit_abs": 2.76315361,
        "trade_duration_s": 1959,
        "trade_duration": 32,
        "profit_ratio": 0.02686,
        "profit_pct": 2.69,
        "profit_abs": 2.76315361,
        "sell_reason": "no longer good",
        "exit_reason": "no longer good",
        "exit_order_status": "closed",
        "stop_loss_abs": 0.1981,
        "stop_loss_ratio": -0.216,
        "stop_loss_pct": -21.6,
        "stoploss_order_id": null,
        "stoploss_last_update": "2022-10-18 09:13:42",
        "stoploss_last_update_timestamp": 1666077222000,
        "initial_stop_loss_abs": 0.1981,
        "initial_stop_loss_ratio": -0.216,
        "initial_stop_loss_pct": -21.6,
        "min_rate": 0.2495,
        "max_rate": 0.2592,
        "leverage": 1.0,
        "interest_rate": 0.0,
        "liquidation_price": null,
        "is_short": false,
        "trading_mode": "spot",
        "funding_fees": 0.0,
        "amount_precision": 1.0,
        "price_precision": 3.0,
        "precision_mode": 2,
        "contract_size": 1.0,
        "open_order_id": null,
        "orders": [
            {
                "amount": 102.0,
                "safe_price": 0.2526,
                "ft_order_side": "buy",
                "order_filled_timestamp": 1666084370887,
                "ft_is_entry": true,
                "pair": "ETH/USDT",
                "order_id": "78404228",
                "status": "closed",
                "average": 0.2526,
                "cost": 25.7652,
                "filled": 102.0,
                "is_open": false,
                "order_date": "2022-10-18 09:12:42",
                "order_timestamp": 1666084362684,
                "order_filled_date": "2022-10-18 09:12:50",
                "order_type": "limit",
                "price": 0.2526,
                "remaining": 0.0
            },
            {
                "amount": 102.0,
                "safe_price": 0.2517,
                "ft_order_side": "buy",
                "order_filled_timestamp": 1666084379056,
                "ft_is_entry": true,
                "pair": "ETH/USDT",
                "order_id": "78405139",
                "status": "closed",
                "average": 0.2517,
                "cost": 25.6734,
                "filled": 102.0,
                "is_open": false,
                "order_date": "2022-10-18 09:12:57",
                "order_timestamp": 1666084377681,
                "order_filled_date": "2022-10-18 09:12:59",
                "order_type": "limit",
                "price": 0.2517,
                "remaining": 0.0
            },
            {
                "amount": 102.0,
                "safe_price": 0.2517,
                "ft_order_side": "buy",
                "order_filled_timestamp": 1666084389644,
                "ft_is_entry": true,
                "pair": "ETH/USDT",
                "order_id": "78405265",
                "status": "closed",
                "average": 0.2517,
                "cost": 25.6734,
                "filled": 102.0,
                "is_open": false,
                "order_date": "2022-10-18 09:13:03",
                "order_timestamp": 1666084383295,
                "order_filled_date": "2022-10-18 09:13:09",
                "order_type": "limit",
                "price": 0.2517,
                "remaining": 0.0
            },
            {
                "amount": 102.0,
                "safe_price": 0.2516,
                "ft_order_side": "buy",
                "order_filled_timestamp": 1666084723521,
                "ft_is_entry": true,
                "pair": "ETH/USDT",
                "order_id": "78405395",
                "status": "closed",
                "average": 0.2516,
                "cost": 25.6632,
                "filled": 102.0,
                "is_open": false,
                "order_date": "2022-10-18 09:13:13",
                "order_timestamp": 1666084393920,
                "order_filled_date": "2022-10-18 09:18:43",
                "order_type": "limit",
                "price": 0.2516,
                "remaining": 0.0
            },
            {
                "amount": 407.0,
                "safe_price": 0.2592,
                "ft_order_side": "sell",
                "order_filled_timestamp": 1666086322198,
                "ft_is_entry": false,
                "pair": "ETH/USDT",
                "order_id": "78432649",
                "status": "closed",
                "average": 0.2592,
                "cost": 105.4944,
                "filled": 407.0,
                "is_open": false,
                "order_date": "2022-10-18 09:45:21",
                "order_timestamp": 1666086321435,
                "order_filled_date": "2022-10-18 09:45:22",
                "order_type": "market",
                "price": 0.2592,
                "remaining": 0.0,
                "funding_fee": -0.055
            }
        ]
    }"""
    trade = Trade.from_json(trade_string)
    Trade.session.add(trade)
    Trade.commit()

    assert trade.id == 25
    assert trade.pair == 'ETH/USDT'
    assert trade.open_date_utc == datetime(2022, 10, 18, 9, 12, 42, tzinfo=timezone.utc)
    assert isinstance(trade.open_date, datetime)
    assert trade.exit_reason == 'no longer good'
    assert trade.realized_profit == 2.76315361
    assert trade.precision_mode == 2
    assert trade.amount_precision == 1.0
    assert trade.contract_size == 1.0

    assert len(trade.orders) == 5
    last_o = trade.orders[-1]
    assert last_o.order_filled_utc == datetime(2022, 10, 18, 9, 45, 22, tzinfo=timezone.utc)
    assert isinstance(last_o.order_date, datetime)
    assert last_o.funding_fee == -0.055


@pytest.mark.usefixtures("init_persistence")
def test_trade_serialize_load_back(fee):

    create_mock_trades_usdt(fee, None)

    t = Trade.get_trades([Trade.id == 1]).first()
    assert t.id == 1
    t.funding_fees = 0.025
    t.orders[0].funding_fee = 0.0125
    Trade.commit()

    tjson = t.to_json(False)
    assert isinstance(tjson, dict)
    trade_string = json.dumps(tjson)
    trade = Trade.from_json(trade_string)

    assert trade.id == t.id
    assert trade.funding_fees == t.funding_fees
    assert trade.orders[0].funding_fee == t.orders[0].funding_fee
    excluded = [
        'trade_id', 'quote_currency', 'open_timestamp', 'close_timestamp',
        'realized_profit_ratio', 'close_profit_pct',
        'trade_duration_s', 'trade_duration',
        'profit_ratio', 'profit_pct', 'profit_abs', 'stop_loss_abs',
        'initial_stop_loss_abs',
        'orders',
    ]
    failed = []
    # Ensure all attributes written can be read.
    for obj, value in tjson.items():
        if obj in excluded:
            continue
        tattr = getattr(trade, obj, None)
        if isinstance(tattr, datetime):
            tattr = tattr.strftime('%Y-%m-%d %H:%M:%S')
        if tattr != value:
            failed.append((obj, tattr, value))

    assert tjson.get('trade_id') == trade.id
    assert tjson.get('quote_currency') == trade.stake_currency
    assert tjson.get('stop_loss_abs') == trade.stop_loss
    assert tjson.get('initial_stop_loss_abs') == trade.initial_stop_loss

    excluded_o = [
        'order_filled_timestamp', 'ft_is_entry', 'pair', 'is_open', 'order_timestamp',
    ]
    order_obj = trade.orders[0]
    for obj, value in tjson['orders'][0].items():
        if obj in excluded_o:
            continue
        tattr = getattr(order_obj, obj, None)
        if isinstance(tattr, datetime):
            tattr = tattr.strftime('%Y-%m-%d %H:%M:%S')
        if tattr != value:
            failed.append((obj, tattr, value))

    assert tjson['orders'][0]['pair'] == order_obj.ft_pair
    print(failed)
    assert not failed
