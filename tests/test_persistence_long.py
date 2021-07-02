import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import FunctionType
from unittest.mock import MagicMock
import arrow
import pytest
from math import isclose
from sqlalchemy import create_engine, inspect, text
from freqtrade import constants
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.persistence import LocalTrade, Order, Trade, clean_dry_run_db, init_db
from tests.conftest import create_mock_trades_with_leverage, log_has, log_has_re


@pytest.mark.usefixtures("init_persistence")
def test_update_with_binance(limit_leveraged_buy_order, limit_leveraged_sell_order, fee, ten_minutes_ago, caplog):
    """
        10 minute leveraged limit trade on binance at 3x leverage

        Leveraged trade
        fee: 0.25% base
        interest_rate: 0.05% per day
        open_rate: 0.00001099 base
        close_rate: 0.00001173 base
        amount: 272.97543219 crypto
        stake_amount: 0.0009999999999226999 base
        borrowed: 0.0019999999998453998  base
        time-periods: 10 minutes(rounds up to 1/24 time-period of 1 day)
        interest: borrowed * interest_rate * time-periods
                    = 0.0019999999998453998 * 0.0005 * 1/24 = 4.166666666344583e-08 base
        open_value: (amount * open_rate) + (amount * open_rate * fee)
            = (272.97543219 * 0.00001099) + (272.97543219 * 0.00001099 * 0.0025)
            = 0.0030074999997675204
        close_value: (amount_closed * close_rate) - (amount_closed * close_rate * fee)
            = (272.97543219 * 0.00001173) - (272.97543219 * 0.00001173 * 0.0025)
            = 0.003193996815039728
        total_profit =  close_value - open_value - interest
            = 0.003193996815039728 - 0.0030074999997675204 - 4.166666666344583e-08
            = 0.00018645514860554435
        total_profit_percentage = total_profit / stake_amount
            = 0.00018645514860554435 / 0.0009999999999226999
            = 0.18645514861995735

    """
    trade = Trade(
        id=2,
        pair='ETH/BTC',
        stake_amount=0.0009999999999226999,
        open_rate=0.01,
        amount=5,
        is_open=True,
        open_date=ten_minutes_ago,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        # borrowed=90.99181073,
        interest_rate=0.0005,
        exchange='binance'
    )
    # assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.borrowed is None
    assert trade.is_short is None
    # trade.open_order_id = 'something'
    trade.update(limit_leveraged_buy_order)
    # assert trade.open_order_id is None
    assert trade.open_rate == 0.00001099
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.borrowed == 0.0019999999998453998
    assert trade.is_short is True
    assert log_has_re(r"LIMIT_BUY has been fulfilled for Trade\(id=2, "
                      r"pair=ETH/BTC, amount=272.97543219, open_rate=0.00001099, open_since=.*\).",
                      caplog)
    caplog.clear()
    # trade.open_order_id = 'something'
    trade.update(limit_leveraged_sell_order)
    # assert trade.open_order_id is None
    assert trade.close_rate == 0.00001173
    assert trade.close_profit == 0.18645514861995735
    assert trade.close_date is not None
    assert log_has_re(r"LIMIT_SELL has been fulfilled for Trade\(id=2, "
                      r"pair=ETH/BTC, amount=272.97543219, open_rate=0.00001099, open_since=.*\).",
                      caplog)


@pytest.mark.usefixtures("init_persistence")
def test_update_market_order(limit_leveraged_buy_order, limit_leveraged_sell_order, fee, ten_minutes_ago, caplog):
    """
        10 minute leveraged market trade on Kraken at 3x leverage
        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per 4 hrs
        open_rate: 0.00004099 base
        close_rate: 0.00004173 base
        amount: 91.99181073 * leverage(3) = 275.97543219 crypto
        stake_amount: 0.0037707443218227
        borrowed: 0.0075414886436454 base
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
        interest: borrowed * interest_rate * time-periods
                    = 0.0075414886436454 * 0.0005 * 1 = 3.7707443218227e-06 crypto
        open_value: (amount * open_rate) + (amount * open_rate * fee)
            = (275.97543219 * 0.00004099) + (275.97543219 * 0.00004099 * 0.0025)
            = 0.01134051354788177
        close_value: (amount_closed * close_rate) - (amount_closed * close_rate * fee)
            = (275.97543219 * 0.00004173) - (275.97543219 * 0.00004173 * 0.0025)
            = 0.011487663648325479
        total_profit = close_value - open_value - interest
            = 0.011487663648325479 - 0.01134051354788177 - 3.7707443218227e-06
            = 0.0001433793561218866
        total_profit_percentage = total_profit / stake_amount
            = 0.0001433793561218866 / 0.0037707443218227
            = 0.03802415223225211
    """
    trade = Trade(
        id=1,
        pair='ETH/BTC',
        stake_amount=0.0037707443218227,
        amount=5,
        open_rate=0.01,
        is_open=True,
        leverage=3,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=ten_minutes_ago,
        interest_rate=0.0005,
        exchange='kraken'
    )
    trade.open_order_id = 'something'
    trade.update(limit_leveraged_buy_order)
    assert trade.leverage == 3.0
    assert trade.is_short == True
    assert trade.open_order_id is None
    assert trade.open_rate == 0.00004099
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.interest_rate == 0.0005
    # TODO: Uncomment the next assert and make it work.
    # The logger also has the exact same but there's some spacing in there
    assert log_has_re(r"MARKET_BUY has been fulfilled for Trade\(id=1, "
                      r"pair=ETH/BTC, amount=275.97543219, open_rate=0.00004099, open_since=.*\).",
                      caplog)
    caplog.clear()
    trade.is_open = True
    trade.open_order_id = 'something'
    trade.update(limit_leveraged_sell_order)
    assert trade.open_order_id is None
    assert trade.close_rate == 0.00004173
    assert trade.close_profit == 0.03802415223225211
    assert trade.close_date is not None
    # TODO: The amount should maybe be the opening amount + the interest
    # TODO: Uncomment the next assert and make it work.
    # The logger also has the exact same but there's some spacing in there
    assert log_has_re(r"MARKET_SELL has been fulfilled for Trade\(id=1, "
                      r"pair=ETH/BTC, amount=275.97543219, open_rate=0.00004099, open_since=.*\).",
                      caplog)


@pytest.mark.usefixtures("init_persistence")
def test_calc_open_close_trade_price(limit_leveraged_buy_order, limit_leveraged_sell_order, five_hours_ago, fee):
    """
        5 hour leveraged trade on Binance

        fee: 0.25% base
        interest_rate: 0.05% per day
        open_rate: 0.00001099 base
        close_rate: 0.00001173 base
        amount: 272.97543219 crypto
        stake_amount: 0.0009999999999226999 base
        borrowed: 0.0019999999998453998  base
        time-periods: 5 hours(rounds up to 5/24 time-period of 1 day)
        interest: borrowed * interest_rate * time-periods
                    = 0.0019999999998453998 * 0.0005 * 5/24 = 2.0833333331722917e-07 base
        open_value: (amount * open_rate) + (amount * open_rate * fee)
            = (272.97543219 * 0.00001099) + (272.97543219 * 0.00001099 * 0.0025)
            = 0.0030074999997675204
        close_value: (amount_closed * close_rate) - (amount_closed * close_rate * fee)
            = (272.97543219 * 0.00001173) - (272.97543219 * 0.00001173 * 0.0025)
            = 0.003193996815039728
        total_profit =  close_value - open_value - interest
            = 0.003193996815039728 - 0.0030074999997675204 - 2.0833333331722917e-07
            = 0.00018628848193889054
        total_profit_percentage = total_profit / stake_amount
            = 0.00018628848193889054 / 0.0009999999999226999
            = 0.18628848195329067
    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.0009999999999226999,
        open_rate=0.01,
        amount=5,
        open_date=five_hours_ago,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        interest_rate=0.0005
    )
    trade.open_order_id = 'something'
    trade.update(limit_leveraged_buy_order)
    assert trade._calc_open_trade_value() == 0.0030074999997675204
    trade.update(limit_leveraged_sell_order)

    # Will be slightly different due to slight changes in compilation time, and the fact that interest depends on time
    assert round(trade.calc_close_trade_value(), 11) == round(0.003193996815039728, 11)
    # Profit in BTC
    assert round(trade.calc_profit(), 8) == round(0.18628848195329067, 8)
    # Profit in percent
    # assert round(trade.calc_profit_ratio(), 11) == round(0.05822425142973869, 11)


@pytest.mark.usefixtures("init_persistence")
def test_trade_close(fee, five_hours_ago):
    """
        5 hour leveraged market trade on Kraken at 3x leverage
        fee: 0.25% base
        interest_rate: 0.05% per 4 hrs
        open_rate: 0.1 base
        close_rate: 0.2 base
        amount: 5 * leverage(3) = 15 crypto
        stake_amount: 0.5
        borrowed: 1 base
        time-periods: 5/4 periods of 4hrs
        interest: borrowed * interest_rate * time-periods
                    = 1 * 0.0005 * 5/4 = 0.000625 crypto
        open_value: (amount * open_rate) + (amount * open_rate * fee)
            = (15 * 0.1) + (15 * 0.1 * 0.0025)
            = 1.50375
        close_value: (amount_closed * close_rate) + (amount_closed * close_rate * fee)
            = (15 * 0.2) - (15 * 0.2 * 0.0025)
            = 2.9925
        total_profit = close_value - open_value - interest
            = 2.9925 - 1.50375 - 0.000625
            = 1.4881250000000001
        total_profit_percentage = total_profit / stake_amount
            = 1.4881250000000001 /  0.5
            = 2.9762500000000003
    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.1,
        open_rate=0.01,
        amount=5,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=five_hours_ago,
        exchange='kraken',
        leverage=3.0,
        interest_rate=0.0005
    )
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.is_open is True
    trade.close(0.02)
    assert trade.is_open is False
    assert trade.close_profit == round(2.9762500000000003, 8)
    assert trade.close_date is not None

    # TODO-mg: Remove these comments probably
    # new_date = arrow.Arrow(2020, 2, 2, 15, 6, 1).datetime,
    # assert trade.close_date != new_date
    # # Close should NOT update close_date if the trade has been closed already
    # assert trade.is_open is False
    # trade.close_date = new_date
    # trade.close(0.02)
    # assert trade.close_date == new_date


@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price_exception(limit_leveraged_buy_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        open_rate=0.1,
        amount=5,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        interest_rate=0.0005,
        borrowed=0.002
    )
    trade.open_order_id = 'something'
    trade.update(limit_leveraged_buy_order)
    assert trade.calc_close_trade_value() == 0.0


@pytest.mark.usefixtures("init_persistence")
def test_update_open_order(limit_leveraged_buy_order):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=1.00,
        open_rate=0.01,
        amount=5,
        fee_open=0.1,
        fee_close=0.1,
        interest_rate=0.0005,
        borrowed=2.00,
        exchange='binance',
    )
    assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None
    limit_leveraged_buy_order['status'] = 'open'
    trade.update(limit_leveraged_buy_order)
    assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None


@pytest.mark.usefixtures("init_persistence")
def test_calc_open_trade_value(market_leveraged_buy_order, ten_minutes_ago, fee):
    """
        10 minute leveraged market trade on Kraken at 3x leverage
        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per 4 hrs
        open_rate: 0.00004099 base
        close_rate: 0.00004173 base
        amount: 91.99181073 * leverage(3) = 275.97543219 crypto
        stake_amount: 0.0037707443218227
        borrowed: 0.0075414886436454 base
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
        interest: borrowed * interest_rate * time-periods
            = 0.0075414886436454 * 0.0005 * 1 = 3.7707443218227e-06 crypto
        open_value: (amount * open_rate) + (amount * open_rate * fee)
            = (275.97543219 * 0.00004099) + (275.97543219 * 0.00004099 * 0.0025)
            = 0.01134051354788177
    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=5,
        open_rate=0.00004099,
        open_date=ten_minutes_ago,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        interest_rate=0.0005,
        exchange='kraken',
        leverage=3
    )
    trade.open_order_id = 'open_trade'
    trade.update(market_leveraged_buy_order)  # Buy @ 0.00001099
    # Get the open rate price with the standard fee rate
    assert trade._calc_open_trade_value() == 0.01134051354788177
    trade.fee_open = 0.003
    # Get the open rate price with a custom fee rate
    assert trade._calc_open_trade_value() == 0.011346169664364504


@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price(market_leveraged_buy_order, market_leveraged_sell_order, ten_minutes_ago, fee):
    """
        10 minute leveraged market trade on Kraken at 3x leverage
        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per 4 hrs
        open_rate: 0.00004099 base
        close_rate: 0.00004173 base
        amount: 91.99181073 * leverage(3) = 275.97543219 crypto
        stake_amount: 0.0037707443218227
        borrowed: 0.0075414886436454 base
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
        interest: borrowed * interest_rate * time-periods
                    = 0.0075414886436454 * 0.0005 * 1 = 3.7707443218227e-06 crypto
        open_value: (amount * open_rate) + (amount * open_rate * fee)
            = (275.97543219 * 0.00004099) + (275.97543219 * 0.00004099 * 0.0025)
            = 0.01134051354788177
        close_value: (amount_closed * close_rate) - (amount_closed * close_rate * fee)
            = (275.97543219 * 0.00001234) - (275.97543219 * 0.00001234 * 0.0025) = 0.0033970229911415386
            = (275.97543219 * 0.00001234) - (275.97543219 * 0.00001234 * 0.003)  = 0.0033953202227249265
            = (275.97543219 * 0.00004173) - (275.97543219 * 0.00004173 * 0.005)  = 0.011458872511362258

    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=5,
        open_rate=0.00001099,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=ten_minutes_ago,
        interest_rate=0.0005,
        is_short=True,
        leverage=3.0,
        exchange='kraken',
    )
    trade.open_order_id = 'close_trade'
    trade.update(market_leveraged_buy_order)  # Buy @ 0.00001099
    # Get the close rate price with a custom close rate and a regular fee rate
    assert isclose(trade.calc_close_trade_value(rate=0.00001234), 0.0033970229911415386)
    # Get the close rate price with a custom close rate and a custom fee rate
    assert isclose(trade.calc_close_trade_value(rate=0.00001234, fee=0.003), 0.0033953202227249265)
    # Test when we apply a Sell order, and ask price with a custom fee rate
    trade.update(market_leveraged_sell_order)
    assert isclose(trade.calc_close_trade_value(fee=0.005), 0.011458872511362258)


@pytest.mark.usefixtures("init_persistence")
def test_calc_profit(market_leveraged_buy_order, market_leveraged_sell_order, ten_minutes_ago, five_hours_ago, fee):
    """
        # TODO: Update this one
        Leveraged trade on Kraken at 3x leverage
        fee: 0.25% base or 0.3%
        interest_rate: 0.05%, 0.25% per 4 hrs
        open_rate: 0.00004099 base
        close_rate: 0.00004173 base
        stake_amount: 0.0037707443218227
        amount: 91.99181073 * leverage(3) = 275.97543219 crypto
        borrowed: 0.0075414886436454 base
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
                        5 hours = 5/4

        interest: borrowed * interest_rate * time-periods
                    = 0.0075414886436454 * 0.0005 * 1 = 3.7707443218227e-06 crypto
                    = 0.0075414886436454 * 0.00025 * 5/4 = 2.3567152011391876e-06 crypto
                    = 0.0075414886436454 * 0.0005 * 5/4 = 4.713430402278375e-06 crypto
                    = 0.0075414886436454 * 0.00025 * 1 = 1.88537216091135e-06 crypto
        open_value: (amount * open_rate) + (amount * open_rate * fee)
            = (275.97543219 * 0.00004099) + (275.97543219 * 0.00004099 * 0.0025) = 0.01134051354788177
        close_value: (amount_closed * close_rate) - (amount_closed * close_rate * fee)
            (275.97543219 * 0.00005374) - (275.97543219 * 0.00005374 * 0.0025) = 0.014793842426575873
            (275.97543219 * 0.00000437) - (275.97543219 * 0.00000437 * 0.0025) = 0.0012029976070736241
            (275.97543219 * 0.00005374) - (275.97543219 * 0.00005374 * 0.003) = 0.014786426966712927
            (275.97543219 * 0.00000437) - (275.97543219 * 0.00000437 * 0.003) = 0.0012023946007542888
        total_profit = close_value - open_value
            = 0.014793842426575873 - 0.01134051354788177  = 0.003453328878694104
            = 0.0012029976070736241 - 0.01134051354788177 = -0.010137515940808145
            = 0.014786426966712927 - 0.01134051354788177  = 0.0034459134188311574
            = 0.0012023946007542888 - 0.01134051354788177 = -0.01013811894712748
        total_profit_percentage = total_profit / stake_amount
            0.003453328878694104/0.0037707443218227   = 0.9158215418394733
            -0.010137515940808145/0.0037707443218227  = -2.6884654793852154
            0.0034459134188311574/0.0037707443218227  = 0.9138549646255183
            -0.01013811894712748/0.0037707443218227   = -2.6886253964381557

    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.0038388182617629,
        amount=5,
        open_rate=0.00004099,
        open_date=ten_minutes_ago,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='kraken',
        leverage=3.0,
        interest_rate=0.0005
    )
    trade.open_order_id = 'something'
    trade.update(market_leveraged_buy_order)  # Buy @ 0.00001099
    # Custom closing rate and regular fee rate

    # Higher than open rate
    assert trade.calc_profit(rate=0.00004374, interest_rate=0.0005) == round(
        0.003453328878694104, 8)
    assert trade.calc_profit_ratio(
        rate=0.00004374, interest_rate=0.0005) == round(0.9158215418394733, 8)

    # Lower than open rate
    trade.open_date = five_hours_ago
    assert trade.calc_profit(
        rate=0.00000437, interest_rate=0.00025) == round(-0.010137515940808145, 8)
    assert trade.calc_profit_ratio(
        rate=0.00000437, interest_rate=0.00025) == round(-2.6884654793852154, 8)

    # Custom closing rate and custom fee rate
    # Higher than open rate
    assert trade.calc_profit(rate=0.00004374, fee=0.003,
                             interest_rate=0.0005) == round(0.0034459134188311574, 8)
    assert trade.calc_profit_ratio(rate=0.00004374, fee=0.003,
                                   interest_rate=0.0005) == round(0.9138549646255183, 8)

    # Lower than open rate
    trade.open_date = ten_minutes_ago
    assert trade.calc_profit(rate=0.00000437, fee=0.003,
                             interest_rate=0.00025) == round(-0.01013811894712748, 8)
    assert trade.calc_profit_ratio(rate=0.00000437, fee=0.003,
                                   interest_rate=0.00025) == round(-2.6886253964381557, 8)

    # Test when we apply a Sell order. Sell higher than open rate @ 0.00001173
    trade.update(market_leveraged_sell_order)
    assert trade.calc_profit() == round(0.0001433793561218866, 8)
    assert trade.calc_profit_ratio() == round(0.03802415223225211, 8)

    # Test with a custom fee rate on the close trade
    # assert trade.calc_profit(fee=0.003) == 0.00006163
    # assert trade.calc_profit_ratio(fee=0.003) == 0.06147824


@pytest.mark.usefixtures("init_persistence")
def test_interest_kraken(market_leveraged_buy_order, ten_minutes_ago, five_hours_ago, fee):
    """
        Market trade on Kraken at 3x and 8x leverage
        Short trade
        interest_rate: 0.05%, 0.25% per 4 hrs
        open_rate: 0.00004099 base
        close_rate: 0.00004173 base
        stake_amount: 0.0037707443218227
        borrowed: 0.0075414886436454
        amount:
            91.99181073 * leverage(3) = 275.97543219 crypto
            91.99181073 * leverage(5) = 459.95905365 crypto
        borrowed:
            0.0075414886436454  base
            0.0150829772872908  base
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
                        5 hours = 5/4

        interest: borrowed * interest_rate * time-periods
                = 0.0075414886436454 * 0.0005 * 1 = 3.7707443218227e-06 base
                = 0.0075414886436454 * 0.00025 * 5/4 = 2.3567152011391876e-06 base
                = 0.0150829772872908 * 0.0005 * 5/4 = 9.42686080455675e-06 base
                = 0.0150829772872908 * 0.00025 * 1 = 3.7707443218227e-06 base
    """

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.0037707443218227,
        amount=91.99181073,
        open_rate=0.00001099,
        open_date=ten_minutes_ago,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='kraken',
        leverage=3.0,
        interest_rate=0.0005
    )

    assert float(round(trade.calculate_interest(), 8)) == 3.7707443218227e-06
    trade.open_date = five_hours_ago
    assert float(round(trade.calculate_interest(interest_rate=0.00025), 8)
                 ) == 2.3567152011391876e-06  # TODO: Fails with 0.08624233

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.0037707443218227,
        amount=91.99181073,
        open_rate=0.00001099,
        open_date=five_hours_ago,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='kraken',
        is_short=True,
        leverage=5.0,
        interest_rate=0.0005
    )

    assert float(round(trade.calculate_interest(), 8)
                 ) == 9.42686080455675e-06  # TODO: Fails with 0.28747445
    trade.open_date = ten_minutes_ago
    assert float(round(trade.calculate_interest(interest_rate=0.00025), 8)) == 3.7707443218227e-06


@pytest.mark.usefixtures("init_persistence")
def test_interest_binance(market_leveraged_buy_order, ten_minutes_ago, five_hours_ago, fee):
    """
        Market trade on Kraken at 3x and 8x leverage
        Short trade
        interest_rate: 0.05%, 0.25% per 4 hrs
        open_rate: 0.00004099 base
        close_rate: 0.00004173 base
        stake_amount: 0.0037707443218227
        borrowed: 0.0075414886436454
        amount:
            91.99181073 * leverage(3) = 275.97543219 crypto
            91.99181073 * leverage(5) = 459.95905365 crypto
        borrowed:
            0.0075414886436454  base
            0.0150829772872908  base
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
                        5 hours = 5/24

        interest: borrowed * interest_rate * time-periods
                = 0.0075414886436454 * 0.0005 * 1/24 = 1.571143467426125e-07 base
                = 0.0075414886436454 * 0.00025 * 5/24 = 3.9278586685653125e-07 base
                = 0.0150829772872908 * 0.0005 * 5/24 = 1.571143467426125e-06 base
                = 0.0150829772872908 * 0.00025 * 1/24 = 1.571143467426125e-07 base
    """

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=275.97543219,
        open_rate=0.00001099,
        open_date=ten_minutes_ago,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        is_short=True,
        borrowed=275.97543219,
        interest_rate=0.0005
    )

    assert float(round(trade.calculate_interest(), 8)) == 1.571143467426125e-07
    trade.open_date = five_hours_ago
    assert float(round(trade.calculate_interest(interest_rate=0.00025), 8)
                 ) == 3.9278586685653125e-07

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=459.95905365,
        open_rate=0.00001099,
        open_date=five_hours_ago,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        is_short=True,
        borrowed=459.95905365,
        interest_rate=0.0005
    )

    assert float(round(trade.calculate_interest(), 8)) == 1.571143467426125e-06
    trade.open_date = ten_minutes_ago
    assert float(round(trade.calculate_interest(interest_rate=0.00025), 8)) == 1.571143467426125e-07
