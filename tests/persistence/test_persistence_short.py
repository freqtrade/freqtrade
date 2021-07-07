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
from freqtrade.enums import InterestMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.persistence import LocalTrade, Order, Trade, clean_dry_run_db, init_db
from tests.conftest import create_mock_trades_with_leverage, log_has, log_has_re


@pytest.mark.usefixtures("init_persistence")
def test_interest_kraken_short(market_short_order, fee):
    """
        Market trade on Kraken at 3x and 8x leverage
        Short trade
        interest_rate: 0.05%, 0.25% per 4 hrs
        open_rate: 0.00004173 base
        close_rate: 0.00004099 base
        amount:
            275.97543219 crypto
            459.95905365 crypto
        borrowed:
            275.97543219  crypto
            459.95905365  crypto
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
                        5 hours = 5/4

        interest: borrowed * interest_rate * time-periods
                    = 275.97543219 * 0.0005 * 1    = 0.137987716095 crypto
                    = 275.97543219 * 0.00025 * 5/4 = 0.086242322559375 crypto
                    = 459.95905365 * 0.0005 * 5/4  = 0.28747440853125 crypto
                    = 459.95905365 * 0.00025 * 1   = 0.1149897634125 crypto
    """

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=275.97543219,
        open_rate=0.00001099,
        open_date=datetime.utcnow() - timedelta(hours=0, minutes=10),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='kraken',
        is_short=True,
        leverage=3.0,
        interest_rate=0.0005,
        interest_mode=InterestMode.HOURSPER4
    )

    assert float(round(trade.calculate_interest(), 8)) == round(0.137987716095, 8)
    trade.open_date = datetime.utcnow() - timedelta(hours=5, minutes=0)
    assert float(round(trade.calculate_interest(interest_rate=0.00025), 8)
                 ) == round(0.086242322559375, 8)

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=459.95905365,
        open_rate=0.00001099,
        open_date=datetime.utcnow() - timedelta(hours=5, minutes=0),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='kraken',
        is_short=True,
        leverage=5.0,
        interest_rate=0.0005,
        interest_mode=InterestMode.HOURSPER4
    )

    assert float(round(trade.calculate_interest(), 8)) == round(0.28747440853125, 8)
    trade.open_date = datetime.utcnow() - timedelta(hours=0, minutes=10)
    assert float(round(trade.calculate_interest(interest_rate=0.00025), 8)
                 ) == round(0.1149897634125, 8)


@ pytest.mark.usefixtures("init_persistence")
def test_interest_binance_short(market_short_order, fee):
    """
        Market trade on Binance at 3x and 5x leverage
        Short trade
        interest_rate: 0.05%, 0.25% per 1 day
        open_rate: 0.00004173 base
        close_rate: 0.00004099 base
        amount:
            91.99181073 * leverage(3) = 275.97543219 crypto
            91.99181073 * leverage(5) = 459.95905365 crypto
        borrowed:
            275.97543219  crypto
            459.95905365  crypto
        time-periods: 10 minutes(rounds up to 1/24 time-period of 1 day)
                        5 hours = 5/24

        interest: borrowed * interest_rate * time-periods
                    = 275.97543219 * 0.0005 * 1/24 = 0.005749488170625 crypto
                    = 275.97543219 * 0.00025 * 5/24 = 0.0143737204265625 crypto
                    = 459.95905365 * 0.0005 * 5/24 = 0.047912401421875 crypto
                    = 459.95905365 * 0.00025 * 1/24 = 0.0047912401421875 crypto
    """

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=275.97543219,
        open_rate=0.00001099,
        open_date=datetime.utcnow() - timedelta(hours=0, minutes=10),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        is_short=True,
        leverage=3.0,
        interest_rate=0.0005,
        interest_mode=InterestMode.HOURSPERDAY
    )

    assert float(round(trade.calculate_interest(), 8)) == 0.00574949
    trade.open_date = datetime.utcnow() - timedelta(hours=5, minutes=0)
    assert float(round(trade.calculate_interest(interest_rate=0.00025), 8)) == 0.01437372

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=459.95905365,
        open_rate=0.00001099,
        open_date=datetime.utcnow() - timedelta(hours=5, minutes=0),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        is_short=True,
        leverage=5.0,
        interest_rate=0.0005,
        interest_mode=InterestMode.HOURSPERDAY
    )

    assert float(round(trade.calculate_interest(), 8)) == 0.04791240
    trade.open_date = datetime.utcnow() - timedelta(hours=0, minutes=10)
    assert float(round(trade.calculate_interest(interest_rate=0.00025), 8)) == 0.00479124


@ pytest.mark.usefixtures("init_persistence")
def test_calc_open_trade_value_short(market_short_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=5,
        open_rate=0.00004173,
        open_date=datetime.utcnow() - timedelta(hours=0, minutes=10),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        interest_rate=0.0005,
        is_short=True,
        leverage=3.0,
        exchange='kraken',
        interest_mode=InterestMode.HOURSPER4
    )
    trade.open_order_id = 'open_trade'
    trade.update(market_short_order)  # Buy @ 0.00001099
    # Get the open rate price with the standard fee rate
    assert trade._calc_open_trade_value() == 0.011487663648325479
    trade.fee_open = 0.003
    # Get the open rate price with a custom fee rate
    assert trade._calc_open_trade_value() == 0.011481905420932834


@ pytest.mark.usefixtures("init_persistence")
def test_update_open_order_short(limit_short_order):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=1.00,
        open_rate=0.01,
        amount=5,
        leverage=3.0,
        fee_open=0.1,
        fee_close=0.1,
        interest_rate=0.0005,
        is_short=True,
        exchange='binance',
        interest_mode=InterestMode.HOURSPERDAY
    )
    assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None
    limit_short_order['status'] = 'open'
    trade.update(limit_short_order)
    assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None


@ pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price_exception_short(limit_short_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        open_rate=0.1,
        amount=15.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        interest_rate=0.0005,
        leverage=3.0,
        is_short=True,
        interest_mode=InterestMode.HOURSPERDAY
    )
    trade.open_order_id = 'something'
    trade.update(limit_short_order)
    assert trade.calc_close_trade_value() == 0.0


@ pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price_short(market_short_order, market_exit_short_order, fee):
    """
        10 minute short market trade on Kraken at 3x leverage
        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per 4 hrs
        open_rate: 0.00004173 base
        close_rate: 0.00001234 base
        amount: = 275.97543219 crypto
        borrowed: 275.97543219  crypto
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
        interest: borrowed * interest_rate * time-periods
                    = 275.97543219 * 0.0005 * 1 = 0.137987716095 crypto
        amount_closed: amount + interest = 275.97543219 + 0.137987716095 = 276.113419906095
        close_value: (amount_closed * close_rate) + (amount_closed * close_rate * fee)
            = (276.113419906095 * 0.00001234) + (276.113419906095 * 0.00001234 * 0.0025)
            = 0.01134618380465571
    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=5,
        open_rate=0.00001099,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.utcnow() - timedelta(hours=0, minutes=10),
        interest_rate=0.0005,
        is_short=True,
        leverage=3.0,
        exchange='kraken',
        interest_mode=InterestMode.HOURSPER4
    )
    trade.open_order_id = 'close_trade'
    trade.update(market_short_order)  # Buy @ 0.00001099
    # Get the close rate price with a custom close rate and a regular fee rate
    assert isclose(trade.calc_close_trade_value(rate=0.00001234), 0.003415757700645315)
    # Get the close rate price with a custom close rate and a custom fee rate
    assert isclose(trade.calc_close_trade_value(rate=0.00001234, fee=0.003), 0.0034174613204461354)
    # Test when we apply a Sell order, and ask price with a custom fee rate
    trade.update(market_exit_short_order)
    assert isclose(trade.calc_close_trade_value(fee=0.005), 0.011374478527360586)


@ pytest.mark.usefixtures("init_persistence")
def test_calc_open_close_trade_price_short(limit_short_order, limit_exit_short_order, fee):
    """
        5 hour short trade on Binance
        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per day
        open_rate: 0.00001173 base
        close_rate: 0.00001099 base
        amount: 90.99181073 crypto
        borrowed: 90.99181073  crypto
        stake_amount: 0.0010673339398629
        time-periods: 5 hours = 5/24
        interest: borrowed * interest_rate * time-periods
                    = 90.99181073 * 0.0005 * 5/24 = 0.009478313617708333 crypto
        open_value: (amount * open_rate) - (amount * open_rate * fee)
            = (90.99181073 * 0.00001173) - (90.99181073 * 0.00001173 * 0.0025)
            = 0.0010646656050132426
        amount_closed: amount + interest = 90.99181073 + 0.009478313617708333 = 91.0012890436177
        close_value: (amount_closed * close_rate) + (amount_closed * close_rate * fee)
            = (91.0012890436177 * 0.00001099) + (91.0012890436177 * 0.00001099 * 0.0025)
            = 0.001002604427005832
        total_profit = open_value - close_value
            = 0.0010646656050132426 - 0.001002604427005832
            = 0.00006206117800741065
        total_profit_percentage = (close_value - open_value) / stake_amount
            = (0.0010646656050132426 - 0.0010025208853391716)/0.0010673339398629
            = 0.05822425142973869
    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.0010673339398629,
        open_rate=0.01,
        amount=5,
        open_date=datetime.utcnow() - timedelta(hours=5, minutes=0),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        interest_rate=0.0005,
        interest_mode=InterestMode.HOURSPERDAY
    )
    trade.open_order_id = 'something'
    trade.update(limit_short_order)
    assert trade._calc_open_trade_value() == 0.0010646656050132426
    trade.update(limit_exit_short_order)

    # Will be slightly different due to slight changes in compilation time, and the fact that interest depends on time
    assert round(trade.calc_close_trade_value(), 11) == round(0.001002604427005832, 11)
    # Profit in BTC
    assert round(trade.calc_profit(), 8) == round(0.00006206117800741065, 8)
    # Profit in percent
    # assert round(trade.calc_profit_ratio(), 11) == round(0.05822425142973869, 11)


@ pytest.mark.usefixtures("init_persistence")
def test_trade_close_short(fee):
    """
        Five hour short trade on Kraken at 3x leverage
        Short trade
        Exchange: Kraken
        fee: 0.25% base
        interest_rate: 0.05% per 4 hours
        open_rate: 0.02 base
        close_rate: 0.01 base
        leverage: 3.0
        amount: 15 crypto
        borrowed: 15 crypto
        time-periods: 5 hours = 5/4

        interest: borrowed * interest_rate * time-periods
                    = 15 * 0.0005 * 5/4 = 0.009375 crypto
        open_value: (amount * open_rate) - (amount * open_rate * fee)
            = (15 * 0.02) - (15 * 0.02 * 0.0025)
            = 0.29925
        amount_closed: amount + interest = 15 + 0.009375 = 15.009375
        close_value: (amount_closed * close_rate) + (amount_closed * close_rate * fee)
            = (15.009375 * 0.01) + (15.009375 * 0.01 * 0.0025)
            = 0.150468984375
        total_profit = open_value - close_value
            = 0.29925 - 0.150468984375
            = 0.148781015625
        total_profit_percentage = total_profit / stake_amount
            = 0.148781015625 / 0.1
            = 1.4878101562500001
    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.1,
        open_rate=0.02,
        amount=15,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.utcnow() - timedelta(hours=5, minutes=0),
        exchange='kraken',
        is_short=True,
        leverage=3.0,
        interest_rate=0.0005,
        interest_mode=InterestMode.HOURSPER4
    )
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.is_open is True
    trade.close(0.01)
    assert trade.is_open is False
    assert trade.close_profit == round(1.4878101562500001, 8)
    assert trade.close_date is not None

    # TODO-mg: Remove these comments probably
    # new_date = arrow.Arrow(2020, 2, 2, 15, 6, 1).datetime,
    # assert trade.close_date != new_date
    # # Close should NOT update close_date if the trade has been closed already
    # assert trade.is_open is False
    # trade.close_date = new_date
    # trade.close(0.02)
    # assert trade.close_date == new_date


@ pytest.mark.usefixtures("init_persistence")
def test_update_with_binance_short(limit_short_order, limit_exit_short_order, fee, caplog):
    """
        10 minute short limit trade on binance

        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per day
        open_rate: 0.00001173 base
        close_rate: 0.00001099 base
        amount: 90.99181073 crypto
        stake_amount: 0.0010673339398629 base
        borrowed: 90.99181073  crypto
        time-periods: 10 minutes(rounds up to 1/24 time-period of 1 day)
        interest: borrowed * interest_rate * time-periods
                    = 90.99181073 * 0.0005 * 1/24 = 0.0018956627235416667 crypto
        open_value: (amount * open_rate) - (amount * open_rate * fee)
            = 90.99181073 * 0.00001173 - 90.99181073 * 0.00001173 * 0.0025
            = 0.0010646656050132426
        amount_closed: amount + interest = 90.99181073 + 0.0018956627235416667 = 90.99370639272354
        close_value: (amount_closed * close_rate) + (amount_closed * close_rate * fee)
            = (90.99370639272354 * 0.00001099) + (90.99370639272354 * 0.00001099 * 0.0025)
            = 0.0010025208853391716
        total_profit = open_value - close_value
            = 0.0010646656050132426 - 0.0010025208853391716
            = 0.00006214471967407108
        total_profit_percentage = (close_value - open_value) / stake_amount
            = 0.00006214471967407108 / 0.0010673339398629
            = 0.05822425142973869

    """
    trade = Trade(
        id=2,
        pair='ETH/BTC',
        stake_amount=0.0010673339398629,
        open_rate=0.01,
        amount=5,
        is_open=True,
        open_date=datetime.utcnow() - timedelta(hours=0, minutes=10),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        # borrowed=90.99181073,
        interest_rate=0.0005,
        exchange='binance',
        interest_mode=InterestMode.HOURSPERDAY
    )
    # assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.borrowed == 0.0
    assert trade.is_short is None
    # trade.open_order_id = 'something'
    trade.update(limit_short_order)
    # assert trade.open_order_id is None
    assert trade.open_rate == 0.00001173
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.borrowed == 90.99181073
    assert trade.is_short is True
    assert trade.stop_loss == 0.00001300
    assert trade.liquidation_price == 0.00001300
    assert log_has_re(r"LIMIT_SELL has been fulfilled for Trade\(id=2, "
                      r"pair=ETH/BTC, amount=90.99181073, open_rate=0.00001173, open_since=.*\).",
                      caplog)
    caplog.clear()
    # trade.open_order_id = 'something'
    trade.update(limit_exit_short_order)
    # assert trade.open_order_id is None
    assert trade.close_rate == 0.00001099
    assert trade.close_profit == 0.05822425
    assert trade.close_date is not None
    assert log_has_re(r"LIMIT_BUY has been fulfilled for Trade\(id=2, "
                      r"pair=ETH/BTC, amount=90.99181073, open_rate=0.00001173, open_since=.*\).",
                      caplog)


@ pytest.mark.usefixtures("init_persistence")
def test_update_market_order_short(
    market_short_order,
    market_exit_short_order,
    fee,
    caplog
):
    """
        10 minute short market trade on Kraken at 3x leverage
        Short trade
        fee: 0.25% base
        interest_rate: 0.05% per 4 hrs
        open_rate: 0.00004173 base
        close_rate: 0.00004099 base
        amount: = 275.97543219 crypto
        stake_amount: 0.0038388182617629
        borrowed: 275.97543219  crypto
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
        interest: borrowed * interest_rate * time-periods
                    = 275.97543219 * 0.0005 * 1 = 0.137987716095 crypto
        open_value: (amount * open_rate) - (amount * open_rate * fee)
            = 275.97543219 * 0.00004173 - 275.97543219 * 0.00004173 * 0.0025
            = 0.011487663648325479
        amount_closed: amount + interest = 275.97543219 + 0.137987716095 = 276.113419906095
        close_value: (amount_closed * close_rate) + (amount_closed * close_rate * fee)
            = (276.113419906095 * 0.00004099) + (276.113419906095 * 0.00004099 * 0.0025)
            = 0.01134618380465571
        total_profit = open_value - close_value
            = 0.011487663648325479 - 0.01134618380465571
            = 0.00014147984366976937
        total_profit_percentage = total_profit / stake_amount
        = 0.00014147984366976937 / 0.0038388182617629
        = 0.036855051222142936
    """
    trade = Trade(
        id=1,
        pair='ETH/BTC',
        stake_amount=0.0038388182617629,
        amount=5,
        open_rate=0.01,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.utcnow() - timedelta(hours=0, minutes=10),
        leverage=3.0,
        interest_rate=0.0005,
        exchange='kraken',
        interest_mode=InterestMode.HOURSPER4
    )
    trade.open_order_id = 'something'
    trade.update(market_short_order)
    assert trade.leverage == 3.0
    assert trade.is_short == True
    assert trade.open_order_id is None
    assert trade.open_rate == 0.00004173
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.interest_rate == 0.0005
    assert trade.stop_loss == 0.00004300
    assert trade.liquidation_price == 0.00004300
    # The logger also has the exact same but there's some spacing in there
    assert log_has_re(r"MARKET_SELL has been fulfilled for Trade\(id=1, "
                      r"pair=ETH/BTC, amount=275.97543219, open_rate=0.00004173, open_since=.*\).",
                      caplog)
    caplog.clear()
    trade.is_open = True
    trade.open_order_id = 'something'
    trade.update(market_exit_short_order)
    assert trade.open_order_id is None
    assert trade.close_rate == 0.00004099
    assert trade.close_profit == 0.03685505
    assert trade.close_date is not None
    # TODO-mg: The amount should maybe be the opening amount + the interest
    # TODO-mg: Uncomment the next assert and make it work.
    # The logger also has the exact same but there's some spacing in there
    assert log_has_re(r"MARKET_BUY has been fulfilled for Trade\(id=1, "
                      r"pair=ETH/BTC, amount=275.97543219, open_rate=0.00004173, open_since=.*\).",
                      caplog)


@ pytest.mark.usefixtures("init_persistence")
def test_calc_profit_short(market_short_order, market_exit_short_order, fee):
    """
        Market trade on Kraken at 3x leverage
        Short trade
        fee: 0.25% base or 0.3%
        interest_rate: 0.05%, 0.25% per 4 hrs
        open_rate: 0.00004173 base
        close_rate: 0.00004099 base
        stake_amount: 0.0038388182617629
        amount: = 275.97543219 crypto
        borrowed: 275.97543219  crypto
        time-periods: 10 minutes(rounds up to 1 time-period of 4hrs)
                        5 hours = 5/4

        interest: borrowed * interest_rate * time-periods
                    = 275.97543219 * 0.0005 * 1 = 0.137987716095 crypto
                    = 275.97543219 * 0.00025 * 5/4 = 0.086242322559375 crypto
                    = 275.97543219 * 0.0005 * 5/4 = 0.17248464511875 crypto
                    = 275.97543219 * 0.00025 * 1 = 0.0689938580475 crypto
        open_value: (amount * open_rate) - (amount * open_rate * fee)
            = (275.97543219 * 0.00004173) - (275.97543219 * 0.00004173 * 0.0025) = 0.011487663648325479
        amount_closed: amount + interest
            = 275.97543219 + 0.137987716095 = 276.113419906095
            = 275.97543219 + 0.086242322559375 = 276.06167451255936
            = 275.97543219 + 0.17248464511875 = 276.14791683511874
            = 275.97543219 + 0.0689938580475 = 276.0444260480475
        close_value: (amount_closed * close_rate) + (amount_closed * close_rate * fee)
            (276.113419906095 * 0.00004374) + (276.113419906095 * 0.00004374 * 0.0025) = 0.012107393989159325
            (276.06167451255936 * 0.00000437) + (276.06167451255936 * 0.00000437 * 0.0025) = 0.0012094054914139338
            (276.14791683511874 * 0.00004374) + (276.14791683511874 * 0.00004374 * 0.003) = 0.012114946012015198
            (276.0444260480475 * 0.00000437) + (276.0444260480475 * 0.00000437 * 0.003) = 0.0012099330842554573
        total_profit = open_value - close_value
            = print(0.011487663648325479 - 0.012107393989159325) = -0.0006197303408338461
            = print(0.011487663648325479 - 0.0012094054914139338) = 0.010278258156911545
            = print(0.011487663648325479 - 0.012114946012015198) = -0.0006272823636897188
            = print(0.011487663648325479 - 0.0012099330842554573) = 0.010277730564070022
        total_profit_percentage = (close_value - open_value) / stake_amount
            (0.011487663648325479 - 0.012107393989159325)/0.0038388182617629  = -0.16143779115744006
            (0.011487663648325479 - 0.0012094054914139338)/0.0038388182617629 = 2.677453699564163
            (0.011487663648325479 - 0.012114946012015198)/0.0038388182617629  = -0.16340506919482353
            (0.011487663648325479 - 0.0012099330842554573)/0.0038388182617629 = 2.677316263299785

    """
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.0038388182617629,
        amount=5,
        open_rate=0.00001099,
        open_date=datetime.utcnow() - timedelta(hours=0, minutes=10),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='kraken',
        is_short=True,
        leverage=3.0,
        interest_rate=0.0005,
        interest_mode=InterestMode.HOURSPER4
    )
    trade.open_order_id = 'something'
    trade.update(market_short_order)  # Buy @ 0.00001099
    # Custom closing rate and regular fee rate

    # Higher than open rate
    assert trade.calc_profit(rate=0.00004374, interest_rate=0.0005) == round(-0.00061973, 8)
    assert trade.calc_profit_ratio(
        rate=0.00004374, interest_rate=0.0005) == round(-0.16143779115744006, 8)

    # Lower than open rate
    trade.open_date = datetime.utcnow() - timedelta(hours=5, minutes=0)
    assert trade.calc_profit(rate=0.00000437, interest_rate=0.00025) == round(0.01027826, 8)
    assert trade.calc_profit_ratio(
        rate=0.00000437, interest_rate=0.00025) == round(2.677453699564163, 8)

    # Custom closing rate and custom fee rate
    # Higher than open rate
    assert trade.calc_profit(rate=0.00004374, fee=0.003,
                             interest_rate=0.0005) == round(-0.00062728, 8)
    assert trade.calc_profit_ratio(rate=0.00004374, fee=0.003,
                                   interest_rate=0.0005) == round(-0.16340506919482353, 8)

    # Lower than open rate
    trade.open_date = datetime.utcnow() - timedelta(hours=0, minutes=10)
    assert trade.calc_profit(rate=0.00000437, fee=0.003,
                             interest_rate=0.00025) == round(0.01027773, 8)
    assert trade.calc_profit_ratio(rate=0.00000437, fee=0.003,
                                   interest_rate=0.00025) == round(2.677316263299785, 8)

    # Test when we apply a Sell order. Sell higher than open rate @ 0.00001173
    trade.update(market_exit_short_order)
    assert trade.calc_profit() == round(0.00014148, 8)
    assert trade.calc_profit_ratio() == round(0.03685505, 8)

    # Test with a custom fee rate on the close trade
    # assert trade.calc_profit(fee=0.003) == 0.00006163
    # assert trade.calc_profit_ratio(fee=0.003) == 0.06147824


def test_adjust_stop_loss_short(fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=5,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        open_rate=1,
        max_rate=1,
        is_short=True,
        interest_mode=InterestMode.HOURSPERDAY
    )
    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 1.05
    assert trade.stop_loss_pct == 0.05
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == 0.05
    # Get percent of profit with a lower rate
    trade.adjust_stop_loss(1.04, 0.05)
    assert trade.stop_loss == 1.05
    assert trade.stop_loss_pct == 0.05
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == 0.05
    # Get percent of profit with a custom rate (Higher than open rate)
    trade.adjust_stop_loss(0.7, 0.1)
    # If the price goes down to 0.7, with a trailing stop of 0.1, the new stoploss at 0.1 above 0.7 would be 0.7*0.1 higher
    assert round(trade.stop_loss, 8) == 0.77
    assert trade.stop_loss_pct == 0.1
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == 0.05
    # current rate lower again ... should not change
    trade.adjust_stop_loss(0.8, -0.1)
    assert round(trade.stop_loss, 8) == 0.77
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == 0.05
    # current rate higher... should raise stoploss
    trade.adjust_stop_loss(0.6, -0.1)
    assert round(trade.stop_loss, 8) == 0.66
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == 0.05
    #  Initial is true but stop_loss set - so doesn't do anything
    trade.adjust_stop_loss(0.3, -0.1, True)
    assert round(trade.stop_loss, 8) == 0.66  # TODO-mg: What is this test?
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == 0.05
    assert trade.stop_loss_pct == 0.1
    trade.set_liquidation_price(0.63)
    trade.adjust_stop_loss(0.59, -0.1)
    assert trade.stop_loss == 0.63
    assert trade.liquidation_price == 0.63

    # TODO-mg: Do a test with a trade that has a liquidation price


@ pytest.mark.usefixtures("init_persistence")
@ pytest.mark.parametrize('use_db', [True, False])
def test_get_open_short(fee, use_db):
    Trade.use_db = use_db
    Trade.reset_trades()
    create_mock_trades_with_leverage(fee, use_db)
    assert len(Trade.get_open_trades()) == 5
    Trade.use_db = True


def test_stoploss_reinitialization_short(default_conf, fee):
    init_db(default_conf['db_url'])
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        open_date=arrow.utcnow().shift(hours=-2).datetime,
        amount=10,
        fee_close=fee.return_value,
        exchange='binance',
        open_rate=1,
        max_rate=1,
        is_short=True,
        leverage=3.0,
        interest_mode=InterestMode.HOURSPERDAY
    )
    trade.adjust_stop_loss(trade.open_rate, -0.05, True)
    assert trade.stop_loss == 1.05
    assert trade.stop_loss_pct == 0.05
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == 0.05
    Trade.query.session.add(trade)
    # Lower stoploss
    Trade.stoploss_reinitialization(-0.06)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 1.06
    assert trade_adj.stop_loss_pct == 0.06
    assert trade_adj.initial_stop_loss == 1.06
    assert trade_adj.initial_stop_loss_pct == 0.06
    # Raise stoploss
    Trade.stoploss_reinitialization(-0.04)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 1.04
    assert trade_adj.stop_loss_pct == 0.04
    assert trade_adj.initial_stop_loss == 1.04
    assert trade_adj.initial_stop_loss_pct == 0.04
    # Trailing stoploss
    trade.adjust_stop_loss(0.98, -0.04)
    assert trade_adj.stop_loss == 1.0192
    assert trade_adj.initial_stop_loss == 1.04
    Trade.stoploss_reinitialization(-0.04)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    # Stoploss should not change in this case.
    assert trade_adj.stop_loss == 1.0192
    assert trade_adj.stop_loss_pct == 0.04
    assert trade_adj.initial_stop_loss == 1.04
    assert trade_adj.initial_stop_loss_pct == 0.04
    # Stoploss can't go above liquidation price
    trade_adj.set_liquidation_price(1.0)
    trade.adjust_stop_loss(0.97, -0.04)
    assert trade_adj.stop_loss == 1.0
    assert trade_adj.stop_loss == 1.0


@ pytest.mark.usefixtures("init_persistence")
@ pytest.mark.parametrize('use_db', [True, False])
def test_total_open_trades_stakes_short(fee, use_db):
    Trade.use_db = use_db
    Trade.reset_trades()
    res = Trade.total_open_trades_stakes()
    assert res == 0
    create_mock_trades_with_leverage(fee, use_db)
    res = Trade.total_open_trades_stakes()
    assert res == 15.133
    Trade.use_db = True


@ pytest.mark.usefixtures("init_persistence")
def test_get_best_pair_short(fee):
    res = Trade.get_best_pair()
    assert res is None
    create_mock_trades_with_leverage(fee)
    res = Trade.get_best_pair()
    assert len(res) == 2
    assert res[0] == 'DOGE/BTC'
    assert res[1] == 0.17524390243902502
