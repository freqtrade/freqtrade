# pragma pylint: disable=missing-docstring, C0103
import logging
from datetime import datetime, timedelta, timezone
from math import isclose
from pathlib import Path
from types import FunctionType
from unittest.mock import MagicMock

import arrow
import pytest
from sqlalchemy import create_engine, inspect, text

from freqtrade import constants
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.persistence import LocalTrade, Order, Trade, clean_dry_run_db, init_db
from tests.conftest import create_mock_trades, log_has, log_has_re


def test_init_create_session(default_conf):
    # Check if init create a session
    init_db(default_conf['db_url'], default_conf['dry_run'])
    assert hasattr(Trade, '_session')
    assert 'scoped_session' in type(Trade._session).__name__


def test_init_custom_db_url(default_conf, tmpdir):
    # Update path to a value other than default, but still in-memory
    filename = f"{tmpdir}/freqtrade2_test.sqlite"
    assert not Path(filename).is_file()

    default_conf.update({'db_url': f'sqlite:///{filename}'})

    init_db(default_conf['db_url'], default_conf['dry_run'])
    assert Path(filename).is_file()


def test_init_invalid_db_url(default_conf):
    # Update path to a value other than default, but still in-memory
    default_conf.update({'db_url': 'unknown:///some.url'})
    with pytest.raises(OperationalException, match=r'.*no valid database URL*'):
        init_db(default_conf['db_url'], default_conf['dry_run'])


def test_init_prod_db(default_conf, mocker):
    default_conf.update({'dry_run': False})
    default_conf.update({'db_url': constants.DEFAULT_DB_PROD_URL})

    create_engine_mock = mocker.patch('freqtrade.persistence.models.create_engine', MagicMock())

    init_db(default_conf['db_url'], default_conf['dry_run'])
    assert create_engine_mock.call_count == 1
    assert create_engine_mock.mock_calls[0][1][0] == 'sqlite:///tradesv3.sqlite'


def test_init_dryrun_db(default_conf, tmpdir):
    filename = f"{tmpdir}/freqtrade2_prod.sqlite"
    assert not Path(filename).is_file()
    default_conf.update({
        'dry_run': True,
        'db_url': f'sqlite:///{filename}'
    })

    init_db(default_conf['db_url'], default_conf['dry_run'])
    assert Path(filename).is_file()


@pytest.mark.usefixtures("init_persistence")
def test_update_limit_order(limit_buy_order_usdt, limit_sell_order_usdt, fee, caplog):
    """
        On this test we will buy and sell a crypto currency.
        fee: 0.25% quote
        open_rate: 2.00 quote
        close_rate: 2.20 quote
        amount: = 30.0 crypto
        stake_amount
            60.0  quote
        borrowed
             0 quote
        open_value: (amount * open_rate) + (amount * open_rate * fee)
             30 * 2 + 30 * 2 * 0.0025 = 60.15 quote
        close_value:
            (amount * close_rate) - (amount * close_rate * fee) - interest
            (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) = 65.835
        total_profit:
            close_value - open_value
            65.835 - 60.15             = 5.685
        total_profit_ratio:
            ((close_value/open_value) - 1) * leverage
            ((65.835 / 60.15) - 1)  * 1 = 0.0945137157107232

    """

    trade = Trade(
        id=2,
        pair='ADA/USDT',
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        open_date=arrow.utcnow().datetime,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
    )
    assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None

    trade.open_order_id = 'something'
    trade.update(limit_buy_order_usdt)
    assert trade.open_order_id is None
    assert trade.open_rate == 2.00
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(r"LIMIT_BUY has been fulfilled for Trade\(id=2, "
                      r"pair=ADA/USDT, amount=30.00000000, open_rate=2.00000000, open_since=.*\).",
                      caplog)

    caplog.clear()
    trade.open_order_id = 'something'
    trade.update(limit_sell_order_usdt)
    assert trade.open_order_id is None
    assert trade.close_rate == 2.20
    assert trade.close_profit == round(0.0945137157107232, 8)
    assert trade.close_date is not None
    assert log_has_re(r"LIMIT_SELL has been fulfilled for Trade\(id=2, "
                      r"pair=ADA/USDT, amount=30.00000000, open_rate=2.00000000, open_since=.*\).",
                      caplog)
    caplog.clear()


@pytest.mark.usefixtures("init_persistence")
def test_update_market_order(market_buy_order_usdt, market_sell_order_usdt, fee, caplog):
    trade = Trade(
        id=1,
        pair='ADA/USDT',
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=arrow.utcnow().datetime,
        exchange='binance',
    )

    trade.open_order_id = 'something'
    trade.update(market_buy_order_usdt)
    assert trade.open_order_id is None
    assert trade.open_rate == 2.0
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(r"MARKET_BUY has been fulfilled for Trade\(id=1, "
                      r"pair=ADA/USDT, amount=30.00000000, open_rate=2.00000000, open_since=.*\).",
                      caplog)

    caplog.clear()
    trade.is_open = True
    trade.open_order_id = 'something'
    trade.update(market_sell_order_usdt)
    assert trade.open_order_id is None
    assert trade.close_rate == 2.2
    assert trade.close_profit == round(0.0945137157107232, 8)
    assert trade.close_date is not None
    assert log_has_re(r"MARKET_SELL has been fulfilled for Trade\(id=1, "
                      r"pair=ADA/USDT, amount=30.00000000, open_rate=2.00000000, open_since=.*\).",
                      caplog)


@pytest.mark.usefixtures("init_persistence")
def test_calc_open_close_trade_price(limit_buy_order_usdt, limit_sell_order_usdt, fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
    )

    trade.open_order_id = 'something'
    trade.update(limit_buy_order_usdt)
    assert trade._calc_open_trade_value() == 60.15
    trade.update(limit_sell_order_usdt)
    assert isclose(trade.calc_close_trade_value(), 65.835)

    # Profit in USDT
    assert trade.calc_profit() == 5.685

    # Profit in percent
    assert trade.calc_profit_ratio() == round(0.0945137157107232, 8)


@pytest.mark.usefixtures("init_persistence")
def test_trade_close(limit_buy_order_usdt, limit_sell_order_usdt, fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=arrow.Arrow(2020, 2, 1, 15, 5, 1).datetime,
        exchange='binance',
    )
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.is_open is True
    trade.close(2.2)
    assert trade.is_open is False
    assert trade.close_profit == round(0.0945137157107232, 8)
    assert trade.close_date is not None

    new_date = arrow.Arrow(2020, 2, 2, 15, 6, 1).datetime,
    assert trade.close_date != new_date
    # Close should NOT update close_date if the trade has been closed already
    assert trade.is_open is False
    trade.close_date = new_date
    trade.close(2.2)
    assert trade.close_date == new_date


@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price_exception(limit_buy_order_usdt, fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
    )

    trade.open_order_id = 'something'
    trade.update(limit_buy_order_usdt)
    assert trade.calc_close_trade_value() == 0.0


@pytest.mark.usefixtures("init_persistence")
def test_update_open_order(limit_buy_order_usdt):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        fee_open=0.1,
        fee_close=0.1,
        exchange='binance',
    )

    assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None

    limit_buy_order_usdt['status'] = 'open'
    trade.update(limit_buy_order_usdt)

    assert trade.open_order_id is None
    assert trade.close_profit is None
    assert trade.close_date is None


@pytest.mark.usefixtures("init_persistence")
def test_update_invalid_order(limit_buy_order_usdt):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        fee_open=0.1,
        fee_close=0.1,
        exchange='binance',
    )
    limit_buy_order_usdt['type'] = 'invalid'
    with pytest.raises(ValueError, match=r'Unknown order type'):
        trade.update(limit_buy_order_usdt)


@pytest.mark.usefixtures("init_persistence")
def test_calc_open_trade_value(limit_buy_order_usdt, fee):
    """
        fee: 0.25 %, 0.3% quote
        open_rate: 2.00 quote
        amount: = 30.0 crypto
        stake_amount
            60.0  quote
        open_value: (amount * open_rate) + (amount * open_rate * fee)
        0.25% fee
            30 * 2 + 30 * 2 * 0.0025 = 60.15 quote
        0.3% fee
            30 * 2 + 30 * 2 * 0.003  = 60.18 quote
    """
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
    )
    trade.open_order_id = 'open_trade'
    trade.update(limit_buy_order_usdt)  # Buy @ 2.0

    # Get the open rate price with the standard fee rate
    assert trade._calc_open_trade_value() == 60.15
    trade.fee_open = 0.003
    # Get the open rate price with a custom fee rate
    assert trade._calc_open_trade_value() == 60.18


@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price(limit_buy_order_usdt, limit_sell_order_usdt, fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
    )
    trade.open_order_id = 'close_trade'
    trade.update(limit_buy_order_usdt)  # Buy @ 2.0

    # Get the close rate price with a custom close rate and a regular fee rate
    assert trade.calc_close_trade_value(rate=2.5) == 74.8125
    # Get the close rate price with a custom close rate and a custom fee rate
    assert trade.calc_close_trade_value(rate=2.5, fee=0.003) == 74.775
    # Test when we apply a Sell order, and ask price with a custom fee rate
    trade.update(limit_sell_order_usdt)
    assert trade.calc_close_trade_value(fee=0.005) == 65.67


@pytest.mark.usefixtures("init_persistence")
def test_calc_profit(limit_buy_order_usdt, limit_sell_order_usdt, fee):
    """
        arguments:
            fee:
                0.25% quote
                0.30% quote
            open_rate: 2.0 quote
            close_rate:
                1.9 quote
                2.1 quote
                2.2 quote
            amount: = 30.0 crypto
            stake_amount
                60.0  quote
        open_value: (amount * open_rate) + (amount * open_rate * fee)
          0.0025 fee
            30 * 2 + 30 * 2 * 0.0025 = 60.15 quote
            30 * 2 - 30 * 2 * 0.0025 = 59.85 quote
          0.003 fee: Is only applied to close rate in this test
        close_value:
            equations:
                (amount_closed * close_rate) - (amount_closed * close_rate * fee)
            2.1 quote
                (30.00 * 2.1) - (30.00 * 2.1 * 0.0025)   = 62.8425
            1.9 quote
                (30.00 * 1.9) - (30.00 * 1.9 * 0.0025)   = 56.8575
            2.2 quote
                (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) = 65.835
        total_profit:
            equations:
                close_value - open_value
            2.1 quote
                62.8425 - 60.15 = 2.6925
            1.9 quote
                56.8575 - 60.15 = -3.2925
            2.2 quote
                65.835  - 60.15 = 5.685
        total_profit_ratio:
            equations:
                ((close_value/open_value) - 1) * leverage
            2.1 quote
                (62.8425 / 60.15) - 1 = 0.04476309226932673
            1.9 quote
                (56.8575 / 60.15) - 1 = -0.05473815461346632
            2.2 quote
                (65.835 / 60.15) - 1  = 0.0945137157107232
        fee: 0.003
            close_value:
                2.1 quote: (30.00 * 2.1) - (30.00 * 2.1 * 0.003) = 62.811
                1.9 quote: (30.00 * 1.9) - (30.00 * 1.9 * 0.003) = 56.829
                2.2 quote: (30.00 * 2.2) - (30.00 * 2.2 * 0.003) = 65.802
            total_profit
                fee: 0.003
                    2.1 quote: 62.811 - 60.15 = 2.6610000000000014
                    1.9 quote: 56.829 - 60.15 = -3.320999999999998
                    2.2 quote: 65.802 - 60.15 = 5.652000000000008
            total_profit_ratio
                fee: 0.003
                    2.1 quote: (62.811 / 60.15) - 1 = 0.04423940149625927
                    1.9 quote: (56.829 / 60.15) - 1 = -0.05521197007481293
                    2.2 quote: (65.802 / 60.15) - 1 = 0.09396508728179565
    """
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
    )
    trade.open_order_id = 'something'
    trade.update(limit_buy_order_usdt)  # Buy @ 2.0

    # Custom closing rate and regular fee rate
    # Higher than open rate - 2.1 quote
    assert trade.calc_profit(rate=2.1) == 2.6925
    # Lower than open rate - 1.9 quote
    assert trade.calc_profit(rate=1.9) == round(-3.292499999999997, 8)

    # fee 0.003
    # Higher than open rate - 2.1 quote
    assert trade.calc_profit(rate=2.1, fee=0.003) == 2.661
    # Lower than open rate - 1.9 quote
    assert trade.calc_profit(rate=1.9, fee=0.003) == round(-3.320999999999998, 8)

    # Test when we apply a Sell order. Sell higher than open rate @ 2.2
    trade.update(limit_sell_order_usdt)
    assert trade.calc_profit() == round(5.684999999999995, 8)

    # Test with a custom fee rate on the close trade
    assert trade.calc_profit(fee=0.003) == round(5.652000000000008, 8)


@pytest.mark.usefixtures("init_persistence")
def test_calc_profit_ratio(limit_buy_order_usdt, limit_sell_order_usdt, fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance'
    )
    trade.open_order_id = 'something'
    trade.update(limit_buy_order_usdt)  # Buy @ 2.0

    # Higher than open rate - 2.1 quote
    assert trade.calc_profit_ratio(rate=2.1) == round(0.04476309226932673, 8)
    # Lower than open rate - 1.9 quote
    assert trade.calc_profit_ratio(rate=1.9) == round(-0.05473815461346632, 8)

    # fee 0.003
    # Higher than open rate - 2.1 quote
    assert trade.calc_profit_ratio(rate=2.1, fee=0.003) == round(0.04423940149625927, 8)
    # Lower than open rate - 1.9 quote
    assert trade.calc_profit_ratio(rate=1.9, fee=0.003) == round(-0.05521197007481293, 8)

    # Test when we apply a Sell order. Sell higher than open rate @ 2.2
    trade.update(limit_sell_order_usdt)
    assert trade.calc_profit_ratio() == round(0.0945137157107232, 8)

    # Test with a custom fee rate on the close trade
    assert trade.calc_profit_ratio(fee=0.003) == round(0.09396508728179565, 8)

    trade.open_trade_value = 0.0
    assert trade.calc_profit_ratio(fee=0.003) == 0.0


@pytest.mark.usefixtures("init_persistence")
def test_clean_dry_run_db(default_conf, fee):

    # Simulate dry_run entries
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=0.001,
        amount=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        exchange='binance',
        open_order_id='dry_run_buy_12345'
    )
    Trade.query.session.add(trade)

    trade = Trade(
        pair='ETC/BTC',
        stake_amount=0.001,
        amount=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        exchange='binance',
        open_order_id='dry_run_sell_12345'
    )
    Trade.query.session.add(trade)

    # Simulate prod entry
    trade = Trade(
        pair='ETC/BTC',
        stake_amount=0.001,
        amount=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        exchange='binance',
        open_order_id='prod_buy_12345'
    )
    Trade.query.session.add(trade)

    # We have 3 entries: 2 dry_run, 1 prod
    assert len(Trade.query.filter(Trade.open_order_id.isnot(None)).all()) == 3

    clean_dry_run_db()

    # We have now only the prod
    assert len(Trade.query.filter(Trade.open_order_id.isnot(None)).all()) == 1


def test_migrate_new(mocker, default_conf, fee, caplog):
    """
    Test Database migration (starting with new pairformat)
    """
    caplog.set_level(logging.DEBUG)
    amount = 103.223
    # Always create all columns apart from the last!
    create_table_old = """CREATE TABLE IF NOT EXISTS "trades" (
                                id INTEGER NOT NULL,
                                exchange VARCHAR NOT NULL,
                                pair VARCHAR NOT NULL,
                                is_open BOOLEAN NOT NULL,
                                fee FLOAT NOT NULL,
                                open_rate FLOAT,
                                close_rate FLOAT,
                                close_profit FLOAT,
                                stake_amount FLOAT NOT NULL,
                                amount FLOAT,
                                open_date DATETIME NOT NULL,
                                close_date DATETIME,
                                open_order_id VARCHAR,
                                stop_loss FLOAT,
                                initial_stop_loss FLOAT,
                                max_rate FLOAT,
                                sell_reason VARCHAR,
                                strategy VARCHAR,
                                ticker_interval INTEGER,
                                stoploss_order_id VARCHAR,
                                PRIMARY KEY (id),
                                CHECK (is_open IN (0, 1))
                                );"""
    insert_table_old = """INSERT INTO trades (exchange, pair, is_open, fee,
                          open_rate, stake_amount, amount, open_date,
                          stop_loss, initial_stop_loss, max_rate, ticker_interval,
                          open_order_id, stoploss_order_id)
                          VALUES ('binance', 'ETC/BTC', 1, {fee},
                          0.00258580, {stake}, {amount},
                          '2019-11-28 12:44:24.000000',
                          0.0, 0.0, 0.0, '5m',
                          'buy_order', 'stop_order_id222')
                          """.format(fee=fee.return_value,
                                     stake=default_conf.get("stake_amount"),
                                     amount=amount
                                     )
    engine = create_engine('sqlite://')
    mocker.patch('freqtrade.persistence.models.create_engine', lambda *args, **kwargs: engine)

    # Create table using the old format
    with engine.begin() as connection:
        connection.execute(text(create_table_old))
        connection.execute(text("create index ix_trades_is_open on trades(is_open)"))
        connection.execute(text("create index ix_trades_pair on trades(pair)"))
        connection.execute(text(insert_table_old))

        # fake previous backup
        connection.execute(text("create table trades_bak as select * from trades"))

        connection.execute(text("create table trades_bak1 as select * from trades"))
    # Run init to test migration
    init_db(default_conf['db_url'], default_conf['dry_run'])

    assert len(Trade.query.filter(Trade.id == 1).all()) == 1
    trade = Trade.query.filter(Trade.id == 1).first()
    assert trade.fee_open == fee.return_value
    assert trade.fee_close == fee.return_value
    assert trade.open_rate_requested is None
    assert trade.close_rate_requested is None
    assert trade.is_open == 1
    assert trade.amount == amount
    assert trade.amount_requested == amount
    assert trade.stake_amount == default_conf.get("stake_amount")
    assert trade.pair == "ETC/BTC"
    assert trade.exchange == "binance"
    assert trade.max_rate == 0.0
    assert trade.min_rate is None
    assert trade.stop_loss == 0.0
    assert trade.initial_stop_loss == 0.0
    assert trade.sell_reason is None
    assert trade.strategy is None
    assert trade.timeframe == '5m'
    assert trade.stoploss_order_id == 'stop_order_id222'
    assert trade.stoploss_last_update is None
    assert log_has("trying trades_bak1", caplog)
    assert log_has("trying trades_bak2", caplog)
    assert log_has("Running database migration for trades - backup: trades_bak2", caplog)
    assert trade.open_trade_value == trade._calc_open_trade_value()
    assert trade.close_profit_abs is None

    assert log_has("Moving open orders to Orders table.", caplog)
    orders = Order.query.all()
    assert len(orders) == 2
    assert orders[0].order_id == 'buy_order'
    assert orders[0].ft_order_side == 'buy'

    assert orders[1].order_id == 'stop_order_id222'
    assert orders[1].ft_order_side == 'stoploss'

    caplog.clear()
    # Drop latest column
    with engine.begin() as connection:
        connection.execute(text("alter table orders rename to orders_bak"))
    inspector = inspect(engine)

    with engine.begin() as connection:
        for index in inspector.get_indexes('orders_bak'):
            connection.execute(text(f"drop index {index['name']}"))
        # Recreate table
        connection.execute(text("""
            CREATE TABLE orders (
                id INTEGER NOT NULL,
                ft_trade_id INTEGER,
                ft_order_side VARCHAR NOT NULL,
                ft_pair VARCHAR NOT NULL,
                ft_is_open BOOLEAN NOT NULL,
                order_id VARCHAR NOT NULL,
                status VARCHAR,
                symbol VARCHAR,
                order_type VARCHAR,
                side VARCHAR,
                price FLOAT,
                amount FLOAT,
                filled FLOAT,
                remaining FLOAT,
                cost FLOAT,
                order_date DATETIME,
                order_filled_date DATETIME,
                order_update_date DATETIME,
                PRIMARY KEY (id),
                CONSTRAINT _order_pair_order_id UNIQUE (ft_pair, order_id),
                FOREIGN KEY(ft_trade_id) REFERENCES trades (id)
            )
            """))

        connection.execute(text("""
        insert into orders ( id, ft_trade_id, ft_order_side, ft_pair, ft_is_open, order_id, status,
            symbol, order_type, side, price, amount, filled, remaining, cost, order_date,
            order_filled_date, order_update_date)
            select id, ft_trade_id, ft_order_side, ft_pair, ft_is_open, order_id, status,
            symbol, order_type, side, price, amount, filled, remaining, cost, order_date,
            order_filled_date, order_update_date
            from orders_bak
        """))

    # Run init to test migration
    init_db(default_conf['db_url'], default_conf['dry_run'])

    assert log_has("trying orders_bak1", caplog)

    orders = Order.query.all()
    assert len(orders) == 2
    assert orders[0].order_id == 'buy_order'
    assert orders[0].ft_order_side == 'buy'

    assert orders[1].order_id == 'stop_order_id222'
    assert orders[1].ft_order_side == 'stoploss'


def test_migrate_mid_state(mocker, default_conf, fee, caplog):
    """
    Test Database migration (starting with new pairformat)
    """
    caplog.set_level(logging.DEBUG)
    amount = 103.223
    create_table_old = """CREATE TABLE IF NOT EXISTS "trades" (
                                id INTEGER NOT NULL,
                                exchange VARCHAR NOT NULL,
                                pair VARCHAR NOT NULL,
                                is_open BOOLEAN NOT NULL,
                                fee_open FLOAT NOT NULL,
                                fee_close FLOAT NOT NULL,
                                open_rate FLOAT,
                                close_rate FLOAT,
                                close_profit FLOAT,
                                stake_amount FLOAT NOT NULL,
                                amount FLOAT,
                                open_date DATETIME NOT NULL,
                                close_date DATETIME,
                                open_order_id VARCHAR,
                                PRIMARY KEY (id),
                                CHECK (is_open IN (0, 1))
                                );"""
    insert_table_old = """INSERT INTO trades (exchange, pair, is_open, fee_open, fee_close,
                          open_rate, stake_amount, amount, open_date)
                          VALUES ('binance', 'ETC/BTC', 1, {fee}, {fee},
                          0.00258580, {stake}, {amount},
                          '2019-11-28 12:44:24.000000')
                          """.format(fee=fee.return_value,
                                     stake=default_conf.get("stake_amount"),
                                     amount=amount
                                     )
    engine = create_engine('sqlite://')
    mocker.patch('freqtrade.persistence.models.create_engine', lambda *args, **kwargs: engine)

    # Create table using the old format
    with engine.begin() as connection:
        connection.execute(text(create_table_old))
        connection.execute(text(insert_table_old))

    # Run init to test migration
    init_db(default_conf['db_url'], default_conf['dry_run'])

    assert len(Trade.query.filter(Trade.id == 1).all()) == 1
    trade = Trade.query.filter(Trade.id == 1).first()
    assert trade.fee_open == fee.return_value
    assert trade.fee_close == fee.return_value
    assert trade.open_rate_requested is None
    assert trade.close_rate_requested is None
    assert trade.is_open == 1
    assert trade.amount == amount
    assert trade.stake_amount == default_conf.get("stake_amount")
    assert trade.pair == "ETC/BTC"
    assert trade.exchange == "binance"
    assert trade.max_rate == 0.0
    assert trade.stop_loss == 0.0
    assert trade.initial_stop_loss == 0.0
    assert trade.open_trade_value == trade._calc_open_trade_value()
    assert log_has("trying trades_bak0", caplog)
    assert log_has("Running database migration for trades - backup: trades_bak0", caplog)


def test_adjust_stop_loss(fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=30.0,
        amount=30,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        open_rate=1,
        max_rate=1,
    )

    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # Get percent of profit with a lower rate
    trade.adjust_stop_loss(0.96, 0.05)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # Get percent of profit with a custom rate (Higher than open rate)
    trade.adjust_stop_loss(1.3, -0.1)
    assert round(trade.stop_loss, 8) == 1.17
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # current rate lower again ... should not change
    trade.adjust_stop_loss(1.2, 0.1)
    assert round(trade.stop_loss, 8) == 1.17
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # current rate higher... should raise stoploss
    trade.adjust_stop_loss(1.4, 0.1)
    assert round(trade.stop_loss, 8) == 1.26
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    #  Initial is true but stop_loss set - so doesn't do anything
    trade.adjust_stop_loss(1.7, 0.1, True)
    assert round(trade.stop_loss, 8) == 1.26
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    assert trade.stop_loss_pct == -0.1


def test_adjust_min_max_rates(fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=30.0,
        amount=30.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='binance',
        open_rate=1,
    )

    trade.adjust_min_max_rates(trade.open_rate, trade.open_rate)
    assert trade.max_rate == 1
    assert trade.min_rate == 1

    # check min adjusted, max remained
    trade.adjust_min_max_rates(0.96, 0.96)
    assert trade.max_rate == 1
    assert trade.min_rate == 0.96

    # check max adjusted, min remains
    trade.adjust_min_max_rates(1.05, 1.05)
    assert trade.max_rate == 1.05
    assert trade.min_rate == 0.96

    # current rate "in the middle" - no adjustment
    trade.adjust_min_max_rates(1.03, 1.03)
    assert trade.max_rate == 1.05
    assert trade.min_rate == 0.96

    # current rate "in the middle" - no adjustment
    trade.adjust_min_max_rates(1.10, 0.91)
    assert trade.max_rate == 1.10
    assert trade.min_rate == 0.91


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize('use_db', [True, False])
def test_get_open(fee, use_db):
    Trade.use_db = use_db
    Trade.reset_trades()

    create_mock_trades(fee, use_db)
    assert len(Trade.get_open_trades()) == 4

    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
def test_to_json(default_conf, fee):

    # Simulate dry_run entries
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=arrow.utcnow().shift(hours=-2).datetime,
        open_rate=0.123,
        exchange='binance',
        buy_tag=None,
        open_order_id='dry_run_buy_12345'
    )
    result = trade.to_json()
    assert isinstance(result, dict)

    assert result == {'trade_id': None,
                      'pair': 'ETH/BTC',
                      'is_open': None,
                      'open_date': trade.open_date.strftime("%Y-%m-%d %H:%M:%S"),
                      'open_timestamp': int(trade.open_date.timestamp() * 1000),
                      'open_order_id': 'dry_run_buy_12345',
                      'close_date': None,
                      'close_timestamp': None,
                      'open_rate': 0.123,
                      'open_rate_requested': None,
                      'open_trade_value': 15.1668225,
                      'fee_close': 0.0025,
                      'fee_close_cost': None,
                      'fee_close_currency': None,
                      'fee_open': 0.0025,
                      'fee_open_cost': None,
                      'fee_open_currency': None,
                      'close_rate': None,
                      'close_rate_requested': None,
                      'amount': 123.0,
                      'amount_requested': 123.0,
                      'stake_amount': 0.001,
                      'trade_duration': None,
                      'trade_duration_s': None,
                      'close_profit': None,
                      'close_profit_pct': None,
                      'close_profit_abs': None,
                      'profit_ratio': None,
                      'profit_pct': None,
                      'profit_abs': None,
                      'sell_reason': None,
                      'sell_order_status': None,
                      'stop_loss_abs': None,
                      'stop_loss_ratio': None,
                      'stop_loss_pct': None,
                      'stoploss_order_id': None,
                      'stoploss_last_update': None,
                      'stoploss_last_update_timestamp': None,
                      'initial_stop_loss_abs': None,
                      'initial_stop_loss_pct': None,
                      'initial_stop_loss_ratio': None,
                      'min_rate': None,
                      'max_rate': None,
                      'strategy': None,
                      'buy_tag': None,
                      'timeframe': None,
                      'exchange': 'binance',
                      }

    # Simulate dry_run entries
    trade = Trade(
        pair='XRP/BTC',
        stake_amount=0.001,
        amount=100.0,
        amount_requested=101.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=arrow.utcnow().shift(hours=-2).datetime,
        close_date=arrow.utcnow().shift(hours=-1).datetime,
        open_rate=0.123,
        close_rate=0.125,
        buy_tag='buys_signal_001',
        exchange='binance',
    )
    result = trade.to_json()
    assert isinstance(result, dict)

    assert result == {'trade_id': None,
                      'pair': 'XRP/BTC',
                      'open_date': trade.open_date.strftime("%Y-%m-%d %H:%M:%S"),
                      'open_timestamp': int(trade.open_date.timestamp() * 1000),
                      'close_date': trade.close_date.strftime("%Y-%m-%d %H:%M:%S"),
                      'close_timestamp': int(trade.close_date.timestamp() * 1000),
                      'open_rate': 0.123,
                      'close_rate': 0.125,
                      'amount': 100.0,
                      'amount_requested': 101.0,
                      'stake_amount': 0.001,
                      'trade_duration': 60,
                      'trade_duration_s': 3600,
                      'stop_loss_abs': None,
                      'stop_loss_pct': None,
                      'stop_loss_ratio': None,
                      'stoploss_order_id': None,
                      'stoploss_last_update': None,
                      'stoploss_last_update_timestamp': None,
                      'initial_stop_loss_abs': None,
                      'initial_stop_loss_pct': None,
                      'initial_stop_loss_ratio': None,
                      'close_profit': None,
                      'close_profit_pct': None,
                      'close_profit_abs': None,
                      'profit_ratio': None,
                      'profit_pct': None,
                      'profit_abs': None,
                      'close_rate_requested': None,
                      'fee_close': 0.0025,
                      'fee_close_cost': None,
                      'fee_close_currency': None,
                      'fee_open': 0.0025,
                      'fee_open_cost': None,
                      'fee_open_currency': None,
                      'is_open': None,
                      'max_rate': None,
                      'min_rate': None,
                      'open_order_id': None,
                      'open_rate_requested': None,
                      'open_trade_value': 12.33075,
                      'sell_reason': None,
                      'sell_order_status': None,
                      'strategy': None,
                      'buy_tag': 'buys_signal_001',
                      'timeframe': None,
                      'exchange': 'binance',
                      }


def test_stoploss_reinitialization(default_conf, fee):
    init_db(default_conf['db_url'])
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=30.0,
        fee_open=fee.return_value,
        open_date=arrow.utcnow().shift(hours=-2).datetime,
        amount=30.0,
        fee_close=fee.return_value,
        exchange='binance',
        open_rate=1,
        max_rate=1,
    )

    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    Trade.query.session.add(trade)

    # Lower stoploss
    Trade.stoploss_reinitialization(0.06)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.94
    assert trade_adj.stop_loss_pct == -0.06
    assert trade_adj.initial_stop_loss == 0.94
    assert trade_adj.initial_stop_loss_pct == -0.06

    # Raise stoploss
    Trade.stoploss_reinitialization(0.04)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.96
    assert trade_adj.stop_loss_pct == -0.04
    assert trade_adj.initial_stop_loss == 0.96
    assert trade_adj.initial_stop_loss_pct == -0.04

    # Trailing stoploss (move stoplos up a bit)
    trade.adjust_stop_loss(1.02, 0.04)
    assert trade_adj.stop_loss == 0.9792
    assert trade_adj.initial_stop_loss == 0.96

    Trade.stoploss_reinitialization(0.04)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    # Stoploss should not change in this case.
    assert trade_adj.stop_loss == 0.9792
    assert trade_adj.stop_loss_pct == -0.04
    assert trade_adj.initial_stop_loss == 0.96
    assert trade_adj.initial_stop_loss_pct == -0.04


def test_update_fee(fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=30.0,
        fee_open=fee.return_value,
        open_date=arrow.utcnow().shift(hours=-2).datetime,
        amount=30.0,
        fee_close=fee.return_value,
        exchange='binance',
        open_rate=1,
        max_rate=1,
    )
    fee_cost = 0.15
    fee_currency = 'BTC'
    fee_rate = 0.0075
    assert trade.fee_open_currency is None
    assert not trade.fee_updated('buy')
    assert not trade.fee_updated('sell')

    trade.update_fee(fee_cost, fee_currency, fee_rate, 'buy')
    assert trade.fee_updated('buy')
    assert not trade.fee_updated('sell')
    assert trade.fee_open_currency == fee_currency
    assert trade.fee_open_cost == fee_cost
    assert trade.fee_open == fee_rate
    # Setting buy rate should "guess" close rate
    assert trade.fee_close == fee_rate
    assert trade.fee_close_currency is None
    assert trade.fee_close_cost is None

    fee_rate = 0.0076
    trade.update_fee(fee_cost, fee_currency, fee_rate, 'sell')
    assert trade.fee_updated('buy')
    assert trade.fee_updated('sell')
    assert trade.fee_close == 0.0076
    assert trade.fee_close_cost == fee_cost
    assert trade.fee_close == fee_rate


def test_fee_updated(fee):
    trade = Trade(
        pair='ADA/USDT',
        stake_amount=30.0,
        fee_open=fee.return_value,
        open_date=arrow.utcnow().shift(hours=-2).datetime,
        amount=30.0,
        fee_close=fee.return_value,
        exchange='binance',
        open_rate=1,
        max_rate=1,
    )

    assert trade.fee_open_currency is None
    assert not trade.fee_updated('buy')
    assert not trade.fee_updated('sell')
    assert not trade.fee_updated('asdf')

    trade.update_fee(0.15, 'BTC', 0.0075, 'buy')
    assert trade.fee_updated('buy')
    assert not trade.fee_updated('sell')
    assert trade.fee_open_currency is not None
    assert trade.fee_close_currency is None

    trade.update_fee(0.15, 'ABC', 0.0075, 'sell')
    assert trade.fee_updated('buy')
    assert trade.fee_updated('sell')
    assert not trade.fee_updated('asfd')


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize('use_db', [True, False])
def test_total_open_trades_stakes(fee, use_db):

    Trade.use_db = use_db
    Trade.reset_trades()
    res = Trade.total_open_trades_stakes()
    assert res == 0
    create_mock_trades(fee, use_db)
    res = Trade.total_open_trades_stakes()
    assert res == 0.004

    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize('use_db', [True, False])
def test_get_total_closed_profit(fee, use_db):

    Trade.use_db = use_db
    Trade.reset_trades()
    res = Trade.get_total_closed_profit()
    assert res == 0
    create_mock_trades(fee, use_db)
    res = Trade.get_total_closed_profit()
    assert res == 0.000739127

    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize('use_db', [True, False])
def test_get_trades_proxy(fee, use_db):
    Trade.use_db = use_db
    Trade.reset_trades()
    create_mock_trades(fee, use_db)
    trades = Trade.get_trades_proxy()
    assert len(trades) == 6

    assert isinstance(trades[0], Trade)

    trades = Trade.get_trades_proxy(is_open=True)
    assert len(trades) == 4
    assert trades[0].is_open
    trades = Trade.get_trades_proxy(is_open=False)

    assert len(trades) == 2
    assert not trades[0].is_open

    opendate = datetime.now(tz=timezone.utc) - timedelta(minutes=15)

    assert len(Trade.get_trades_proxy(open_date=opendate)) == 3

    Trade.use_db = True


def test_get_trades_backtest():
    Trade.use_db = False
    with pytest.raises(NotImplementedError, match=r"`Trade.get_trades\(\)` not .*"):
        Trade.get_trades([])
    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
def test_get_overall_performance(fee):

    create_mock_trades(fee)
    res = Trade.get_overall_performance()

    assert len(res) == 2
    assert 'pair' in res[0]
    assert 'profit' in res[0]
    assert 'count' in res[0]


@pytest.mark.usefixtures("init_persistence")
def test_get_best_pair(fee):

    res = Trade.get_best_pair()
    assert res is None

    create_mock_trades(fee)
    res = Trade.get_best_pair()
    assert len(res) == 2
    assert res[0] == 'XRP/BTC'
    assert res[1] == 0.01


@pytest.mark.usefixtures("init_persistence")
def test_update_order_from_ccxt(caplog):
    # Most basic order return (only has orderid)
    o = Order.parse_from_ccxt_object({'id': '1234'}, 'ETH/BTC', 'buy')
    assert isinstance(o, Order)
    assert o.ft_pair == 'ETH/BTC'
    assert o.ft_order_side == 'buy'
    assert o.order_id == '1234'
    assert o.ft_is_open
    ccxt_order = {
        'id': '1234',
        'side': 'buy',
        'symbol': 'ETH/BTC',
        'type': 'limit',
        'price': 1234.5,
        'amount':  20.0,
        'filled': 9,
        'remaining': 11,
        'status': 'open',
        'timestamp': 1599394315123
    }
    o = Order.parse_from_ccxt_object(ccxt_order, 'ETH/BTC', 'buy')
    assert isinstance(o, Order)
    assert o.ft_pair == 'ETH/BTC'
    assert o.ft_order_side == 'buy'
    assert o.order_id == '1234'
    assert o.order_type == 'limit'
    assert o.price == 1234.5
    assert o.filled == 9
    assert o.remaining == 11
    assert o.order_date is not None
    assert o.ft_is_open
    assert o.order_filled_date is None

    # Order is unfilled, "filled" not set
    # https://github.com/freqtrade/freqtrade/issues/5404
    ccxt_order.update({'filled': None, 'remaining': 20.0, 'status': 'canceled'})
    o.update_from_ccxt_object(ccxt_order)

    # Order has been closed
    ccxt_order.update({'filled': 20.0, 'remaining': 0.0, 'status': 'closed'})
    o.update_from_ccxt_object(ccxt_order)

    assert o.filled == 20.0
    assert o.remaining == 0.0
    assert not o.ft_is_open
    assert o.order_filled_date is not None

    ccxt_order.update({'id': 'somethingelse'})
    with pytest.raises(DependencyException, match=r"Order-id's don't match"):
        o.update_from_ccxt_object(ccxt_order)

    message = "aaaa is not a valid response object."
    assert not log_has(message, caplog)
    Order.update_orders([o], 'aaaa')
    assert log_has(message, caplog)

    # Call regular update - shouldn't fail.
    Order.update_orders([o], {'id': '1234'})


@pytest.mark.usefixtures("init_persistence")
def test_select_order(fee):
    create_mock_trades(fee)

    trades = Trade.get_trades().all()

    # Open buy order, no sell order
    order = trades[0].select_order('buy', True)
    assert order is None
    order = trades[0].select_order('buy', False)
    assert order is not None
    order = trades[0].select_order('sell', None)
    assert order is None

    # closed buy order, and open sell order
    order = trades[1].select_order('buy', True)
    assert order is None
    order = trades[1].select_order('buy', False)
    assert order is not None
    order = trades[1].select_order('buy', None)
    assert order is not None
    order = trades[1].select_order('sell', True)
    assert order is None
    order = trades[1].select_order('sell', False)
    assert order is not None

    # Has open buy order
    order = trades[3].select_order('buy', True)
    assert order is not None
    order = trades[3].select_order('buy', False)
    assert order is None

    # Open sell order
    order = trades[4].select_order('buy', True)
    assert order is None
    order = trades[4].select_order('buy', False)
    assert order is not None

    order = trades[4].select_order('sell', True)
    assert order is not None
    assert order.ft_order_side == 'stoploss'
    order = trades[4].select_order('sell', False)
    assert order is None


def test_Trade_object_idem():

    assert issubclass(Trade, LocalTrade)

    trade = vars(Trade)
    localtrade = vars(LocalTrade)

    excludes = (
        'delete',
        'session',
        'commit',
        'query',
        'open_date',
        'get_best_pair',
        'get_overall_performance',
        'get_total_closed_profit',
        'total_open_trades_stakes',
        'get_sold_trades_without_assigned_fees',
        'get_open_trades_without_assigned_fees',
        'get_open_order_trades',
        'get_trades',
    )

    # Parent (LocalTrade) should have the same attributes
    for item in trade:
        # Exclude private attributes and open_date (as it's not assigned a default)
        if (not item.startswith('_') and item not in excludes):
            assert item in localtrade

    # Fails if only a column is added without corresponding parent field
    for item in localtrade:
        if (not item.startswith('__')
                and item not in ('trades', 'trades_open', 'total_profit')
                and type(getattr(LocalTrade, item)) not in (property, FunctionType)):
            assert item in trade
