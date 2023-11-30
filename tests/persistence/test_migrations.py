# pragma pylint: disable=missing-docstring, C0103
import logging
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.schema import CreateTable

from freqtrade.constants import DEFAULT_DB_PROD_URL
from freqtrade.enums import TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import Trade, init_db
from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.migrations import get_last_sequence_ids, set_sequence_ids
from freqtrade.persistence.models import PairLock
from freqtrade.persistence.trade_model import Order
from tests.conftest import log_has


spot, margin, futures = TradingMode.SPOT, TradingMode.MARGIN, TradingMode.FUTURES


def test_init_create_session(default_conf):
    # Check if init create a session
    init_db(default_conf['db_url'])
    assert hasattr(Trade, 'session')
    assert 'scoped_session' in type(Trade.session).__name__


def test_init_custom_db_url(default_conf, tmp_path):
    # Update path to a value other than default, but still in-memory
    filename = tmp_path / "freqtrade2_test.sqlite"
    assert not filename.is_file()

    default_conf.update({'db_url': f'sqlite:///{filename}'})

    init_db(default_conf['db_url'])
    assert filename.is_file()
    r = Trade.session.execute(text("PRAGMA journal_mode"))
    assert r.first() == ('wal',)


def test_init_invalid_db_url():
    # Update path to a value other than default, but still in-memory
    with pytest.raises(OperationalException, match=r'.*no valid database URL*'):
        init_db('unknown:///some.url')

    with pytest.raises(OperationalException, match=r'Bad db-url.*For in-memory database, pl.*'):
        init_db('sqlite:///')


def test_init_prod_db(default_conf, mocker):
    default_conf.update({'dry_run': False})
    default_conf.update({'db_url': DEFAULT_DB_PROD_URL})

    create_engine_mock = mocker.patch('freqtrade.persistence.models.create_engine', MagicMock())

    init_db(default_conf['db_url'])
    assert create_engine_mock.call_count == 1
    assert create_engine_mock.mock_calls[0][1][0] == 'sqlite:///tradesv3.sqlite'


def test_init_dryrun_db(default_conf, tmpdir):
    filename = f"{tmpdir}/freqtrade2_prod.sqlite"
    assert not Path(filename).is_file()
    default_conf.update({
        'dry_run': True,
        'db_url': f'sqlite:///{filename}'
    })

    init_db(default_conf['db_url'])
    assert Path(filename).is_file()


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
    create_table_order = """CREATE TABLE orders (
                                id INTEGER NOT NULL,
                                ft_trade_id INTEGER,
                                ft_order_side VARCHAR(25) NOT NULL,
                                ft_pair VARCHAR(25) NOT NULL,
                                ft_is_open BOOLEAN NOT NULL,
                                order_id VARCHAR(255) NOT NULL,
                                status VARCHAR(255),
                                symbol VARCHAR(25),
                                order_type VARCHAR(50),
                                side VARCHAR(25),
                                price FLOAT,
                                amount FLOAT,
                                filled FLOAT,
                                remaining FLOAT,
                                cost FLOAT,
                                order_date DATETIME,
                                order_filled_date DATETIME,
                                order_update_date DATETIME,
                                PRIMARY KEY (id)
                            );"""
    insert_table_old = """INSERT INTO trades (exchange, pair, is_open, fee,
                          open_rate, stake_amount, amount, open_date,
                          stop_loss, initial_stop_loss, max_rate, ticker_interval,
                          open_order_id, stoploss_order_id)
                          VALUES ('binance', 'ETC/BTC', 1, {fee},
                          0.00258580, {stake}, {amount},
                          '2019-11-28 12:44:24.000000',
                          0.0, 0.0, 0.0, '5m',
                          'buy_order', 'dry_stop_order_id222')
                          """.format(fee=fee.return_value,
                                     stake=default_conf.get("stake_amount"),
                                     amount=amount
                                     )
    insert_orders = f"""
        insert into orders (
            ft_trade_id,
            ft_order_side,
            ft_pair,
            ft_is_open,
            order_id,
            status,
            symbol,
            order_type,
            side,
            price,
            amount,
            filled,
            remaining,
            cost)
        values (
            1,
            'buy',
            'ETC/BTC',
            0,
            'dry_buy_order',
            'closed',
            'ETC/BTC',
            'limit',
            'buy',
            0.00258580,
            {amount},
            {amount},
            0,
            {amount * 0.00258580}
        ),
        (
            1,
            'buy',
            'ETC/BTC',
            1,
            'dry_buy_order22',
            'canceled',
            'ETC/BTC',
            'limit',
            'buy',
            0.00258580,
            {amount},
            {amount},
            0,
            {amount * 0.00258580}
        ),
         (
            1,
            'stoploss',
            'ETC/BTC',
            1,
            'dry_stop_order_id11X',
            'canceled',
            'ETC/BTC',
            'limit',
            'sell',
            0.00258580,
            {amount},
            {amount},
            0,
            {amount * 0.00258580}
        ),
        (
            1,
            'stoploss',
            'ETC/BTC',
            1,
            'dry_stop_order_id222',
            'open',
            'ETC/BTC',
            'limit',
            'sell',
            0.00258580,
            {amount},
            {amount},
            0,
            {amount * 0.00258580}
        ),
        (
            -- Order without reference trade
            2,
            'buy',
            'ETC/BTC',
            1,
            'dry_buy_order55',
            'canceled',
            'ETC/BTC',
            'limit',
            'buy',
            0.00258580,
            {amount},
            {amount},
            0,
            {amount * 0.00258580}
        )
    """
    engine = create_engine('sqlite://')
    mocker.patch('freqtrade.persistence.models.create_engine', lambda *args, **kwargs: engine)

    # Create table using the old format
    with engine.begin() as connection:
        connection.execute(text(create_table_old))
        connection.execute(text(create_table_order))
        connection.execute(text("create index ix_trades_is_open on trades(is_open)"))
        connection.execute(text("create index ix_trades_pair on trades(pair)"))
        connection.execute(text(insert_table_old))
        connection.execute(text(insert_orders))

        # fake previous backup
        connection.execute(text("create table trades_bak as select * from trades"))

        connection.execute(text("create table trades_bak1 as select * from trades"))
    # Run init to test migration
    init_db(default_conf['db_url'])

    trades = Trade.session.scalars(select(Trade)).all()
    assert len(trades) == 1
    trade = trades[0]
    assert trade.id == 1
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
    assert trade.exit_reason is None
    assert trade.strategy is None
    assert trade.timeframe == '5m'
    assert trade.stoploss_order_id == 'dry_stop_order_id222'
    assert trade.stoploss_last_update is None
    assert log_has("trying trades_bak1", caplog)
    assert log_has("trying trades_bak2", caplog)
    assert log_has("Running database migration for trades - backup: trades_bak2, orders_bak0",
                   caplog)
    assert log_has("Database migration finished.", caplog)
    assert pytest.approx(trade.open_trade_value) == trade._calc_open_trade_value(
        trade.amount, trade.open_rate)
    assert trade.close_profit_abs is None
    assert trade.stake_amount == trade.max_stake_amount

    orders = trade.orders
    assert len(orders) == 4
    assert orders[0].order_id == 'dry_buy_order'
    assert orders[0].ft_order_side == 'buy'

    assert orders[-1].order_id == 'dry_stop_order_id222'
    assert orders[-1].ft_order_side == 'stoploss'
    assert orders[-1].ft_is_open is True

    assert orders[1].order_id == 'dry_buy_order22'
    assert orders[1].ft_order_side == 'buy'
    assert orders[1].ft_is_open is True

    assert orders[2].order_id == 'dry_stop_order_id11X'
    assert orders[2].ft_order_side == 'stoploss'
    assert orders[2].ft_is_open is False

    orders1 = Order.session.scalars(select(Order)).all()
    assert len(orders1) == 5
    order = orders1[4]
    assert order.ft_trade_id == 2
    assert order.ft_is_open is False


def test_migrate_too_old(mocker, default_conf, fee, caplog):
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
    with pytest.raises(OperationalException, match=r'Your database seems to be very old'):
        init_db(default_conf['db_url'])


def test_migrate_get_last_sequence_ids():
    engine = MagicMock()
    engine.begin = MagicMock()
    engine.name = 'postgresql'
    get_last_sequence_ids(engine, 'trades_bak', 'orders_bak')

    assert engine.begin.call_count == 2
    engine.reset_mock()
    engine.begin.reset_mock()

    engine.name = 'somethingelse'
    get_last_sequence_ids(engine, 'trades_bak', 'orders_bak')

    assert engine.begin.call_count == 0


def test_migrate_set_sequence_ids():
    engine = MagicMock()
    engine.begin = MagicMock()
    engine.name = 'postgresql'
    set_sequence_ids(engine, 22, 55, 5)

    assert engine.begin.call_count == 1
    engine.reset_mock()
    engine.begin.reset_mock()

    engine.name = 'somethingelse'
    set_sequence_ids(engine, 22, 55, 6)

    assert engine.begin.call_count == 0


def test_migrate_pairlocks(mocker, default_conf, fee, caplog):
    """
    Test Database migration (starting with new pairformat)
    """
    caplog.set_level(logging.DEBUG)
    # Always create all columns apart from the last!
    create_table_old = """CREATE TABLE pairlocks (
                            id INTEGER NOT NULL,
                            pair VARCHAR(25) NOT NULL,
                            reason VARCHAR(255),
                            lock_time DATETIME NOT NULL,
                            lock_end_time DATETIME NOT NULL,
                            active BOOLEAN NOT NULL,
                            PRIMARY KEY (id)
                        )
                                """
    create_index1 = "CREATE INDEX ix_pairlocks_pair ON pairlocks (pair)"
    create_index2 = "CREATE INDEX ix_pairlocks_lock_end_time ON pairlocks (lock_end_time)"
    create_index3 = "CREATE INDEX ix_pairlocks_active ON pairlocks (active)"
    insert_table_old = """INSERT INTO pairlocks (
        id, pair, reason, lock_time, lock_end_time, active)
        VALUES (1, 'ETH/BTC', 'Auto lock', '2021-07-12 18:41:03', '2021-07-11 18:45:00', 1)
                          """
    insert_table_old2 = """INSERT INTO pairlocks (
        id, pair, reason, lock_time, lock_end_time, active)
        VALUES (2, '*', 'Lock all', '2021-07-12 18:41:03', '2021-07-12 19:00:00', 1)
                          """
    engine = create_engine('sqlite://')
    mocker.patch('freqtrade.persistence.models.create_engine', lambda *args, **kwargs: engine)
    # Create table using the old format
    with engine.begin() as connection:
        connection.execute(text(create_table_old))

        connection.execute(text(insert_table_old))
        connection.execute(text(insert_table_old2))
        connection.execute(text(create_index1))
        connection.execute(text(create_index2))
        connection.execute(text(create_index3))

    init_db(default_conf['db_url'])

    assert len(PairLock.get_all_locks().all()) == 2
    assert len(PairLock.session.scalars(select(PairLock).filter(PairLock.pair == '*')).all()) == 1
    pairlocks = PairLock.session.scalars(select(PairLock).filter(PairLock.pair == 'ETH/BTC')).all()
    assert len(pairlocks) == 1
    pairlocks[0].pair == 'ETH/BTC'
    pairlocks[0].side == '*'


@pytest.mark.parametrize('dialect', [
    'sqlite', 'postgresql', 'mysql', 'oracle', 'mssql',
    ])
def test_create_table_compiles(dialect):

    dialect_mod = import_module(f"sqlalchemy.dialects.{dialect}")
    for table in ModelBase.metadata.tables.values():
        create_sql = str(CreateTable(table).compile(dialect=dialect_mod.dialect()))
        assert 'CREATE TABLE' in create_sql
