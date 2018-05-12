# pragma pylint: disable=missing-docstring, C0103
import os

import pytest
from sqlalchemy import create_engine

from freqtrade.persistence import Trade, init, clean_dry_run_db


@pytest.fixture(scope='function')
def init_persistence(default_conf):
    init(default_conf)


def test_init_create_session(default_conf, mocker):
    mocker.patch.dict('freqtrade.persistence._CONF', default_conf)

    # Check if init create a session
    init(default_conf)
    assert hasattr(Trade, 'session')
    assert 'Session' in type(Trade.session).__name__


def test_init_dry_run_db(default_conf, mocker):
    default_conf.update({'dry_run_db': True})
    mocker.patch.dict('freqtrade.persistence._CONF', default_conf)

    # First, protect the existing 'tradesv3.dry_run.sqlite' (Do not delete user data)
    dry_run_db = 'tradesv3.dry_run.sqlite'
    dry_run_db_swp = dry_run_db + '.swp'

    if os.path.isfile(dry_run_db):
        os.rename(dry_run_db, dry_run_db_swp)

    # Check if the new tradesv3.dry_run.sqlite was created
    init(default_conf)
    assert os.path.isfile(dry_run_db) is True

    # Delete the file made for this unitest and rollback to the previous
    # tradesv3.dry_run.sqlite file

    # 1. Delete file from the test
    if os.path.isfile(dry_run_db):
        os.remove(dry_run_db)

    # 2. Rollback to the initial file
    if os.path.isfile(dry_run_db_swp):
        os.rename(dry_run_db_swp, dry_run_db)


def test_init_dry_run_without_db(default_conf, mocker):
    default_conf.update({'dry_run_db': False})
    mocker.patch.dict('freqtrade.persistence._CONF', default_conf)

    # First, protect the existing 'tradesv3.dry_run.sqlite' (Do not delete user data)
    dry_run_db = 'tradesv3.dry_run.sqlite'
    dry_run_db_swp = dry_run_db + '.swp'

    if os.path.isfile(dry_run_db):
        os.rename(dry_run_db, dry_run_db_swp)

    # Check if the new tradesv3.dry_run.sqlite was created
    init(default_conf)
    assert os.path.isfile(dry_run_db) is False

    # Rollback to the initial 'tradesv3.dry_run.sqlite' file
    if os.path.isfile(dry_run_db_swp):
        os.rename(dry_run_db_swp, dry_run_db)


def test_init_prod_db(default_conf, mocker):
    default_conf.update({'dry_run': False})
    mocker.patch.dict('freqtrade.persistence._CONF', default_conf)

    # First, protect the existing 'tradesv3.sqlite' (Do not delete user data)
    prod_db = 'tradesv3.sqlite'
    prod_db_swp = prod_db + '.swp'

    if os.path.isfile(prod_db):
        os.rename(prod_db, prod_db_swp)

    # Check if the new tradesv3.sqlite was created
    init(default_conf)
    assert os.path.isfile(prod_db) is True

    # Delete the file made for this unitest and rollback to the previous tradesv3.sqlite file

    # 1. Delete file from the test
    if os.path.isfile(prod_db):
        os.remove(prod_db)

    # Rollback to the initial 'tradesv3.sqlite' file
    if os.path.isfile(prod_db_swp):
        os.rename(prod_db_swp, prod_db)


@pytest.mark.usefixtures("init_persistence")
def test_update_with_bittrex(limit_buy_order, limit_sell_order, fee):
    """
    On this test we will buy and sell a crypto currency.

    Buy
    - Buy: 90.99181073 Crypto at 0.00001099 BTC
        (90.99181073*0.00001099 = 0.0009999 BTC)
    - Buying fee: 0.25%
    - Total cost of buy trade: 0.001002500 BTC
        ((90.99181073*0.00001099) + ((90.99181073*0.00001099)*0.0025))

    Sell
    - Sell: 90.99181073 Crypto at 0.00001173 BTC
        (90.99181073*0.00001173 = 0,00106733394 BTC)
    - Selling fee: 0.25%
    - Total cost of sell trade: 0.001064666 BTC
        ((90.99181073*0.00001173) - ((90.99181073*0.00001173)*0.0025))

    Profit/Loss: +0.000062166 BTC
        (Sell:0.001064666 - Buy:0.001002500)
    Profit/Loss percentage: 0.0620
        ((0.001064666/0.001002500)-1 = 6.20%)

    :param limit_buy_order:
    :param limit_sell_order:
    :return:
    """

    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
    )
    assert trade.open_order_id is None
    assert trade.open_rate is None
    assert trade.close_profit is None
    assert trade.close_date is None

    trade.open_order_id = 'something'
    trade.update(limit_buy_order)
    assert trade.open_order_id is None
    assert trade.open_rate == 0.00001099
    assert trade.close_profit is None
    assert trade.close_date is None

    trade.open_order_id = 'something'
    trade.update(limit_sell_order)
    assert trade.open_order_id is None
    assert trade.close_rate == 0.00001173
    assert trade.close_profit == 0.06201057
    assert trade.close_date is not None


@pytest.mark.usefixtures("init_persistence")
def test_calc_open_close_trade_price(limit_buy_order, limit_sell_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
    )

    trade.open_order_id = 'something'
    trade.update(limit_buy_order)
    assert trade.calc_open_trade_price() == 0.001002500

    trade.update(limit_sell_order)
    assert trade.calc_close_trade_price() == 0.0010646656

    # Profit in BTC
    assert trade.calc_profit() == 0.00006217

    # Profit in percent
    assert trade.calc_profit_percent() == 0.06201057


@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price_exception(limit_buy_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
    )

    trade.open_order_id = 'something'
    trade.update(limit_buy_order)
    assert trade.calc_close_trade_price() == 0.0


@pytest.mark.usefixtures("init_persistence")
def test_update_open_order(limit_buy_order):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=1.00,
        fee_open=0.1,
        fee_close=0.1,
        exchange='bittrex',
    )

    assert trade.open_order_id is None
    assert trade.open_rate is None
    assert trade.close_profit is None
    assert trade.close_date is None

    limit_buy_order['status'] = 'open'
    trade.update(limit_buy_order)

    assert trade.open_order_id is None
    assert trade.open_rate is None
    assert trade.close_profit is None
    assert trade.close_date is None


@pytest.mark.usefixtures("init_persistence")
def test_update_invalid_order(limit_buy_order):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=1.00,
        fee_open=0.1,
        fee_close=0.1,
        exchange='bittrex',
    )
    limit_buy_order['type'] = 'invalid'
    with pytest.raises(ValueError, match=r'Unknown order type'):
        trade.update(limit_buy_order)


@pytest.mark.usefixtures("init_persistence")
def test_calc_open_trade_price(limit_buy_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
    )
    trade.open_order_id = 'open_trade'
    trade.update(limit_buy_order)  # Buy @ 0.00001099

    # Get the open rate price with the standard fee rate
    assert trade.calc_open_trade_price() == 0.001002500

    # Get the open rate price with a custom fee rate
    assert trade.calc_open_trade_price(fee=0.003) == 0.001003000


@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price(limit_buy_order, limit_sell_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
    )
    trade.open_order_id = 'close_trade'
    trade.update(limit_buy_order)  # Buy @ 0.00001099

    # Get the close rate price with a custom close rate and a regular fee rate
    assert trade.calc_close_trade_price(rate=0.00001234) == 0.0011200318

    # Get the close rate price with a custom close rate and a custom fee rate
    assert trade.calc_close_trade_price(rate=0.00001234, fee=0.003) == 0.0011194704

    # Test when we apply a Sell order, and ask price with a custom fee rate
    trade.update(limit_sell_order)
    assert trade.calc_close_trade_price(fee=0.005) == 0.0010619972


@pytest.mark.usefixtures("init_persistence")
def test_calc_profit(limit_buy_order, limit_sell_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
    )
    trade.open_order_id = 'profit_percent'
    trade.update(limit_buy_order)  # Buy @ 0.00001099

    # Custom closing rate and regular fee rate
    # Higher than open rate
    assert trade.calc_profit(rate=0.00001234) == 0.00011753
    # Lower than open rate
    assert trade.calc_profit(rate=0.00000123) == -0.00089086

    # Custom closing rate and custom fee rate
    # Higher than open rate
    assert trade.calc_profit(rate=0.00001234, fee=0.003) == 0.00011697
    # Lower than open rate
    assert trade.calc_profit(rate=0.00000123, fee=0.003) == -0.00089092

    # Test when we apply a Sell order. Sell higher than open rate @ 0.00001173
    trade.update(limit_sell_order)
    assert trade.calc_profit() == 0.00006217

    # Test with a custom fee rate on the close trade
    assert trade.calc_profit(fee=0.003) == 0.00006163


@pytest.mark.usefixtures("init_persistence")
def test_calc_profit_percent(limit_buy_order, limit_sell_order, fee):
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
    )
    trade.open_order_id = 'profit_percent'
    trade.update(limit_buy_order)  # Buy @ 0.00001099

    # Get percent of profit with a custom rate (Higher than open rate)
    assert trade.calc_profit_percent(rate=0.00001234) == 0.1172387

    # Get percent of profit with a custom rate (Lower than open rate)
    assert trade.calc_profit_percent(rate=0.00000123) == -0.88863827

    # Test when we apply a Sell order. Sell higher than open rate @ 0.00001173
    trade.update(limit_sell_order)
    assert trade.calc_profit_percent() == 0.06201057

    # Test with a custom fee rate on the close trade
    assert trade.calc_profit_percent(fee=0.003) == 0.0614782


def test_clean_dry_run_db(default_conf, fee):
    init(default_conf, create_engine('sqlite://'))

    # Simulate dry_run entries
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        amount=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        exchange='bittrex',
        open_order_id='dry_run_buy_12345'
    )
    Trade.session.add(trade)

    trade = Trade(
        pair='ETC/BTC',
        stake_amount=0.001,
        amount=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        exchange='bittrex',
        open_order_id='dry_run_sell_12345'
    )
    Trade.session.add(trade)

    # Simulate prod entry
    trade = Trade(
        pair='ETC/BTC',
        stake_amount=0.001,
        amount=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        exchange='bittrex',
        open_order_id='prod_buy_12345'
    )
    Trade.session.add(trade)

    # We have 3 entries: 2 dry_run, 1 prod
    assert len(Trade.query.filter(Trade.open_order_id.isnot(None)).all()) == 3

    clean_dry_run_db()

    # We have now only the prod
    assert len(Trade.query.filter(Trade.open_order_id.isnot(None)).all()) == 1


def test_migrate_old(default_conf, fee):
    """
    Test Database migration(starting with old pairformat)
    """
    amount = 103.223
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
                                PRIMARY KEY (id),
                                CHECK (is_open IN (0, 1))
                                );"""
    insert_table_old = """INSERT INTO trades (exchange, pair, is_open, fee,
                          open_rate, stake_amount, amount, open_date)
                          VALUES ('BITTREX', 'BTC_ETC', 1, {fee},
                          0.00258580, {stake}, {amount},
                          '2017-11-28 12:44:24.000000')
                          """.format(fee=fee.return_value,
                                     stake=default_conf.get("stake_amount"),
                                     amount=amount
                                     )
    engine = create_engine('sqlite://')
    # Create table using the old format
    engine.execute(create_table_old)
    engine.execute(insert_table_old)
    # Run init to test migration
    init(default_conf, engine)

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
    assert trade.exchange == "bittrex"


def test_migrate_new(default_conf, fee):
    """
    Test Database migration (starting with new pairformat)
    """
    amount = 103.223
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
                                PRIMARY KEY (id),
                                CHECK (is_open IN (0, 1))
                                );"""
    insert_table_old = """INSERT INTO trades (exchange, pair, is_open, fee,
                          open_rate, stake_amount, amount, open_date)
                          VALUES ('binance', 'ETC/BTC', 1, {fee},
                          0.00258580, {stake}, {amount},
                          '2019-11-28 12:44:24.000000')
                          """.format(fee=fee.return_value,
                                     stake=default_conf.get("stake_amount"),
                                     amount=amount
                                     )
    engine = create_engine('sqlite://')
    # Create table using the old format
    engine.execute(create_table_old)
    engine.execute(insert_table_old)
    # Run init to test migration
    init(default_conf, engine)

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
