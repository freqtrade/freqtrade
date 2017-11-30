"""
a caribou migration

name: v2-to-v3
version: 20171111135341
"""

def upgrade(connection):
    # Rename old table
    sql = """
        ALTER TABLE trades RENAME TO trades_orig;
        """
    connection.execute(sql)

    # Create new table
    sql = """
        CREATE TABLE trades (
            id INTEGER NOT NULL,
            exchange VARCHAR NOT NULL,
            pair VARCHAR NOT NULL,
            fee FLOAT NOT NULL DEFAULT 0.0,
            is_open BOOLEAN NOT NULL,
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
        );
        """
    connection.execute(sql)

    # Copy data from old table to the new one
    sql = """
        INSERT INTO trades(id, exchange, pair, is_open, open_rate, close_rate,
        close_profit, stake_amount, amount, open_date, close_date, open_order_id)
        SELECT id, exchange, pair, is_open, open_rate, close_rate, close_profit,
        btc_amount, amount, open_date, close_date, open_order_id
        FROM trades_orig;
        """
    connection.execute(sql)

    # Remove old table
    sql = """
        DROP TABLE trades_orig;
        """
    connection.execute(sql)

    connection.commit()


def downgrade(connection):
    sql = """
        ALTER TABLE trades RENAME TO trades_orig;
        """
    connection.execute(sql)

    # Create new table
    sql = """
        CREATE TABLE trades (
            id INTEGER NOT NULL,
            exchange VARCHAR NOT NULL,
            pair VARCHAR NOT NULL,
            is_open BOOLEAN NOT NULL,
            open_rate FLOAT NOT NULL,
            close_rate FLOAT,
            close_profit FLOAT,
            btc_amount FLOAT NOT NULL,
            amount FLOAT NOT NULL,
            open_date DATETIME NOT NULL,
            close_date DATETIME,
            open_order_id VARCHAR,
            PRIMARY KEY (id),
            CHECK (is_open IN (0, 1))
        );
        """
    connection.execute(sql)

    # Copy data from old table to the new one
    sql = """
        INSERT INTO trades(id, exchange, pair, is_open, open_rate, close_rate,
        close_profit, btc_amount, amount, open_date, close_date, open_order_id)
        SELECT id, exchange, pair, is_open, open_rate, close_rate, close_profit,
        stake_amount, amount, open_date, close_date, open_order_id
        FROM trades_orig;
        """
    connection.execute(sql)

    # Remove old table
    sql = """
        DROP TABLE trades_orig;
        """
    connection.execute(sql)

    connection.commit()
