# SQL Helper

This page contains some help if you want to edit your sqlite db.

## Install sqlite3

Sqlite3 is a terminal based sqlite application.
Feel free to use a visual Database editor like SqliteBrowser if you feel more comfortable with that.

### Ubuntu/Debian installation

```bash
sudo apt-get install sqlite3
```

## Open the DB

```bash
sqlite3
.open <filepath>
```

## Table structure

### List tables

```bash
.tables
```

### Display table structure

```bash
.schema <table_name>
```

### Trade table structure

```sql
CREATE TABLE trades
    id INTEGER NOT NULL,
    exchange VARCHAR NOT NULL,
    pair VARCHAR NOT NULL,
    is_open BOOLEAN NOT NULL,
    fee_open FLOAT NOT NULL,
    fee_open_cost FLOAT,
    fee_open_currency VARCHAR,
    fee_close FLOAT NOT NULL,
    fee_close_cost FLOAT,
    fee_close_currency VARCHAR,
    open_rate FLOAT,
    open_rate_requested FLOAT,
    open_trade_price FLOAT,
    close_rate FLOAT,
    close_rate_requested FLOAT,
    close_profit FLOAT,
    close_profit_abs FLOAT,
    stake_amount FLOAT NOT NULL,
    amount FLOAT,
    open_date DATETIME NOT NULL,
    close_date DATETIME,
    open_order_id VARCHAR,
    stop_loss FLOAT,
    stop_loss_pct FLOAT,
    initial_stop_loss FLOAT,
    initial_stop_loss_pct FLOAT,
    stoploss_order_id VARCHAR,
    stoploss_last_update DATETIME,
    max_rate FLOAT,
    min_rate FLOAT,
    sell_reason VARCHAR,
    strategy VARCHAR,
    timeframe INTEGER,
    PRIMARY KEY (id),
    CHECK (is_open IN (0, 1))
);
CREATE INDEX ix_trades_stoploss_order_id ON trades (stoploss_order_id);
CREATE INDEX ix_trades_pair ON trades (pair);
CREATE INDEX ix_trades_is_open ON trades (is_open);

```

## Get all trades in the table

```sql
SELECT * FROM trades;
```

## Fix trade still open after a manual sell on the exchange

!!! Warning
  	Manually selling a pair on the exchange will not be detected by the bot and it will try to sell anyway. Whenever possible, forcesell <tradeid> should be used to accomplish the same thing.  
	It is strongly advised to backup your database file before making any manual changes.

!!! Note
  	This should not be necessary after /forcesell, as forcesell orders are closed automatically by the bot on the next iteration.

```sql
UPDATE trades
SET is_open=0,
  close_date=<close_date>,
  close_rate=<close_rate>,
  close_profit=close_rate/open_rate-1,
  close_profit_abs = (amount * <close_rate> * (1 - fee_close) - (amount * open_rate * 1 - fee_open)),
  sell_reason=<sell_reason>
WHERE id=<trade_ID_to_update>;
```

### Example

```sql
UPDATE trades
SET is_open=0,
  close_date='2017-12-20 03:08:45.103418',
  close_rate=0.19638016,
  close_profit=0.0496,
  close_profit_abs = (amount * 0.19638016 * (1 - fee_close) - (amount * open_rate * 1 - fee_open))
  sell_reason='force_sell'  
WHERE id=31;
```

## Insert manually a new trade

```sql
INSERT INTO trades (exchange, pair, is_open, fee_open, fee_close, open_rate, stake_amount, amount, open_date)
VALUES ('bittrex', 'ETH/BTC', 1, 0.0025, 0.0025, <open_rate>, <stake_amount>, <amount>, '<datetime>')
```

##### Example:

```sql
INSERT INTO trades (exchange, pair, is_open, fee_open, fee_close, open_rate, stake_amount, amount, open_date)
VALUES ('bittrex', 'ETH/BTC', 1, 0.0025, 0.0025, 0.00258580, 0.002, 0.7715262081, '2017-11-28 12:44:24.000000')
```
