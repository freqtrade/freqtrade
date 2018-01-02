# SQL Helper
This page constains some help if you want to edit your sqlite db.

## Install sqlite3
**Ubuntu/Debian installation**
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
CREATE TABLE trades (
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
);
```

## Get all trades in the table

```sql
SELECT * FROM trades;
```

## Fix trade still open after a /forcesell

```sql
UPDATE trades
SET is_open=0, close_date=<close_date>, close_rate=<close_rate>, close_profit=close_rate/open_rate  
WHERE id=<trade_ID_to_update>;
```

**Example:**
```sql
UPDATE trades
SET is_open=0, close_date='2017-12-20 03:08:45.103418', close_rate=0.19638016, close_profit=0.0496  
WHERE id=31;
```


## Fix wrong fees in the table
If your DB was created before 
[PR#200](https://github.com/gcarq/freqtrade/pull/200) was merged
(before 12/23/17).

```sql
UPDATE trades SET fee=0.0025 WHERE fee=0.005;
```