# SQL Helper

This page contains some help if you want to edit your sqlite db.

## Install sqlite3

Sqlite3 is a terminal based sqlite application.
Feel free to use a visual Database editor like SqliteBrowser if you feel more comfortable with that.

### Ubuntu/Debian installation

```bash
sudo apt-get install sqlite3
```

### Using sqlite3 via docker

The freqtrade docker image does contain sqlite3, so you can edit the database without having to install anything on the host system.

``` bash
docker compose exec freqtrade /bin/bash
sqlite3 <database-file>.sqlite
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

## Get all trades in the table

```sql
SELECT * FROM trades;
```

## Fix trade still open after a manual exit on the exchange

!!! Warning
    Manually selling a pair on the exchange will not be detected by the bot and it will try to sell anyway. Whenever possible, /forceexit <tradeid> should be used to accomplish the same thing.  
    It is strongly advised to backup your database file before making any manual changes.

!!! Note
    This should not be necessary after /forceexit, as force_exit orders are closed automatically by the bot on the next iteration.

```sql
UPDATE trades
SET is_open=0,
  close_date=<close_date>,
  close_rate=<close_rate>,
  close_profit = close_rate / open_rate - 1,
  close_profit_abs = (amount * <close_rate> * (1 - fee_close) - (amount * (open_rate * (1 - fee_open)))),
  exit_reason=<exit_reason>
WHERE id=<trade_ID_to_update>;
```

### Example

```sql
UPDATE trades
SET is_open=0,
  close_date='2020-06-20 03:08:45.103418',
  close_rate=0.19638016,
  close_profit=0.0496,
  close_profit_abs = (amount * 0.19638016 * (1 - fee_close) - (amount * (open_rate * (1 - fee_open)))),
  exit_reason='force_exit'  
WHERE id=31;
```

## Remove trade from the database

!!! Tip "Use RPC Methods to delete trades"
    Consider using `/delete <tradeid>` via telegram or rest API. That's the recommended way to deleting trades.

If you'd still like to remove a trade from the database directly, you can use the below query.

!!! Danger
    Some systems (Ubuntu) disable foreign keys in their sqlite3 packaging. When using sqlite - please ensure that foreign keys are on by running `PRAGMA foreign_keys = ON` before the above query.

```sql
DELETE FROM trades WHERE id = <tradeid>;

DELETE FROM trades WHERE id = 31;
```

!!! Warning
    This will remove this trade from the database. Please make sure you got the correct id and **NEVER** run this query without the `where` clause.

## Use a different database system

Freqtrade is using SQLAlchemy, which supports multiple different database systems. As such, a multitude of database systems should be supported.
Freqtrade does not depend or install any additional database driver. Please refer to the [SQLAlchemy docs](https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls) on installation instructions for the respective database systems.

The following systems have been tested and are known to work with freqtrade:

* sqlite (default)
* PostgreSQL
* MariaDB

!!! Warning
    By using one of the below database systems, you acknowledge that you know how to manage such a system. The freqtrade team will not provide any support with setup or maintenance (or backups) of the below database systems.

### PostgreSQL

Installation:
`pip install psycopg2-binary`

Usage:
`... --db-url postgresql+psycopg2://<username>:<password>@localhost:5432/<database>`

Freqtrade will automatically create the tables necessary upon startup.

If you're running different instances of Freqtrade, you must either setup one database per Instance or use different users / schemas for your connections.

### MariaDB / MySQL

Freqtrade supports MariaDB by using SQLAlchemy, which supports multiple different database systems.

Installation:
`pip install pymysql`

Usage:
`... --db-url mysql+pymysql://<username>:<password>@localhost:3306/<database>`
