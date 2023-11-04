# Trade Object

## Trade

A position freqtrade enters is stored in a `Trade` object - which is persisted to the database.
It's a core concept of freqtrade - and something you'll come across in many sections of the documentation, which will most likely point you to this location.

It will be passed to the strategy in many [strategy callbacks](strategy-callbacks.md). The object passed to the strategy cannot be modified directly. Indirect modifications may occur based on callback results.

## Trade - Available attributes

The following attributes / properties are available for each individual trade - and can be used with `trade.<property>` (e.g. `trade.pair`).

|  Attribute | DataType | Description |
|------------|-------------|-------------|
`pair`| string | Pair of this trade
`is_open`| boolean | Is the trade currently open, or has it been concluded
`open_rate`| float | Rate this trade was entered at (Avg. entry rate in case of trade-adjustments)
`close_rate`| float | Close rate - only set when is_open = False
`stake_amount`| float | Amount in Stake (or Quote) currency.
`amount`| float | Amount in Asset / Base currency that is currently owned.
`open_date`| datetime | Timestamp when trade was opened **use `open_date_utc` instead**
`open_date_utc`| datetime | Timestamp when trade was opened - in UTC
`close_date`| datetime | Timestamp when trade was closed **use `close_date_utc` instead**
`close_date_utc`| datetime | Timestamp when trade was closed - in UTC
`close_profit`| float | Relative profit at the time of trade closure. `0.01` == 1%
`close_profit_abs`| float | Absolute profit (in stake currency) at the time of trade closure.
`leverage` | float | Leverage used for this trade - defaults to 1.0 in spot markets.
`enter_tag`| string | Tag provided on entry via the `enter_tag` column in the dataframe
`is_short` | boolean | True for short trades, False otherwise
`orders` | Order[] | List of order objects attached to this trade (includes both filled and cancelled orders)
`date_last_filled_utc` | datetime | Time of the last filled order
`entry_side` | "buy" / "sell" | Order Side the trade was entered
`exit_side` | "buy" / "sell" | Order Side that will result in a trade exit / position reduction.
`trade_direction` | "long" / "short" | Trade direction in text - long or short.
`nr_of_successful_entries` | int | Number of successful (filled) entry orders
`nr_of_successful_exits` | int | Number of successful (filled) exit orders

## Class methods

The following are class methods - which return generic information, and usually result in an explicit query against the database.
They can be used as `Trade.<method>` - e.g. `open_trades = Trade.get_open_trade_count()`

!!! Warning "Backtesting/hyperopt"
    Most methods will work in both backtesting / hyperopt and live/dry modes.
    During backtesting, it's limited to usage in [strategy callbacks](strategy-callbacks.md). Usage in `populate_*()` methods is not supported and will result in wrong results.

### get_trades_proxy

When your strategy needs some information on existing (open or close) trades - it's best to use `Trade.get_trades_proxy()`.

Usage:

``` python
from freqtrade.persistence import Trade
from datetime import timedelta

# ...
trade_hist = Trade.get_trades_proxy(pair='ETH/USDT', is_open=False, open_date=current_date - timedelta(days=2))

```

`get_trades_proxy()` supports the following keyword arguments. All arguments are optional - calling `get_trades_proxy()` without arguments will return a list of all trades in the database.

* `pair` e.g. `pair='ETH/USDT'`
* `is_open` e.g. `is_open=False`
* `open_date` e.g. `open_date=current_date - timedelta(days=2)`
* `close_date` e.g. `close_date=current_date - timedelta(days=5)`

### get_open_trade_count

Get the number of currently open trades

``` python
from freqtrade.persistence import Trade
# ...
open_trades = Trade.get_open_trade_count()
```

### get_total_closed_profit

Retrieve the total profit the bot has generated so far.
Aggregates `close_profit_abs` for all closed trades.

``` python
from freqtrade.persistence import Trade

# ...
profit = Trade.get_total_closed_profit()
```

### total_open_trades_stakes

Retrieve the total stake_amount that's currently in trades.

``` python
from freqtrade.persistence import Trade

# ...
profit = Trade.total_open_trades_stakes()
```

### get_overall_performance

Retrieve the overall performance - similar to the `/performance` telegram command.

``` python
from freqtrade.persistence import Trade

# ...
if self.config['runmode'].value in ('live', 'dry_run'):
    performance = Trade.get_overall_performance()
```

Sample return value: ETH/BTC had 5 trades, with a total profit of 1.5% (ratio of 0.015).

``` json
{"pair": "ETH/BTC", "profit": 0.015, "count": 5}
```

## Order Object

An `Order` object represents an order on the exchange (or a simulated order in dry-run mode).
An `Order` object will always be tied to it's corresponding [`Trade`](#trade-object), and only really makes sense in the context of a trade.

### Order - Available attributes

an Order object is typically attached to a trade.
Most properties here can be None as they are dependant on the exchange response.

|  Attribute | DataType | Description |
|------------|-------------|-------------|
`trade` | Trade | Trade object this order is attached to
`ft_pair` | string | Pair this order is for
`ft_is_open` | boolean | is the order filled?
`order_type` | string | Order type as defined on the exchange - usually market, limit or stoploss
`status` | string | Status as defined by ccxt. Usually open, closed, expired or canceled
`side` | string | Buy or Sell
`price` | float | Price the order was placed at
`average` | float | Average price the order filled at
`amount` | float | Amount in base currency
`filled` | float | Filled amount (in base currency)
`remaining` | float | Remaining amount
`cost` | float | Cost of the order - usually average * filled (*Exchange dependant on futures, may contain the cost with or without leverage and may be in contracts.*)
`stake_amount` | float | Stake amount used for this order. *Added in 2023.7.*
`order_date` | datetime | Order creation date **use `order_date_utc` instead**
`order_date_utc` | datetime | Order creation date (in UTC)
`order_fill_date` | datetime |  Order fill date **use `order_fill_utc` instead**
`order_fill_date_utc` | datetime | Order fill date
