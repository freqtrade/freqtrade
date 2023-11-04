## Prices used for orders

Prices for regular orders can be controlled via the parameter structures `entry_pricing` for trade entries and `exit_pricing` for trade exits.
Prices are always retrieved right before an order is placed, either by querying the exchange tickers or by using the orderbook data.

!!! Note
    Orderbook data used by Freqtrade are the data retrieved from exchange by the ccxt's function `fetch_order_book()`, i.e. are usually data from the L2-aggregated orderbook, while the ticker data are the structures returned by the ccxt's `fetch_ticker()`/`fetch_tickers()` functions. Refer to the ccxt library [documentation](https://github.com/ccxt/ccxt/wiki/Manual#market-data) for more details.

!!! Warning "Using market orders"
    Please read the section [Market order pricing](#market-order-pricing) section when using market orders.

### Entry price

#### Enter price side

The configuration setting `entry_pricing.price_side` defines the side of the orderbook the bot looks for when buying.

The following displays an orderbook.

``` explanation
...
103
102
101  # ask
-------------Current spread
99   # bid
98
97
...
```

If `entry_pricing.price_side` is set to `"bid"`, then the bot will use 99 as entry price.  
In line with that, if `entry_pricing.price_side` is set to `"ask"`, then the bot will use 101 as entry price.

Depending on the order direction (_long_/_short_), this will lead to different results. Therefore we recommend to use `"same"` or `"other"` for this configuration instead.
This would result in the following pricing matrix:

| direction | Order | setting | price | crosses spread |
|------ |--------|-----|-----|-----|
| long  | buy  | ask   | 101 | yes |
| long  | buy  | bid   | 99  | no  |
| long  | buy  | same  | 99  | no  |
| long  | buy  | other | 101 | yes |
| short | sell | ask   | 101 | no  |
| short | sell | bid   | 99  | yes |
| short | sell | same  | 101 | no  |
| short | sell | other | 99  | yes |

Using the other side of the orderbook often guarantees quicker filled orders, but the bot can also end up paying more than what would have been necessary.
Taker fees instead of maker fees will most likely apply even when using limit buy orders.
Also, prices at the "other" side of the spread are higher than prices at the "bid" side in the orderbook, so the order behaves similar to a market order (however with a maximum price).

#### Entry price with Orderbook enabled

When entering a trade with the orderbook enabled (`entry_pricing.use_order_book=True`), Freqtrade fetches the `entry_pricing.order_book_top` entries from the orderbook and uses the entry specified as `entry_pricing.order_book_top` on the configured side (`entry_pricing.price_side`) of the orderbook. 1 specifies the topmost entry in the orderbook, while 2 would use the 2nd entry in the orderbook, and so on.

#### Entry price without Orderbook enabled

The following section uses `side` as the configured `entry_pricing.price_side` (defaults to `"same"`).

When not using orderbook (`entry_pricing.use_order_book=False`), Freqtrade uses the best `side` price from the ticker if it's below the `last` traded price from the ticker. Otherwise (when the `side` price is above the `last` price), it calculates a rate between `side` and `last` price based on `entry_pricing.price_last_balance`.

The `entry_pricing.price_last_balance` configuration parameter controls this. A value of `0.0` will use `side` price, while `1.0` will use the `last` price and values between those interpolate between ask and last price.

#### Check depth of market

When check depth of market is enabled (`entry_pricing.check_depth_of_market.enabled=True`), the entry signals are filtered based on the orderbook depth (sum of all amounts) for each orderbook side.

Orderbook `bid` (buy) side depth is then divided by the orderbook `ask` (sell) side depth and the resulting delta is compared to the value of the `entry_pricing.check_depth_of_market.bids_to_ask_delta` parameter. The entry order is only executed if the orderbook delta is greater than or equal to the configured delta value.

!!! Note
    A delta value below 1 means that `ask` (sell) orderbook side depth is greater than the depth of the `bid` (buy) orderbook side, while a value greater than 1 means opposite (depth of the buy side is higher than the depth of the sell side).

### Exit price

#### Exit price side

The configuration setting `exit_pricing.price_side` defines the side of the spread the bot looks for when exiting a trade.

The following displays an orderbook:

``` explanation
...
103
102
101  # ask
-------------Current spread
99   # bid
98
97
...
```

If `exit_pricing.price_side` is set to `"ask"`, then the bot will use 101 as exiting price.  
In line with that, if `exit_pricing.price_side` is set to `"bid"`, then the bot will use 99 as exiting price.

Depending on the order direction (_long_/_short_), this will lead to different results. Therefore we recommend to use `"same"` or `"other"` for this configuration instead.
This would result in the following pricing matrix:

| Direction | Order | setting | price | crosses spread |
|------ |--------|-----|-----|-----|
| long  | sell | ask   | 101 | no  |
| long  | sell | bid   | 99  | yes |
| long  | sell | same  | 101 | no  |
| long  | sell | other | 99  | yes |
| short | buy  | ask   | 101 | yes |
| short | buy  | bid   | 99  | no  |
| short | buy  | same  | 99  | no  |
| short | buy  | other | 101 | yes |

#### Exit price with Orderbook enabled

When exiting with the orderbook enabled (`exit_pricing.use_order_book=True`), Freqtrade fetches the `exit_pricing.order_book_top` entries in the orderbook and uses the entry specified as `exit_pricing.order_book_top` from the configured side (`exit_pricing.price_side`) as trade exit price.

1 specifies the topmost entry in the orderbook, while 2 would use the 2nd entry in the orderbook, and so on.

#### Exit price without Orderbook enabled

The following section uses `side` as the configured `exit_pricing.price_side` (defaults to `"ask"`).

When not using orderbook (`exit_pricing.use_order_book=False`), Freqtrade uses the best `side` price from the ticker if it's above the `last` traded price from the ticker. Otherwise (when the `side` price is below the `last` price), it calculates a rate between `side` and `last` price based on `exit_pricing.price_last_balance`.

The `exit_pricing.price_last_balance` configuration parameter controls this. A value of `0.0` will use `side` price, while `1.0` will use the last price and values between those interpolate between `side` and last price.

### Market order pricing

When using market orders, prices should be configured to use the "correct" side of the orderbook to allow realistic pricing detection.
Assuming both entry and exits are using market orders, a configuration similar to the following must be used

``` jsonc
  "order_types": {
    "entry": "market",
    "exit": "market"
    // ...
  },
  "entry_pricing": {
    "price_side": "other",
    // ...
  },
  "exit_pricing":{
    "price_side": "other",
    // ...
  },
```

Obviously, if only one side is using limit orders, different pricing combinations can be used.
