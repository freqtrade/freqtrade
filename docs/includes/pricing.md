## Prices used for orders

Prices for regular orders can be controlled via the parameter structures `bid_strategy` for buying and `ask_strategy` for selling.
Prices are always retrieved right before an order is placed, either by querying the exchange tickers or by using the orderbook data.

!!! Note
    Orderbook data used by Freqtrade are the data retrieved from exchange by the ccxt's function `fetch_order_book()`, i.e. are usually data from the L2-aggregated orderbook, while the ticker data are the structures returned by the ccxt's `fetch_ticker()`/`fetch_tickers()` functions. Refer to the ccxt library [documentation](https://github.com/ccxt/ccxt/wiki/Manual#market-data) for more details.

!!! Warning "Using market orders"
    Please read the section [Market order pricing](#market-order-pricing) section when using market orders.

### Buy price

#### Check depth of market

When check depth of market is enabled (`bid_strategy.check_depth_of_market.enabled=True`), the buy signals are filtered based on the orderbook depth (sum of all amounts) for each orderbook side.

Orderbook `bid` (buy) side depth is then divided by the orderbook `ask` (sell) side depth and the resulting delta is compared to the value of the `bid_strategy.check_depth_of_market.bids_to_ask_delta` parameter. The buy order is only executed if the orderbook delta is greater than or equal to the configured delta value.

!!! Note
    A delta value below 1 means that `ask` (sell) orderbook side depth is greater than the depth of the `bid` (buy) orderbook side, while a value greater than 1 means opposite (depth of the buy side is higher than the depth of the sell side).

#### Buy price side

The configuration setting `bid_strategy.price_side` defines the side of the spread the bot looks for when buying.

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

If `bid_strategy.price_side` is set to `"bid"`, then the bot will use 99 as buying price.  
In line with that, if `bid_strategy.price_side` is set to `"ask"`, then the bot will use 101 as buying price.

Using `ask` price often guarantees quicker filled orders, but the bot can also end up paying more than what would have been necessary.
Taker fees instead of maker fees will most likely apply even when using limit buy orders.
Also, prices at the "ask" side of the spread are higher than prices at the "bid" side in the orderbook, so the order behaves similar to a market order (however with a maximum price).

#### Buy price with Orderbook enabled

When buying with the orderbook enabled (`bid_strategy.use_order_book=True`), Freqtrade fetches the `bid_strategy.order_book_top` entries from the orderbook and then uses the entry specified as `bid_strategy.order_book_top` on the configured side (`bid_strategy.price_side`) of the orderbook. 1 specifies the topmost entry in the orderbook, while 2 would use the 2nd entry in the orderbook, and so on.

#### Buy price without Orderbook enabled

The following section uses `side` as the configured `bid_strategy.price_side`.

When not using orderbook (`bid_strategy.use_order_book=False`), Freqtrade uses the best `side` price from the ticker if it's below the `last` traded price from the ticker. Otherwise (when the `side` price is above the `last` price), it calculates a rate between `side` and `last` price.

The `bid_strategy.ask_last_balance` configuration parameter controls this. A value of `0.0` will use `side` price, while `1.0` will use the `last` price and values between those interpolate between ask and last price.

### Sell price

#### Sell price side

The configuration setting `ask_strategy.price_side` defines the side of the spread the bot looks for when selling.

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

If `ask_strategy.price_side` is set to `"ask"`, then the bot will use 101 as selling price.  
In line with that, if `ask_strategy.price_side` is set to `"bid"`, then the bot will use 99 as selling price.

#### Sell price with Orderbook enabled

When selling with the orderbook enabled (`ask_strategy.use_order_book=True`), Freqtrade fetches the `ask_strategy.order_book_max` entries in the orderbook. Then each of the orderbook steps between `ask_strategy.order_book_min` and `ask_strategy.order_book_max` on the configured orderbook side are validated for a profitable sell-possibility based on the strategy configuration (`minimal_roi` conditions) and the sell order is placed at the first profitable spot.

!!! Note
    Using `order_book_max` higher than `order_book_min` only makes sense when ask_strategy.price_side is set to `"ask"`.

The idea here is to place the sell order early, to be ahead in the queue.

A fixed slot (mirroring `bid_strategy.order_book_top`) can be defined by setting `ask_strategy.order_book_min` and `ask_strategy.order_book_max` to the same number.

!!! Warning "Order_book_max > 1 - increased risks for stoplosses!"
    Using `ask_strategy.order_book_max` higher than 1 will increase the risk the stoploss on exchange is cancelled too early, since an eventual [stoploss on exchange](#understand-order_types) will be cancelled as soon as the order is placed.
    Also, the sell order will remain on the exchange for `unfilledtimeout.sell` (or until it's filled) - which can lead to missed stoplosses (with or without using stoploss on exchange).

!!! Warning "Order_book_max > 1 in dry-run"
    Using `ask_strategy.order_book_max` higher than 1 will result in improper dry-run results (significantly better than real orders executed on exchange), since dry-run assumes orders to be filled almost instantly.
    It is therefore advised to not use this setting for dry-runs.

#### Sell price without Orderbook enabled

When not using orderbook (`ask_strategy.use_order_book=False`), the price at the `ask_strategy.price_side` side (defaults to `"ask"`) from the ticker will be used as the sell price.

When not using orderbook (`ask_strategy.use_order_book=False`), Freqtrade uses the best `side` price from the ticker if it's below the `last` traded price from the ticker. Otherwise (when the `side` price is above the `last` price), it calculates a rate between `side` and `last` price.

The `ask_strategy.bid_last_balance` configuration parameter controls this. A value of `0.0` will use `side` price, while `1.0` will use the last price and values between those interpolate between `side` and last price.

### Market order pricing

When using market orders, prices should be configured to use the "correct" side of the orderbook to allow realistic pricing detection.
Assuming both buy and sell are using market orders, a configuration similar to the following might be used

``` jsonc
  "order_types": {
    "buy": "market",
    "sell": "market"
    // ...
  },
  "bid_strategy": {
    "price_side": "ask",
    // ...
  },
  "ask_strategy":{
    "price_side": "bid",
    // ...
  },
```

Obviously, if only one side is using limit orders, different pricing combinations can be used.
