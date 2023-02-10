# Stop Loss

The `stoploss` configuration parameter is loss as ratio that should trigger a sale.
For example, value `-0.10` will cause immediate sell if the profit dips below -10% for a given trade. This parameter is optional.
Stoploss calculations do include fees, so a stoploss of -10% is placed exactly 10% below the entry point.

Most of the strategy files already include the optimal `stoploss` value.

!!! Info
    All stoploss properties mentioned in this file can be set in the Strategy, or in the configuration.  
    <ins>Configuration values will override the strategy values.</ins>

## Stop Loss On-Exchange/Freqtrade

Those stoploss modes can be *on exchange* or *off exchange*.

These modes can be configured with these values:

``` python
    'emergency_exit': 'market',
    'stoploss_on_exchange': False
    'stoploss_on_exchange_interval': 60,
    'stoploss_on_exchange_limit_ratio': 0.99
```

!!! Note
    Stoploss on exchange is only supported for Binance (stop-loss-limit), Huobi (stop-limit), Kraken (stop-loss-market, stop-loss-limit), Gate (stop-limit), and Kucoin (stop-limit and stop-market) as of now.  
    <ins>Do not set too low/tight stoploss value if using stop loss on exchange!</ins>  
    If set to low/tight then you have greater risk of missing fill on the order and stoploss will not work.

### stoploss_on_exchange and stoploss_on_exchange_limit_ratio

Enable or Disable stop loss on exchange.
If the stoploss is *on exchange* it means a stoploss limit order is placed on the exchange immediately after buy order fills. This will protect you against sudden crashes in market, as the order execution happens purely within the exchange, and has no potential network overhead.

If `stoploss_on_exchange` uses limit orders, the exchange needs 2 prices, the stoploss_price and the Limit price.  
`stoploss` defines the stop-price where the limit order is placed - and limit should be slightly below this.  
If an exchange supports both limit and market stoploss orders, then the value of `stoploss` will be used to determine the stoploss type.  

Calculation example: we bought the asset at 100\$.  
Stop-price is 95\$, then limit would be `95 * 0.99 = 94.05$` - so the limit order fill can happen between 95$ and 94.05$.  

For example, assuming the stoploss is on exchange, and trailing stoploss is enabled, and the market is going up, then the bot automatically cancels the previous stoploss order and puts a new one with a stop value higher than the previous stoploss order.

!!! Note
    If `stoploss_on_exchange` is enabled and the stoploss is cancelled manually on the exchange, then the bot will create a new stoploss order.

### stoploss_on_exchange_interval

In case of stoploss on exchange there is another parameter called `stoploss_on_exchange_interval`. This configures the interval in seconds at which the bot will check the stoploss and update it if necessary.  
The bot cannot do these every 5 seconds (at each iteration), otherwise it would get banned by the exchange.
So this parameter will tell the bot how often it should update the stoploss order. The default value is 60 (1 minute).
This same logic will reapply a stoploss order on the exchange should you cancel it accidentally.

### stoploss_price_type

!!! Warning "Only applies to futures"
    `stoploss_price_type` only applies to futures markets (on exchanges where it's available).
    Freqtrade will perform a validation of this setting on startup, failing to start if an invalid setting for your exchange has been selected.
    Supported price types are gonna differs between each exchanges. Please check with your exchange on which price types it supports.

Stoploss on exchange on futures markets can trigger on different price types.
The naming for these prices in exchange terminology often varies, but is usually something around "last" (or "contract price" ), "mark" and "index".

Acceptable values for this setting are `"last"`, `"mark"` and `"index"` - which freqtrade will transfer automatically to the corresponding API type, and place the [stoploss on exchange](#stoploss_on_exchange-and-stoploss_on_exchange_limit_ratio) order correspondingly.

### force_exit

`force_exit` is an optional value, which defaults to the same value as `exit` and is used when sending a `/forceexit` command from Telegram or from the Rest API.

### force_entry

`force_entry` is an optional value, which defaults to the same value as `entry` and is used when sending a `/forceentry` command from Telegram or from the Rest API.

### emergency_exit

`emergency_exit` is an optional value, which defaults to `market` and is used when creating stop loss on exchange orders fails.
The below is the default which is used if not changed in strategy or configuration file.

Example from strategy file:

``` python
order_types = {
    "entry": "limit",
    "exit": "limit",
    "emergency_exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": True,
    "stoploss_on_exchange_interval": 60,
    "stoploss_on_exchange_limit_ratio": 0.99
}
```

## Stop Loss Types

At this stage the bot contains the following stoploss support modes:

1. Static stop loss.
2. Trailing stop loss.
3. Trailing stop loss, custom positive loss.
4. Trailing stop loss only once the trade has reached a certain offset.
5. [Custom stoploss function](strategy-callbacks.md#custom-stoploss)

### Static Stop Loss

This is very simple, you define a stop loss of x (as a ratio of price, i.e. x * 100% of price). This will try to sell the asset once the loss exceeds the defined loss.

Example of stop loss:

``` python
    stoploss = -0.10
```

For example, simplified math:

* the bot buys an asset at a price of 100$
* the stop loss is defined at -10%
* the stop loss would get triggered once the asset drops below 90$

### Trailing Stop Loss

The initial value for this is `stoploss`, just as you would define your static Stop loss.
To enable trailing stoploss:

``` python
    stoploss = -0.10
    trailing_stop = True
```

This will now activate an algorithm, which automatically moves the stop loss up every time the price of your asset increases.

For example, simplified math:

* the bot buys an asset at a price of 100$
* the stop loss is defined at -10%
* the stop loss would get triggered once the asset drops below 90$
* assuming the asset now increases to 102$
* the stop loss will now be -10% of 102$ = 91.8$
* now the asset drops in value to 101\$, the stop loss will still be 91.8$ and would trigger at 91.8$.

In summary: The stoploss will be adjusted to be always be -10% of the highest observed price.

### Trailing stop loss, custom positive loss

You could also have a default stop loss when you are in the red with your buy (buy - fee), but once you hit a positive result (or an offset you define) the system will utilize a new stop loss, which can have a different value.
For example, your default stop loss is -10%, but once you have more than 0% profit (example 0.1%) a different trailing stoploss will be used.

!!! Note
    If you want the stoploss to only be changed when you break even of making a profit (what most users want) please refer to next section with [offset enabled](#Trailing-stop-loss-only-once-the-trade-has-reached-a-certain-offset).

Both values require `trailing_stop` to be set to true and `trailing_stop_positive` with a value.

``` python
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False  # Default - not necessary for this example
```

For example, simplified math:

* the bot buys an asset at a price of 100$
* the stop loss is defined at -10%
* the stop loss would get triggered once the asset drops below 90$
* assuming the asset now increases to 102$
* the stop loss will now be -2% of 102$ = 99.96$ (99.96$ stop loss will be locked in and will follow asset price increments with -2%)
* now the asset drops in value to 101\$, the stop loss will still be 99.96$ and would trigger at 99.96$

The 0.02 would translate to a -2% stop loss.
Before this, `stoploss` is used for the trailing stoploss.

!!! Tip "Use an offset to change your stoploss"
    Use `trailing_stop_positive_offset` to ensure that your new trailing stoploss will be in profit by setting `trailing_stop_positive_offset` higher than `trailing_stop_positive`. Your first new stoploss value will then already have locked in profits.

    Example with simplified math:

    ``` python
        stoploss = -0.10
        trailing_stop = True
        trailing_stop_positive = 0.02
        trailing_stop_positive_offset = 0.03
    ```

    * the bot buys an asset at a price of 100$
    * the stop loss is defined at -10%, so the stop loss would get triggered once the asset drops below 90$
    * assuming the asset now increases to 102$
    * the stoploss will now be at 91.8$ - 10% below the highest observed rate
    * assuming the asset now increases to 103.5$ (above the offset configured)
    * the stop loss will now be -2% of 103.5$ = 101.43$
    * now the asset drops in value to 102\$, the stop loss will still be 101.43$ and would trigger once price breaks below 101.43$

### Trailing stop loss only once the trade has reached a certain offset

You can also keep a static stoploss until the offset is reached, and then trail the trade to take profits once the market turns.

If `trailing_only_offset_is_reached = True` then the trailing stoploss is only activated once the offset is reached. Until then, the stoploss remains at the configured `stoploss`.
This option can be used with or without `trailing_stop_positive`, but uses `trailing_stop_positive_offset` as offset.

``` python
    trailing_stop_positive_offset = 0.011
    trailing_only_offset_is_reached = True
```

Configuration (offset is buy-price + 3%):

``` python
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
```

For example, simplified math:

* the bot buys an asset at a price of 100$
* the stop loss is defined at -10%
* the stop loss would get triggered once the asset drops below 90$
* stoploss will remain at 90$ unless asset increases to or above the configured offset
* assuming the asset now increases to 103$ (where we have the offset configured)
* the stop loss will now be -2% of 103$ = 100.94$
* now the asset drops in value to 101\$, the stop loss will still be 100.94$ and would trigger at 100.94$

!!! Tip
    Make sure to have this value (`trailing_stop_positive_offset`) lower than minimal ROI, otherwise minimal ROI will apply first and sell the trade.

## Stoploss and Leverage

Stoploss should be thought of as "risk on this trade" - so a stoploss of 10% on a 100$ trade means you are willing to lose 10$ (10%) on this trade - which would trigger if the price moves 10% to the downside.

When using leverage, the same principle is applied - with stoploss defining the risk on the trade (the amount you are willing to lose).

Therefore, a stoploss of 10% on a 10x trade would trigger on a 1% price move.
If your stake amount (own capital) was 100$ - this trade would be 1000$ at 10x (after leverage).
If price moves 1% - you've lost 10$ of your own capital - therfore stoploss will trigger in this case.

Make sure to be aware of this, and avoid using too tight stoploss (at 10x leverage, 10% risk may be too little to allow the trade to "breath" a little).

## Changing stoploss on open trades

A stoploss on an open trade can be changed by changing the value in the configuration or strategy and use the `/reload_config` command (alternatively, completely stopping and restarting the bot also works).

The new stoploss value will be applied to open trades (and corresponding log-messages will be generated).

### Limitations

Stoploss values cannot be changed if `trailing_stop` is enabled and the stoploss has already been adjusted, or if [Edge](edge.md) is enabled (since Edge would recalculate stoploss based on the current market situation).
