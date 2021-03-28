# Stop Loss

The `stoploss` configuration parameter is loss as ratio that should trigger a sale.
For example, value `-0.10` will cause immediate sell if the profit dips below -10% for a given trade. This parameter is optional.

Most of the strategy files already include the optimal `stoploss` value.

!!! Info
    All stoploss properties mentioned in this file can be set in the Strategy, or in the configuration.  
    <ins>Configuration values will override the strategy values.</ins>

## Stop Loss On-Exchange/Freqtrade

Those stoploss modes can be *on exchange* or *off exchange*.

These modes can be configured with these values:

``` python
    'emergencysell': 'market',
    'stoploss_on_exchange': False
    'stoploss_on_exchange_interval': 60,
    'stoploss_on_exchange_limit_ratio': 0.99
```

!!! Note
    Stoploss on exchange is only supported for Binance (stop-loss-limit), Kraken (stop-loss-market, stop-loss-limit) and FTX (stop limit and stop-market) as of now.  
    <ins>Do not set too low/tight stoploss value if using stop loss on exchange!</ins>  
    If set to low/tight then you have greater risk of missing fill on the order and stoploss will not work.

### stoploss_on_exchange and stoploss_on_exchange_limit_ratio

Enable or Disable stop loss on exchange.
If the stoploss is *on exchange* it means a stoploss limit order is placed on the exchange immediately after buy order happens successfully. This will protect you against sudden crashes in market as the order will be in the queue immediately and if market goes down then the order has more chance of being fulfilled.

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

### forcesell

`forcesell` is an optional value, which defaults to the same value as `sell` and is used when sending a `/forcesell` command from Telegram or from the Rest API.

### forcebuy

`forcebuy` is an optional value, which defaults to the same value as `buy` and is used when sending a `/forcebuy` command from Telegram or from the Rest API.

### emergencysell

`emergencysell` is an optional value, which defaults to `market` and is used when creating stop loss on exchange orders fails.
The below is the default which is used if not changed in strategy or configuration file.

Example from strategy file:

``` python
order_types = {
    'buy': 'limit',
    'sell': 'limit',
    'emergencysell': 'market',
    'stoploss': 'market',
    'stoploss_on_exchange': True,
    'stoploss_on_exchange_interval': 60,
    'stoploss_on_exchange_limit_ratio': 0.99
}
```

## Stop Loss Types

At this stage the bot contains the following stoploss support modes:

1. Static stop loss.
2. Trailing stop loss.
3. Trailing stop loss, custom positive loss.
4. Trailing stop loss only once the trade has reached a certain offset.
5. [Custom stoploss function](strategy-advanced.md#custom-stoploss)

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

It is also possible to have a default stop loss, when you are in the red with your buy (buy - fee), but once you hit positive result the system will utilize a new stop loss, which can have a different value.
For example, your default stop loss is -10%, but once you have more than 0% profit (example 0.1%) a different trailing stoploss will be used.

!!! Note
    If you want the stoploss to only be changed when you break even of making a profit (what most users want) please refer to next section with [offset enabled](#Trailing-stop-loss-only-once-the-trade-has-reached-a-certain-offset).

Both values require `trailing_stop` to be set to true and `trailing_stop_positive` with a value.

``` python
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
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

### Trailing stop loss only once the trade has reached a certain offset

It is also possible to use a static stoploss until the offset is reached, and then trail the trade to take profits once the market turns.

If `"trailing_only_offset_is_reached": true` then the trailing stoploss is only activated once the offset is reached. Until then, the stoploss remains at the configured `stoploss`.
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
* stoploss will remain at 90$ unless asset increases to or above our configured offset
* assuming the asset now increases to 103$ (where we have the offset configured)
* the stop loss will now be -2% of 103$ = 100.94$
* now the asset drops in value to 101\$, the stop loss will still be 100.94$ and would trigger at 100.94$

!!! Tip
    Make sure to have this value (`trailing_stop_positive_offset`) lower than minimal ROI, otherwise minimal ROI will apply first and sell the trade.

## Changing stoploss on open trades

A stoploss on an open trade can be changed by changing the value in the configuration or strategy and use the `/reload_config` command (alternatively, completely stopping and restarting the bot also works).

The new stoploss value will be applied to open trades (and corresponding log-messages will be generated).

### Limitations

Stoploss values cannot be changed if `trailing_stop` is enabled and the stoploss has already been adjusted, or if [Edge](edge.md) is enabled (since Edge would recalculate stoploss based on the current market situation).
