# Stop Loss

The `stoploss` configuration parameter is loss in percentage that should trigger a sale.
For example, value `-0.10` will cause immediate sell if the profit dips below -10% for a given trade. This parameter is optional.

Most of the strategy files already include the optimal `stoploss` value.
Stoploss parameters need to be set in either strategy or configuration file.
Parameters in the configuration will overwrite settings within the strategy.

## Stop Loss Types

At this stage the bot contains the following stoploss support modes:

1. static stop loss, defined in either the strategy or configuration.
2. trailing stop loss, defined in the configuration.
3. trailing stop loss, custom positive loss, defined in configuration.
4. trailing stop loss only once the trade has reached a certain offset,

!!! Note
    All stoploss properties can be configured in either Strategy or configuration. Configuration values override strategy values.

Those stoploss modes can be *on exchange* or *off exchange*. If the stoploss is *on exchange* it means a stoploss limit order is placed on the exchange immediately after buy order happens successfully. This will protect you against sudden crashes in market as the order will be in the queue immediately and if market goes down then the order has more chance of being fulfilled.

In case of stoploss on exchange there is another parameter called `stoploss_on_exchange_interval`. This configures the interval in seconds at which the bot will check the stoploss and update it if necessary.

For example, assuming the stoploss is on exchange, and trailing stoploss is enabled, and the market is going up, then the bot automatically cancels the previous stoploss order and puts a new one with a stop value higher than the previous stoploss order.
The bot cannot do this every 5 seconds, otherwise it would get banned by the exchange.
So this parameter will tell the bot how often it should update the stoploss order. The default value is 60 (1 minute).
This same logic will reapply a stoploss order on the exchange should you cancel it accidentally.

!!! Note
    Stoploss on exchange is only supported for Binance as of now.

## Static Stop Loss

This is very simple, you define a stop loss of x. This will try to sell the asset once the loss exceeds the defined loss.

## Trailing Stop Loss

The initial value for this is `stoploss`, set either in the strategy or in the configuration file. Just as you would define your Stop loss normally.
To enable this Feauture all you have to do is to define the configuration element:

``` json
"trailing_stop" : True
```

This will now activate an algorithm, which automatically moves the stop loss up every time the price of your asset increases.

For example, simplified math:

* the bot buys an asset at a price of 100$
* the stop loss is defined at 2%
* the stop loss would get triggered once the asset dropps below 98$
* assuming the asset now increases to 102$
* the stop loss will now be 2% of 102$ or 99.96$
* now the asset drops in value to 101$, the stop loss, will still be 99.96$, and would trigger at 99.96$.

In summary: The stoploss will be adjusted to be always be 2% of the highest observed price.

### Custom positive stoploss

It is also possible to have a default stop loss, when you are in the red with your buy, but once your profit surpasses a certain percentage, the system will utilize a new stop loss, which can have a different value.
For example your default stop loss is 5%, but once you have 1.1% profit, it will be changed to be only a 1% stop loss, which trails the green candles until it goes below them.

Both values can be configured in the strategy or configuration file and requires `"trailing_stop": true` to be set to true.

``` json
    "trailing_stop_positive":  0.01,
    "trailing_stop_positive_offset":  0.011,
```

The 0.01 would translate to a 1% stop loss, once you hit 1.1% profit.

You should also make sure to have this value (`trailing_stop_positive_offset`) lower than your minimal ROI, otherwise minimal ROI will apply first and sell your trade.

### Trailing only once offset is reached

It is also possible to use a static stoploss until the offset is reached, and then trail the trade to take profits once the market goes down again.

If `"trailing_only_offset_is_reached": true` then the trailing stoploss is only activated once the offset is reached. Until then, the stoploss remains at the configured `stoploss`.
This option can be used with or without `trailing_stop_positive`, but uses `trailing_stop_positive_offset` as offset.

``` json
    "trailing_only_offset_is_reached": true,
```

Simplified example:

```python
    stoploss = 0.05
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
```

* the bot buys an asset at a price of 100$
* the stop loss is defined at 5%
* the stop loss will remain at 95% until profit reaches +3%

## Changing stoploss on open trades

A stoploss on an open trade can be changed by changing the value in the configuration or strategy and use the `/reload_conf` command (alternatively, completely stopping and restarting the bot also works).

The new stoploss value will be applied to open trades (and corresponding log-messages will be generated).

### Limitations

Stoploss values cannot be changed if `trailing_stop` is enabled and the stoploss has already been adjusted, or if [Edge](edge.md) is enabled (since Edge would recalculate stoploss based on the current market situation).
