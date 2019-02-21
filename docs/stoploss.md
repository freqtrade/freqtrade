# Stop Loss support

At this stage the bot contains the following stoploss support modes:

1. static stop loss, defined in either the strategy or configuration.
2. trailing stop loss, defined in the configuration.
3. trailing stop loss, custom positive loss, defined in configuration.

!!! Note
    All stoploss properties can be configured in either Strategy or configuration. Configuration values override strategy values.

Those stoploss modes can be *on exchange* or *off exchange*. If the stoploss is *on exchange* it means a stoploss limit order is placed on the exchange immediately after buy order happens successfuly. This will protect you against sudden crashes in market as the order will be in the queue immediately and if market goes down then the order has more chance of being fulfilled.

In case of stoploss on exchange there is another parameter called `stoploss_on_exchange_interval`. This configures the interval in seconds at which the bot will check the stoploss and update it if necessary. As an example in case of trailing stoploss if the order is on the exchange and the market is going up then the bot automatically cancels the previous stoploss order and put a new one with a stop value higher than previous one. It is clear that the bot cannot do it every 5 seconds otherwise it gets banned. So this parameter will tell the bot how often it should update the stoploss order. The default value is 60 (1 minute).

!!! Note
    Stoploss on exchange is only supported for Binance as of now.


## Static Stop Loss

This is very simple, basically you define a stop loss of x in your strategy file or alternative in the configuration, which
will overwrite the strategy definition. This will basically try to sell your asset, the second the loss exceeds the defined loss.

## Trail Stop Loss

The initial value for this stop loss, is defined in your strategy or configuration. Just as you would define your Stop Loss normally.
To enable this Feauture all you have to do is to define the configuration element:

``` json
"trailing_stop" : True
```

This will now activate an algorithm, which automatically moves your stop loss up every time the price of your asset increases.

For example, simplified math,

* you buy an asset at a price of 100$
* your stop loss is defined at 2%
* which means your stop loss, gets triggered once your asset dropped below 98$
* assuming your asset now increases to 102$
* your stop loss, will now be 2% of 102$ or 99.96$
* now your asset drops in value to 101$, your stop loss, will still be 99.96$

basically what this means is that your stop loss will be adjusted to be always be 2% of the highest observed price

### Custom positive loss

Due to demand, it is possible to have a default stop loss, when you are in the red with your buy, but once your profit surpasses a certain percentage,
the system will utilize a new stop loss, which can be a different value. For example your default stop loss is 5%, but once you have 1.1% profit,
it will be changed to be only a 1% stop loss, which trails the green candles until it goes below them.

Both values can be configured in the main configuration file and requires `"trailing_stop": true` to be set to true.

``` json
    "trailing_stop_positive":  0.01,
    "trailing_stop_positive_offset":  0.011,
```

The 0.01 would translate to a 1% stop loss, once you hit 1.1% profit.

You should also make sure to have this value (`trailing_stop_positive_offset`) lower than your minimal ROI, otherwise minimal ROI will apply first and sell your trade.
