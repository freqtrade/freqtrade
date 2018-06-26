# Stop Loss support

At this stage the bot contains the following stoploss support modes:

1. static stop loss, defined in either the strategy or configuration
2. trailing stop loss, defined in the configuration
3. trailing stop loss, custom positive loss, defined in configuration

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

Due to demand, it is possible to have a default stop loss, when you are in the red with your buy, but once your buy turns positive,
the system will utilize a new stop loss, which can be a different value. For example your default stop loss is 5%, but once you are in the
black, it will be changed to be only a 1% stop loss

this can be configured in the main configuration file, the following way:

``` json
    "trailing_stop": {
        "positive" : 0.01
    },
```

The 0.01 would translate to a 1% stop loss, once you hit profit.
