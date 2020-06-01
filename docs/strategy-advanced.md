# Advanced Strategies

This page explains some advanced concepts available for strategies.
If you're just getting started, please be familiar with the methods described in the [Strategy Customization](strategy-customization.md) documentation first.

## Custom order timeout rules

Simple, timebased order-timeouts can be configured either via strategy or in the configuration in the `unfilledtimeout` section.

However, freqtrade also offers a custom callback for both ordertypes, which allows you to decide based on custom criteria if a order did time out or not.

!!! Note
    Unfilled order timeouts are not relevant during backtesting or hyperopt, and are only relevant during real (live) trading. Therefore these methods are only called in these circumstances.

### Custom order timeout example

A simple example, which applies different unfilled-timeouts depending on the price of the asset can be seen below.
It applies a tight timeout for higher priced assets, while allowing more time to fill on cheap coins.

The function must return either `True` (cancel order) or `False` (keep order alive).

``` python
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

class Awesomestrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since our maximum timeout from below is 24 hours.
    unfilledtimeout = {
        'buy': 60 * 25,
        'sell': 60 * 25
    }

    def check_buy_timeout(self, pair: str, trade: 'Trade', order: dict, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date < datetime.utcnow() - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date < datetime.utcnow() - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date < datetime.utcnow() - timedelta(hours=24):
           return True
        return False


    def check_sell_timeout(self, pair: str, trade: 'Trade', order: dict, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date < datetime.utcnow() - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date < datetime.utcnow() - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date < datetime.utcnow() - timedelta(hours=24):
           return True
        return False
```

!!! Note
    For the above example, `unfilledtimeout` must be set to something bigger than 24h, otherwise that type of timeout will apply first.

### Custom order timeout example (using additional data)

``` python
from datetime import datetime
from freqtrade.persistence import Trade

class Awesomestrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since our maximum timeout from below is 24 hours.
    unfilledtimeout = {
        'buy': 60 * 25,
        'sell': 60 * 25
    }

    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['bids'][0][0]
        # Cancel buy order if price is more than 2% above the order.
        if current_price > order['price'] * 1.02:
            return True
        return False


    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel sell order if price is more than 2% below the order.
        if current_price < order['price'] * 0.98:
            return True
        return False
```
