# Advanced Strategies

This page explains some advanced concepts available for strategies.
If you're just getting started, please be familiar with the methods described in the [Strategy Customization](strategy-customization.md) documentation and with the [Freqtrade basics](bot-basics.md) first.

[Freqtrade basics](bot-basics.md) describes in which sequence each method described below is called, which can be helpful to understand which method to use for your custom needs.

!!! Note
    All callback methods described below should only be implemented in a strategy if they are actually used.

!!! Tip
    You can get a strategy template containing all below methods by running `freqtrade new-strategy --strategy MyAwesomeStrategy --template advanced`

## Custom stoploss

A stoploss can only ever move upwards - so if you set it to an absolute profit of 2%, you can never move it below this price.
Also, the traditional `stoploss` value serves as an absolute lower level and will be instated as the initial stoploss.

The usage of the custom stoploss method must be enabled by setting `use_custom_stoploss=True` on the strategy object.
The method must return a stoploss value (float / number) with a relative ratio below the current price.
E.g. `current_profit = 0.05` (5% profit) - stoploss returns `0.02` - then you "locked in" a profit of 3% (`0.05 - 0.02 = 0.03`).

To simulate a regular trailing stoploss of 4% (trailing 4% behind the maximum reached price) you would use the following very simple method:

``` python
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the currentrate
        """
        return -0.04
```

Stoploss on exchange works similar to `trailing_stop`, and the stoploss on exchange is updated as configured in `stoploss_on_exchange_interval` ([More details about stoploss on exchange](stoploss.md#stop-loss-on-exchange-freqtrade)).

!!! Note "Use of dates"
    All time-based calculations should be done based on `current_time` - using `datetime.now()` or `datetime.utcnow()` is discouraged, as this will break backtesting support.

!!! Tip "Trailing stoploss"
    It's recommended to disable `trailing_stop` when using custom stoploss values. Both can work in tandem, but you might encounter the trailing stop to move the price higher while your custom function would not want this, causing conflicting behavior.

### Custom stoploss examples

The next section will show some examples on what's possible with the custom stoploss function.
Of course, many more things are possible, and all examples can be combined at will.

#### Time based trailing stop

Use the initial stoploss for the first 60 minutes, after this change to 10% trailing stoploss, and after 2 hours (120 minutes) we use a 5% trailing stoploss.

``` python
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=120) > trade.open_date:
            return -0.05
        elif current_time - timedelta(minutes=60) > trade.open_date:
            return -0.10
        return 1
```

#### Different stoploss per pair

Use a different stoploss depending on the pair.
In this example, we'll trail the highest price with 10% trailing stoploss for `ETH/BTC` and `XRP/BTC`, with 5% trailing stoploss for `LTC/BTC` and with 15% for all other pairs.

``` python
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        if pair in ('ETH/BTC', 'XRP/BTC'):
            return -0.10
        elif pair in ('LTC/BTC'):
            return -0.05
        return -0.15
```

#### Absolute stoploss

The below example sets absolute profit levels based on the current profit.

* Use the regular stoploss until 20% profit is reached
* Once profit is > 40%, stoploss will be at 25%, locking in at least 25% of the profit.
* Once profit is > 25% - stoploss will be 15%.
* Once profit is > 20% - stoploss will be set to 7%.

``` python
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        # Calculate as `-desired_stop_from_open + current_profit` to get the distance between current_profit and initial price
        if current_profit > 0.40:
            return (-0.25 + current_profit)
        if current_profit > 0.25:
            return (-0.15 + current_profit)
        if current_profit > 0.20:
            return (-0.7 + current_profit)
        return 1
```

---

## Custom order timeout rules

Simple, time-based order-timeouts can be configured either via strategy or in the configuration in the `unfilledtimeout` section.

However, freqtrade also offers a custom callback for both order types, which allows you to decide based on custom criteria if a order did time out or not.

!!! Note
    Unfilled order timeouts are not relevant during backtesting or hyperopt, and are only relevant during real (live) trading. Therefore these methods are only called in these circumstances.

### Custom order timeout example

A simple example, which applies different unfilled-timeouts depending on the price of the asset can be seen below.
It applies a tight timeout for higher priced assets, while allowing more time to fill on cheap coins.

The function must return either `True` (cancel order) or `False` (keep order alive).

``` python
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

class AwesomeStrategy(IStrategy):

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

class AwesomeStrategy(IStrategy):

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

---

## Bot loop start callback

A simple callback which is called once at the start of every bot throttling iteration.
This can be used to perform calculations which are pair independent (apply to all pairs), loading of external data, etc.

``` python
import requests

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        if self.config['runmode'].value in ('live', 'dry_run'):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods
            self.remote_data = requests.get('https://some_remote_source.example.com')

```

## Bot order confirmation

### Trade entry (buy order) confirmation

`confirm_trade_entry()` can be used to abort a trade entry at the latest second (maybe because the price is not what we expect).

``` python
class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, **kwargs) -> bool:
        """
        Called right before placing a buy order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (quote) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        return True

```

### Trade exit (sell order) confirmation

`confirm_trade_exit()` can be used to abort a trade exit (sell) at the latest second (maybe because the price is not what we expect).

``` python
from freqtrade.persistence import Trade


class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        """
        Called right before placing a regular sell order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be sold.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in quote currency.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param sell_reason: Sell reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'sell_signal', 'force_sell', 'emergency_sell']
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the sell-order is placed on the exchange.
            False aborts the process
        """
        if sell_reason == 'force_sell' and trade.calc_profit_ratio(rate) < 0:
            # Reject force-sells with negative profit
            # This is just a sample, please adjust to your needs
            # (this does not necessarily make sense, assuming you know when you're force-selling)
            return False
        return True

```

---

## Derived strategies

The strategies can be derived from other strategies. This avoids duplication of your custom strategy code. You can use this technique to override small parts of your main strategy, leaving the rest untouched:

``` python
class MyAwesomeStrategy(IStrategy):
    ...
    stoploss = 0.13
    trailing_stop = False
    # All other attributes and methods are here as they
    # should be in any custom strategy...
    ...

class MyAwesomeStrategy2(MyAwesomeStrategy):
    # Override something
    stoploss = 0.08
    trailing_stop = True
```

Both attributes and methods may be overridden, altering behavior of the original strategy in a way you need.
