# Advanced Strategies

This page explains some advanced concepts available for strategies.
If you're just getting started, please be familiar with the methods described in the [Strategy Customization](strategy-customization.md) documentation and with the [Freqtrade basics](bot-basics.md) first.

[Freqtrade basics](bot-basics.md) describes in which sequence each method described below is called, which can be helpful to understand which method to use for your custom needs.

!!! Note
    All callback methods described below should only be implemented in a strategy if they are actually used.

!!! Tip
    You can get a strategy template containing all below methods by running `freqtrade new-strategy --strategy MyAwesomeStrategy --template advanced`

## Storing information

Storing information can be accomplished by creating a new dictionary within the strategy class.

The name of the variable can be chosen at will, but should be prefixed with `cust_` to avoid naming collisions with predefined strategy variables.

```python
class AwesomeStrategy(IStrategy):
    # Create custom dictionary
    custom_info = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Check if the entry already exists
        if not metadata["pair"] in self.custom_info:
            # Create empty entry for this pair
            self.custom_info[metadata["pair"]] = {}

        if "crosstime" in self.custom_info[metadata["pair"]]:
            self.custom_info[metadata["pair"]]["crosstime"] += 1
        else:
            self.custom_info[metadata["pair"]]["crosstime"] = 1
```

!!! Warning
    The data is not persisted after a bot-restart (or config-reload). Also, the amount of data should be kept smallish (no DataFrames and such), otherwise the bot will start to consume a lot of memory and eventually run out of memory and crash.

!!! Note
    If the data is pair-specific, make sure to use pair as one of the keys in the dictionary.

***

### Storing custom information using DatetimeIndex from `dataframe`

Imagine you need to store an indicator like `ATR` or `RSI` into `custom_info`. To use this in a meaningful way, you will not only need the raw data of the indicator, but probably also need to keep the right timestamps.

```python
import talib.abstract as ta
class AwesomeStrategy(IStrategy):
    # Create custom dictionary
    custom_info = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # using "ATR" here as example
        dataframe['atr'] = ta.ATR(dataframe)
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
          # add indicator mapped to correct DatetimeIndex to custom_info
          self.custom_info[metadata['pair']] = dataframe[['date', 'atr']].set_index('date')
        return dataframe
```

!!! Warning
    The data is not persisted after a bot-restart (or config-reload). Also, the amount of data should be kept smallish (no DataFrames and such), otherwise the bot will start to consume a lot of memory and eventually run out of memory and crash.

!!! Note
    If the data is pair-specific, make sure to use pair as one of the keys in the dictionary.

See `custom_stoploss` examples below on how to access the saved dataframe columns

## Custom stoploss

The stoploss price can only ever move upwards - if the stoploss value returned from `custom_stoploss` would result in a lower stoploss price than was previously set, it will be ignored. The traditional `stoploss` value serves as an absolute lower level and will be instated as the initial stoploss.

The usage of the custom stoploss method must be enabled by setting `use_custom_stoploss=True` on the strategy object.
The method must return a stoploss value (float / number) as a percentage of the current price.
E.g. If the `current_rate` is 200 USD, then returning `0.02` will set the stoploss price 2% lower, at 196 USD.

The absolute value of the return value is used (the sign is ignored), so returning `0.05` or `-0.05` have the same result, a stoploss 5% below the current price.

To simulate a regular trailing stoploss of 4% (trailing 4% behind the maximum reached price) you would use the following very simple method:

``` python
# additional imports required
from datetime import datetime
from freqtrade.persistence import Trade

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
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
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=120) > trade.open_date_utc:
            return -0.05
        elif current_time - timedelta(minutes=60) > trade.open_date_utc:
            return -0.10
        return 1
```

#### Different stoploss per pair

Use a different stoploss depending on the pair.
In this example, we'll trail the highest price with 10% trailing stoploss for `ETH/BTC` and `XRP/BTC`, with 5% trailing stoploss for `LTC/BTC` and with 15% for all other pairs.

``` python
from datetime import datetime
from freqtrade.persistence import Trade

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if pair in ('ETH/BTC', 'XRP/BTC'):
            return -0.10
        elif pair in ('LTC/BTC'):
            return -0.05
        return -0.15
```

#### Trailing stoploss with positive offset

Use the initial stoploss until the profit is above 4%, then use a trailing stoploss of 50% of the current profit with a minimum of 2.5% and a maximum of 5%.

Please note that the stoploss can only increase, values lower than the current stoploss are ignored.

``` python
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit < 0.04:
            return -1 # return a value bigger than the inital stoploss to keep using the inital stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 2

        # Use a minimum of 2.5% and a maximum of 5%
        return max(min(desired_stoploss, 0.05), 0.025)
```

#### Calculating stoploss relative to open price

Stoploss values returned from `custom_stoploss()` always specify a percentage relative to `current_rate`. In order to set a stoploss relative to the *open* price, we need to use `current_profit` to calculate what percentage relative to the `current_rate` will give you the same result as if the percentage was specified from the open price.

The helper function [`stoploss_from_open()`](strategy-customization.md#stoploss_from_open) can be used to convert from an open price relative stop, to a current price relative stop which can be returned from `custom_stoploss()`.

#### Stepped stoploss

Instead of continuously trailing behind the current price, this example sets fixed stoploss price levels based on the current profit.

* Use the regular stoploss until 20% profit is reached
* Once profit is > 20% - set stoploss to 7% above open price.
* Once profit is > 25% - set stoploss to 15% above open price.
* Once profit is > 40% - set stoploss to 25% above open price.


``` python
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # evaluate highest to lowest, so that highest possible stop is used
        if current_profit > 0.40:
            return stoploss_from_open(0.25, current_profit)
        elif current_profit > 0.25:
            return stoploss_from_open(0.15, current_profit)
        elif current_profit > 0.20:
            return stoploss_from_open(0.07, current_profit)

        # return maximum stoploss value, keeping current stoploss price unchanged
        return 1
```
#### Custom stoploss using an indicator from dataframe example

Imagine you want to use `custom_stoploss()` to use a trailing indicator like e.g. "ATR"

See: "Storing custom information using DatetimeIndex from `dataframe`" example above) on how to store the indicator into `custom_info`

!!! Warning
    only use .iat[-1] in live mode, not in backtesting/hyperopt
    otherwise you will look into the future
    see [Common mistakes when developing strategies](strategy-customization.md#common-mistakes-when-developing-strategies) for more info.

``` python
from freqtrade.persistence import Trade
from freqtrade.state import RunMode

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        result = 1
        if self.custom_info and pair in self.custom_info and trade:
            # using current_time directly (like below) will only work in backtesting.
            # so check "runmode" to make sure that it's only used in backtesting/hyperopt
            if self.dp and self.dp.runmode.value in ('backtest', 'hyperopt'):
              relative_sl = self.custom_info[pair].loc[current_time]['atr']
            # in live / dry-run, it'll be really the current time
            else:
              # but we can just use the last entry from an already analyzed dataframe instead
              dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                       timeframe=self.timeframe)
              # WARNING
              # only use .iat[-1] in live mode, not in backtesting/hyperopt
              # otherwise you will look into the future
              # see: https://www.freqtrade.io/en/latest/strategy-customization/#common-mistakes-when-developing-strategies
              relative_sl = dataframe['atr'].iat[-1]

            if (relative_sl is not None):
                # new stoploss relative to current_rate
                new_stoploss = (current_rate-relative_sl)/current_rate
                # turn into relative negative offset required by `custom_stoploss` return implementation
                result = new_stoploss - 1

        return result
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
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since our maximum timeout from below is 24 hours.
    unfilledtimeout = {
        'buy': 60 * 25,
        'sell': 60 * 25
    }

    def check_buy_timeout(self, pair: str, trade: 'Trade', order: dict, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date_utc < datetime.now(timezone.utc) - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date_utc < datetime.now(timezone.utc) - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date_utc < datetime.now(timezone.utc) - timedelta(hours=24):
           return True
        return False


    def check_sell_timeout(self, pair: str, trade: 'Trade', order: dict, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date_utc < datetime.now(timezone.utc) - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date_utc < datetime.now(timezone.utc) - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date_utc < datetime.now(timezone.utc) - timedelta(hours=24):
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

!!! Note "Parent-strategy in different files"
    If you have the parent-strategy in a different file, you'll need to add the following to the top of your "child"-file to ensure proper loading, otherwise freqtrade may not be able to load the parent strategy correctly.

    ``` python
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))

    from myawesomestrategy import MyAwesomeStrategy
    ```

## Embedding Strategies

Freqtrade provides you with with an easy way to embed the strategy into your configuration file.
This is done by utilizing BASE64 encoding and providing this string at the strategy configuration field,
in your chosen config file.

### Encoding a string as BASE64

This is a quick example, how to generate the BASE64 string in python

```python
from base64 import urlsafe_b64encode

with open(file, 'r') as f:
    content = f.read()
content = urlsafe_b64encode(content.encode('utf-8'))
```

The variable 'content', will contain the strategy file in a BASE64 encoded form. Which can now be set in your configurations file as following

```json
"strategy": "NameOfStrategy:BASE64String"
```

Please ensure that 'NameOfStrategy' is identical to the strategy name!
