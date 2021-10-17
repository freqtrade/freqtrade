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

## Dataframe access

You may access dataframe in various strategy functions by querying it from dataprovider.

``` python
from freqtrade.exchange import timeframe_to_prev_date

class AwesomeStrategy(IStrategy):
    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: 'datetime', **kwargs) -> bool:
        # Obtain pair dataframe.
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Obtain last available candle. Do not use current_time to look up latest candle, because 
        # current_time points to current incomplete candle whose data is not available.
        last_candle = dataframe.iloc[-1].squeeze()
        # <...>

        # In dry/live runs trade open date will not match candle open date therefore it must be 
        # rounded.
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # Look up trade candle.
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        # trade_candle may be empty for trades that just opened as it is still incomplete.
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            # <...>
```

!!! Warning "Using .iloc[-1]"
    You can use `.iloc[-1]` here because `get_analyzed_dataframe()` only returns candles that backtesting is allowed to see.
    This will not work in `populate_*` methods, so make sure to not use `.iloc[]` in that area.
    Also, this will only work starting with version 2021.5.

***

## Custom sell signal

It is possible to define custom sell signals, indicating that specified position should be sold. This is very useful when we need to customize sell conditions for each individual trade, or if you need the trade profit to take the sell decision. 

For example you could implement a 1:2 risk-reward ROI with `custom_sell()`.

Using custom_sell() signals in place of stoploss though *is not recommended*. It is a inferior method to using `custom_stoploss()` in this regard - which also allows you to keep the stoploss on exchange.

!!! Note
    Returning a `string` or `True` from this method is equal to setting sell signal on a candle at specified time. This method is not called when sell signal is set already, or if sell signals are disabled (`use_sell_signal=False` or `sell_profit_only=True` while profit is below `sell_profit_offset`). `string` max length is 64 characters. Exceeding this limit will cause the message to be truncated to 64 characters.

An example of how we can use different indicators depending on the current profit and also sell trades that were open longer than one day:

``` python
class AwesomeStrategy(IStrategy):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Above 20% profit, sell when rsi < 80
        if current_profit > 0.2:
            if last_candle['rsi'] < 80:
                return 'rsi_below_80'

        # Between 2% and 10%, sell if EMA-long above EMA-short
        if 0.02 < current_profit < 0.1:
            if last_candle['emalong'] > last_candle['emashort']:
                return 'ema_long_below_80'

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 1:
            return 'unclog'
```

See [Dataframe access](#dataframe-access) for more information about dataframe use in strategy callbacks.

## Buy Tag

When your strategy has multiple buy signals, you can name the signal that triggered.
Then you can access you buy signal on `custom_sell`

```python
def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['rsi'] < 35) &
            (dataframe['volume'] > 0)
        ),
        ['buy', 'buy_tag']] = (1, 'buy_signal_rsi')

    return dataframe

def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last_candle = dataframe.iloc[-1].squeeze()
    if trade.buy_tag == 'buy_signal_rsi' and last_candle['rsi'] > 80:
        return 'sell_signal_rsi'
    return None

```

!!! Note
    `buy_tag` is limited to 100 characters, remaining data will be truncated.


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
        :return float: New stoploss value, relative to the current rate
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
            return -1 # return a value bigger than the initial stoploss to keep using the initial stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 2

        # Use a minimum of 2.5% and a maximum of 5%
        return max(min(desired_stoploss, 0.05), 0.025)
```

#### Calculating stoploss relative to open price

Stoploss values returned from `custom_stoploss()` always specify a percentage relative to `current_rate`. In order to set a stoploss relative to the *open* price, we need to use `current_profit` to calculate what percentage relative to the `current_rate` will give you the same result as if the percentage was specified from the open price.

The helper function [`stoploss_from_open()`](strategy-customization.md#stoploss_from_open) can be used to convert from an open price relative stop, to a current price relative stop which can be returned from `custom_stoploss()`.

### Calculating stoploss percentage from absolute price

Stoploss values returned from `custom_stoploss()` always specify a percentage relative to `current_rate`. In order to set a stoploss at specified absolute price level, we need to use `stop_rate` to calculate what percentage relative to the `current_rate` will give you the same result as if the percentage was specified from the open price.

The helper function [`stoploss_from_absolute()`](strategy-customization.md#stoploss_from_absolute) can be used to convert from an absolute price, to a current price relative stop which can be returned from `custom_stoploss()`.

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

Absolute stoploss value may be derived from indicators stored in dataframe. Example uses parabolic SAR below the price as stoploss.

``` python
class AwesomeStrategy(IStrategy):

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # <...>
        dataframe['sar'] = ta.SAR(dataframe)

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Use parabolic sar as absolute stoploss price
        stoploss_price = last_candle['sar']

        # Convert absolute price to percentage relative to current_rate
        if stoploss_price < current_rate:
            return (stoploss_price / current_rate) - 1

        # return maximum stoploss value, keeping current stoploss price unchanged
        return 1
```

See [Dataframe access](#dataframe-access) for more information about dataframe use in strategy callbacks.

---

## Custom order price rules

By default, freqtrade use the orderbook to automatically set an order price([Relevant documentation](configuration.md#prices-used-for-orders)), you also have the option to create custom order prices based on your strategy.

You can use this feature by creating a `custom_entry_price()` function in your strategy file to customize entry prices and `custom_exit_price()` for exits.

!!! Note
    If your custom pricing function return None or an invalid value, price will fall back to `proposed_rate`, which is based on the regular pricing configuration.

### Custom order entry and exit price example

``` python
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def custom_entry_price(self, pair: str, current_time: datetime,
                           proposed_rate, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_entryprice = dataframe['bollinger_10_lowerband'].iat[-1]
        
        return new_entryprice

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_exitprice = dataframe['bollinger_10_upperband'].iat[-1]
        
        return new_exitprice

```

!!! Warning
    Modifying entry and exit prices will only work for limit orders. Depending on the price chosen, this can result in a lot of unfilled orders. By default the maximum allowed distance between the current price and the custom price is 2%, this value can be changed in config with the `custom_price_max_distance_ratio` parameter.

!!! Example
    If the new_entryprice is 97, the proposed_rate is 100 and the `custom_price_max_distance_ratio` is set to 2%, The retained valid custom entry price will be 98.

!!! Warning "No backtesting support"
    Custom entry-prices are currently not supported during backtesting.

## Custom order timeout rules

Simple, time-based order-timeouts can be configured either via strategy or in the configuration in the `unfilledtimeout` section.

However, freqtrade also offers a custom callback for both order types, which allows you to decide based on custom criteria if an order did time out or not.

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
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
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
        :param current_time: datetime object, containing the current datetime
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
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
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
        :param current_time: datetime object, containing the current datetime
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

### Stake size management

It is possible to manage your risk by reducing or increasing stake amount when placing a new trade.

```python
class AwesomeStrategy(IStrategy):
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['fastk_rsi_1h'] > current_candle['fastd_rsi_1h']:
            if self.config['stake_amount'] == 'unlimited':
                # Use entire available wallet during favorable conditions when in compounding mode.
                return max_stake
            else:
                # Compound profits during favorable conditions instead of using a static stake.
                return self.wallets.get_total_stake_amount() / self.config['max_open_trades']

        # Use default stake amount.
        return proposed_stake
```

Freqtrade will fall back to the `proposed_stake` value should your code raise an exception. The exception itself will be logged.

!!! Tip
    You do not _have_ to ensure that `min_stake <= returned_value <= max_stake`. Trades will succeed as the returned value will be clamped to supported range and this acton will be logged.

!!! Tip
    Returning `0` or `None` will prevent trades from being placed.

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

Freqtrade provides you with an easy way to embed the strategy into your configuration file.
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

## Performance warning

When executing a strategy, one can sometimes be greeted by the following in the logs

> PerformanceWarning: DataFrame is highly fragmented.

This is a warning from [`pandas`](https://github.com/pandas-dev/pandas) and as the warning continues to say:
use `pd.concat(axis=1)`.
This can have slight performance implications, which are usually only visible during hyperopt (when optimizing an indicator).

For example:

```python
for val in self.buy_ema_short.range:
    dataframe[f'ema_short_{val}'] = ta.EMA(dataframe, timeperiod=val)
```

should be rewritten to

```python
frames = [dataframe]
for val in self.buy_ema_short.range:
    frames.append({
        f'ema_short_{val}': ta.EMA(dataframe, timeperiod=val)
    })

# Append columns to existing dataframe
merged_frame = pd.concat(frames, axis=1)
```
