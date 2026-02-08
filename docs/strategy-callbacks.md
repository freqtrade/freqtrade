# Strategy Callbacks

While the main strategy functions (`populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`) should be used in a vectorized way, and are only called [once during backtesting](bot-basics.md#backtesting-hyperopt-execution-logic), callbacks are called "whenever needed".

As such, you should avoid doing heavy calculations in callbacks to avoid delays during operations.
Depending on the callback used, they may be called when entering / exiting a trade, or throughout the duration of a trade.

Currently available callbacks:

* [`bot_start()`](#bot-start)
* [`bot_loop_start()`](#bot-loop-start)
* [`custom_stake_amount()`](#stake-size-management)
* [`custom_exit()`](#custom-exit-signal)
* [`custom_stoploss()`](#custom-stoploss)
* [`custom_roi()`](#custom-roi)
* [`custom_entry_price()` and `custom_exit_price()`](#custom-order-price-rules)
* [`check_entry_timeout()` and `check_exit_timeout()`](#custom-order-timeout-rules)
* [`confirm_trade_entry()`](#trade-entry-buy-order-confirmation)
* [`confirm_trade_exit()`](#trade-exit-sell-order-confirmation)
* [`adjust_trade_position()`](#adjust-trade-position)
* [`adjust_entry_price()`](#adjust-entry-price)
* [`leverage()`](#leverage-callback)
* [`order_filled()`](#order-filled-callback)

!!! Tip "Callback calling sequence"
    You can find the callback calling sequence in [bot-basics](bot-basics.md#bot-execution-logic)

--8<-- "includes/strategy-imports.md"

--8<-- "includes/strategy-exit-comparisons.md"


## Bot start

A simple callback which is called once when the strategy is loaded.
This can be used to perform actions that must only be performed once and runs after dataprovider and wallet are set

``` python
import requests

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        if self.config["runmode"].value in ("live", "dry_run"):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods
            self.custom_remote_data = requests.get("https://some_remote_source.example.com")

```

During hyperopt, this runs only once at startup.

## Bot loop start

A simple callback which is called once at the start of every bot throttling iteration in dry/live mode (roughly every 5
seconds, unless configured differently) or once per candle in backtest/hyperopt mode.
This can be used to perform calculations which are pair independent (apply to all pairs), loading of external data, etc.

``` python
# Default imports
import requests

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        if self.config["runmode"].value in ("live", "dry_run"):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods
            self.remote_data = requests.get("https://some_remote_source.example.com")

```

## Stake size management

Called before entering a trade, makes it possible to manage your position size when placing a new trade.

```python
# Default imports

class AwesomeStrategy(IStrategy):
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle["fastk_rsi_1h"] > current_candle["fastd_rsi_1h"]:
            if self.config["stake_amount"] == "unlimited":
                # Use entire available wallet during favorable conditions when in compounding mode.
                return max_stake
            else:
                # Compound profits during favorable conditions instead of using a static stake.
                return self.wallets.get_total_stake_amount() / self.config["max_open_trades"]

        # Use default stake amount.
        return proposed_stake
```

Freqtrade will fall back to the `proposed_stake` value should your code raise an exception. The exception itself will be logged.

!!! Tip
    You do not _have_ to ensure that `min_stake <= returned_value <= max_stake`. Trades will succeed as the returned value will be clamped to supported range and this action will be logged.

!!! Tip
    Returning `0` or `None` will prevent trades from being placed.

## Custom exit signal

Called for open trade every throttling iteration (roughly every 5 seconds) until a trade is closed.

Allows to define custom exit signals, indicating that specified position should be closed (full exit). This is very useful when we need to customize exit conditions for each individual trade, or if you need trade data to make an exit decision.

For example you could implement a 1:2 risk-reward ROI with `custom_exit()`.

Using `custom_exit()` signals in place of stoploss though *is not recommended*. It is a inferior method to using `custom_stoploss()` in this regard - which also allows you to keep the stoploss on exchange.

!!! Note
    Returning a (none-empty) `string` or `True` from this method is equal to setting exit signal on a candle at specified time. This method is not called when exit signal is set already, or if exit signals are disabled (`use_exit_signal=False`). `string` max length is 64 characters. Exceeding this limit will cause the message to be truncated to 64 characters.
    `custom_exit()` will ignore `exit_profit_only`, and will always be called unless `use_exit_signal=False`, even if there is a new enter signal.

An example of how we can use different indicators depending on the current profit and also exit trades that were open longer than one day:

``` python
# Default imports

class AwesomeStrategy(IStrategy):
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Above 20% profit, sell when rsi < 80
        if current_profit > 0.2:
            if last_candle["rsi"] < 80:
                return "rsi_below_80"

        # Between 2% and 10%, sell if EMA-long above EMA-short
        if 0.02 < current_profit < 0.1:
            if last_candle["emalong"] > last_candle["emashort"]:
                return "ema_long_below_80"

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 1:
            return "unclog"
```

See [Dataframe access](strategy-advanced.md#dataframe-access) for more information about dataframe use in strategy callbacks.

## Custom stoploss

Called for open trade every iteration (roughly every 5 seconds) until a trade is closed.

The usage of the custom stoploss method must be enabled by setting `use_custom_stoploss=True` on the strategy object.

The stoploss price can only ever move upwards - if the stoploss value returned from `custom_stoploss` would result in a lower stoploss price than was previously set, it will be ignored. The traditional `stoploss` value serves as an absolute lower level and will be instated as the initial stoploss (before this method is called for the first time for a trade), and is still mandatory.  
As custom stoploss acts as regular, changing stoploss, it will behave similar to `trailing_stop` - and trades exiting due to this will have the exit_reason of `"trailing_stop_loss"`.

The method must return a stoploss value (float / number) as a percentage of the current price.
E.g. If the `current_rate` is 200 USD, then returning `0.02` will set the stoploss price 2% lower, at 196 USD.
During backtesting, `current_rate` (and `current_profit`) are provided against the candle's high (or low for short trades) - while the resulting stoploss is evaluated against the candle's low (or high for short trades).

The absolute value of the return value is used (the sign is ignored), so returning `0.05` or `-0.05` have the same result, a stoploss 5% below the current price.
Returning `None` will be interpreted as "no desire to change", and is the only safe way to return when you'd like to not modify the stoploss.
`NaN` and `inf` values are considered invalid and will be ignored (identical to `None`).

Stoploss on exchange works similar to `trailing_stop`, and the stoploss on exchange is updated as configured in `stoploss_on_exchange_interval` ([More details about stoploss on exchange](stoploss.md#stop-loss-on-exchangefreqtrade)).

If you're on futures markets, please take note of the [stoploss and leverage](stoploss.md#stoploss-and-leverage) section, as the stoploss value returned from `custom_stoploss` is the risk for this trade - not the relative price movement.

!!! Note "Use of dates"
    All time-based calculations should be done based on `current_time` - using `datetime.now()` or `datetime.utcnow()` is discouraged, as this will break backtesting support.

!!! Tip "Trailing stoploss"
    It's recommended to disable `trailing_stop` when using custom stoploss values. Both can work in tandem, but you might encounter the trailing stop to move the price higher while your custom function would not want this, causing conflicting behavior.

### Adjust stoploss after position adjustments

Depending on your strategy, you may encounter the need to adjust the stoploss in both directions after a [position adjustment](#adjust-trade-position).
For this, freqtrade will make an additional call with `after_fill=True` after an order fills, which will allow the strategy to move the stoploss in any direction (also widening the gap between stoploss and current price, which is otherwise forbidden).

!!! Note "backwards compatibility"
    This call will only be made if the `after_fill` parameter is part of the function definition of your `custom_stoploss` function.
    As such, this will not impact (and with that, surprise) existing, running strategies.

### Custom stoploss examples

The next section will show some examples on what's possible with the custom stoploss function.
Of course, many more things are possible, and all examples can be combined at will.

#### Trailing stop via custom stoploss

To simulate a regular trailing stoploss of 4% (trailing 4% behind the maximum reached price) you would use the following very simple method:

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> float | None:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value.
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param after_fill: True if the stoploss is called after the order was filled.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current_rate
        """
        return -0.04 * trade.leverage
```

#### Time based trailing stop

Use the initial stoploss for the first 60 minutes, after this change to 10% trailing stoploss, and after 2 hours (120 minutes) we use a 5% trailing stoploss.

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> float | None:

        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=120) > trade.open_date_utc:
            return -0.05 * trade.leverage
        elif current_time - timedelta(minutes=60) > trade.open_date_utc:
            return -0.10 * trade.leverage
        return None
```

#### Time based trailing stop with after-fill adjustments

Use the initial stoploss for the first 60 minutes, after this change to 10% trailing stoploss, and after 2 hours (120 minutes) we use a 5% trailing stoploss.
If an additional order fills, set stoploss to -10% below the new `open_rate` ([Averaged across all entries](#position-adjust-calculations)).

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> float | None:

        if after_fill: 
            # After an additional order, start with a stoploss of 10% below the new open rate
            return stoploss_from_open(0.10, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=120) > trade.open_date_utc:
            return -0.05 * trade.leverage
        elif current_time - timedelta(minutes=60) > trade.open_date_utc:
            return -0.10 * trade.leverage
        return None
```

#### Different stoploss per pair

Use a different stoploss depending on the pair.
In this example, we'll trail the highest price with 10% trailing stoploss for `ETH/BTC` and `XRP/BTC`, with 5% trailing stoploss for `LTC/BTC` and with 15% for all other pairs.

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        if pair in ("ETH/BTC", "XRP/BTC"):
            return -0.10 * trade.leverage
        elif pair in ("LTC/BTC"):
            return -0.05 * trade.leverage
        return -0.15 * trade.leverage
```

#### Trailing stoploss with positive offset

Use the initial stoploss until the profit is above 4%, then use a trailing stoploss of 50% of the current profit with a minimum of 2.5% and a maximum of 5%.

Please note that the stoploss can only increase, values lower than the current stoploss are ignored.

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        if current_profit < 0.04:
            return None # return None to keep using the initial stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 2

        # Use a minimum of 2.5% and a maximum of 5%
        return max(min(desired_stoploss, 0.05), 0.025) * trade.leverage
```

#### Stepped stoploss

Instead of continuously trailing behind the current price, this example sets fixed stoploss price levels based on the current profit.

* Use the regular stoploss until 20% profit is reached
* Once profit is > 20% - set stoploss to 7% above open price.
* Once profit is > 25% - set stoploss to 15% above open price.
* Once profit is > 40% - set stoploss to 25% above open price.

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        # evaluate highest to lowest, so that highest possible stop is used
        if current_profit > 0.40:
            return stoploss_from_open(0.25, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.25:
            return stoploss_from_open(0.15, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.20:
            return stoploss_from_open(0.07, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # return maximum stoploss value, keeping current stoploss price unchanged
        return None
```

#### Custom stoploss using an indicator from dataframe example

Absolute stoploss value may be derived from indicators stored in dataframe. Example uses parabolic SAR below the price as stoploss.

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # <...>
        dataframe["sar"] = ta.SAR(dataframe)

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Use parabolic sar as absolute stoploss price
        stoploss_price = last_candle["sar"]

        # Convert absolute price to percentage relative to current_rate
        if stoploss_price < current_rate:
            return stoploss_from_absolute(stoploss_price, current_rate, is_short=trade.is_short)

        # return maximum stoploss value, keeping current stoploss price unchanged
        return None
```

See [Dataframe access](strategy-advanced.md#dataframe-access) for more information about dataframe use in strategy callbacks.

### Common helpers for stoploss calculations

#### Stoploss relative to open price

Stoploss values returned from `custom_stoploss()` must specify a percentage relative to `current_rate`, but sometimes you may want to specify a stoploss relative to the _entry_ price instead.
`stoploss_from_open()` is a helper function to calculate a stoploss value that can be returned from `custom_stoploss` which will be equivalent to the desired trade profit above the entry point.

??? Example "Returning a stoploss relative to the open price from the custom stoploss function"

    Say the open price was $100, and `current_price` is $121 (`current_profit` will be `0.21`).  

    If we want a stop price at 7% above the open price we can call `stoploss_from_open(0.07, current_profit, False)` which will return `0.1157024793`.  11.57% below $121 is $107, which is the same as 7% above $100.

    This function will consider leverage - so at 10x leverage, the actual stoploss would be 0.7% above $100 (0.7% * 10x = 7%).


    ``` python
    # Default imports

    class AwesomeStrategy(IStrategy):

        # ... populate_* methods

        use_custom_stoploss = True

        def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float, after_fill: bool,
                            **kwargs) -> float | None:

            # once the profit has risen above 10%, keep the stoploss at 7% above the open price
            if current_profit > 0.10:
                return stoploss_from_open(0.07, current_profit, is_short=trade.is_short, leverage=trade.leverage)

            return 1

    ```

    Full examples can be found in the [Custom stoploss](strategy-callbacks.md#custom-stoploss) section of the Documentation.

!!! Note
    Providing invalid input to `stoploss_from_open()` may produce "CustomStoploss function did not return valid stoploss" warnings.
    This may happen if `current_profit` parameter is below specified `open_relative_stop`. Such situations may arise when closing trade
    is blocked by `confirm_trade_exit()` method. Warnings can be solved by never blocking stop loss sells by checking `exit_reason` in
    `confirm_trade_exit()`, or by using `return stoploss_from_open(...) or 1` idiom, which will request to not change stop loss when
    `current_profit < open_relative_stop`.

#### Stoploss percentage from absolute price

Stoploss values returned from `custom_stoploss()` always specify a percentage relative to `current_rate`. In order to set a stoploss at specified absolute price level, we need to use `stop_rate` to calculate what percentage relative to the `current_rate` will give you the same result as if the percentage was specified from the open price.

The helper function `stoploss_from_absolute()` can be used to convert from an absolute price, to a current price relative stop which can be returned from `custom_stoploss()`.

??? Example "Returning a stoploss using absolute price from the custom stoploss function"

    If we want to trail a stop price at 2xATR below current price we can call `stoploss_from_absolute(current_rate + (side * candle["atr"] * 2), current_rate=current_rate, is_short=trade.is_short, leverage=trade.leverage)`.
    For futures, we need to adjust the direction (up or down), as well as adjust for leverage, since the [`custom_stoploss`](strategy-callbacks.md#custom-stoploss) callback  returns the ["risk for this trade"](stoploss.md#stoploss-and-leverage) - not the relative price movement.

    ``` python
    # Default imports

    class AwesomeStrategy(IStrategy):

        use_custom_stoploss = True

        def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
            return dataframe

        def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float, after_fill: bool,
                            **kwargs) -> float | None:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            candle = dataframe.iloc[-1].squeeze()
            side = 1 if trade.is_short else -1
            return stoploss_from_absolute(current_rate + (side * candle["atr"] * 2), 
                                          current_rate=current_rate, 
                                          is_short=trade.is_short,
                                          leverage=trade.leverage)

    ```

---

## Custom ROI

Called for open trade every iteration (roughly every 5 seconds) until a trade is closed.

The usage of the custom ROI method must be enabled by setting `use_custom_roi=True` on the strategy object.

This method allows you to define a custom minimum ROI threshold for exiting a trade, expressed as a ratio (e.g., `0.05` for 5% profit). If both `minimal_roi` and `custom_roi` are defined, the lower of the two thresholds will trigger an exit. For example, if `minimal_roi` is set to `{"0": 0.10}` (10% at 0 minutes) and `custom_roi` returns `0.05`, the trade will exit when the profit reaches 5%. Also, if `custom_roi` returns `0.10` and `minimal_roi` is set to `{"0": 0.05}` (5% at 0 minutes), the trade will be closed when the profit reaches 5%.

The method must return a float representing the new ROI threshold as a ratio, or `None` to fall back to the `minimal_roi` logic. Returning `NaN` or `inf` values is considered invalid and will be treated as `None`, causing the bot to use the `minimal_roi` configuration.

### Custom ROI examples

The following examples illustrate how to use the `custom_roi` function to implement different ROI logics.

#### Custom ROI per side

Use different ROI thresholds depending on the `side`. In this example, 5% for long entries and 2% for short entries.

```python
# Default imports

class AwesomeStrategy(IStrategy):

    use_custom_roi = True

    # ... populate_* methods

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:
        """
        Custom ROI logic, returns a new minimum ROI threshold (as a ratio, e.g., 0.05 for +5%).
        Only called when use_custom_roi is set to True.

        If used at the same time as minimal_roi, an exit will be triggered when the lower
        threshold is reached. Example: If minimal_roi = {"0": 0.01} and custom_roi returns 0.05,
        an exit will be triggered if profit reaches 5%.

        :param pair: Pair that's currently analyzed.
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime.
        :param trade_duration: Current trade duration in minutes.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the current trade.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New ROI value as a ratio, or None to fall back to minimal_roi logic.
        """
        return 0.05 if side == "long" else 0.02
```

#### Custom ROI per pair

Use different ROI thresholds depending on the `pair`.

```python
# Default imports

class AwesomeStrategy(IStrategy):

    use_custom_roi = True

    # ... populate_* methods

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:

        stake = trade.stake_currency
        roi_map = {
            f"BTC/{stake}": 0.02, # 2% for BTC
            f"ETH/{stake}": 0.03, # 3% for ETH
            f"XRP/{stake}": 0.04, # 4% for XRP
        }

        return roi_map.get(pair, 0.01) # 1% for any other pair
```

#### Custom ROI per entry tag

Use different ROI thresholds depending on the `entry_tag` provided with the buy signal.

```python
# Default imports

class AwesomeStrategy(IStrategy):

    use_custom_roi = True

    # ... populate_* methods

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:

        roi_by_tag = {
            "breakout": 0.08,       # 8% if tag is "breakout"
            "rsi_overbought": 0.05, # 5% if tag is "rsi_overbought"
            "mean_reversion": 0.03, # 3% if tag is "mean_reversion"
        }

        return roi_by_tag.get(entry_tag, 0.01)  # 1% if tag is unknown
```

#### Custom ROI based on ATR

ROI value may be derived from indicators stored in dataframe. This example uses the ATR ratio as ROI.

``` python
# Default imports
# <...>
import talib.abstract as ta

class AwesomeStrategy(IStrategy):

    use_custom_roi = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # <...>
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=10)

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        atr_ratio = last_candle["atr"] / last_candle["close"]

        return atr_ratio # Returns the ATR value as ratio
```

---

## Custom order price rules

By default, freqtrade use the orderbook to automatically set an order price ([Relevant documentation](configuration.md#prices-used-for-orders)), you also have the option to create custom order prices based on your strategy.

You can use this feature by creating a `custom_entry_price()` function in your strategy file to customize entry prices and `custom_exit_price()` for exits.

Each of these methods are called right before placing an order on the exchange.

!!! Note
    If your custom pricing function return None or an invalid value, price will fall back to `proposed_rate`, which is based on the regular pricing configuration.

!!! Note
    When using `custom_entry_price()`, the Trade object will be available as soon as the first entry order associated with the trade is created, for the first entry, `trade` parameter value will be `None`.

### Custom order entry and exit price example

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def custom_entry_price(self, pair: str, trade: Trade | None, current_time: datetime, proposed_rate: float,
                           entry_tag: str | None, side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_entryprice = dataframe["bollinger_10_lowerband"].iat[-1]

        return new_entryprice

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, exit_tag: str | None, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_exitprice = dataframe["bollinger_10_upperband"].iat[-1]

        return new_exitprice

```

!!! Warning
    Modifying entry and exit prices will only work for limit orders. Depending on the price chosen, this can result in a lot of unfilled orders. By default the maximum allowed distance between the current price and the custom price is 2%, this value can be changed in config with the `custom_price_max_distance_ratio` parameter.
    **Example**:
    If the new_entryprice is 97, the proposed_rate is 100 and the `custom_price_max_distance_ratio` is set to 2%, The retained valid custom entry price will be 98, which is 2% below the current (proposed) rate.

!!! Warning "Backtesting"
    Custom prices are supported in backtesting (starting with 2021.12), and orders will fill if the price falls within the candle's low/high range.
    Orders that don't fill immediately are subject to regular timeout handling, which happens once per (detail) candle.
    `custom_exit_price()` is only called for sells of type exit_signal, Custom exit and partial exits. All other exit-types will use regular backtesting prices.

## Custom order timeout rules

Simple, time-based order-timeouts can be configured either via strategy or in the configuration in the `unfilledtimeout` section.

However, freqtrade also offers a custom callback for both order types, which allows you to decide based on custom criteria if an order did time out or not.

!!! Note
    Backtesting fills orders if their price falls within the candle's low/high range.
    The below callbacks will be called once per (detail) candle for orders that don't fill immediately (which use custom pricing).

### Custom order timeout example

Called for every open order until that order is either filled or cancelled.
`check_entry_timeout()` is called for trade entries, while `check_exit_timeout()` is called for trade exit orders.

A simple example, which applies different unfilled-timeouts depending on the price of the asset can be seen below.
It applies a tight timeout for higher priced assets, while allowing more time to fill on cheap coins.

The function must return either `True` (cancel order) or `False` (keep order alive).

``` python
    # Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since the maximum timeout from below is 24 hours.
    unfilledtimeout = {
        "entry": 60 * 25,
        "exit": 60 * 25
    }

    def check_entry_timeout(self, pair: str, trade: Trade, order: Order,
                            current_time: datetime, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date_utc < current_time - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date_utc < current_time - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date_utc < current_time - timedelta(hours=24):
           return True
        return False


    def check_exit_timeout(self, pair: str, trade: Trade, order: Order,
                           current_time: datetime, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date_utc < current_time - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date_utc < current_time - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date_utc < current_time - timedelta(hours=24):
           return True
        return False
```

!!! Note
    For the above example, `unfilledtimeout` must be set to something bigger than 24h, otherwise that type of timeout will apply first.

### Custom order timeout example (using additional data)

``` python
    # Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since the maximum timeout from below is 24 hours.
    unfilledtimeout = {
        "entry": 60 * 25,
        "exit": 60 * 25
    }

    def check_entry_timeout(self, pair: str, trade: Trade, order: Order,
                            current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob["bids"][0][0]
        # Cancel buy order if price is more than 2% above the order.
        if current_price > order.price * 1.02:
            return True
        return False


    def check_exit_timeout(self, pair: str, trade: Trade, order: Order,
                           current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob["asks"][0][0]
        # Cancel sell order if price is more than 2% below the order.
        if current_price < order.price * 0.98:
            return True
        return False
```

---

## Bot order confirmation

Confirm trade entry / exits.
This are the last methods that will be called before an order is placed.

### Trade entry (buy order) confirmation

`confirm_trade_entry()` can be used to abort a trade entry at the latest second (maybe because the price is not what we expect).

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders 
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        return True

```

### Trade exit (sell order) confirmation

`confirm_trade_exit()` can be used to abort a trade exit (sell) at the latest second (maybe because the price is not what we expect).

`confirm_trade_exit()` may be called multiple times within one iteration for the same trade if different exit-reasons apply.
The exit-reasons (if applicable) will be in the following sequence:

* `exit_signal` / `custom_exit`
* `stop_loss`
* `roi`
* `trailing_stop_loss`

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ["roi", "stop_loss", "stoploss_on_exchange", "trailing_stop_loss",
                           "exit_signal", "force_exit", "emergency_exit"]
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        if exit_reason == "force_exit" and trade.calc_profit_ratio(rate) < 0:
            # Reject force-sells with negative profit
            # This is just a sample, please adjust to your needs
            # (this does not necessarily make sense, assuming you know when you're force-selling)
            return False
        return True

```

!!! Warning
    `confirm_trade_exit()` can prevent stoploss exits, causing significant losses as this would ignore stoploss exits.
    `confirm_trade_exit()` will not be called for Liquidations - as liquidations are forced by the exchange, and therefore cannot be rejected.

## Adjust trade position

The `position_adjustment_enable` strategy property enables the usage of `adjust_trade_position()` callback in the strategy.
For performance reasons, it's disabled by default and freqtrade will show a warning message on startup if enabled.
`adjust_trade_position()` can be used to perform additional orders, for example to manage risk with DCA (Dollar Cost Averaging) or to increase or decrease positions.

Additional orders also result in additional fees and those orders don't count towards `max_open_trades`.

This callback is also called when there is an open order (either buy or sell) waiting for execution - and will cancel the existing open order to place a new order if the amount, price or direction is different. Also partially filled orders will be canceled, and will be replaced with the new amount as returned by the callback.

`adjust_trade_position()` is called very frequently for the duration of a trade, so you must keep your implementation as performant as possible.

Position adjustments will always be applied in the direction of the trade, so a positive value will always increase your position (negative values will decrease your position), no matter if it's a long or short trade.
Adjustment orders can be assigned with a tag by returning a 2 element Tuple, with the first element being the adjustment amount, and the 2nd element the tag (e.g. `return 250, "increase_favorable_conditions"`).

Modifications to leverage are not possible, and the stake-amount returned is assumed to be before applying leverage.

The combined stake currently allocated to the position is held in `trade.stake_amount`. Therefore `trade.stake_amount` will always be updated on every additional entry and partial exit made through `adjust_trade_position()`.

!!! Danger "Loose Logic"
    On dry and live run, this function will be called every `throttle_process_secs` (default to 5s). If you have a loose logic, (e.g. increase position if RSI of the last candle is below 30), your bot will do extra re-entry every 5 secs until you either it run out of money, hit the `max_position_adjustment` limit, or a new candle with RSI more than 30 arrived.

    Same thing also can happen with partial exit.  
    So be sure to have a strict logic and/or check for the last filled order and if an order is already open.

!!! Warning "Performance with many position adjustments"
    Position adjustments can be a good approach to increase a strategy's output - but it can also have drawbacks if using this feature extensively.  
    Each of the orders will be attached to the trade object for the duration of the trade - hence increasing memory usage.
    Trades with long duration and 10s or even 100ds of position adjustments are therefore not recommended, and should be closed at regular intervals to not affect performance.

!!! Warning "Backtesting"
    During backtesting this callback is called for each candle in `timeframe` or `timeframe_detail`, so run-time performance will be affected.
    This can also cause deviating results between live and backtesting, since backtesting can adjust the trade only once per candle, whereas live could adjust the trade multiple times per candle.

### Increase position

The strategy is expected to return a positive **stake_amount** (in stake currency) between `min_stake` and `max_stake` if and when an additional entry order should be made (position is increased -> buy order for long trades, sell order for short trades).

If there are not enough funds in the wallet (the return value is above `max_stake`) then the signal will be ignored.
`max_entry_position_adjustment` property is used to limit the number of additional entries per trade (on top of the first entry order) that the bot can execute. By default, the value is -1 which means the bot have no limit on number of adjustment entries.

Additional entries are ignored once you have reached the maximum amount of extra entries that you have set on `max_entry_position_adjustment`, but the callback is called anyway looking for partial exits.

!!! Note "About stake size"
    Using fixed stake size means it will be the amount used for the first order, just like without position adjustment.
    If you wish to buy additional orders with DCA, then make sure to leave enough funds in the wallet for that.
    Using `"unlimited"` stake amount with DCA orders requires you to also implement the `custom_stake_amount()` callback to avoid allocating all funds to the initial order.

### Decrease position

The strategy is expected to return a negative stake_amount (in stake currency) for a partial exit.
Returning the full owned stake at that point (`-trade.stake_amount`) results in a full exit.  
Returning a value more than the above (so remaining stake_amount would become negative) will result in the bot ignoring the signal.

For a partial exit, it's important to know that the formula used to calculate the amount of the coin for the partial exit order is `amount to be exited partially = negative_stake_amount * trade.amount / trade.stake_amount`, where `negative_stake_amount` is the value returned from the `adjust_trade_position` function. As seen in the formula, the formula doesn't care about current profit/loss of the position. It only cares about `trade.amount` and `trade.stake_amount` which aren't affected by the price movement at all.

For example, let's say you buy 2 SHITCOIN/USDT at open rate of 50, which means the trade's stake amount is 100 USDT. Now the price raises to 200 and you want to sell half of it. In that case, you have to return -50% of `trade.stake_amount` (0.5 * 100 USDT) which equals to -50. The bot will calculate the amount it needed to sell, which is `50 * 2 / 100` which equals 1 SHITCOIN/USDT. If you return -200 (50% of 2 * 200), the bot will ignore it since `trade.stake_amount` is only 100 USDT but you asked to sell 200 USDT which means you are asking to sell 4 SHITCOIN/USDT.

Back to the example above, since current rate is 200, the current USDT value of your trade is now 400 USDT. Let's say you want to partially sell 100 USDT to take out the initial investment and leave the profit in the trade hoping that the price keeps rising. In that case, you have to do a different approach. First, you need to calculate the exact amount you needed to sell. In this case, since you want to sell 100 USDT worth based of current rate, the exact amount you need to partially sell is `100 * 2 / 400` which equals 0.5 SHITCOIN/USDT. Since we know now the exact amount we want to sell (0.5), the value you need to return in the `adjust_trade_position` function is `-amount to be exited partially * trade.stake_amount / trade.amount`, which equals -25. The bot will sell 0.5 SHITCOIN/USDT, keeping 1.5 in trade. You will receive 100 USDT from the partial exit.

!!! Warning "Stoploss calculation"
    Stoploss is still calculated from the initial opening price, not averaged price.
    Regular stoploss rules still apply (cannot move down).

    While `/stopentry` command stops the bot from entering new trades, the position adjustment feature will continue buying new orders on existing trades.

``` python
# Default imports

class DigDeeperStrategy(IStrategy):

    position_adjustment_enable = True

    # Attempts to handle large drops with DCA. High stoploss is required.
    stoploss = -0.30

    # ... populate_* methods

    # Example specific variables
    max_entry_position_adjustment = 3
    # This number is explained a bit further down
    max_dca_multiplier = 5.5

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> float | None | tuple[float | None, str | None]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra entry or exit orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current entry rate (same as current_entry_profit)
        :param current_profit: Current profit (as ratio), calculated based on current_rate 
                               (same as current_entry_profit).
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
                       Optionally, return a tuple with a 2nd element with an order reason
        """
        if trade.has_open_orders:
            # Only act if no orders are open
            return

        if current_profit > 0.05 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 2), "half_profit_5%"

        if current_profit > -0.05:
            return None

        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when not actively falling price.
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle["close"] < previous_candle["close"]:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # That is why max_dca_multiplier is 5.5
        # Hope you have a deep wallet!
        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].stake_amount_filled
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
            return stake_amount, "1/3rd_increase"
        except Exception as exception:
            return None

        return None

```

### Position adjust calculations

* Entry rates are calculated using weighted averages.
* Exits will not influence the average entry rate.
* Partial exit relative profit is relative to the average entry price at this point.
* Final exit relative profit is calculated based on the total invested capital. (See example below)

??? example "Calculation example"
    *This example assumes 0 fees for simplicity, and a long position on an imaginary coin.*  
    
    * Buy 100@8\$ 
    * Buy 100@9\$ -> Avg price: 8.5\$
    * Sell 100@10\$ -> Avg price: 8.5\$, realized profit 150\$, 17.65%
    * Buy 150@11\$ -> Avg price: 10\$, realized profit 150\$, 17.65%
    * Sell 100@12\$ -> Avg price: 10\$, total realized profit 350\$, 20%
    * Sell 150@14\$ -> Avg price: 10\$, total realized profit 950\$, 40%  <- *This will be the last "Exit" message*

    The total profit for this trade was 950$ on a 3350$ investment (`100@8$ + 100@9$ + 150@11$`). As such - the final relative profit is 28.35% (`950 / 3350`).

## Adjust order Price

The `adjust_order_price()` callback may be used by strategy developer to refresh/replace limit orders upon arrival of new candles.  
This callback is called once every iteration unless the order has been (re)placed within the current candle - limiting the maximum (re)placement of each order to once per candle.
This also means that the first call will be at the start of the next candle after the initial order was placed.

Be aware that `custom_entry_price()`/`custom_exit_price()` is still the one dictating initial limit order price target at the time of the signal.

Orders can be cancelled out of this callback by returning `None`.

Returning `current_order_rate` will keep the order on the exchange "as is".
Returning any other price will cancel the existing order, and replace it with a new order.

If the cancellation of the original order fails, then the order will not be replaced - though the order will most likely have been canceled on exchange. Having this happen on initial entries will result in the deletion of the order, while on position adjustment orders, it'll result in the trade size remaining as is.  
If the order has been partially filled, the order will not be replaced. You can however use [`adjust_trade_position()`](#adjust-trade-position) to adjust the trade size to the expected position size, should this be necessary / desired.

!!! Warning "Regular timeout"
    Entry `unfilledtimeout` mechanism (as well as `check_entry_timeout()`/`check_exit_timeout()`) takes precedence over this callback.
    Orders that are cancelled via the above methods will not have this callback called. Be sure to update timeout values to match your expectations.

```python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def adjust_order_price(
        self,
        trade: Trade,
        order: Order | None,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: str | None,
        side: str,
        is_entry: bool,
        **kwargs,
    ) -> float | None:
        """
        Exit and entry order price re-adjustment logic, returning the user desired limit price.
        This only executes when a order was already placed, still open (unfilled fully or partially)
        and not timed out on subsequent candles after entry trigger.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-callbacks/

        When not implemented by a strategy, returns current_order_rate as default.
        If current_order_rate is returned then the existing order is maintained.
        If None is returned then order gets canceled but not replaced by a new one.

        :param pair: Pair that's currently analyzed
        :param trade: Trade object.
        :param order: Order object
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate, calculated based on pricing settings in entry_pricing.
        :param current_order_rate: Rate of the existing order in place.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param is_entry: True if the order is an entry order, False if it's an exit order.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float or None: New entry price value if provided
        """

        # Limit entry orders to use and follow SMA200 as price target for the first 10 minutes since entry trigger for BTC/USDT pair.
        if (
            is_entry
            and pair == "BTC/USDT" 
            and entry_tag == "long_sma200" 
            and side == "long" 
            and (current_time - timedelta(minutes=10)) <= trade.open_date_utc
        ):
            # just cancel the order if it has been filled more than half of the amount
            if order.filled > order.remaining:
                return None
            else:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
                current_candle = dataframe.iloc[-1].squeeze()
                # desired price
                return current_candle["sma_200"]
        # default: maintain existing order
        return current_order_rate
```

!!! danger "Incompatibility with `adjust_*_price()`"
    If you have both `adjust_order_price()` and `adjust_entry_price()`/`adjust_exit_price()` implemented, only `adjust_order_price()` will be used.
    If you need to adjust entry/exit prices, you can either implement the logic in `adjust_order_price()`, or use the split `adjust_entry_price()` / `adjust_exit_price()` callbacks, but not both.
    Mixing these is not supported and will raise an error during bot startup.

### Adjust Entry Price

The `adjust_entry_price()` callback may be used by strategy developer to refresh/replace entry limit orders upon arrival.
It's a sub-set of `adjust_order_price()` and is called only for entry orders.
All remaining behavior is identical to `adjust_order_price()`.

The trade open-date (`trade.open_date_utc`) will remain at the time of the very first order placed.
Please make sure to be aware of this - and eventually adjust your logic in other callbacks to account for this, and use the date of the first filled order instead.

### Adjust Exit Price

The `adjust_exit_price()` callback may be used by strategy developer to refresh/replace exit limit orders upon arrival.
It's a sub-set of `adjust_order_price()` and is called only for exit orders.
All remaining behavior is identical to `adjust_order_price()`.

## Leverage Callback

When trading in markets that allow leverage, this method must return the desired Leverage (Defaults to 1 -> No leverage).

Assuming a capital of 500USDT, a trade with leverage=3 would result in a position with 500 x 3 = 1500 USDT.

Values that are above `max_leverage` will be adjusted to `max_leverage`.
For markets / exchanges that don't support leverage, this method is ignored.

``` python
# Default imports

class AwesomeStrategy(IStrategy):
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 1.0
```

All profit calculations include leverage. Stoploss / ROI also include leverage in their calculation.
Defining a stoploss of 10% at 10x leverage would trigger the stoploss with a 1% move to the downside.

## Order filled Callback

The `order_filled()` callback may be used to perform specific actions based on the current trade state after an order is filled.
It will be called independent of the order type (entry, exit, stoploss or position adjustment).

Assuming that your strategy needs to store the high value of the candle at trade entry, this is possible with this callback as the following example show.

``` python
# Default imports

class AwesomeStrategy(IStrategy):
    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
        Called right after an order fills. 
        Will be called for all order types (entry, exit, stoploss, position adjustment).
        :param pair: Pair for trade
        :param trade: trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if (trade.nr_of_successful_entries == 1) and (order.ft_order_side == trade.entry_side):
            trade.set_custom_data(key="entry_candle_high", value=last_candle["high"])

        return None

```

!!! Tip "Learn more about storing data"
    You can learn more about storing data on the [Storing custom trade data](strategy-advanced.md#storing-information-persistent) section.
    Please keep in mind that this is considered advanced usage, and should be used with care.

## Plot annotations callback

The plot annotations callback is called whenever freqUI requests data to display a chart.
This callback has no meaning in the trade cycle context and is only used for charting purposes.

The strategy can then return a list of `AnnotationType` objects to be displayed on the chart.
Depending on the content returned - the chart can display horizontal areas, vertical areas, boxes or lines.

### Annotation types

Currently two types of annotations are supported, `area` and `line`.

#### Area

``` json
{
    "type": "area", // Type of the annotation, currently only "area" is supported
    "start": "2024-01-01 15:00:00", // Start date of the area
    "end": "2024-01-01 16:00:00",  // End date of the area
    "y_start": 94000.2,  // Price / y axis value
    "y_end": 98000, // Price / y axis value
    "color": "",
    "z_level": 5, // z-level, higher values are drawn on top of lower values. Positions relative to the Chart elements need to be set in freqUI.
    "label": "some label"
}
```

#### Line

``` json
{
    "type": "line", // Type of the annotation, currently only "line" is supported
    "start": "2024-01-01 15:00:00", // Start date of the line
    "end": "2024-01-01 16:00:00",  // End date of the line
    "y_start": 94000.2,  // Price / y axis value
    "y_end": 98000, // Price / y axis value
    "color": "",
    "z_level": 5, // z-level, higher values are drawn on top of lower values. Positions relative to the Chart elements need to be set in freqUI.
    "label": "some label",
    "width": 2, // Optional, line width in pixels. Defaults to 1
    "line_style": "dashed", // Optional, can be "solid", "dashed" or "dotted". Defaults to "solid"

}
```

#### Point

``` json
{
    "type": "point", // Type of the annotation, currently only "point" is supported
    "x": "2024-01-01 15:00:00", // Start date of the point
    "y": 94000.2,  // Price / y axis value
    "color": "",
    "z_level": 5, // z-level, higher values are drawn on top of lower values. Positions relative to the Chart elements need to be set in freqUI.
    "label": "some label",
    "size": 2, // Optional, line width in pixels. Defaults to 10
    "shape": "circle", // Optional, can be "circle", "rect", "roundRect", "triangle", "pin", "arrow", "none".
    "rotate": 0, // Optional, rotation of the shape/symbol in degrees. Defaults to 0

}
```

The below example will mark the chart with areas for the hours 8 and 15, with a grey color, highlighting the market open and close hours.
This is obviously a very basic example.

``` python
# Default imports

class AwesomeStrategy(IStrategy):
    def plot_annotations(
        self, pair: str, start_date: datetime, end_date: datetime, dataframe: DataFrame, **kwargs
    ) -> list[AnnotationType]:
        """
        Retrieve area annotations for a chart.
        Must be returned as array, with type, label, color, start, end, y_start, y_end.
        All settings except for type are optional - though it usually makes sense to include either
        "start and end" or "y_start and y_end" for either horizontal or vertical plots
        (or all 4 for boxes).
        :param pair: Pair that's currently analyzed
        :param start_date: Start date of the chart data being requested
        :param end_date: End date of the chart data being requested
        :param dataframe: DataFrame with the analyzed data for the chart
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: List of AnnotationType objects
        """
        annotations = []
        while start_dt < end_date:
            start_dt += timedelta(hours=1)
            if start_dt.hour in (8, 15):
                annotations.append(
                    {
                        "type": "area",
                        "label": "Trade open and close hours",
                        "start": start_dt,
                        "end": start_dt + timedelta(hours=1),
                        # Omitting y_start and y_end will result in a vertical area spanning the whole height of the main Chart
                        "color": "rgba(133, 133, 133, 0.4)",
                    }
                )

        return annotations

```

Entries will be validated, and won't be passed to the UI if they don't correspond to the expected schema and will log an error if they don't.

!!! Warning "Many annotations"
    Using too many annotations can cause the UI to hang, especially when plotting large amounts of historic data.
    Use the annotation feature with care.

### Plot annotations example

![FreqUI - plot Annotations](assets/freqUI-chart-annotations-dark.png#only-dark)
![FreqUI - plot Annotations](assets/freqUI-chart-annotations-light.png#only-light)

??? Info "Code used for the plot above"
    This is an example code and should be treated as such.

    ``` python
    # Default imports

    class AwesomeStrategy(IStrategy):
        def plot_annotations(
            self, pair: str, start_date: datetime, end_date: datetime, dataframe: DataFrame, **kwargs
        ) -> list[AnnotationType]:
            annotations = []
            while start_dt < end_date:
                start_dt += timedelta(hours=1)
                if (start_dt.hour % 4) == 0:
                    annotations.append(
                        {
                            "type": "area",
                            "label": "4h",
                            "start": start_dt,
                            "end": start_dt + timedelta(hours=1),
                            "color": "rgba(133, 133, 133, 0.4)",
                        }
                    )
                elif (start_dt.hour % 2) == 0:
                price = dataframe.loc[dataframe["date"] == start_dt, "close"].mean()
                    annotations.append(
                        {
                            "type": "area",
                            "label": "2h",
                            "start": start_dt,
                            "end": start_dt + timedelta(hours=1),
                            "y_end": price * 1.01,
                            "y_start": price * 0.99,
                            "color": "rgba(0, 255, 0, 0.4)",
                            "z_level": 5,
                        }
                    )

            return annotations

    ```
