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
* [`custom_entry_price()` and `custom_exit_price()`](#custom-order-price-rules)
* [`check_entry_timeout()` and `check_exit_timeout()`](#custom-order-timeout-rules)
* [`confirm_trade_entry()`](#trade-entry-buy-order-confirmation)
* [`confirm_trade_exit()`](#trade-exit-sell-order-confirmation)
* [`adjust_trade_position()`](#adjust-trade-position)
* [`adjust_entry_price()`](#adjust-entry-price)
* [`leverage()`](#leverage-callback)

!!! Tip "Callback calling sequence"
    You can find the callback calling sequence in [bot-basics](bot-basics.md#bot-execution-logic)

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
        if self.config['runmode'].value in ('live', 'dry_run'):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods
            self.custom_remote_data = requests.get('https://some_remote_source.example.com')

```

During hyperopt, this runs only once at startup.

## Bot loop start

A simple callback which is called once at the start of every bot throttling iteration in dry/live mode (roughly every 5
seconds, unless configured differently) or once per candle in backtest/hyperopt mode.
This can be used to perform calculations which are pair independent (apply to all pairs), loading of external data, etc.

``` python
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
        if self.config['runmode'].value in ('live', 'dry_run'):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods
            self.remote_data = requests.get('https://some_remote_source.example.com')

```

## Stake size management

Called before entering a trade, makes it possible to manage your position size when placing a new trade.

```python
class AwesomeStrategy(IStrategy):
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
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
    You do not _have_ to ensure that `min_stake <= returned_value <= max_stake`. Trades will succeed as the returned value will be clamped to supported range and this action will be logged.

!!! Tip
    Returning `0` or `None` will prevent trades from being placed.

## Custom exit signal

Called for open trade every throttling iteration (roughly every 5 seconds) until a trade is closed.

Allows to define custom exit signals, indicating that specified position should be sold. This is very useful when we need to customize exit conditions for each individual trade, or if you need trade data to make an exit decision.

For example you could implement a 1:2 risk-reward ROI with `custom_exit()`.

Using `custom_exit()` signals in place of stoploss though *is not recommended*. It is a inferior method to using `custom_stoploss()` in this regard - which also allows you to keep the stoploss on exchange.

!!! Note
    Returning a (none-empty) `string` or `True` from this method is equal to setting exit signal on a candle at specified time. This method is not called when exit signal is set already, or if exit signals are disabled (`use_exit_signal=False`). `string` max length is 64 characters. Exceeding this limit will cause the message to be truncated to 64 characters.
    `custom_exit()` will ignore `exit_profit_only`, and will always be called unless `use_exit_signal=False`, even if there is a new enter signal.

An example of how we can use different indicators depending on the current profit and also exit trades that were open longer than one day:

``` python
class AwesomeStrategy(IStrategy):
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
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

See [Dataframe access](strategy-advanced.md#dataframe-access) for more information about dataframe use in strategy callbacks.

## Custom stoploss

Called for open trade every iteration (roughly every 5 seconds) until a trade is closed.

The usage of the custom stoploss method must be enabled by setting `use_custom_stoploss=True` on the strategy object.

The stoploss price can only ever move upwards - if the stoploss value returned from `custom_stoploss` would result in a lower stoploss price than was previously set, it will be ignored. The traditional `stoploss` value serves as an absolute lower level and will be instated as the initial stoploss (before this method is called for the first time for a trade), and is still mandatory.

The method must return a stoploss value (float / number) as a percentage of the current price.
E.g. If the `current_rate` is 200 USD, then returning `0.02` will set the stoploss price 2% lower, at 196 USD.
During backtesting, `current_rate` (and `current_profit`) are provided against the candle's high (or low for short trades) - while the resulting stoploss is evaluated against the candle's low (or high for short trades).

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
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
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
            return stoploss_from_open(0.25, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.25:
            return stoploss_from_open(0.15, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.20:
            return stoploss_from_open(0.07, current_profit, is_short=trade.is_short, leverage=trade.leverage)

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
            return stoploss_from_absolute(stoploss_price, current_rate, is_short=trade.is_short)

        # return maximum stoploss value, keeping current stoploss price unchanged
        return 1
```

See [Dataframe access](strategy-advanced.md#dataframe-access) for more information about dataframe use in strategy callbacks.

### Common helpers for stoploss calculations

#### Stoploss relative to open price

Stoploss values returned from `custom_stoploss()` always specify a percentage relative to `current_rate`. In order to set a stoploss relative to the *open* price, we need to use `current_profit` to calculate what percentage relative to the `current_rate` will give you the same result as if the percentage was specified from the open price.

The helper function [`stoploss_from_open()`](strategy-customization.md#stoploss_from_open) can be used to convert from an open price relative stop, to a current price relative stop which can be returned from `custom_stoploss()`.

#### Stoploss percentage from absolute price

Stoploss values returned from `custom_stoploss()` always specify a percentage relative to `current_rate`. In order to set a stoploss at specified absolute price level, we need to use `stop_rate` to calculate what percentage relative to the `current_rate` will give you the same result as if the percentage was specified from the open price.

The helper function [`stoploss_from_absolute()`](strategy-customization.md#stoploss_from_absolute) can be used to convert from an absolute price, to a current price relative stop which can be returned from `custom_stoploss()`.

---

## Custom order price rules

By default, freqtrade use the orderbook to automatically set an order price([Relevant documentation](configuration.md#prices-used-for-orders)), you also have the option to create custom order prices based on your strategy.

You can use this feature by creating a `custom_entry_price()` function in your strategy file to customize entry prices and `custom_exit_price()` for exits.

Each of these methods are called right before placing an order on the exchange.

!!! Note
    If your custom pricing function return None or an invalid value, price will fall back to `proposed_rate`, which is based on the regular pricing configuration.

### Custom order entry and exit price example

``` python
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_entryprice = dataframe['bollinger_10_lowerband'].iat[-1]

        return new_entryprice

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, exit_tag: Optional[str], **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_exitprice = dataframe['bollinger_10_upperband'].iat[-1]

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
from datetime import datetime, timedelta
from freqtrade.persistence import Trade, Order

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since the maximum timeout from below is 24 hours.
    unfilledtimeout = {
        'entry': 60 * 25,
        'exit': 60 * 25
    }

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                            current_time: datetime, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date_utc < current_time - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date_utc < current_time - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date_utc < current_time - timedelta(hours=24):
           return True
        return False


    def check_exit_timeout(self, pair: str, trade: Trade, order: 'Order',
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
from datetime import datetime
from freqtrade.persistence import Trade, Order

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since the maximum timeout from below is 24 hours.
    unfilledtimeout = {
        'entry': 60 * 25,
        'exit': 60 * 25
    }

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                            current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['bids'][0][0]
        # Cancel buy order if price is more than 2% above the order.
        if current_price > order.price * 1.02:
            return True
        return False


    def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                           current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
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
class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
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
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
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
from freqtrade.persistence import Trade


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
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        if exit_reason == 'force_exit' and trade.calc_profit_ratio(rate) < 0:
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

`max_entry_position_adjustment` property is used to limit the number of additional buys per trade (on top of the first buy) that the bot can execute. By default, the value is -1 which means the bot have no limit on number of adjustment buys.

The strategy is expected to return a stake_amount (in stake currency) between `min_stake` and `max_stake` if and when an additional buy order should be made (position is increased).
If there are not enough funds in the wallet (the return value is above `max_stake`) then the signal will be ignored.
Additional orders also result in additional fees and those orders don't count towards `max_open_trades`.

This callback is **not** called when there is an open order (either buy or sell) waiting for execution.

`adjust_trade_position()` is called very frequently for the duration of a trade, so you must keep your implementation as performant as possible.

Additional Buys are ignored once you have reached the maximum amount of extra buys that you have set on `max_entry_position_adjustment`, but the callback is called anyway looking for partial exits.

Position adjustments will always be applied in the direction of the trade, so a positive value will always increase your position (negative values will decrease your position), no matter if it's a long or short trade. Modifications to leverage are not possible, and the stake-amount is assumed to be before applying leverage.

!!! Note "About stake size"
    Using fixed stake size means it will be the amount used for the first order, just like without position adjustment.
    If you wish to buy additional orders with DCA, then make sure to leave enough funds in the wallet for that.
    Using 'unlimited' stake amount with DCA orders requires you to also implement the `custom_stake_amount()` callback to avoid allocating all funds to the initial order.

!!! Warning
    Stoploss is still calculated from the initial opening price, not averaged price.
    Regular stoploss rules still apply (cannot move down).

    While `/stopentry` command stops the bot from entering new trades, the position adjustment feature will continue buying new orders on existing trades.

!!! Warning "Backtesting"
    During backtesting this callback is called for each candle in `timeframe` or `timeframe_detail`, so run-time performance will be affected.
    This can also cause deviating results between live and backtesting, since backtesting can adjust the trade only once per candle, whereas live could adjust the trade multiple times per candle.

``` python
from freqtrade.persistence import Trade


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
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra buy or sell orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
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
        """

        if current_profit > 0.05 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 2)

        if current_profit > -0.05:
            return None

        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when not actively falling price.
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
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
            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
            return stake_amount
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

## Adjust Entry Price

The `adjust_entry_price()` callback may be used by strategy developer to refresh/replace limit orders upon arrival of new candles.
Be aware that `custom_entry_price()` is still the one dictating initial entry limit order price target at the time of entry trigger.

Orders can be cancelled out of this callback by returning `None`.

Returning `current_order_rate` will keep the order on the exchange "as is".
Returning any other price will cancel the existing order, and replace it with a new order.

The trade open-date (`trade.open_date_utc`) will remain at the time of the very first order placed.
Please make sure to be aware of this - and eventually adjust your logic in other callbacks to account for this, and use the date of the first filled order instead.

!!! Warning "Regular timeout"
    Entry `unfilledtimeout` mechanism (as well as `check_entry_timeout()`) takes precedence over this.
    Entry Orders that are cancelled via the above methods will not have this callback called. Be sure to update timeout values to match your expectations.

```python
from freqtrade.persistence import Trade
from datetime import timedelta

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def adjust_entry_price(self, trade: Trade, order: Optional[Order], pair: str,
                           current_time: datetime, proposed_rate: float, current_order_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Entry price re-adjustment logic, returning the user desired limit price.
        This only executes when a order was already placed, still open (unfilled fully or partially)
        and not timed out on subsequent candles after entry trigger.

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
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New entry price value if provided

        """
        # Limit orders to use and follow SMA200 as price target for the first 10 minutes since entry trigger for BTC/USDT pair.
        if pair == 'BTC/USDT' and entry_tag == 'long_sma200' and side == 'long' and (current_time - timedelta(minutes=10)) > trade.open_date_utc:
            # just cancel the order if it has been filled more than half of the amount
            if order.filled > order.remaining:
                return None
            else:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
                current_candle = dataframe.iloc[-1].squeeze()
                # desired price
                return current_candle['sma_200']
        # default: maintain existing order
        return current_order_rate
```

## Leverage Callback

When trading in markets that allow leverage, this method must return the desired Leverage (Defaults to 1 -> No leverage).

Assuming a capital of 500USDT, a trade with leverage=3 would result in a position with 500 x 3 = 1500 USDT.

Values that are above `max_leverage` will be adjusted to `max_leverage`.
For markets / exchanges that don't support leverage, this method is ignored.

``` python
class AwesomeStrategy(IStrategy):
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 1.0
```

All profit calculations include leverage. Stoploss / ROI also include leverage in their calculation.
Defining a stoploss of 10% at 10x leverage would trigger the stoploss with a 1% move to the downside.
