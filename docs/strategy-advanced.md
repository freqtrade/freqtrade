# Advanced Strategies

This page explains some advanced concepts available for strategies.
If you're just getting started, please familiarize yourself with the [Freqtrade basics](bot-basics.md) and methods described in [Strategy Customization](strategy-customization.md) first.

The call sequence of the methods described here is covered under [bot execution logic](bot-basics.md#bot-execution-logic). Those docs are also helpful in deciding which method is most suitable for your customisation needs.

!!! Note
    Callback methods should *only* be implemented if a strategy uses them.

!!! Tip
    Start off with a strategy template containing all available callback methods by running `freqtrade new-strategy --strategy MyAwesomeStrategy --template advanced`

## Storing information (Persistent)

Freqtrade allows storing/retrieving user custom information associated with a specific trade in the database.

Using a trade object, information can be stored using `trade.set_custom_data(key='my_key', value=my_value)` and retrieved using `trade.get_custom_data(key='my_key')`. Each data entry is associated with a trade and a user supplied key (of type `string`). This means that this can only be used in callbacks that also provide a trade object.

For the data to be able to be stored within the database, freqtrade must serialized the data. This is done by converting the data to a JSON formatted string.
Freqtrade will attempt to reverse this action on retrieval, so from a strategy perspective, this should not be relevant.

```python
from freqtrade.persistence import Trade
from datetime import timedelta

class AwesomeStrategy(IStrategy):

    def bot_loop_start(self, **kwargs) -> None:
        for trade in Trade.get_open_order_trades():
            fills = trade.select_filled_orders(trade.entry_side)
            if trade.pair == 'ETH/USDT':
                trade_entry_type = trade.get_custom_data(key='entry_type')
                if trade_entry_type is None:
                    trade_entry_type = 'breakout' if 'entry_1' in trade.enter_tag else 'dip'
                elif fills > 1:
                    trade_entry_type = 'buy_up'
                trade.set_custom_data(key='entry_type', value=trade_entry_type)
        return super().bot_loop_start(**kwargs)

    def adjust_entry_price(self, trade: Trade, order: Optional[Order], pair: str,
                           current_time: datetime, proposed_rate: float, current_order_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        # Limit orders to use and follow SMA200 as price target for the first 10 minutes since entry trigger for BTC/USDT pair.
        if (
            pair == 'BTC/USDT' 
            and entry_tag == 'long_sma200' 
            and side == 'long' 
            and (current_time - timedelta(minutes=10)) > trade.open_date_utc 
            and order.filled == 0.0
        ):
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()
            # store information about entry adjustment
            existing_count = trade.get_custom_data('num_entry_adjustments', default=0)
            if not existing_count:
                existing_count = 1
            else:
                existing_count += 1
            trade.set_custom_data(key='num_entry_adjustments', value=existing_count)

            # adjust order price
            return current_candle['sma_200']

        # default: maintain existing order
        return current_order_rate

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):

        entry_adjustment_count = trade.get_custom_data(key='num_entry_adjustments')
        trade_entry_type = trade.get_custom_data(key='entry_type')
        if entry_adjustment_count is None:
            if current_profit > 0.01 and (current_time - timedelta(minutes=100) > trade.open_date_utc):
                return True, 'exit_1'
        else
            if entry_adjustment_count > 0 and if current_profit > 0.05:
                return True, 'exit_2'
            if trade_entry_type == 'breakout' and current_profit > 0.1:
                return True, 'exit_3

        return False, None
```

The above is a simple example - there are simpler ways to retrieve trade data like entry-adjustments.

!!! Note
    It is recommended that simple data types are used `[bool, int, float, str]` to ensure no issues when serializing the data that needs to be stored.
    Storing big junks of data may lead to unintended side-effects, like a database becoming big (and as a consequence, also slow).

!!! Warning "Non-serializable data"
    If supplied data cannot be serialized a warning is logged and the entry for the specified `key` will contain `None` as data.

??? Note "All attributes"
    custom-data has the following accessors through the Trade object (assumed as `trade` below):

    * `trade.get_custom_data(key='something', default=0)` - Returns the actual value given in the type provided.
    * `trade.get_custom_data_entry(key='something')` - Returns the entry - including metadata. The value is accessible via `.value` property.
    * `trade.set_custom_data(key='something', value={'some': 'value'})` - set or update the corresponding key for this trade. Value must be serializable - and we recommend to keep the stored data relatively small.

    "value" can be any type (both in setting and receiving) - but must be json serializable.

## Storing information (Non-Persistent)

!!! Warning "Deprecated"
    This method of storing information is deprecated and we do advise against using non-persistent storage.  
    Please use [Persistent Storage](#storing-information-persistent) instead.

    It's content has therefore been collapsed.

??? Abstract "Storing information"
    Storing information can be accomplished by creating a new dictionary within the strategy class.

    The name of the variable can be chosen at will, but should be prefixed with `custom_` to avoid naming collisions with predefined strategy variables.

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
                           rate: float, time_in_force: str, exit_reason: str,
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

## Enter Tag

When your strategy has multiple buy signals, you can name the signal that triggered.
Then you can access your buy signal on `custom_exit`

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['rsi'] < 35) &
            (dataframe['volume'] > 0)
        ),
        ['enter_long', 'enter_tag']] = (1, 'buy_signal_rsi')

    return dataframe

def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs):
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last_candle = dataframe.iloc[-1].squeeze()
    if trade.enter_tag == 'buy_signal_rsi' and last_candle['rsi'] > 80:
        return 'sell_signal_rsi'
    return None

```

!!! Note
    `enter_tag` is limited to 100 characters, remaining data will be truncated.

!!! Warning
    There is only one `enter_tag` column, which is used for both long and short trades.
    As a consequence, this column must be treated as "last write wins" (it's just a dataframe column after all).
    In fancy situations, where multiple signals collide (or if signals are deactivated again based on different conditions), this can lead to odd results with the wrong tag applied to an entry signal.
    These results are a consequence of the strategy overwriting prior tags - where the last tag will "stick" and will be the one freqtrade will use.

## Exit tag

Similar to [Entry Tagging](#enter-tag), you can also specify an exit tag.

``` python
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['rsi'] > 70) &
            (dataframe['volume'] > 0)
        ),
        ['exit_long', 'exit_tag']] = (1, 'exit_rsi')

    return dataframe
```

The provided exit-tag is then used as sell-reason - and shown as such in backtest results.

!!! Note
    `exit_reason` is limited to 100 characters, remaining data will be truncated.

## Strategy version

You can implement custom strategy versioning by using the "version" method, and returning the version you would like this strategy to have.

``` python
def version(self) -> str:
    """
    Returns version of the strategy.
    """
    return "1.1"
```

!!! Note
    You should make sure to implement proper version control (like a git repository) alongside this, as freqtrade will not keep historic versions of your strategy, so it's up to the user to be able to eventually roll back to a prior version of the strategy.

## Derived strategies

The strategies can be derived from other strategies. This avoids duplication of your custom strategy code. You can use this technique to override small parts of your main strategy, leaving the rest untouched:

``` python title="user_data/strategies/myawesomestrategy.py"
class MyAwesomeStrategy(IStrategy):
    ...
    stoploss = 0.13
    trailing_stop = False
    # All other attributes and methods are here as they
    # should be in any custom strategy...
    ...

```

``` python title="user_data/strategies/MyAwesomeStrategy2.py"
from myawesomestrategy import MyAwesomeStrategy
class MyAwesomeStrategy2(MyAwesomeStrategy):
    # Override something
    stoploss = 0.08
    trailing_stop = True
```

Both attributes and methods may be overridden, altering behavior of the original strategy in a way you need.

While keeping the subclass in the same file is technically possible, it can lead to some problems with hyperopt parameter files, we therefore recommend to use separate strategy files, and import the parent strategy as shown above.

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
    frames.append(DataFrame({
        f'ema_short_{val}': ta.EMA(dataframe, timeperiod=val)
    }))

# Combine all dataframes, and reassign the original dataframe column
dataframe = pd.concat(frames, axis=1)
```

Freqtrade does however also counter this by running `dataframe.copy()` on the dataframe right after the `populate_indicators()` method - so performance implications of this should be low to non-existent.
