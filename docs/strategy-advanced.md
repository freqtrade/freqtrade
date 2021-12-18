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

## Exit tag

Similar to [Buy Tagging](#buy-tag), you can also specify a sell tag.

``` python
def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['rsi'] > 70) &
            (dataframe['volume'] > 0)
        ),
        ['sell', 'exit_tag']] = (1, 'exit_rsi')

    return dataframe
```

The provided exit-tag is then used as sell-reason - and shown as such in backtest results.

!!! Note
    `sell_reason` is limited to 100 characters, remaining data will be truncated.

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

## Adjust trade position

`adjust_trade_position()` can be used to perform additional orders to manage risk with DCA (Dollar Cost Averaging) for example.

!!! Note
    The `position_adjustment_enable` configuration parameter must be enabled to use adjust_trade_position callback in strategy.

!!! Warning
    Additional orders also mean additional fees.

!!! Warning
    Stoploss is still calculated from the initial opening price, not averaged price.

``` python
from freqtrade.persistence import Trade


class DigDeeperStrategy(IStrategy):
    
    # Attempts to handle large drops with DCA. High stoploss is required.
    stoploss = -0.30
    
    # ... populate_* methods

    def adjust_trade_position(self, pair: str, trade: Trade,
                              current_time: datetime, current_rate: float, current_profit: float,
                                  **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.
 
        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """

        if current_profit > -0.05:
            return None

        # Obtain pair dataframe.
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
        # Only buy when not actively falling price.
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None

        count_of_buys = 0
        for order in trade.orders:
            # Instantly stop when there's an open order
            if order.ft_is_open:
                return None
            if order.ft_order_side == 'buy' and order.status == "closed":
                count_of_buys += 1

        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falles down to -5% again, we buy 1.5x more
        # If that falles once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # Hope you have a deep wallet!
        if 0 < count_of_buys <= 3:
            try:
                stake_amount = self.wallets.get_trade_stake_amount(pair, None)
                stake_amount = stake_amount * (1 + (count_of_buys * 0.25))
                return stake_amount
            except Exception as exception:
                return None

        return None

```
