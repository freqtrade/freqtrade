# Strategy Migration between V2 and V3

To support new markets and trade-types (namely short trades / trades with leverage), some things had to change in the interface.
If you intend on using markets other than spot markets, please migrate your strategy to the new format.

We have put a great effort into keeping compatibility with existing strategies, so if you just want to continue using freqtrade in __spot markets__, there should be no changes necessary for now.

You can use the quick summary as checklist. Please refer to the detailed sections below for full migration details.

## Quick summary / migration checklist

Note : `forcesell`, `forcebuy`, `emergencysell` are changed to `force_exit`, `force_enter`, `emergency_exit` respectively.

* Strategy methods:
  * [`populate_buy_trend()` -> `populate_entry_trend()`](#populate_buy_trend)
  * [`populate_sell_trend()` -> `populate_exit_trend()`](#populate_sell_trend)
  * [`custom_sell()` -> `custom_exit()`](#custom_sell)
  * [`check_buy_timeout()` -> `check_entry_timeout()`](#custom_entry_timeout)
  * [`check_sell_timeout()` -> `check_exit_timeout()`](#custom_entry_timeout)
  * New `side` argument to callbacks without trade object
    * [`custom_stake_amount`](#custom_stake_amount)
    * [`confirm_trade_entry`](#confirm_trade_entry)
    * [`custom_entry_price`](#custom_entry_price)
  * [Changed argument name in `confirm_trade_exit`](#confirm_trade_exit)
* Dataframe columns:
  * [`buy` -> `enter_long`](#populate_buy_trend)
  * [`sell` -> `exit_long`](#populate_sell_trend)
  * [`buy_tag` -> `enter_tag` (used for both long and short trades)](#populate_buy_trend)
  * [New column `enter_short` and corresponding new column `exit_short`](#populate_sell_trend)
* trade-object now has the following new properties:
  * `is_short`
  * `entry_side`
  * `exit_side`
  * `trade_direction`
  * renamed: `sell_reason` -> `exit_reason`
* [Renamed `trade.nr_of_successful_buys` to `trade.nr_of_successful_entries` (mostly relevant for `adjust_trade_position()`)](#adjust-trade-position-changes)
* Introduced new [`leverage` callback](strategy-callbacks.md#leverage-callback).
* Informative pairs can now pass a 3rd element in the Tuple, defining the candle type.
* `@informative` decorator now takes an optional `candle_type` argument.
* [helper methods](#helper-methods) `stoploss_from_open` and `stoploss_from_absolute` now take `is_short` as additional argument.
* `INTERFACE_VERSION` should be set to 3.
* [Strategy/Configuration settings](#strategyconfiguration-settings).
  * `order_time_in_force` buy -> entry, sell -> exit.
  * `order_types` buy -> entry, sell -> exit.
  * `unfilledtimeout` buy -> entry, sell -> exit.
  * `ignore_buying_expired_candle_after` -> moved to root level instead of "ask_strategy/exit_pricing"
* Terminology changes
  * Sell reasons changed to reflect the new naming of "exit" instead of sells. Be careful in your strategy if you're using `exit_reason` checks and eventually update your strategy.
    * `sell_signal` -> `exit_signal`
    * `custom_sell` -> `custom_exit`
    * `force_sell` -> `force_exit`
    * `emergency_sell` -> `emergency_exit`
  * Order pricing
    * `bid_strategy` -> `entry_pricing`
    * `ask_strategy` -> `exit_pricing`
    * `ask_last_balance` -> `price_last_balance`
    * `bid_last_balance` -> `price_last_balance`
  * Webhook terminology changed from "sell" to "exit", and from "buy" to entry
    * `webhookbuy` -> `entry`
    * `webhookbuyfill` -> `entry_fill`
    * `webhookbuycancel` -> `entry_cancel`
    * `webhooksell` -> `exit`
    * `webhooksellfill` -> `exit_fill`
    * `webhooksellcancel` -> `exit_cancel`
  * Telegram notification settings
    * `buy` -> `entry`
    * `buy_fill` -> `entry_fill`
    * `buy_cancel` -> `entry_cancel`
    * `sell` -> `exit`
    * `sell_fill` -> `exit_fill`
    * `sell_cancel` -> `exit_cancel`
  * Strategy/config settings:
    * `use_sell_signal` -> `use_exit_signal`
    * `sell_profit_only` -> `exit_profit_only`
    * `sell_profit_offset` -> `exit_profit_offset`
    * `ignore_roi_if_buy_signal` -> `ignore_roi_if_entry_signal`
    * `forcebuy_enable` -> `force_entry_enable`

## Extensive explanation

### `populate_buy_trend`

In `populate_buy_trend()` - you will want to change the columns you assign from `'buy`' to `'enter_long'`, as well as the method name from `populate_buy_trend` to `populate_entry_trend`.

```python hl_lines="1 9"
def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
            (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['buy', 'buy_tag']] = (1, 'rsi_cross')

    return dataframe
```

After:

```python hl_lines="1 9"
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
            (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['enter_long', 'enter_tag']] = (1, 'rsi_cross')

    return dataframe
```

Please refer to the [Strategy documentation](strategy-customization.md#entry-signal-rules) on how to enter and exit short trades.

### `populate_sell_trend`

Similar to `populate_buy_trend`, `populate_sell_trend()` will be renamed to `populate_exit_trend()`.
We'll also change the column from `'sell'` to `'exit_long'`.

``` python hl_lines="1 9"
def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
            (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['sell', 'exit_tag']] = (1, 'some_exit_tag')
    return dataframe
```

After

``` python hl_lines="1 9"
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
            (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['exit_long', 'exit_tag']] = (1, 'some_exit_tag')
    return dataframe
```

Please refer to the [Strategy documentation](strategy-customization.md#exit-signal-rules) on how to enter and exit short trades.

### `custom_sell`

`custom_sell` has been renamed to `custom_exit`.
It's now also being called for every iteration, independent of current profit and `exit_profit_only` settings.

``` python hl_lines="2"
class AwesomeStrategy(IStrategy):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # ...
```

``` python hl_lines="2"
class AwesomeStrategy(IStrategy):
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # ...
```

### `custom_entry_timeout`

`check_buy_timeout()` has been renamed to `check_entry_timeout()`, and `check_sell_timeout()` has been renamed to `check_exit_timeout()`.

``` python hl_lines="2 6"
class AwesomeStrategy(IStrategy):
    def check_buy_timeout(self, pair: str, trade: 'Trade', order: dict, 
                            current_time: datetime, **kwargs) -> bool:
        return False

    def check_sell_timeout(self, pair: str, trade: 'Trade', order: dict, 
                            current_time: datetime, **kwargs) -> bool:
        return False 
```

``` python hl_lines="2 6"
class AwesomeStrategy(IStrategy):
    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order', 
                            current_time: datetime, **kwargs) -> bool:
        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order', 
                            current_time: datetime, **kwargs) -> bool:
        return False 
```

### `custom_stake_amount`

New string argument `side` - which can be either `"long"` or `"short"`.

``` python hl_lines="4"
class AwesomeStrategy(IStrategy):
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], **kwargs) -> float:
        # ... 
        return proposed_stake
```

``` python hl_lines="4"
class AwesomeStrategy(IStrategy):
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        # ... 
        return proposed_stake
```

### `confirm_trade_entry`

New string argument `side` - which can be either `"long"` or `"short"`.

``` python hl_lines="4"
class AwesomeStrategy(IStrategy):
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str], 
                            **kwargs) -> bool:
      return True
```

After: 

``` python hl_lines="4"
class AwesomeStrategy(IStrategy):
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str], 
                            side: str, **kwargs) -> bool:
      return True
```

### `confirm_trade_exit`

Changed argument `sell_reason` to `exit_reason`.
For compatibility, `sell_reason` will still be provided for a limited time.

``` python hl_lines="3"
class AwesomeStrategy(IStrategy):
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
    return True
```

After:

``` python hl_lines="3"
class AwesomeStrategy(IStrategy):
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
    return True
```

### `custom_entry_price`

New string argument `side` - which can be either `"long"` or `"short"`.

``` python hl_lines="3"
class AwesomeStrategy(IStrategy):
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], **kwargs) -> float:
      return proposed_rate
```

After:

``` python hl_lines="3"
class AwesomeStrategy(IStrategy):
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
      return proposed_rate
```

### Adjust trade position changes

While adjust-trade-position itself did not change, you should no longer use `trade.nr_of_successful_buys` - and instead use `trade.nr_of_successful_entries`, which will also include short entries.

### Helper methods

Added argument "is_short" to `stoploss_from_open` and `stoploss_from_absolute`.
This should be given the value of `trade.is_short`.

``` python hl_lines="5 7"
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # once the profit has risen above 10%, keep the stoploss at 7% above the open price
        if current_profit > 0.10:
            return stoploss_from_open(0.07, current_profit)

        return stoploss_from_absolute(current_rate - (candle['atr'] * 2), current_rate)

        return 1

```

After:

``` python hl_lines="5 7"
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # once the profit has risen above 10%, keep the stoploss at 7% above the open price
        if current_profit > 0.10:
            return stoploss_from_open(0.07, current_profit, is_short=trade.is_short)

        return stoploss_from_absolute(current_rate - (candle['atr'] * 2), current_rate, is_short=trade.is_short)


```

### Strategy/Configuration settings

#### `order_time_in_force`

`order_time_in_force` attributes changed from `"buy"` to `"entry"` and `"sell"` to `"exit"`.

``` python
    order_time_in_force: Dict = {
        "buy": "gtc",
        "sell": "gtc",
    }
```

After:

``` python hl_lines="2 3"
    order_time_in_force: Dict = {
        "entry": "GTC",
        "exit": "GTC",
    }
```

#### `order_types`

`order_types` have changed all wordings from `buy` to `entry` - and `sell` to `exit`.
And two words are joined with `_`. 

``` python hl_lines="2-6"
    order_types = {
        "buy": "limit",
        "sell": "limit",
        "emergencysell": "market",
        "forcesell": "market",
        "forcebuy": "market",
        "stoploss": "market",
        "stoploss_on_exchange": false,
        "stoploss_on_exchange_interval": 60
    }
```

After:

``` python hl_lines="2-6"
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": false,
        "stoploss_on_exchange_interval": 60
    }
```

#### Strategy level settings

* `use_sell_signal` -> `use_exit_signal`
* `sell_profit_only` -> `exit_profit_only`
* `sell_profit_offset` -> `exit_profit_offset`
* `ignore_roi_if_buy_signal` -> `ignore_roi_if_entry_signal`

``` python hl_lines="2-5"
    # These values can be overridden in the config.
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset: 0.01
    ignore_roi_if_buy_signal = False
```

After:

``` python hl_lines="2-5"
    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset: 0.01
    ignore_roi_if_entry_signal = False
```

#### `unfilledtimeout`

`unfilledtimeout` have changed all wordings from `buy` to `entry` - and `sell` to `exit`.

``` python hl_lines="2-3"
unfilledtimeout = {
        "buy": 10,
        "sell": 10,
        "exit_timeout_count": 0,
        "unit": "minutes"
    }
```

After:

``` python hl_lines="2-3"
unfilledtimeout = {
        "entry": 10,
        "exit": 10,
        "exit_timeout_count": 0,
        "unit": "minutes"
    }
```

#### `order pricing`

Order pricing changed in 2 ways. `bid_strategy` was renamed to `entry_pricing` and `ask_strategy` was renamed to `exit_pricing`.
The attributes `ask_last_balance` -> `price_last_balance` and `bid_last_balance` -> `price_last_balance` were renamed as well.
Also, price-side can now be defined as `ask`, `bid`, `same` or `other`.
Please refer to the [pricing documentation](configuration.md#prices-used-for-orders) for more information.

``` json hl_lines="2-3 6 12-13 16"
{
    "bid_strategy": {
        "price_side": "bid",
        "use_order_book": true,
        "order_book_top": 1,
        "ask_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": false,
            "bids_to_ask_delta": 1
        }
    },
    "ask_strategy":{
        "price_side": "ask",
        "use_order_book": true,
        "order_book_top": 1,
        "bid_last_balance": 0.0
        "ignore_buying_expired_candle_after": 120
    }
}
```

after:

``` json  hl_lines="2-3 6 12-13 16"
{
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": false,
            "bids_to_ask_delta": 1
        }
    },
    "exit_pricing":{
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1,
        "price_last_balance": 0.0
    },
    "ignore_buying_expired_candle_after": 120
}
```

## FreqAI strategy

The `populate_any_indicators()` method has been split into `feature_engineering_expand_all()`, `feature_engineering_expand_basic()`, `feature_engineering_standard()` and`set_freqai_targets()`.

For each new function, the pair (and timeframe where necessary) will be automatically added to the column.
As such, the definition of features becomes much simpler with the new logic.

For a full explanation of each method, please go to the corresponding [freqAI documentation page](freqai-feature-engineering.md#defining-the-features)

``` python linenums="1" hl_lines="12-37 39-42 63-65 67-75"

def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            informative[f"%-{pair}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{pair}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{pair}adx-period_{t}"] = ta.ADX(informative, timeperiod=t)
            informative[f"%-{pair}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            informative[f"%-{pair}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)

            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(informative), window=t, stds=2.2
            )
            informative[f"{pair}bb_lowerband-period_{t}"] = bollinger["lower"]
            informative[f"{pair}bb_middleband-period_{t}"] = bollinger["mid"]
            informative[f"{pair}bb_upperband-period_{t}"] = bollinger["upper"]

            informative[f"%-{pair}bb_width-period_{t}"] = (
                informative[f"{pair}bb_upperband-period_{t}"]
                - informative[f"{pair}bb_lowerband-period_{t}"]
            ) / informative[f"{pair}bb_middleband-period_{t}"]
            informative[f"%-{pair}close-bb_lower-period_{t}"] = (
                informative["close"] / informative[f"{pair}bb_lowerband-period_{t}"]
            )

            informative[f"%-{pair}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)

            informative[f"%-{pair}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            ) # (1)

        informative[f"%-{pair}pct-change"] = informative["close"].pct_change()
        informative[f"%-{pair}raw_volume"] = informative["volume"]
        informative[f"%-{pair}raw_price"] = informative["close"]
        # (2)

        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # Add generalized indicators here (because in live, it will call this
        # function to populate indicators during training). Notice how we ensure not to
        # add them multiple times
        if set_generalized_indicators:
            df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
            df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25
            # (3)

            # user adds targets here by prepending them with &- (see convention below)
            df["&-s_close"] = (
                df["close"]
                .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
                .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
                .mean()
                / df["close"]
                - 1
            )  # (4)

        return df
```

1. Features - Move to `feature_engineering_expand_all`
2. Basic features, not expanded across `include_periods_candles` - move to`feature_engineering_expand_basic()`.
3. Standard features which should not be expanded - move to `feature_engineering_standard()`.
4. Targets - Move this part to `set_freqai_targets()`.

### freqai - feature engineering expand all

Features will now expand automatically. As such, the expansion loops, as well as the `{pair}` / `{timeframe}` parts will need to be removed.

``` python linenums="1"
    def feature_engineering_expand_all(self, dataframe, period, **kwargs) -> DataFrame::
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param df: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

```

### Freqai - feature engineering basic

Basic features. Make sure to remove the `{pair}` part from your features.

``` python linenums="1"
    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame::
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param df: strategy dataframe which will receive the features
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe
```

### FreqAI - feature engineering standard

``` python linenums="1"
    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param df: strategy dataframe which will receive the features
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe
```

### FreqAI - set Targets

Targets now get their own, dedicated method.

``` python linenums="1"
    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param df: strategy dataframe which will receive the targets
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
            )

        return dataframe
```


### FreqAI - New data Pipeline

If you have created your own custom `IFreqaiModel` with a custom `train()`/`predict()` function, *and* you still rely on `data_cleaning_train/predict()`, then you will need to migrate to the new pipeline. If your model does *not* rely on `data_cleaning_train/predict()`, then you do not need to worry about this migration. That means that this migration guide is relevant for a very small percentage of power-users. If you stumbled upon this guide by mistake, feel free to inquire in depth about your problem in the Freqtrade discord server.

The conversion involves first removing `data_cleaning_train/predict()` and replacing them with a `define_data_pipeline()` and `define_label_pipeline()` function to your `IFreqaiModel` class:

```python  linenums="1" hl_lines="11-14 47-49 55-57"
class MyCoolFreqaiModel(BaseRegressionModel):
    """
    Some cool custom IFreqaiModel you made before Freqtrade version 2023.6
    """
    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:

        # ... your custom stuff

        # Remove these lines
        # data_dictionary = dk.make_train_test_datasets(features_filtered, labels_filtered)
        # self.data_cleaning_train(dk)
        # data_dictionary = dk.normalize_data(data_dictionary)
        # (1)

        # Add these lines. Now we control the pipeline fit/transform ourselves
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        (dd["train_features"],
         dd["train_labels"],
         dd["train_weights"]) = dk.feature_pipeline.fit_transform(dd["train_features"],
                                                                  dd["train_labels"],
                                                                  dd["train_weights"])

        (dd["test_features"],
         dd["test_labels"],
         dd["test_weights"]) = dk.feature_pipeline.transform(dd["test_features"],
                                                             dd["test_labels"],
                                                             dd["test_weights"])

        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])
        dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])

        # ... your custom code

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:

        # ... your custom stuff

        # Remove these lines:
        # self.data_cleaning_predict(dk)
        # (2)

        # Add these lines:
        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)

        # Remove this line
        # pred_df = dk.denormalize_labels_from_metadata(pred_df)
        # (3)

        # Replace with these lines
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
        if self.freqai_info.get("DI_threshold", 0) > 0:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(len(outliers.index))
        dk.do_predict = outliers.to_numpy()

        # ... your custom code
        return (pred_df, dk.do_predict)
```


1. Data normalization and cleaning is now homogenized with the new pipeline definition. This is created in the new `define_data_pipeline()` and `define_label_pipeline()` functions. The `data_cleaning_train()` and `data_cleaning_predict()` functions are no longer used. You can override `define_data_pipeline()` to create your own custom pipeline if you wish.
2. Data normalization and cleaning is now homogenized with the new pipeline definition. This is created in the new `define_data_pipeline()` and `define_label_pipeline()` functions. The `data_cleaning_train()` and `data_cleaning_predict()` functions are no longer used. You can override `define_data_pipeline()` to create your own custom pipeline if you wish.
3. Data denormalization is done with the new pipeline. Replace this with the lines below.
