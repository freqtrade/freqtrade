# Strategy Customization

This page explains how to customize your strategies, add new indicators and set up trading rules.

If you haven't already, please familiarize yourself with:

- the [Freqtrade strategy 101](freqtrade-101.md), which provides a quick start to strategy development
- the [Freqtrade bot basics](bot-basics.md), which provides overall info on how the bot operates

## Develop your own strategy

The bot includes a default strategy file.

Also, several other strategies are available in the [strategy repository](https://github.com/freqtrade/freqtrade-strategies).

You will however most likely have your own idea for a strategy.

This document intends to help you convert your ideas into a working strategy.

### Generating a strategy template

To get started, you can use the command:

```bash
freqtrade new-strategy --strategy AwesomeStrategy
```

This will create a new strategy called `AwesomeStrategy` from a template, which will be located using the filename `user_data/strategies/AwesomeStrategy.py`.

!!! Note
    There is a difference between the *name* of the strategy and the filename. In most commands, Freqtrade uses the *name* of the strategy, *not the filename*.

!!! Note
    The `new-strategy` command generates starting examples which will not be profitable out of the box.

??? Hint "Different template levels"
    `freqtrade new-strategy` has an additional parameter, `--template`, which controls the amount of pre-build information you get in the created strategy. Use `--template minimal` to get an empty strategy without any indicator examples, or `--template advanced` to get a template with more complicated features defined.

### Anatomy of a strategy

A strategy file contains all the information needed to build the strategy logic:

- Candle data in OHLCV format
- Indicators
- Entry logic
  - Signals
- Exit logic
  - Signals
  - Minimal ROI
  - Callbacks ("custom functions")
- Stoploss
  - Fixed/absolute
  - Trailing
  - Callbacks ("custom functions")
- Pricing [optional]
- Position adjustment [optional]

The bot includes a sample strategy called `SampleStrategy` that you can use as a basis: `user_data/strategies/sample_strategy.py`.
You can test it with the parameter: `--strategy SampleStrategy`. Remember that you use the strategy class name, not the filename.

Additionally, there is an attribute called `INTERFACE_VERSION`, which defines the version of the strategy interface the bot should use.
The current version is 3 - which is also the default when it's not set explicitly in the strategy.

You may see older strategies set to interface version 2, and these will need to be updated to v3 terminology as future versions will require this to be set.

Starting the bot in dry or live mode is accomplished using the `trade` command:

```bash
freqtrade trade --strategy AwesomeStrategy
```

### Bot modes

Freqtrade strategies can be processed by the Freqtrade bot in 5 main modes:

- backtesting
- hyperopting
- dry ("forward testing")
- live
- FreqAI (not covered here)

Check the [configuration documentation](configuration.md) about how to set the bot to dry or live mode.

**Always use dry mode when testing as this gives you an idea of how your strategy will work in reality without risking capital.**

## Diving in deeper
**For the following section we will use the [user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_strategy.py)
file as reference.**

!!! Note "Strategies and Backtesting"
    To avoid problems and unexpected differences between backtesting and dry/live modes, please be aware
    that during backtesting the full time range is passed to the `populate_*()` methods at once.
    It is therefore best to use vectorized operations (across the whole dataframe, not loops) and
    avoid index referencing (`df.iloc[-1]`), but instead use `df.shift()` to get to the previous candle.

!!! Warning "Warning: Using future data"
    Since backtesting passes the full time range to the `populate_*()` methods, the strategy author
    needs to take care to avoid having the strategy utilize data from the future.
    Some common patterns for this are listed in the [Common Mistakes](#common-mistakes-when-developing-strategies) section of this document.

??? Hint "Lookahead and recursive analysis"
    Freqtrade includes two helpful commands to help assess common lookahead (using future data) and 
    recursive bias (variance in indicator values) issues. Before running a strategy in dry or live more, 
    you should always use these commands first. Please check the relevant documentation for 
    [lookahead](lookahead-analysis.md) and [recursive](recursive-analysis.md) analysis.

### Dataframe

Freqtrade uses [pandas](https://pandas.pydata.org/) to store/provide the candlestick (OHLCV) data.
Pandas is a great library developed for processing large amounts of data in tabular format.

Each row in a dataframe corresponds to one candle on a chart, with the latest complete candle always being the last in the dataframe (sorted by date).

If we were to look at the first few rows of the main dataframe using the pandas `head()` function, we would see:

```output
> dataframe.head()
                       date      open      high       low     close     volume
0 2021-11-09 23:25:00+00:00  67279.67  67321.84  67255.01  67300.97   44.62253
1 2021-11-09 23:30:00+00:00  67300.97  67301.34  67183.03  67187.01   61.38076
2 2021-11-09 23:35:00+00:00  67187.02  67187.02  67031.93  67123.81  113.42728
3 2021-11-09 23:40:00+00:00  67123.80  67222.40  67080.33  67160.48   78.96008
4 2021-11-09 23:45:00+00:00  67160.48  67160.48  66901.26  66943.37  111.39292
```

A dataframe is a table where columns are not single values, but a series of data values. As such, simple python comparisons like the following will not work:

``` python
    if dataframe['rsi'] > 30:
        dataframe['enter_long'] = 1
```

The above section will fail with `The truth value of a Series is ambiguous [...]`.

This must instead be written in a pandas-compatible way, so the operation is performed across the whole dataframe, i.e. `vectorisation`.

``` python
    dataframe.loc[
        (dataframe['rsi'] > 30)
    , 'enter_long'] = 1
```

With this section, you have a new column in your dataframe, which has `1` assigned whenever RSI is above 30.

Freqtrade uses this new column as an entry signal, where it is assumed that a trade will subsequently open on the next open candle.

Pandas provides fast ways to calculate metrics, i.e. "vectorisation". To benefit from this speed, it is advised to not use loops, but use vectorized methods instead.

Vectorized operations perform calculations across the whole range of data and are therefore, compared to looping through each row, a lot faster when calculating indicators.

??? Hint "Signals vs Trades"
    - Signals are generated from indicators at candle close, and are intentions to enter a trade.
    - Trades are orders that are executed (on the exchange in live mode) where a trade will then open as close to next candle open as possible.

!!! Warning "Trade order assumptions"
    In backtesting, signals are generated on candle close. Trades are then initiated immeditely on next candle open.
    
    In dry and live, this may be delayed due to all pair dataframes needing to be analysed first, then trade processing 
    for each of those pairs happens. This means that in dry/live you need to be mindful of having as low a computation 
    delay as possible, usually by running a low number of pairs and having a CPU with a good clock speed.

#### Why can't I see "real time" candle data?

Freqtrade does not store incomplete/unfinished candles in the dataframe.

The use of incomplete data for making strategy decisions is called "repainting" and you might see other platforms allow this.

Freqtrade does not. Only complete/finished candle data is available in the dataframe.

### Customize Indicators

Entry and exit signals need indicators. You can add more indicators by extending the list contained in the method `populate_indicators()` from your strategy file.

You should only add the indicators used in either `populate_entry_trend()`, `populate_exit_trend()`, or to populate another indicator, otherwise performance may suffer.

It's important to always return the dataframe from these three functions without removing/modifying the columns `"open", "high", "low", "close", "volume"`, otherwise these fields would contain something unexpected.

Sample:

```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame

    Performance Note: For the best performance be frugal on the number of indicators
    you are using. Let uncomment only the indicator you are using in your strategies
    or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
    :param dataframe: Dataframe with data from the exchange
    :param metadata: Additional information, like the currently traded pair
    :return: a Dataframe with all mandatory indicators for the strategies
    """
    dataframe['sar'] = ta.SAR(dataframe)
    dataframe['adx'] = ta.ADX(dataframe)
    stoch = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch['fastd']
    dataframe['fastk'] = stoch['fastk']
    dataframe['bb_lower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
    dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
    dataframe['mfi'] = ta.MFI(dataframe)
    dataframe['rsi'] = ta.RSI(dataframe)
    dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
    dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
    dataframe['ao'] = awesome_oscillator(dataframe)
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    hilbert = ta.HT_SINE(dataframe)
    dataframe['htsine'] = hilbert['sine']
    dataframe['htleadsine'] = hilbert['leadsine']
    dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe)
    dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe)

    # remember to always return the dataframe
    return dataframe
```

!!! Note "Want more indicator examples?"
    Look into the [user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_strategy.py).
    Then uncomment indicators you need.

#### Indicator libraries

Out of the box, freqtrade installs the following technical libraries:

- [ta-lib](https://ta-lib.github.io/ta-lib-python/)
- [pandas-ta](https://twopirllc.github.io/pandas-ta/)
- [technical](https://technical.freqtrade.io)

Additional technical libraries can be installed as necessary, or custom indicators may be written / invented by the strategy author.

### Strategy startup period

Some indicators have an unstable startup period in which there isn't enough candle data to calculate any values (NaN), or the calculation is incorrect. This can lead to inconsistencies, since Freqtrade does not know how long this unstable period is and uses whatever indicator values are in the dataframe.

To account for this, the strategy can be assigned the `startup_candle_count` attribute.

This should be set to the maximum number of candles that the strategy requires to calculate stable indicators. In the case where a user includes higher timeframes with informative pairs, the `startup_candle_count` does not necessarily change. The value is the maximum period (in candles) that any of the informatives timeframes need to compute stable indicators.

You can use [recursive-analysis](recursive-analysis.md) to check and find the correct `startup_candle_count` to be used. When recursive analysis shows a variance of 0%, then you can be sure that you have enough startup candle data.

In this example strategy, this should be set to 400 (`startup_candle_count = 400`), since the minimum needed history for ema100 calculation to make sure the value is correct is 400 candles.

``` python
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
```

By letting the bot know how much history is needed, backtest trades can start at the specified timerange during backtesting and hyperopt.

!!! Warning "Using x calls to get OHLCV"
    If you receive a warning like `WARNING - Using 3 calls to get OHLCV. This can result in slower operations for the bot. Please check if you really need 1500 candles for your strategy` - you should consider if you really need this much historic data for your signals.
    Having this will cause Freqtrade to make multiple calls for the same pair, which will obviously be slower than one network request.
    As a consequence, Freqtrade will take longer to refresh candles - and should therefore be avoided if possible.
    This is capped to 5 total calls to avoid overloading the exchange, or make freqtrade too slow.

!!! Warning
    `startup_candle_count` should be below `ohlcv_candle_limit * 5` (which is 500 * 5 for most exchanges) - since only this amount of candles will be available during Dry-Run/Live Trade operations.

#### Example

Let's try to backtest 1 month (January 2019) of 5m candles using an example strategy with EMA100, as above.

``` bash
freqtrade backtesting --timerange 20190101-20190201 --timeframe 5m
```

Assuming `startup_candle_count` is set to 400, backtesting knows it needs 400 candles to generate valid entry signals. It will load data from `20190101 - (400 * 5m)` - which is ~2018-12-30 11:40:00.

If this data is available, indicators will be calculated with this extended timerange. The unstable startup period (up to 2019-01-01 00:00:00) will then be removed before backtesting is carried out.

!!! Note "Unavailable startup candle data"
    If data for the startup period is not available, then the timerange will be adjusted to account for this startup period. In our example, backtesting would then start from 2019-01-02 09:20:00.

### Entry signal rules

Edit the method `populate_entry_trend()` in your strategy file to update your entry strategy.

It's important to always return the dataframe without removing/modifying the columns `"open", "high", "low", "close", "volume"`, otherwise these fields would contain something unexpected. The strategy may then produce invalid values, or cease to work entirely.

This method will also define a new column, `"enter_long"` (`"enter_short"` for shorts), which needs to contain `1` for entries, and `0` for "no action". `enter_long` is a mandatory column that must be set even if the strategy is shorting only.

You can name your entry signals by using the `"enter_tag"` column, which can help debug and assess your strategy later. 

Sample from `user_data/strategies/sample_strategy.py`:

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Based on TA indicators, populates the buy signal for the given dataframe
    :param dataframe: DataFrame populated with indicators
    :param metadata: Additional information, like the currently traded pair
    :return: DataFrame with buy column
    """
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

??? Note "Enter short trades"
    Short entries can be created by setting `enter_short` (corresponds to `enter_long` for long trades).
    The `enter_tag` column remains identical.
    Shorting needs to be supported by your exchange and market configuration!
    Also, make sure you set [`can_short`](#can-short) appropriately on your strategy if you intend to short.

    ```python
    # allow both long and short trades
    can_short = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
                (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'rsi_cross')

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsi'], 70)) &  # Signal: RSI crosses below 70
                (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
                (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_short', 'enter_tag']] = (1, 'rsi_cross')

        return dataframe
    ```

!!! Note
    Buying requires sellers to buy from. Therefore volume needs to be > 0 (`dataframe['volume'] > 0`) to make sure that the bot does not buy/sell in no-activity periods.

### Exit signal rules

Edit the method `populate_exit_trend()` into your strategy file to update your exit strategy.

The exit-signal can be suppressed by setting `use_exit_signal` to false in the configuration or strategy.

`use_exit_signal` will not influence [signal collision rules](#colliding-signals) - which will still apply and can prevent entries.

It's important to always return the dataframe without removing/modifying the columns `"open", "high", "low", "close", "volume"`, otherwise these fields would contain something unexpected. The strategy may then produce invalid values, or cease to work entirely.

This method will also define a new column, `"exit_long"` (`"exit_short"` for shorts), which needs to contain `1` for exits, and `0` for "no action".

You can name your exit signals by using the `"exit_tag"` column, which can help debug and assess your strategy later.

Sample from `user_data/strategies/sample_strategy.py`:

```python
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Based on TA indicators, populates the exit signal for the given dataframe
    :param dataframe: DataFrame populated with indicators
    :param metadata: Additional information, like the currently traded pair
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
            (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['exit_long', 'exit_tag']] = (1, 'rsi_too_high')
    return dataframe
```

??? Note "Exit short trades"
    Short exits can be created by setting `exit_short` (corresponds to `exit_long`).
    The `exit_tag` column remains identical.
    Shorting needs to be supported by your exchange and market configuration!
    Also, make sure you set [`can_short`](#can-short) appropriately on your strategy if you intend to short.

    ```python
    # allow both long and short trades
    can_short = True

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
                (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
                (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'rsi_too_high')
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsi'], 30)) &  # Signal: RSI crosses below 30
                (dataframe['tema'] < dataframe['bb_middleband']) &  # Guard
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_short', 'exit_tag']] = (1, 'rsi_too_low')
        return dataframe
    ```

### Minimal ROI

The `minimal_roi` strategy variable defines the minimal Return On Investment (ROI) a trade should reach before exiting, independent from the exit signal.

It is of the following format, i.e. a python `dict`, with the dict key (left side of the colon) being the minutes passed since the trade opened, and the value (right side of the colon) being the percentage.

```python
minimal_roi = {
    "40": 0.0,
    "30": 0.01,
    "20": 0.02,
    "0": 0.04
}
```

The above configuration would therefore mean:

- Exit whenever 4% profit was reached
- Exit when 2% profit was reached (in effect after 20 minutes)
- Exit when 1% profit was reached (in effect after 30 minutes)
- Exit when trade is non-loosing (in effect after 40 minutes)

The calculation does include fees.

#### Disabling minimal ROI

To disable ROI completely, set it to an empty dictionary:

```python
minimal_roi = {}
```

#### Using calculations in minimal ROI

To use times based on candle duration (timeframe), the following snippet can be handy.

This will allow you to change the timeframe for the strategy, but the minimal ROI times will still be set as candles, e.g. after 3 candles.

``` python
from freqtrade.exchange import timeframe_to_minutes

class AwesomeStrategy(IStrategy):

    timeframe = "1d"
    timeframe_mins = timeframe_to_minutes(timeframe)
    minimal_roi = {
        "0": 0.05,                      # 5% for the first 3 candles
        str(timeframe_mins * 3): 0.02,  # 2% after 3 candles
        str(timeframe_mins * 6): 0.01,  # 1% After 6 candles
    }
```

??? info "Orders that don't fill immediately"
    `minimal_roi` will take the `trade.open_date` as reference, which is the time the trade was initialized, i.e. when the first order for this trade was placed.
    This will also hold true for limit orders that don't fill immediately (usually in combination with "off-spot" prices through `custom_entry_price()`), as well as for cases where the initial order price is replaced through `adjust_entry_price()`.
    The time used will still be from the initial `trade.open_date` (when the initial order was first placed), not from the newly placed or adjusted order date.

### Stoploss

Setting a stoploss is highly recommended to protect your capital from strong moves against you.

Sample of setting a 10% stoploss:

``` python
stoploss = -0.10
```

For the full documentation on stoploss features, look at the dedicated [stoploss page](stoploss.md).

### Timeframe

This is the periodicity of candles the bot should use in the strategy.

Common values are `"1m"`, `"5m"`, `"15m"`, `"1h"`, however all values supported by your exchange should work.

Please note that the same entry/exit signals may work well with one timeframe, but not with others.

This setting is accessible within the strategy methods as the `self.timeframe` attribute.

### Can short

To use short signals in futures markets, you will have to set `can_short = True`.

Strategies which enable this will fail to load on spot markets.

If you have `1` values in the `enter_short` column to raise short signals, setting `can_short = False` (which is the default) will mean that these short signals are ignored, even if you have specified futures markets in your configuration.

### Metadata dict

The `metadata` dict (available for `populate_entry_trend`, `populate_exit_trend`, `populate_indicators`) contains additional information.
Currently this is `pair`, which can be accessed using `metadata['pair']`, and will return a pair in the format `XRP/BTC` (or `XRP/BTC:BTC` for futures markets).

The metadata dict should not be modified and does not persist information across multiple functions in your strategy.

Instead, please check the [Storing information](strategy-advanced.md#storing-information-persistent) section.

--8<-- "includes/strategy-imports.md"

## Strategy file loading

By default, freqtrade will attempt to load strategies from all `.py` files within the `userdir` (default `user_data/strategies`).

Assuming your strategy is called `AwesomeStrategy`, stored in the file `user_data/strategies/AwesomeStrategy.py`, then you can start freqtrade in dry (or live, depending on your configuration) mode with:

```bash
    freqtrade trade --strategy AwesomeStrategy`
```

Note that we're using the class name, not the file name.

You can use `freqtrade list-strategies` to see a list of all strategies Freqtrade is able to load (all strategies in the correct folder).
It will also include a "status" field, highlighting potential problems.

??? Hint "Customize strategy directory"
    You can use a different directory by using `--strategy-path user_data/otherPath`. This parameter is available to all commands that require a strategy.

## Informative Pairs

### Get data for non-tradeable pairs

Data for additional, informative pairs (reference pairs) can be beneficial for some strategies to see data on a wider timeframe.

OHLCV data for these pairs will be downloaded as part of the regular whitelist refresh process and is available via `DataProvider` just as other pairs (see below).

These pairs will **not** be traded unless they are also specified in the pair whitelist, or have been selected by Dynamic Whitelisting, e.g. `VolumePairlist`.

The pairs need to be specified as tuples in the format `("pair", "timeframe")`, with pair as the first and timeframe as the second argument.

Sample:

``` python
def informative_pairs(self):
    return [("ETH/USDT", "5m"),
            ("BTC/TUSD", "15m"),
            ]
```

A full sample can be found [in the DataProvider section](#complete-data-provider-sample).

!!! Warning
    As these pairs will be refreshed as part of the regular whitelist refresh, it's best to keep this list short.
    All timeframes and all pairs can be specified as long as they are available (and active) on the used exchange.
    It is however better to use resampling to longer timeframes whenever possible
    to avoid hammering the exchange with too many requests and risk being blocked.

??? Note "Alternative candle types"
    Informative_pairs can also provide a 3rd tuple element defining the candle type explicitly.
    Availability of alternative candle-types will depend on the trading-mode and the exchange. 
    In general, spot pairs cannot be used in futures markets, and futures candles can't be used as informative pairs for spot bots.
    Details about this may vary, if they do, this can be found in the exchange documentation.

    ``` python
    def informative_pairs(self):
        return [
            ("ETH/USDT", "5m", ""),   # Uses default candletype, depends on trading_mode (recommended)
            ("ETH/USDT", "5m", "spot"),   # Forces usage of spot candles (only valid for bots running on spot markets).
            ("BTC/TUSD", "15m", "futures"),  # Uses futures candles (only bots with `trading_mode=futures`)
            ("BTC/TUSD", "15m", "mark"),  # Uses mark candles (only bots with `trading_mode=futures`)
        ]
    ```
***

### Informative pairs decorator (`@informative()`)

To easily define informative pairs, use the `@informative` decorator. All decorated `populate_indicators_*` methods run in isolation,
and do not have access to data from other informative pairs. However, all informative dataframes for each pair are merged and passed to main `populate_indicators()` method.

!!! Note
    Do not use the `@informative` decorator if you need to use data from one informative pair when generating another informative pair. Instead, define informative pairs manually as described [in the DataProvider section](#complete-data-provider-sample).

When hyperopting, use of the hyperoptable parameter `.value` attribute is not supported. Please use the `.range` attribute. See [optimizing an indicator parameter](hyperopt.md#optimizing-an-indicator-parameter) for more information.

??? info "Full documentation"
    ``` python
    def informative(timeframe: str, asset: str = '',
                    fmt: Optional[Union[str, Callable[[KwArg(str)], str]]] = None,
                    *,
                    candle_type: Optional[CandleType] = None,
                    ffill: bool = True) -> Callable[[PopulateIndicators], PopulateIndicators]:
        """
        A decorator for populate_indicators_Nn(self, dataframe, metadata), allowing these functions to
        define informative indicators.

        Example usage:

            @informative('1h')
            def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
                dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
                return dataframe

        :param timeframe: Informative timeframe. Must always be equal or higher than strategy timeframe.
        :param asset: Informative asset, for example BTC, BTC/USDT, ETH/BTC. Do not specify to use
                    current pair. Also supports limited pair format strings (see below)
        :param fmt: Column format (str) or column formatter (callable(name, asset, timeframe)). When not
        specified, defaults to:
        * {base}_{quote}_{column}_{timeframe} if asset is specified.
        * {column}_{timeframe} if asset is not specified.
        Pair format supports these format variables:
        * {base} - base currency in lower case, for example 'eth'.
        * {BASE} - same as {base}, except in upper case.
        * {quote} - quote currency in lower case, for example 'usdt'.
        * {QUOTE} - same as {quote}, except in upper case.
        Format string additionally supports this variables.
        * {asset} - full name of the asset, for example 'BTC/USDT'.
        * {column} - name of dataframe column.
        * {timeframe} - timeframe of informative dataframe.
        :param ffill: ffill dataframe after merging informative pair.
        :param candle_type: '', mark, index, premiumIndex, or funding_rate
        """
    ```

??? Example "Fast and easy way to define informative pairs"

    Most of the time we do not need power and flexibility offered by `merge_informative_pair()`, therefore we can use a decorator to quickly define informative pairs.

    ``` python

    from datetime import datetime
    from freqtrade.persistence import Trade
    from freqtrade.strategy import IStrategy, informative

    class AwesomeStrategy(IStrategy):
        
        # This method is not required. 
        # def informative_pairs(self): ...

        # Define informative upper timeframe for each pair. Decorators can be stacked on same 
        # method. Available in populate_indicators as 'rsi_30m' and 'rsi_1h'.
        @informative('30m')
        @informative('1h')
        def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe

        # Define BTC/STAKE informative pair. Available in populate_indicators and other methods as
        # 'btc_rsi_1h'. Current stake currency should be specified as {stake} format variable 
        # instead of hard-coding actual stake currency. Available in populate_indicators and other 
        # methods as 'btc_usdt_rsi_1h' (when stake currency is USDT).
        @informative('1h', 'BTC/{stake}')
        def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe

        # Define BTC/ETH informative pair. You must specify quote currency if it is different from
        # stake currency. Available in populate_indicators and other methods as 'eth_btc_rsi_1h'.
        @informative('1h', 'ETH/BTC')
        def populate_indicators_eth_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe
    
        # Define BTC/STAKE informative pair. A custom formatter may be specified for formatting
        # column names. A callable `fmt(**kwargs) -> str` may be specified, to implement custom
        # formatting. Available in populate_indicators and other methods as 'rsi_upper_1h'.
        @informative('1h', 'BTC/{stake}', '{column}_{timeframe}')
        def populate_indicators_btc_1h_2(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi_upper'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe
    
        def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            # Strategy timeframe indicators for current pair.
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            # Informative pairs are available in this method.
            dataframe['rsi_less'] = dataframe['rsi'] < dataframe['rsi_1h']
            return dataframe

    ```

!!! Note
    Use string formatting when accessing informative dataframes of other pairs. This will allow easily changing stake currency in config without having to adjust strategy code.

    ``` python
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stake = self.config['stake_currency']
        dataframe.loc[
            (
                (dataframe[f'btc_{stake}_rsi_1h'] < 35)
                &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_signal_rsi')
    
        return dataframe
    ```

    Alternatively column renaming may be used to remove stake currency from column names: `@informative('1h', 'BTC/{stake}', fmt='{base}_{column}_{timeframe}')`.

!!! Warning "Duplicate method names"
    Methods tagged with the `@informative()` decorator must always have unique names! Reusing the same name (for example when copy-pasting already defined informative methods) will overwrite previously defined methods and not produce any errors due to limitations of Python programming language. In such cases you will find that indicators created in methods higher up in the strategy file are not available in the dataframe. Carefully review method names and make sure they are unique!

### *merge_informative_pair()*

This method helps you merge an informative pair to the regular main dataframe safely and consistently, without lookahead bias.

Options:

- Rename the columns to create unique columns
- Merge the dataframe without lookahead bias
- Forward-fill (optional)

For a full sample, please refer to the [complete data provider example](#complete-data-provider-sample) below.

All columns of the informative dataframe will be available on the returning dataframe in a renamed fashion:

!!! Example "Column renaming"
    Assuming `inf_tf = '1d'` the resulting columns will be:

    ``` python
    'date', 'open', 'high', 'low', 'close', 'rsi'                     # from the original dataframe
    'date_1d', 'open_1d', 'high_1d', 'low_1d', 'close_1d', 'rsi_1d'   # from the informative dataframe
    ```

??? Example "Column renaming - 1h"
    Assuming `inf_tf = '1h'` the resulting columns will be:

    ``` python
    'date', 'open', 'high', 'low', 'close', 'rsi'                     # from the original dataframe
    'date_1h', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 'rsi_1h'   # from the informative dataframe
    ```

??? Example "Custom implementation"
    A custom implementation for this is possible, and can be done as follows:

    ``` python

    # Shift date by 1 candle
    # This is necessary since the data is always the "open date"
    # and a 15m candle starting at 12:15 should not know the close of the 1h candle from 12:00 to 13:00
    minutes = timeframe_to_minutes(inf_tf)
    # Only do this if the timeframes are different:
    informative['date_merge'] = informative["date"] + pd.to_timedelta(minutes, 'm')

    # Rename columns to be unique
    informative.columns = [f"{col}_{inf_tf}" for col in informative.columns]
    # Assuming inf_tf = '1d' - then the columns will now be:
    # date_1d, open_1d, high_1d, low_1d, close_1d, rsi_1d

    # Combine the 2 dataframes
    # all indicators on the informative sample MUST be calculated before this point
    dataframe = pd.merge(dataframe, informative, left_on='date', right_on=f'date_merge_{inf_tf}', how='left')
    # FFill to have the 1d value available in every row throughout the day.
    # Without this, comparisons would only work once per day.
    dataframe = dataframe.ffill()

    ```

!!! Warning "Informative timeframe < timeframe"
    Using informative timeframes smaller than the main dataframe timeframe is not recommended with this method, as it will not use any of the additional information this would provide.
    To use the more detailed information properly, more advanced methods should be applied (which are out of scope for this documentation).

## Additional data (DataProvider)

The strategy provides access to the `DataProvider`. This allows you to get additional data to use in your strategy.

All methods return `None` in case of failure, i.e. failures do not raise an exception.

Please always check the mode of operation to select the correct method to get data (see below for examples).

!!! Warning "Hyperopt Limitations"
    The DataProvider is available during hyperopt, however it can only be used in `populate_indicators()` **within a strategy**, not within a hyperopt class file.
    It is also not available in `populate_entry_trend()` and `populate_exit_trend()` methods.

### Possible options for DataProvider

- [`available_pairs`](#available_pairs) - Property with tuples listing cached pairs with their timeframe (pair, timeframe).
- [`current_whitelist()`](#current_whitelist) - Returns a current list of whitelisted pairs. Useful for accessing dynamic whitelists (i.e. VolumePairlist)
- [`get_pair_dataframe(pair, timeframe)`](#get_pair_dataframepair-timeframe) - This is a universal method, which returns either historical data (for backtesting) or cached live data (for the Dry-Run and Live-Run modes).
- [`get_analyzed_dataframe(pair, timeframe)`](#get_analyzed_dataframepair-timeframe) - Returns the analyzed dataframe (after calling `populate_indicators()`, `populate_buy()`, `populate_sell()`) and the time of the latest analysis.
- `historic_ohlcv(pair, timeframe)` - Returns historical data stored on disk.
- `market(pair)` - Returns market data for the pair: fees, limits, precisions, activity flag, etc. See [ccxt documentation](https://github.com/ccxt/ccxt/wiki/Manual#markets) for more details on the Market data structure.
- `ohlcv(pair, timeframe)` - Currently cached candle (OHLCV) data for the pair, returns DataFrame or empty DataFrame.
- [`orderbook(pair, maximum)`](#orderbookpair-maximum) - Returns latest orderbook data for the pair, a dict with bids/asks with a total of `maximum` entries.
- [`ticker(pair)`](#tickerpair) - Returns current ticker data for the pair. See [ccxt documentation](https://github.com/ccxt/ccxt/wiki/Manual#price-tickers) for more details on the Ticker data structure.
- `runmode` - Property containing the current runmode.

### Example Usages

### *available_pairs*

``` python
for pair, timeframe in self.dp.available_pairs:
    print(f"available {pair}, {timeframe}")
```

### *current_whitelist()*

Imagine you've developed a strategy that trades the `5m` timeframe using signals generated from a `1d` timeframe on the top 10 exchange pairs by volume.

The strategy logic might look something like this:

*Scan through the top 10 pairs by volume using the `VolumePairList` every 5 minutes and use a 14 day RSI to enter and exit.*

Due to the limited available data, it's very difficult to resample `5m` candles into daily candles for use in a 14 day RSI. Most exchanges limit users to just 500-1000 candles which effectively gives us around 1.74 daily candles. We need 14 days at least!

Since we can't resample the data we will have to use an informative pair, and since the whitelist will be dynamic we don't know which pair(s) to use! We have a problem!

This is where calling `self.dp.current_whitelist()` comes in handy to retrieve only those pairs in the whitelist.

```python
    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign timeframe to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        return informative_pairs
```

??? Note "Plotting with current_whitelist"
    Current whitelist is not supported for `plot-dataframe`, as this command is usually used by providing an explicit pairlist and would therefore make the return values of this method misleading.
    It's also not supported for FreqUI visualization in [webserver mode](utils.md#webserver-mode), as the configuration for webserver mode doesn't require a pairlist to be set.

### *get_pair_dataframe(pair, timeframe)*

``` python
# fetch live / historical candle (OHLCV) data for the first informative pair
inf_pair, inf_timeframe = self.informative_pairs()[0]
informative = self.dp.get_pair_dataframe(pair=inf_pair,
                                         timeframe=inf_timeframe)
```

!!! Warning "Warning about backtesting"
    In backtesting, `dp.get_pair_dataframe()` behavior differs depending on where it's called.
    Within `populate_*()` methods, `dp.get_pair_dataframe()` returns the full timerange. Please make sure to not "look into the future" to avoid surprises when running in dry/live mode.
    Within [callbacks](strategy-callbacks.md), you'll get the full timerange up to the current (simulated) candle.

### *get_analyzed_dataframe(pair, timeframe)*

This method is used by freqtrade internally to determine the last signal.
It can also be used in specific callbacks to get the signal that caused the action (see [Advanced Strategy Documentation](strategy-advanced.md) for more details on available callbacks).

``` python
# fetch current dataframe
dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=metadata['pair'],
                                                         timeframe=self.timeframe)
```

!!! Note "No data available"
    Returns an empty dataframe if the requested pair was not cached.
    You can check for this with `if dataframe.empty:` and handle this case accordingly.
    This should not happen when using whitelisted pairs.

### *orderbook(pair, maximum)*

``` python
if self.dp.runmode.value in ('live', 'dry_run'):
    ob = self.dp.orderbook(metadata['pair'], 1)
    dataframe['best_bid'] = ob['bids'][0][0]
    dataframe['best_ask'] = ob['asks'][0][0]
```

The orderbook structure is aligned with the order structure from [ccxt](https://github.com/ccxt/ccxt/wiki/Manual#order-book-structure), so the result will be formatted as follows:

``` js
{
    'bids': [
        [ price, amount ], // [ float, float ]
        [ price, amount ],
        ...
    ],
    'asks': [
        [ price, amount ],
        [ price, amount ],
        //...
    ],
    //...
}
```

Therefore, using `ob['bids'][0][0]` as demonstrated above will use the best bid price. `ob['bids'][0][1]` would look at the amount at this orderbook position.

!!! Warning "Warning about backtesting"
    The order book is not part of the historic data which means backtesting and hyperopt will not work correctly if this method is used, as the method will return up-to-date values.

### *ticker(pair)*

``` python
if self.dp.runmode.value in ('live', 'dry_run'):
    ticker = self.dp.ticker(metadata['pair'])
    dataframe['last_price'] = ticker['last']
    dataframe['volume24h'] = ticker['quoteVolume']
    dataframe['vwap'] = ticker['vwap']
```

!!! Warning
    Although the ticker data structure is a part of the ccxt Unified Interface, the values returned by this method can
    vary for different exchanges. For instance, many exchanges do not return `vwap` values, and some exchanges
    do not always fill in the `last` field (so it can be None), etc. So you need to carefully verify the ticker
    data returned from the exchange and add appropriate error handling / defaults.

!!! Warning "Warning about backtesting"
    This method will always return up-to-date / real-time values. As such, usage during backtesting / hyperopt without runmode checks will lead to wrong results, e.g. your whole dataframe will contain the same single value in all rows.

### Send Notification

The dataprovider `.send_msg()` function allows you to send custom notifications from your strategy.
Identical notifications will only be sent once per candle, unless the 2nd argument (`always_send`) is set to True.

``` python
    self.dp.send_msg(f"{metadata['pair']} just got hot!")

    # Force send this notification, avoid caching (Please read warning below!)
    self.dp.send_msg(f"{metadata['pair']} just got hot!", always_send=True)
```

Notifications will only be sent in trading modes (Live/Dry-run) - so this method can be called without conditions for backtesting.

!!! Warning "Spamming"
    You can spam yourself pretty good by setting `always_send=True` in this method. Use this with great care and only in conditions you know will not happen throughout a candle to avoid a message every 5 seconds.

### Complete DataProvider sample

```python
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame

class SampleStrategy(IStrategy):
    # strategy init stuff...

    timeframe = '5m'

    # more strategy init stuff..

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        # Optionally Add additional "static" pairs
        informative_pairs += [("ETH/USDT", "5m"),
                              ("BTC/TUSD", "15m"),
                            ]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        inf_tf = '1d'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        # Get the 14 day rsi
        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # Calculate rsi of the original dataframe (5m timeframe)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Do other stuff
        # ...

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
                (dataframe['rsi_1d'] < 30) &                     # Ensure daily RSI is < 30
                (dataframe['volume'] > 0)                        # Ensure this candle had volume (important for backtesting)
            ),
            ['enter_long', 'enter_tag']] = (1, 'rsi_cross')

```

***

## Additional data (Wallets)

The strategy provides access to the `wallets` object. This contains the current balances of your wallets/accounts on the exchange.

!!! Note "Backtesting / Hyperopt"
    Wallets behaves differently depending on the function from which it is called.
    Within `populate_*()` methods, it'll return the full wallet as configured.
    Within [callbacks](strategy-callbacks.md), you'll get the wallet state corresponding to the actual simulated wallet at that point in the simulation process.

Always check if `wallets` is available to avoid failures during backtesting.

``` python
if self.wallets:
    free_eth = self.wallets.get_free('ETH')
    used_eth = self.wallets.get_used('ETH')
    total_eth = self.wallets.get_total('ETH')
```

### Possible options for Wallets

- `get_free(asset)` - currently available balance to trade
- `get_used(asset)` - currently tied up balance (open orders)
- `get_total(asset)` - total available balance - sum of the 2 above

***

## Additional data (Trades)

A history of trades can be retrieved in the strategy by querying the database.

At the top of the file, import the required object:

```python
from freqtrade.persistence import Trade
```

The following example queries trades from today for the current pair (`metadata['pair']`). Other filters can easily be added.

``` python
trades = Trade.get_trades_proxy(pair=metadata['pair'],
                                open_date=datetime.now(timezone.utc) - timedelta(days=1),
                                is_open=False,
            ]).order_by(Trade.close_date).all()
# Summarize profit for this pair.
curdayprofit = sum(trade.close_profit for trade in trades)
```

For a full list of available methods, please consult the [Trade object](trade-object.md) documentation.

!!! Warning
    Trade history is not available in `populate_*` methods during backtesting or hyperopt, and will result in empty results.

## Prevent trades from happening for a specific pair

Freqtrade locks pairs automatically for the current candle (until that candle is over) when a pair exits, preventing an immediate re-entry of that pair.

This is to prevent "waterfalls" of many and frequent trades within a single candle.

Locked pairs will show the message `Pair <pair> is currently locked.`.

### Locking pairs from within the strategy

Sometimes it may be desired to lock a pair after certain events happen (e.g. multiple losing trades in a row).

Freqtrade has an easy method to do this from within the strategy, by calling `self.lock_pair(pair, until, [reason])`.
`until` must be a datetime object in the future, after which trading will be re-enabled for that pair, while `reason` is an optional string detailing why the pair was locked.

Locks can also be lifted manually, by calling `self.unlock_pair(pair)` or `self.unlock_reason(<reason>)`, providing the reason the pair was unlocked.
`self.unlock_reason(<reason>)` will unlock all pairs currently locked with the provided reason.

To verify if a pair is currently locked, use `self.is_pair_locked(pair)`.

!!! Note
    Locked pairs will always be rounded up to the next candle. So assuming a `5m` timeframe, a lock with `until` set to 10:18 will lock the pair until the candle from 10:15-10:20 will be finished.

!!! Warning
    Manually locking pairs is not available during backtesting. Only locks via Protections are allowed.

#### Pair locking example

``` python
from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone
# Put the above lines a the top of the strategy file, next to all the other imports
# --------

# Within populate indicators (or populate_entry_trend):
if self.config['runmode'].value in ('live', 'dry_run'):
    # fetch closed trades for the last 2 days
    trades = Trade.get_trades_proxy(
        pair=metadata['pair'], is_open=False, 
        open_date=datetime.now(timezone.utc) - timedelta(days=2))
    # Analyze the conditions you'd like to lock the pair .... will probably be different for every strategy
    sumprofit = sum(trade.close_profit for trade in trades)
    if sumprofit < 0:
        # Lock pair for 12 hours
        self.lock_pair(metadata['pair'], until=datetime.now(timezone.utc) + timedelta(hours=12))
```

## Print the main dataframe

To inspect the current main dataframe, you can issue a print-statement in either `populate_entry_trend()` or `populate_exit_trend()`.
You may also want to print the pair so it's clear what data is currently shown.

``` python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            #>> whatever condition<<<
        ),
        ['enter_long', 'enter_tag']] = (1, 'somestring')

    # Print the Analyzed pair
    print(f"result for {metadata['pair']}")

    # Inspect the last 5 rows
    print(dataframe.tail())

    return dataframe
```

Printing more than a few rows is also possible by using `print(dataframe)` instead of `print(dataframe.tail())`. However this is not recommended, as can results in a lot of output (~500 lines per pair every 5 seconds).

## Common mistakes when developing strategies

### Looking into the future while backtesting

Backtesting analyzes the whole dataframe timerange at once for performance reasons. Because of this, strategy authors need to make sure that strategies do not lookahead into the future, i.e. using data that would not be available in dry or live mode.

This is a common pain-point, which can cause huge differences between backtesting and dry/live run methods. Strategies that look into the future will perform well during backtesting, often with incredible profits or winrates, but will fail or perform badly in real conditions.

The following list contains some common patterns which should be avoided to prevent frustration:

- don't use `shift(-1)` or other negative values. This uses data from the future in backtesting, which is not available in dry or live modes.
- don't use `.iloc[-1]` or any other absolute position in the dataframe within `populate_` functions, as this will be different between dry-run and backtesting. Absolute `iloc` indexing is safe to use in callbacks however - see [Strategy Callbacks](strategy-callbacks.md).
- don't use functions that use all dataframe or column values, e.g. `dataframe['mean_volume'] = dataframe['volume'].mean()`. As backtesting uses the full dataframe, at any point in the dataframe, the `'mean_volume'` series would include data from the future. Use rolling() calculations instead, e.g. `dataframe['volume'].rolling(<window>).mean()`.
- don't use `.resample('1h')`. This uses the left border of the period interval, so moves data from an hour boundary to the start of the hour. Use `.resample('1h', label='right')` instead.

!!! Tip "Identifying problems"
    You should always use the two helper commands [lookahead-analysis](lookahead-analysis.md) and [recursive-analysis](recursive-analysis.md), which can each help you figure out problems with your strategy in different ways.
    Please treat them as what they are - helpers to identify most common problems. A negative result of each does not guarantee that there are none of the above errors included.

### Colliding signals

When conflicting signals collide (e.g. both `'enter_long'` and `'exit_long'` are set to `1`), freqtrade will do nothing and ignore the entry signal. This will avoid trades that enter, and exit immediately. Obviously, this can potentially lead to missed entries.

The following rules apply, and entry signals will be ignored if more than one of the 3 signals is set:

- `enter_long` -> `exit_long`, `enter_short`
- `enter_short` -> `exit_short`, `enter_long`

## Further strategy ideas

To get additional ideas for strategies, head over to the [strategy repository](https://github.com/freqtrade/freqtrade-strategies). Feel free to use them as examples, but results will depend on the current market situation, pairs used, etc. Therefore, these strategies should be considered only for learning purposes, not real world trading. Please backtest the strategy for your exchange/desired pairs first, then dry run to evaluate carefully, and use at your own risk.

Feel free to use any of them as inspiration for your own strategies. We're happy to accept Pull Requests containing new strategies to the repository.

## Next steps

Now you have a perfect strategy you probably want to backtest it.
Your next step is to learn [how to use backtesting](backtesting.md).
