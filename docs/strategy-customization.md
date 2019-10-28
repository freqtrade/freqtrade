# Optimization

This page explains where to customize your strategies, and add new
indicators.

## Install a custom strategy file

This is very simple. Copy paste your strategy file into the directory `user_data/strategies`.

Let assume you have a class called `AwesomeStrategy` in the file `awesome-strategy.py`:

1. Move your file into `user_data/strategies` (you should have `user_data/strategies/awesome-strategy.py`
2. Start the bot with the param `--strategy AwesomeStrategy` (the parameter is the class name)

```bash
freqtrade --strategy AwesomeStrategy
```

## Change your strategy

The bot includes a default strategy file. However, we recommend you to
use your own file to not have to lose your parameters every time the default
strategy file will be updated on Github. Put your custom strategy file
into the directory `user_data/strategies`.

Best copy the test-strategy and modify this copy to avoid having bot-updates override your changes.
`cp  user_data/strategies/sample_strategy.py user_data/strategies/awesome-strategy.py`

### Anatomy of a strategy

A strategy file contains all the information needed to build a good strategy:

- Indicators
- Buy strategy rules
- Sell strategy rules
- Minimal ROI recommended
- Stoploss strongly recommended

The bot also include a sample strategy called `SampleStrategy` you can update: `user_data/strategies/sample_strategy.py`.
You can test it with the parameter: `--strategy SampleStrategy`

Additionally, there is an attribute called `INTERFACE_VERSION`, which defines the version of the strategy interface the bot should use.
The current version is 2 - which is also the default when it's not set explicitly in the strategy.

Future versions will require this to be set.

```bash
freqtrade --strategy AwesomeStrategy
```

**For the following section we will use the [user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/user_data/strategies/sample_strategy.py)
file as reference.**

!!! Note Strategies and Backtesting
    To avoid problems and unexpected differences between Backtesting and dry/live modes, please be aware
    that during backtesting the full time-interval is passed to the `populate_*()` methods at once.
    It is therefore best to use vectorized operations (across the whole dataframe, not loops) and
    avoid index referencing (`df.iloc[-1]`), but instead use `df.shift()` to get to the previous candle.

!!! Warning Using future data
    Since backtesting passes the full time interval to the `populate_*()` methods, the strategy author
    needs to take care to avoid having the strategy utilize data from the future.
    Some common patterns for this are listed in the [Common Mistakes](#common-mistakes-when-developing-strategies) section of this document.

### Customize Indicators

Buy and sell strategies need indicators. You can add more indicators by extending the list contained in the method `populate_indicators()` from your strategy file.

You should only add the indicators used in either `populate_buy_trend()`, `populate_sell_trend()`, or to populate another indicator, otherwise performance may suffer.

It's important to always return the dataframe without removing/modifying the columns `"open", "high", "low", "close", "volume"`, otherwise these fields would contain something unexpected.

Sample:

```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame

    Performance Note: For the best performance be frugal on the number of indicators
    you are using. Let uncomment only the indicator you are using in your strategies
    or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
    :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
    :param metadata: Additional information, like the currently traded pair
    :return: a Dataframe with all mandatory indicators for the strategies
    """
    dataframe['sar'] = ta.SAR(dataframe)
    dataframe['adx'] = ta.ADX(dataframe)
    stoch = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch['fastd']
    dataframe['fastk'] = stoch['fastk']
    dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
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
    return dataframe
```

!!! Note "Want more indicator examples?"
    Look into the [user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/user_data/strategies/sample_strategy.py).
    Then uncomment indicators you need.

### Strategy startup period

Most indicators have an instable startup period, in which they are either not available, or the calculation is incorrect. This can lead to inconsistencies, since Freqtrade does not know how long this instable period should be.
To account for this, the strategy can be assigned the `startup_candle_count` attribute.
This should be set to the maximum number of candles that the strategy requires to calculate stable indicators.

In this example strategy, this should be set to 100 (`startup_candle_count = 100`), since the longest needed history is 100 candles.

``` python
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
```

By letting the bot know how much history is needed, backtest trades can start at the specified timerange during backtesting and hyperopt.

!!! Warning
    `startup_candle_count` should be below `ohlcv_candle_limit` (which is 500 for most exchanges) - since only this amount of candles will be available during Dry-Run/Live Trade operations.

#### Example

Let's try to backtest 1 month (January 2019) of 5m candles using the an example strategy with EMA100, as above.

``` bash
freqtrade backtesting --timerange 20190101-20190201 --ticker-interval 5m
```

Assuming `startup_candle_count` is set to 100, backtesting knows it needs 100 candles to generate valid buy signals. It will load data from `20190101 - (100 * 5m)` - which is ~2019-12-31 15:30:00.
If this data is available, indicators will be calculated with this extended timerange. The instable startup period (up to 2019-01-01 00:00:00) will then be removed before starting backtesting.

!!! Note
    If data for the startup period is not available, then the timerange will be adjusted to account for this startup period - so Backtesting would start at 2019-01-01 08:30:00.

### Buy signal rules

Edit the method `populate_buy_trend()` in your strategy file to update your buy strategy.

It's important to always return the dataframe without removing/modifying the columns `"open", "high", "low", "close", "volume"`, otherwise these fields would contain something unexpected.

This will method will also define a new column, `"buy"`, which needs to contain 1 for buys, and 0 for "no action".

Sample from `user_data/strategies/sample_strategy.py`:

```python
def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
        'buy'] = 1

    return dataframe
```

!!! Note
    Buying requires sellers to buy from - therefore volume needs to be > 0 (`dataframe['volume'] > 0`) to make sure that the bot does not buy/sell in no-activity periods.

### Sell signal rules

Edit the method `populate_sell_trend()` into your strategy file to update your sell strategy.
Please note that the sell-signal is only used if `use_sell_signal` is set to true in the configuration.

It's important to always return the dataframe without removing/modifying the columns `"open", "high", "low", "close", "volume"`, otherwise these fields would contain something unexpected.

This will method will also define a new column, `"sell"`, which needs to contain 1 for sells, and 0 for "no action".

Sample from `user_data/strategies/sample_strategy.py`:

```python
def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Based on TA indicators, populates the sell signal for the given dataframe
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
        'sell'] = 1
    return dataframe
```

### Minimal ROI

This dict defines the minimal Return On Investment (ROI) a trade should reach before selling, independent from the sell signal.

It is of the following format, with the dict key (left side of the colon) being the minutes passed since the trade opened, and the value (right side of the colon) being the percentage.

```python
minimal_roi = {
    "40": 0.0,
    "30": 0.01,
    "20": 0.02,
    "0": 0.04
}
```

The above configuration would therefore mean:

- Sell whenever 4% profit was reached
- Sell when 2% profit was reached (in effect after 20 minutes)
- Sell when 1% profit was reached (in effect after 30 minutes)
- Sell when trade is non-loosing (in effect after 40 minutes)

The calculation does include fees.

To disable ROI completely, set it to an insanely high number:

```python
minimal_roi = {
    "0": 100
}
```

While technically not completely disabled, this would sell once the trade reaches 10000% Profit.

### Stoploss

Setting a stoploss is highly recommended to protect your capital from strong moves against you.

Sample:

``` python
stoploss = -0.10
```

This would signify a stoploss of -10%.

For the full documentation on stoploss features, look at the dedicated [stoploss page](stoploss.md).

If your exchange supports it, it's recommended to also set `"stoploss_on_exchange"` in the order_types dictionary, so your stoploss is on the exchange and cannot be missed due to network problems, high load or other reasons.

For more information on order_types please look [here](configuration.md#understand-order_types).

### Ticker interval

This is the set of candles the bot should download and use for the analysis.
Common values are `"1m"`, `"5m"`, `"15m"`, `"1h"`, however all values supported by your exchange should work.

Please note that the same buy/sell signals may work with one interval, but not the other.
This setting is accessible within the strategy by using `self.ticker_interval`.

### Metadata dict

The metadata-dict (available for `populate_buy_trend`, `populate_sell_trend`, `populate_indicators`) contains additional information.
Currently this is `pair`, which can be accessed using `metadata['pair']` - and will return a pair in the format `XRP/BTC`.

The Metadata-dict should not be modified and does not persist information across multiple calls.
Instead, have a look at the section [Storing information](#Storing-information)

### Storing information

Storing information can be accomplished by creating a new dictionary within the strategy class.

The name of the variable can be chosen at will, but should be prefixed with `cust_` to avoid naming collisions with predefined strategy variables.

```python
class Awesomestrategy(IStrategy):
    # Create custom dictionary
    cust_info = {}
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Check if the entry already exists
        if "crosstime" in self.cust_info[metadata["pair"]:
            self.cust_info[metadata["pair"]["crosstime"] += 1
        else:
            self.cust_info[metadata["pair"]["crosstime"] = 1
```

!!! Warning
    The data is not persisted after a bot-restart (or config-reload). Also, the amount of data should be kept smallish (no DataFrames and such), otherwise the bot will start to consume a lot of memory and eventually run out of memory and crash.

!!! Note
    If the data is pair-specific, make sure to use pair as one of the keys in the dictionary.

### Additional data (DataProvider)

The strategy provides access to the `DataProvider`. This allows you to get additional data to use in your strategy.

All methods return `None` in case of failure (do not raise an exception).

Please always check the mode of operation to select the correct method to get data (samples see below).

#### Possible options for DataProvider

- `available_pairs` - Property with tuples listing cached pairs with their intervals (pair, interval).
- `ohlcv(pair, ticker_interval)` - Currently cached ticker data for the pair, returns DataFrame or empty DataFrame.
- `historic_ohlcv(pair, ticker_interval)` - Returns historical data stored on disk.
- `get_pair_dataframe(pair, ticker_interval)` - This is a universal method, which returns either historical data (for backtesting) or cached live data (for the Dry-Run and Live-Run modes).
- `orderbook(pair, maximum)` - Returns latest orderbook data for the pair, a dict with bids/asks with a total of `maximum` entries.
- `market(pair)` - Returns market data for the pair: fees, limits, precisions, activity flag, etc. See [ccxt documentation](https://github.com/ccxt/ccxt/wiki/Manual#markets) for more details on Market data structure.
- `runmode` - Property containing the current runmode.

#### Example: fetch live ohlcv / historic data for the first informative pair

``` python
if self.dp:
    inf_pair, inf_timeframe = self.informative_pairs()[0]
    informative = self.dp.get_pair_dataframe(pair=inf_pair,
                                             ticker_interval=inf_timeframe)
```

!!! Warning Warning about backtesting
    Be carefull when using dataprovider in backtesting. `historic_ohlcv()` (and `get_pair_dataframe()`
    for the backtesting runmode) provides the full time-range in one go,
    so please be aware of it and make sure to not "look into the future" to avoid surprises when running in dry/live mode).

!!! Warning Warning in hyperopt
    This option cannot currently be used during hyperopt.

#### Orderbook

``` python
if self.dp:
    if self.dp.runmode in ('live', 'dry_run'):
        ob = self.dp.orderbook(metadata['pair'], 1)
        dataframe['best_bid'] = ob['bids'][0][0]
        dataframe['best_ask'] = ob['asks'][0][0]
```

!!! Warning
    The order book is not part of the historic data which means backtesting and hyperopt will not work if this
    method is used.

#### Available Pairs

``` python
if self.dp:
    for pair, ticker in self.dp.available_pairs:
        print(f"available {pair}, {ticker}")
```

#### Get data for non-tradeable pairs

Data for additional, informative pairs (reference pairs) can be beneficial for some strategies.
Ohlcv data for these pairs will be downloaded as part of the regular whitelist refresh process and is available via `DataProvider` just as other pairs (see above).
These parts will **not** be traded unless they are also specified in the pair whitelist, or have been selected by Dynamic Whitelisting.

The pairs need to be specified as tuples in the format `("pair", "interval")`, with pair as the first and time interval as the second argument.

Sample:

``` python
def informative_pairs(self):
    return [("ETH/USDT", "5m"),
            ("BTC/TUSD", "15m"),
            ]
```

!!! Warning
    As these pairs will be refreshed as part of the regular whitelist refresh, it's best to keep this list short.
    All intervals and all pairs can be specified as long as they are available (and active) on the used exchange.
    It is however better to use resampling to longer time-intervals when possible
    to avoid hammering the exchange with too many requests and risk being blocked.

### Additional data (Wallets)

The strategy provides access to the `Wallets` object. This contains the current balances on the exchange.

!!! Note
    Wallets is not available during backtesting / hyperopt.

Please always check if `Wallets` is available to avoid failures during backtesting.

``` python
if self.wallets:
    free_eth = self.wallets.get_free('ETH')
    used_eth = self.wallets.get_used('ETH')
    total_eth = self.wallets.get_total('ETH')
```

#### Possible options for Wallets

- `get_free(asset)` - currently available balance to trade
- `get_used(asset)` - currently tied up balance (open orders)
- `get_total(asset)` - total available balance - sum of the 2 above

### Print created dataframe

To inspect the created dataframe, you can issue a print-statement in either `populate_buy_trend()` or `populate_sell_trend()`.
You may also want to print the pair so it's clear what data is currently shown.

``` python
def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            #>> whatever condition<<<
        ),
        'buy'] = 1

    # Print the Analyzed pair
    print(f"result for {metadata['pair']}")

    # Inspect the last 5 rows
    print(dataframe.tail())

    return dataframe
```

Printing more than a few rows is also possible (simply use  `print(dataframe)` instead of `print(dataframe.tail())`), however not recommended, as that will be very verbose (~500 lines per pair every 5 seconds).

### Where can i find a strategy template?

The strategy template is located in the file
[user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/user_data/strategies/sample_strategy.py).

### Specify custom strategy location

If you want to use a strategy from a different directory you can pass `--strategy-path`

```bash
freqtrade --strategy AwesomeStrategy --strategy-path /some/directory
```

### Common mistakes when developing strategies

Backtesting analyzes the whole time-range at once for performance reasons. Because of this, strategy authors need to make sure that strategies do not look-ahead into the future.
This is a common pain-point, which can cause huge differences between backtesting and dry/live run methods, since they all use data which is not available during dry/live runs, so these strategies will perform well during backtesting, but will fail / perform badly in real conditions.

The following lists some common patterns which should be avoided to prevent frustration:

- don't use `shift(-1)`. This uses data from the future, which is not available.
- don't use `.iloc[-1]` or any other absolute position in the dataframe, this will be different between dry-run and backtesting.
- don't use `dataframe['volume'].mean()`. This uses the full DataFrame for backtesting, including data from the future. Use `dataframe['volume'].rolling(<window>).mean()` instead
- don't use `.resample('1h')`. This uses the left border of the interval, so moves data from an hour to the start of the hour. Use `.resample('1h', label='right')` instead.

### Further strategy ideas

To get additional Ideas for strategies, head over to our [strategy repository](https://github.com/freqtrade/freqtrade-strategies). Feel free to use them as they are - but results will depend on the current market situation, pairs used etc. - therefore please backtest the strategy for your exchange/desired pairs first, evaluate carefully, use at your own risk.
Feel free to use any of them as inspiration for your own strategies.
We're happy to accept Pull Requests containing new Strategies to that repo.

We also got a *strategy-sharing* channel in our [Slack community](https://join.slack.com/t/highfrequencybot/shared_invite/enQtNjU5ODcwNjI1MDU3LTU1MTgxMjkzNmYxNWE1MDEzYzQ3YmU4N2MwZjUyNjJjODRkMDVkNjg4YTAyZGYzYzlhOTZiMTE4ZjQ4YzM0OGE) which is a great place to get and/or share ideas.

## Next step

Now you have a perfect strategy you probably want to backtest it.
Your next step is to learn [How to use the Backtesting](backtesting.md).
