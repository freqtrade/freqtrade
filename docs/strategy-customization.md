# Strategy Customization

This page explains how to customize your strategies, add new indicators and set up trading rules.

Please familiarize yourself with [Freqtrade basics](bot-basics.md) first, which provides overall info on how the bot operates.

## Install a custom strategy file

This is very simple. Copy paste your strategy file into the directory `user_data/strategies`.

Let assume you have a class called `AwesomeStrategy` in the file `AwesomeStrategy.py`:

1. Move your file into `user_data/strategies` (you should have `user_data/strategies/AwesomeStrategy.py`
2. Start the bot with the param `--strategy AwesomeStrategy` (the parameter is the class name)

```bash
freqtrade trade --strategy AwesomeStrategy
```

## Develop your own strategy

The bot includes a default strategy file.
Also, several other strategies are available in the [strategy repository](https://github.com/freqtrade/freqtrade-strategies).

You will however most likely have your own idea for a strategy.
This document intends to help you develop one for yourself.

To get started, use `freqtrade new-strategy --strategy AwesomeStrategy`.
This will create a new strategy file from a template, which will be located under `user_data/strategies/AwesomeStrategy.py`.

!!! Note
    This is just a template file, which will most likely not be profitable out of the box.

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
freqtrade trade --strategy AwesomeStrategy
```

**For the following section we will use the [user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_strategy.py)
file as reference.**

!!! Note "Strategies and Backtesting"
    To avoid problems and unexpected differences between Backtesting and dry/live modes, please be aware
    that during backtesting the full time range is passed to the `populate_*()` methods at once.
    It is therefore best to use vectorized operations (across the whole dataframe, not loops) and
    avoid index referencing (`df.iloc[-1]`), but instead use `df.shift()` to get to the previous candle.

!!! Warning "Warning: Using future data"
    Since backtesting passes the full time range to the `populate_*()` methods, the strategy author
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
    :param dataframe: Dataframe with data from the exchange
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
    Look into the [user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_strategy.py).
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

Let's try to backtest 1 month (January 2019) of 5m candles using an example strategy with EMA100, as above.

``` bash
freqtrade backtesting --timerange 20190101-20190201 --timeframe 5m
```

Assuming `startup_candle_count` is set to 100, backtesting knows it needs 100 candles to generate valid buy signals. It will load data from `20190101 - (100 * 5m)` - which is ~2018-12-31 15:30:00.
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

To use times based on candle duration (timeframe), the following snippet can be handy.
This will allow you to change the timeframe for the strategy, and ROI times will still be set as candles (e.g. after 3 candles ...)

``` python
from freqtrade.exchange import timeframe_to_minutes

class AwesomeStrategy(IStrategy):

    timeframe = "1d"
    timeframe_mins = timeframe_to_minutes(timeframe)
    minimal_roi = {
        "0": 0.05,                             # 5% for the first 3 candles
        str(timeframe_mins * 3)): 0.02,  # 2% after 3 candles
        str(timeframe_mins * 6)): 0.01,  # 1% After 6 candles
    }
```

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

### Timeframe (formerly ticker interval)

This is the set of candles the bot should download and use for the analysis.
Common values are `"1m"`, `"5m"`, `"15m"`, `"1h"`, however all values supported by your exchange should work.

Please note that the same buy/sell signals may work well with one timeframe, but not with the others.

This setting is accessible within the strategy methods as the `self.timeframe` attribute.

### Metadata dict

The metadata-dict (available for `populate_buy_trend`, `populate_sell_trend`, `populate_indicators`) contains additional information.
Currently this is `pair`, which can be accessed using `metadata['pair']` - and will return a pair in the format `XRP/BTC`.

The Metadata-dict should not be modified and does not persist information across multiple calls.
Instead, have a look at the section [Storing information](strategy-advanced.md#Storing-information)

## Additional data (informative_pairs)

### Get data for non-tradeable pairs

Data for additional, informative pairs (reference pairs) can be beneficial for some strategies.
OHLCV data for these pairs will be downloaded as part of the regular whitelist refresh process and is available via `DataProvider` just as other pairs (see below).
These parts will **not** be traded unless they are also specified in the pair whitelist, or have been selected by Dynamic Whitelisting.

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

***

## Additional data (DataProvider)

The strategy provides access to the `DataProvider`. This allows you to get additional data to use in your strategy.

All methods return `None` in case of failure (do not raise an exception).

Please always check the mode of operation to select the correct method to get data (samples see below).

!!! Warning "Hyperopt"
    Dataprovider is available during hyperopt, however it can only be used in `populate_indicators()` within a strategy.
    It is not available in `populate_buy()` and `populate_sell()` methods, nor in `populate_indicators()`, if this method located in the hyperopt file.

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
if self.dp:
    for pair, timeframe in self.dp.available_pairs:
        print(f"available {pair}, {timeframe}")
```

### *current_whitelist()*

Imagine you've developed a strategy that trades the `5m` timeframe using signals generated from a `1d` timeframe on the top 10 volume pairs by volume.

The strategy might look something like this:

*Scan through the top 10 pairs by volume using the `VolumePairList` every 5 minutes and use a 14 day RSI to buy and sell.*

Due to the limited available data, it's very difficult to resample our `5m` candles into daily candles for use in a 14 day RSI. Most exchanges limit us to just 500 candles which effectively gives us around 1.74 daily candles. We need 14 days at least!

Since we can't resample our data we will have to use an informative pair; and since our whitelist will be dynamic we don't know which pair(s) to use.

This is where calling `self.dp.current_whitelist()` comes in handy.

```python
    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        return informative_pairs
```

### *get_pair_dataframe(pair, timeframe)*

``` python
# fetch live / historical candle (OHLCV) data for the first informative pair
if self.dp:
    inf_pair, inf_timeframe = self.informative_pairs()[0]
    informative = self.dp.get_pair_dataframe(pair=inf_pair,
                                             timeframe=inf_timeframe)
```

!!! Warning "Warning about backtesting"
    Be careful when using dataprovider in backtesting. `historic_ohlcv()` (and `get_pair_dataframe()`
    for the backtesting runmode) provides the full time-range in one go,
    so please be aware of it and make sure to not "look into the future" to avoid surprises when running in dry/live mode.

### *get_analyzed_dataframe(pair, timeframe)*

This method is used by freqtrade internally to determine the last signal.
It can also be used in specific callbacks to get the signal that caused the action (see [Advanced Strategy Documentation](strategy-advanced.md) for more details on available callbacks).

``` python
# fetch current dataframe
if self.dp:
    if self.dp.runmode.value in ('live', 'dry_run'):
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=metadata['pair'],
                                                                 timeframe=self.timeframe)
```

!!! Note "No data available"
    Returns an empty dataframe if the requested pair was not cached.
    This should not happen when using whitelisted pairs.


!!! Warning "Warning about backtesting"
    This method will return an empty dataframe during backtesting.

### *orderbook(pair, maximum)*

``` python
if self.dp:
    if self.dp.runmode.value in ('live', 'dry_run'):
        ob = self.dp.orderbook(metadata['pair'], 1)
        dataframe['best_bid'] = ob['bids'][0][0]
        dataframe['best_ask'] = ob['asks'][0][0]
```

The orderbook structure is aligned with the order structure from [ccxt](https://github.com/ccxt/ccxt/wiki/Manual#order-book-structure), so the result will look as follows:

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

Therefore, using `ob['bids'][0][0]` as demonstrated above will result in using the best bid price. `ob['bids'][0][1]` would look at the amount at this orderbook position.

!!! Warning "Warning about backtesting"
    The order book is not part of the historic data which means backtesting and hyperopt will not work correctly if this method is used, as the method will return uptodate values.

### *ticker(pair)*

``` python
if self.dp:
    if self.dp.runmode.value in ('live', 'dry_run'):
        ticker = self.dp.ticker(metadata['pair'])
        dataframe['last_price'] = ticker['last']
        dataframe['volume24h'] = ticker['quoteVolume']
        dataframe['vwap'] = ticker['vwap']
```

!!! Warning
    Although the ticker data structure is a part of the ccxt Unified Interface, the values returned by this method can
    vary for different exchanges. For instance, many exchanges do not return `vwap` values, the FTX exchange
    does not always fills in the `last` field (so it can be None), etc. So you need to carefully verify the ticker
    data returned from the exchange and add appropriate error handling / defaults.

!!! Warning "Warning about backtesting"
    This method will always return up-to-date values - so usage during backtesting / hyperopt will lead to wrong results.

### Complete Data-provider sample

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

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
                (dataframe['rsi_1d'] < 30) &                     # Ensure daily RSI is < 30
                (dataframe['volume'] > 0)                        # Ensure this candle had volume (important for backtesting)
            ),
            'buy'] = 1

```

***

## Helper functions

### *merge_informative_pair()*

This method helps you merge an informative pair to a regular dataframe without lookahead bias.
It's there to help you merge the dataframe in a safe and consistent way.

Options:

- Rename the columns for you to create unique columns
- Merge the dataframe without lookahead bias
- Forward-fill (optional)

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
    Using informative timeframes smaller than the dataframe timeframe is not recommended with this method, as it will not use any of the additional information this would provide.
    To use the more detailed information properly, more advanced methods should be applied (which are out of scope for freqtrade documentation, as it'll depend on the respective need).

***

### *stoploss_from_open()*

Stoploss values returned from `custom_stoploss` must specify a percentage relative to `current_rate`, but sometimes you may want to specify a stoploss relative to the open price instead. `stoploss_from_open()` is a helper function to calculate a stoploss value that can be returned from `custom_stoploss` which will be equivalent to the desired percentage above the open price.

??? Example "Returning a stoploss relative to the open price from the custom stoploss function"

    Say the open price was $100, and `current_price` is $121 (`current_profit` will be `0.21`).  

    If we want a stop price at 7% above the open price we can call `stoploss_from_open(0.07, current_profit)` which will return `0.1157024793`.  11.57% below $121 is $107, which is the same as 7% above $100.


    ``` python

    from datetime import datetime
    from freqtrade.persistence import Trade
    from freqtrade.strategy import IStrategy, stoploss_from_open

    class AwesomeStrategy(IStrategy):

        # ... populate_* methods

        use_custom_stoploss = True

        def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                            current_rate: float, current_profit: float, **kwargs) -> float:

            # once the profit has risin above 10%, keep the stoploss at 7% above the open price
            if current_profit > 0.10:
                return stoploss_from_open(0.07, current_profit)

            return 1

    ```

    Full examples can be found in the [Custom stoploss](strategy-advanced.md#custom-stoploss) section of the Documentation.


## Additional data (Wallets)

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

### Possible options for Wallets

- `get_free(asset)` - currently available balance to trade
- `get_used(asset)` - currently tied up balance (open orders)
- `get_total(asset)` - total available balance - sum of the 2 above

***

## Additional data (Trades)

A history of Trades can be retrieved in the strategy by querying the database.

At the top of the file, import Trade.

```python
from freqtrade.persistence import Trade
```

The following example queries for the current pair and trades from today, however other filters can easily be added.

``` python
if self.config['runmode'].value in ('live', 'dry_run'):
    trades = Trade.get_trades([Trade.pair == metadata['pair'],
                               Trade.open_date > datetime.utcnow() - timedelta(days=1),
                               Trade.is_open.is_(False),
                ]).order_by(Trade.close_date).all()
    # Summarize profit for this pair.
    curdayprofit = sum(trade.close_profit for trade in trades)
```

Get amount of stake_currency currently invested in Trades:

``` python
if self.config['runmode'].value in ('live', 'dry_run'):
    total_stakes = Trade.total_open_trades_stakes()
```

Retrieve performance per pair.
Returns a List of dicts per pair.

``` python
if self.config['runmode'].value in ('live', 'dry_run'):
    performance = Trade.get_overall_performance()
```

Sample return value: ETH/BTC had 5 trades, with a total profit of 1.5% (ratio of 0.015).

``` json
{'pair': "ETH/BTC", 'profit': 0.015, 'count': 5}
```

!!! Warning
    Trade history is not available during backtesting or hyperopt.

## Prevent trades from happening for a specific pair

Freqtrade locks pairs automatically for the current candle (until that candle is over) when a pair is sold, preventing an immediate re-buy of that pair.

Locked pairs will show the message `Pair <pair> is currently locked.`.

### Locking pairs from within the strategy

Sometimes it may be desired to lock a pair after certain events happen (e.g. multiple losing trades in a row).

Freqtrade has an easy method to do this from within the strategy, by calling `self.lock_pair(pair, until, [reason])`.
`until` must be a datetime object in the future, after which trading will be re-enabled for that pair, while `reason` is an optional string detailing why the pair was locked.

Locks can also be lifted manually, by calling `self.unlock_pair(pair)`.

To verify if a pair is currently locked, use `self.is_pair_locked(pair)`.

!!! Note
    Locked pairs will always be rounded up to the next candle. So assuming a `5m` timeframe, a lock with `until` set to 10:18 will lock the pair until the candle from 10:15-10:20 will be finished.

!!! Warning
    Manually locking pairs is not available during backtesting, only locks via Protections are allowed.

#### Pair locking example

``` python
from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone
# Put the above lines a the top of the strategy file, next to all the other imports
# --------

# Within populate indicators (or populate_buy):
if self.config['runmode'].value in ('live', 'dry_run'):
   # fetch closed trades for the last 2 days
    trades = Trade.get_trades([Trade.pair == metadata['pair'],
                               Trade.open_date > datetime.utcnow() - timedelta(days=2),
                               Trade.is_open.is_(False),
                ]).all()
    # Analyze the conditions you'd like to lock the pair .... will probably be different for every strategy
    sumprofit = sum(trade.close_profit for trade in trades)
    if sumprofit < 0:
        # Lock pair for 12 hours
        self.lock_pair(metadata['pair'], until=datetime.now(timezone.utc) + timedelta(hours=12))
```

## Print created dataframe

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

## Common mistakes when developing strategies

Backtesting analyzes the whole time-range at once for performance reasons. Because of this, strategy authors need to make sure that strategies do not look-ahead into the future.
This is a common pain-point, which can cause huge differences between backtesting and dry/live run methods, since they all use data which is not available during dry/live runs, so these strategies will perform well during backtesting, but will fail / perform badly in real conditions.

The following lists some common patterns which should be avoided to prevent frustration:

- don't use `shift(-1)`. This uses data from the future, which is not available.
- don't use `.iloc[-1]` or any other absolute position in the dataframe, this will be different between dry-run and backtesting.
- don't use `dataframe['volume'].mean()`. This uses the full DataFrame for backtesting, including data from the future. Use `dataframe['volume'].rolling(<window>).mean()` instead
- don't use `.resample('1h')`. This uses the left border of the interval, so moves data from an hour to the start of the hour. Use `.resample('1h', label='right')` instead.

## Further strategy ideas

To get additional Ideas for strategies, head over to our [strategy repository](https://github.com/freqtrade/freqtrade-strategies). Feel free to use them as they are - but results will depend on the current market situation, pairs used etc. - therefore please backtest the strategy for your exchange/desired pairs first, evaluate carefully, use at your own risk.
Feel free to use any of them as inspiration for your own strategies.
We're happy to accept Pull Requests containing new Strategies to that repo.

## Next step

Now you have a perfect strategy you probably want to backtest it.
Your next step is to learn [How to use the Backtesting](backtesting.md).
