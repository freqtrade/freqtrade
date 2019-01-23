# Optimization

This page explains where to customize your strategies, and add new
indicators.

## Install a custom strategy file

This is very simple. Copy paste your strategy file into the folder
`user_data/strategies`.

Let assume you have a class called `AwesomeStrategy` in the file `awesome-strategy.py`:

1. Move your file into `user_data/strategies` (you should have `user_data/strategies/awesome-strategy.py`
2. Start the bot with the param `--strategy AwesomeStrategy` (the parameter is the class name)

```bash
python3 ./freqtrade/main.py --strategy AwesomeStrategy
```

## Change your strategy

The bot includes a default strategy file. However, we recommend you to
use your own file to not have to lose your parameters every time the default
strategy file will be updated on Github. Put your custom strategy file
into the folder `user_data/strategies`.

Best copy the test-strategy and modify this copy to avoid having bot-updates override your changes.
`cp  user_data/strategies/test_strategy.py user_data/strategies/awesome-strategy.py`

### Anatomy of a strategy

A strategy file contains all the information needed to build a good strategy:

- Indicators
- Buy strategy rules
- Sell strategy rules
- Minimal ROI recommended
- Stoploss strongly recommended

The bot also include a sample strategy called `TestStrategy` you can update: `user_data/strategies/test_strategy.py`.
You can test it with the parameter: `--strategy TestStrategy`

```bash
python3 ./freqtrade/main.py --strategy AwesomeStrategy
```

**For the following section we will use the [user_data/strategies/test_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/user_data/strategies/test_strategy.py)
file as reference.**

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
    Look into the [user_data/strategies/test_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/user_data/strategies/test_strategy.py).<br/>
    Then uncomment indicators you need.

### Buy signal rules

Edit the method `populate_buy_trend()` in your strategy file to update your buy strategy.

It's important to always return the dataframe without removing/modifying the columns `"open", "high", "low", "close", "volume"`, otherwise these fields would contain something unexpected.

This will method will also define a new column, `"buy"`, which needs to contain 1 for buys, and 0 for "no action".

Sample from `user_data/strategies/test_strategy.py`:

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
            (dataframe['adx'] > 30) &
            (dataframe['tema'] <= dataframe['bb_middleband']) &
            (dataframe['tema'] > dataframe['tema'].shift(1))
        ),
        'buy'] = 1

    return dataframe
```

### Sell signal rules

Edit the method `populate_sell_trend()` into your strategy file to update your sell strategy.
Please note that the sell-signal is only used if `use_sell_signal` is set to true in the configuration.

It's important to always return the dataframe without removing/modifying the columns `"open", "high", "low", "close", "volume"`, otherwise these fields would contain something unexpected.

This will method will also define a new column, `"sell"`, which needs to contain 1 for sells, and 0 for "no action".

Sample from `user_data/strategies/test_strategy.py`:

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
            (dataframe['adx'] > 70) &
            (dataframe['tema'] > dataframe['bb_middleband']) &
            (dataframe['tema'] < dataframe['tema'].shift(1))
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
If your exchange supports it, it's recommended to also set `"stoploss_on_exchange"` in the order dict, so your stoploss is on the exchange and cannot be missed for network-problems (or other problems).

For more information on order_types please look [here](https://github.com/freqtrade/freqtrade/blob/develop/docs/configuration.md#understand-order_types).

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

Storing information can be accomplished by crating a new dictionary within the strategy class.

The name of the variable can be choosen at will, but should be prefixed with `cust_` to avoid naming collisions with predefined strategy variables.

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

!!! Warning:
  The data is not persisted after a bot-restart (or config-reload). Also, the amount of data should be kept smallish (no DataFrames and such), otherwise the bot will start to consume a lot of memory and eventually run out of memory and crash.

!!! Note:
  If the data is pair-specific, make sure to use pair as one of the keys in the dictionary.

### Where is the default strategy?

The default buy strategy is located in the file
[freqtrade/default_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/strategy/default_strategy.py).

### Specify custom strategy location

If you want to use a strategy from a different folder you can pass `--strategy-path`

```bash
python3 ./freqtrade/main.py --strategy AwesomeStrategy --strategy-path /some/folder
```

### Further strategy ideas

To get additional Ideas for strategies, head over to our [strategy repository](https://github.com/freqtrade/freqtrade-strategies). Feel free to use them as they are - but results will depend on the current market situation, pairs used etc. - therefore please backtest the strategy for your exchange/desired pairs first, evaluate carefully, use at your own risk.
Feel free to use any of them as inspiration for your own strategies.
We're happy to accept Pull Requests containing new Strategies to that repo.

We also got a *strategy-sharing* channel in our [Slack community](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE) which is a great place to get and/or share ideas.

## Next step

Now you have a perfect strategy you probably want to backtest it.
Your next step is to learn [How to use the Backtesting](backtesting.md).
