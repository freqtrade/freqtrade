# Bot Optimization
This page explains where to customize your strategies, and add new 
indicators. 

## Table of Contents
- [Install a custom strategy file](#install-a-custom-strategy-file)
- [Customize your strategy](#change-your-strategy)
- [Add more Indicator](#add-more-indicator)
- [Where is the default strategy](#where-is-the-default-strategy)

Since the version `0.16.0` the bot allows using custom strategy file.

## Install a custom strategy file
This is very simple. Copy paste your strategy file into the folder 
`user_data/strategies`.

Let assume you have a strategy file `awesome-strategy.py`:
1. Move your file into `user_data/strategies` (you should have `user_data/strategies/awesome-strategy.py`
2. Start the bot with the param `--strategy awesome-strategy` (the parameter is the name of the file without '.py') 

```bash
python3 ./freqtrade/main.py --strategy awesome_strategy
```

## Change your strategy
The bot includes a default strategy file. However, we recommend you to 
use your own file to not have to lose your parameters everytime the default
strategy file will be updated on Github. Put your custom strategy file
into the folder `user_data/strategies`.

A strategy file contains all the information needed to build a good strategy:
- Buy strategy rules
- Sell strategy rules
- Minimal ROI recommended
- Stoploss recommended
- Hyperopt parameter

The bot also include a sample strategy you can update: `user_data/strategies/test_strategy.py`.  
You can test it with the parameter: `--strategy test_strategy`
 
```bash
python3 ./freqtrade/main.py --strategy awesome_strategy
```

**For the following section we will use the [user_data/strategies/test_strategy.py](https://github.com/gcarq/freqtrade/blob/develop/user_data/strategies/test_strategy.py)
file as reference.**

### Buy strategy 
Edit the method `populate_buy_trend()` into your strategy file to 
update your buy strategy.

Sample from `user_data/strategies/test_strategy.py`:  
```python
def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the buy signal for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (
            (dataframe['adx'] > 30) &
            (dataframe['tema'] <= dataframe['blower']) &
            (dataframe['tema'] > dataframe['tema'].shift(1))
        ),
        'buy'] = 1

    return dataframe
```

### Sell strategy
Edit the method `populate_sell_trend()` into your strategy file to 
update your sell strategy.

Sample from `user_data/strategies/test_strategy.py`:  
```python
def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the sell signal for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (
            (dataframe['adx'] > 70) &
            (dataframe['tema'] > dataframe['blower']) &
            (dataframe['tema'] < dataframe['tema'].shift(1))
        ),
        'sell'] = 1
    return dataframe
```

## Add more Indicator
As you have seen, buy and sell strategies need indicators. You can add 
more indicators by extending the list contained in
the method `populate_indicators()` from your strategy file.

Sample:
```python
def populate_indicators(dataframe: DataFrame) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame
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

**Want more indicators example?**  
Look into the [user_data/strategies/test_strategy.py](https://github.com/gcarq/freqtrade/blob/develop/user_data/strategies/test_strategy.py).  
Then uncomment indicators you need.


### Where is the default strategy?
The default buy strategy is located in the file 
[freqtrade/default_strategy.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/strategy/default_strategy.py). 


## Next step
Now you have a perfect strategy you probably want to backtesting it. 
Your next step is to learn [How to use the Backtesting](https://github.com/gcarq/freqtrade/blob/develop/docs/backtesting.md).
