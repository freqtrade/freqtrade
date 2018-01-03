# Bot Optimization
This page explains where to customize your strategies, and add new 
indicators. 

## Table of Contents
- [Change your strategy](#change-your-strategy)
- [Add more Indicator](#add-more-indicator)

## Change your strategy
The bot is using buy and sell strategies to buy and sell your trades. 
Both are customizable.

### Buy strategy
The default buy strategy is located in the file 
[freqtrade/analyze.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/analyze.py#L73-L92). 
Edit the function `populate_buy_trend()` to update your buy strategy.

Sample:
```python
def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the buy signal for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (
            (dataframe['rsi'] < 35) &
            (dataframe['fastd'] < 35) &
            (dataframe['adx'] > 30) &
            (dataframe['plus_di'] > 0.5)
        ) |
        (
            (dataframe['adx'] > 65) &
            (dataframe['plus_di'] > 0.5)
        ),
        'buy'] = 1

    return dataframe
```

### Sell strategy
The default buy strategy is located in the file 
[freqtrade/analyze.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/analyze.py#L95-L115)
Edit the function `populate_sell_trend()` to update your buy strategy.

Sample:
```python
def populate_sell_trend(dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the sell signal for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (
            (
                (crossed_above(dataframe['rsi'], 70)) |
                (crossed_above(dataframe['fastd'], 70))
            ) &
            (dataframe['adx'] > 10) &
            (dataframe['minus_di'] > 0)
        ) |
        (
            (dataframe['adx'] > 70) &
            (dataframe['minus_di'] > 0.5)
        ),
        'sell'] = 1
    return dataframe
```

## Add more Indicator
As you have seen, buy and sell strategies need indicators. You can see 
the indicators in the file 
[freqtrade/analyze.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/analyze.py#L95-L115).
Of course you can add more indicators by extending the list contained in
the function `populate_indicators()`.

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


## Next step
Now you have a perfect strategy you probably want to backtesting it. 
Your next step is to learn [How to use ](https://github.com/gcarq/freqtrade/blob/develop/docs/backtesting.md).
