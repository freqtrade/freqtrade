
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa

import random

# Update this variable if you change the class name
class_name = 'DefaultStrategy'


# This class is a sample. Feel free to customize it.




def Select():
    param = []
    random_items = []
    param.append(str('[' + 'uptrend_long_ema' + '[' + 'enabled' + ']'))
    param.append(str('[' + 'macd_below_zero' +  '][' + 'enabled' + ']'))
    param.append(str('[' + 'uptrend_short_ema' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'mfi' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'fastd' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'adx' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'rsi' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'over_sar' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'green_candle' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'uptrend_sma' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'closebb' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'temabb' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'fastdt' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'ao' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'ema3' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'macd' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'closesar' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'htsine' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'has' '][' + 'enabled'+ ']'))
    param.append(str('[' + 'plusdi' '][' + 'enabled'+ ']'))
    howmany = random.randint(1,20)
    random_items = random.choices(population=param, k=howmany)
    print(' ')
    print('The Parameters Enabled Are As Follows!!!:  ' + str(random_items))
    print(' ')
    return random_items




class DefaultStrategy(IStrategy):
    """
    This is a test strategy to inspire you.
    More information in https://github.com/gcarq/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "40": 0.0,
        "30": 0.01,
        "20": 0.02,
        "0": 0.04
    }

    ticker_interval = 5

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.10

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # Momentum Indicator
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)


        # Awesome oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # Commodity Channel Index: values Oversold:<-100, Overbought:>100
        dataframe['cci'] = ta.CCI(dataframe)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Minus Directional Indicator / Movement
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # Plus Directional Indicator / Movement
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # ROC
        dataframe['roc'] = ta.ROC(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # Stoch
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # Stoch RSI
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']


        # Overlap Studies
        # ------------------------------------

        """
        # Previous Bollinger bands
        # Because ta.BBANDS implementation is broken with small numbers, it actually
        # returns middle band for all the three bands. Switch to qtpylib.bollinger_bands
        # and use middle band instead.

        dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
        """

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']


        # EMA - Exponential Moving Average
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)


        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------

        # Hammer: values [0, 100]
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # Inverted Hammer: values [0, 100]
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # Dragonfly Doji: values [0, 100]
        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # Piercing Line: values [0, 100]
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # Morningstar: values [0, 100]
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # Three White Soldiers: values [0, 100]
        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]


        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------

        # Hanging Man: values [0, 100]
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # Shooting Star: values [0, 100]
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # Gravestone Doji: values [0, 100]
        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # Dark Cloud Cover: values [0, 100]
        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # Evening Doji Star: values [0, 100]
        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # Evening Star: values [0, 100]
        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)


        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------

        # Three Line Strike: values [0, -100, 100]
        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # Spinning Top: values [0, -100, 100]
        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # Engulfing: values [0, -100, 100]
        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # Harami: values [0, -100, 100]
        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # Three Outside Up/Down: values [0, -100, 100]
        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # Three Inside Up/Down: values [0, -100, 100]
        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]


        # Chart type
        # ------------------------------------

        # Heikinashi stategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']


        return dataframe

    params = Select()
    valm = random.randint(1,100)
    valfast = random.randint(1,100)
    valadx = random.randint(1,100)
    valrsi = random.randint(1,100)
    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:

        conditions = []
        # GUARDS AND TRENDS
        if 'uptrend_long_ema' in str(self.params):
            conditions.append(dataframe['ema50'] > dataframe['ema100'])
        if 'macd_below_zero' in str(self.params):
            conditions.append(dataframe['macd'] < 0)
        if 'uptrend_short_ema' in str(self.params):
            conditions.append(dataframe['ema5'] > dataframe['ema10'])
        if 'mfi' in str(self.params):
            print('MFI Value:  ' + str(self.valm))
            conditions.append(dataframe['mfi'] < self.valm)
        if 'fastd' in str(self.params):
            print('FASTD Value  :' + str(self.valfast))
            conditions.append(dataframe['fastd'] < self.valfast)
        if 'adx' in str(self.params):
            print('ADX Value  :' + str(self.valadx))
            conditions.append(dataframe['adx'] > self.valadx)
        if 'rsi' in str(self.params):
            print('RSI Value  :' + str(self.valrsi))
            conditions.append(dataframe['rsi'] < self.valrsi)
        if 'over_sar' in str(self.params):
            conditions.append(dataframe['close'] > dataframe['sar'])
        if 'green_candle' in str(self.params):
            conditions.append(dataframe['close'] > dataframe['open'])
        if 'uptrend_sma' in str(self.params):
            prevsma = dataframe['sma'].shift(1)
            conditions.append(dataframe['sma'] > prevsma)
        if 'closebb' in str(self.params):
            conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
        if 'temabb' in str(self.params):
            conditions.append(dataframe['tema'] < dataframe['bb_lowerband'])
        if 'fastdt' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['fastd'], 10.0))
        if 'ao' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['ao'], 0.0))
        if 'ema3' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['ema3'], dataframe['ema10']))
        if 'macd' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
        if 'closesar' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['close'], dataframe['sar']))
        if 'htsine' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['htleadsine'], dataframe['htsine']))
        if 'has' in str(self.params):
            conditions.append((qtpylib.crossed_above(dataframe['ha_close'], dataframe['ha_open'])) & (dataframe['ha_low'] == dataframe['ha_open']))
        if 'plusdi' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['plus_di'], dataframe['minus_di']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe

