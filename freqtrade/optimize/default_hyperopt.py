# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame
from typing import Dict, Any, Callable
from functools import reduce

import numpy
from hyperopt import hp

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.optimize.interface import IHyperOpt

class_name = 'DefaultHyperOpts'


class DefaultHyperOpts(IHyperOpt):
    """
    Default hyperopt provided by freqtrade bot.
    You can override it with your own hyperopt
    """

    @staticmethod
    def populate_indicators(dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        dataframe['cci'] = ta.CCI(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['roc'] = ta.ROC(dataframe)
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
        # SAR Parabolic
        dataframe['sar'] = ta.SAR(dataframe)
        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        """
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
        """

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        """
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
        """

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        """
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
        """

        # Chart type
        # ------------------------------------
        # Heikinashi stategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by hyperopt
        """
        def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use
            """
            conditions = []
            # GUARDS AND TRENDS
            if 'uptrend_long_ema' in params and params['uptrend_long_ema']['enabled']:
                conditions.append(dataframe['ema50'] > dataframe['ema100'])
            if 'macd_below_zero' in params and params['macd_below_zero']['enabled']:
                conditions.append(dataframe['macd'] < 0)
            if 'uptrend_short_ema' in params and params['uptrend_short_ema']['enabled']:
                conditions.append(dataframe['ema5'] > dataframe['ema10'])
            if 'mfi' in params and params['mfi']['enabled']:
                conditions.append(dataframe['mfi'] < params['mfi']['value'])
            if 'fastd' in params and params['fastd']['enabled']:
                conditions.append(dataframe['fastd'] < params['fastd']['value'])
            if 'adx' in params and params['adx']['enabled']:
                conditions.append(dataframe['adx'] > params['adx']['value'])
            if 'rsi' in params and params['rsi']['enabled']:
                conditions.append(dataframe['rsi'] < params['rsi']['value'])
            if 'over_sar' in params and params['over_sar']['enabled']:
                conditions.append(dataframe['close'] > dataframe['sar'])
            if 'green_candle' in params and params['green_candle']['enabled']:
                conditions.append(dataframe['close'] > dataframe['open'])
            if 'uptrend_sma' in params and params['uptrend_sma']['enabled']:
                prevsma = dataframe['sma'].shift(1)
                conditions.append(dataframe['sma'] > prevsma)

            # TRIGGERS
            triggers = {
                'lower_bb': (
                    dataframe['close'] < dataframe['bb_lowerband']
                ),
                'lower_bb_tema': (
                    dataframe['tema'] < dataframe['bb_lowerband']
                ),
                'faststoch10': (qtpylib.crossed_above(
                    dataframe['fastd'], 10.0
                )),
                'ao_cross_zero': (qtpylib.crossed_above(
                    dataframe['ao'], 0.0
                )),
                'ema3_cross_ema10': (qtpylib.crossed_above(
                    dataframe['ema3'], dataframe['ema10']
                )),
                'macd_cross_signal': (qtpylib.crossed_above(
                    dataframe['macd'], dataframe['macdsignal']
                )),
                'sar_reversal': (qtpylib.crossed_above(
                    dataframe['close'], dataframe['sar']
                )),
                'ht_sine': (qtpylib.crossed_above(
                    dataframe['htleadsine'], dataframe['htsine']
                )),
                'heiken_reversal_bull': (
                    (qtpylib.crossed_above(dataframe['ha_close'], dataframe['ha_open'])) &
                    (dataframe['ha_low'] == dataframe['ha_open'])
                ),
                'di_cross': (qtpylib.crossed_above(
                    dataframe['plus_di'], dataframe['minus_di']
                )),
            }
            conditions.append(triggers.get(params['trigger']['type']))

            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def indicator_space() -> Dict[str, Any]:
        """
        Define your Hyperopt space for searching strategy parameters
        """
        return {
            'macd_below_zero': hp.choice('macd_below_zero', [
                {'enabled': False},
                {'enabled': True}
            ]),
            'mfi': hp.choice('mfi', [
                {'enabled': False},
                {'enabled': True, 'value': hp.quniform('mfi-value', 10, 25, 5)}
            ]),
            'fastd': hp.choice('fastd', [
                {'enabled': False},
                {'enabled': True, 'value': hp.quniform('fastd-value', 15, 45, 5)}
            ]),
            'adx': hp.choice('adx', [
                {'enabled': False},
                {'enabled': True, 'value': hp.quniform('adx-value', 20, 50, 5)}
            ]),
            'rsi': hp.choice('rsi', [
                {'enabled': False},
                {'enabled': True, 'value': hp.quniform('rsi-value', 20, 40, 5)}
            ]),
            'uptrend_long_ema': hp.choice('uptrend_long_ema', [
                {'enabled': False},
                {'enabled': True}
            ]),
            'uptrend_short_ema': hp.choice('uptrend_short_ema', [
                {'enabled': False},
                {'enabled': True}
            ]),
            'over_sar': hp.choice('over_sar', [
                {'enabled': False},
                {'enabled': True}
            ]),
            'green_candle': hp.choice('green_candle', [
                {'enabled': False},
                {'enabled': True}
            ]),
            'uptrend_sma': hp.choice('uptrend_sma', [
                {'enabled': False},
                {'enabled': True}
            ]),
            'trigger': hp.choice('trigger', [
                {'type': 'lower_bb'},
                {'type': 'lower_bb_tema'},
                {'type': 'faststoch10'},
                {'type': 'ao_cross_zero'},
                {'type': 'ema3_cross_ema10'},
                {'type': 'macd_cross_signal'},
                {'type': 'sar_reversal'},
                {'type': 'ht_sine'},
                {'type': 'heiken_reversal_bull'},
                {'type': 'di_cross'},
            ]),
        }

    @staticmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:
        """
        Generate the ROI table that will be used by Hyperopt
        """
        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    @staticmethod
    def stoploss_space() -> Dict[str, Any]:
        """
        Stoploss Value to search
        """
        return {
            'stoploss': hp.quniform('stoploss', -0.5, -0.02, 0.02),
        }

    @staticmethod
    def roi_space() -> Dict[str, Any]:
        """
        Values to search for each ROI steps
        """
        return {
            'roi_t1': hp.quniform('roi_t1', 10, 120, 20),
            'roi_t2': hp.quniform('roi_t2', 10, 60, 15),
            'roi_t3': hp.quniform('roi_t3', 10, 40, 10),
            'roi_p1': hp.quniform('roi_p1', 0.01, 0.04, 0.01),
            'roi_p2': hp.quniform('roi_p2', 0.01, 0.07, 0.01),
            'roi_p3': hp.quniform('roi_p3', 0.01, 0.20, 0.01),
        }
