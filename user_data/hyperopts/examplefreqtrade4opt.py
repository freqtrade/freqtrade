# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy  # noqa
import pandas  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Examplestrategy4opt(IHyperOpt):
    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['slowadx'] = ta.ADX(dataframe, 35)

        # Commodity Channel Index: values Oversold:<-100, Overbought:>100
        dataframe['cci'] = ta.CCI(dataframe)

        # Stoch
        stoch = ta.STOCHF(dataframe, 5)
        dataframe['fastd'] = stoch['fastd']
        dataframe['fastk'] = stoch['fastk']
        dataframe['fastk-previous'] = dataframe.fastk.shift(1)
        dataframe['fastd-previous'] = dataframe.fastd.shift(1)

        # Slow Stoch
        slowstoch = ta.STOCHF(dataframe, 50)
        dataframe['slowfastd'] = slowstoch['fastd']
        dataframe['slowfastk'] = slowstoch['fastk']
        dataframe['slowfastk-previous'] = dataframe.slowfastk.shift(1)
        dataframe['slowfastd-previous'] = dataframe.slowfastd.shift(1)

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)

        dataframe['mean-volume'] = dataframe['volume'].mean()

        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use.
            """
            conditions = []

            # GUARDS AND TRENDS
            if params.get('adx-enabled'):
                conditions.append(
                    (dataframe['adx'] > params['adx-value']) |
                    (dataframe['slowadx'] > params['slowadx-value'])
                )

            if params.get('cci-enabled'):
                conditions.append(dataframe['cci'] < params['cci-value'])

            if params.get('fastk-previous-enabled'):
                conditions.append(dataframe['fastk-previous'] < params['fastk-previous-value'])
            if params.get('fastd-previous-enabled'):
                conditions.append(dataframe['fastd-previous'] < params['fastd-previous-value'])

            if params.get('slowfastk-previous-enabled'):
                conditions.append(dataframe['slowfastk-previous'] <
                                  params['slowfastk-previous-value'])
            if params.get('slowfastd-previous-enabled'):
                conditions.append(dataframe['slowfastd-previous'] <
                                  params['slowfastd-previous-value'])

            if params.get('mean_volume-enabled'):
                conditions.append(dataframe['mean-volume'] < params['mean-volume-value'])

            if params.get('close-enabled'):
                conditions.append(dataframe['close'] < params['close-value'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'fast-previous':
                    conditions.append(dataframe['fastk-previous'] < dataframe['fastd-previous'])
                if params['trigger'] == 'fast':
                    conditions.append(dataframe['fastk'] > dataframe['fastd'])

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching buy strategy parameters.
        """
        return [
            Integer(0, 100, name='adx-value'),
            Integer(-150, -50, name='cci-value'),
            Integer(0, 50, name='fastk-previous-value'),
            Integer(0, 50, name='fastd-previous-value'),
            Integer(0, 50, name='slowfastk-previous-value'),
            Integer(0, 50, name='slowfastd-previous-value'),
            Integer(0.0, 1.5, name='mean-volume-value'),
            Integer(0.00000000, 0.00000500, name='close-value'),
            Categorical([True, False], name='adx-enabled'),
            Categorical([True, False], name='cci-enabled'),
            Categorical([True, False], name='fastk-previous-enabled'),
            Categorical([True, False], name='fastd-previous-enabled'),
            Categorical([True, False], name='slowfastk-previous-enabled'),
            Categorical([True, False], name='slowfastd-previous-enabled'),
            Categorical([True, False], name='mean-volume-enabled'),
            Categorical([True, False], name='close-enabled'),
            Categorical(['fast', 'fast-previous'], name='trigger')
        ]

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the sell strategy parameters to be used by Hyperopt.
        """
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Sell strategy Hyperopt will build and use.
            """
            conditions = []

            # GUARDS AND TRENDS
            if params.get('sell-slowadx-enabled'):
                conditions.append(dataframe['slowadx'] < dataframe['sell-slowadx-value'])
            if params.get('sell-fast-enabled'):
                conditions.append((dataframe['fastk'] > dataframe['sell-fastk-value'])
                                  | (dataframe['fastd'] > dataframe['sell-fastd-value']))

            # TRIGGERS
            if 'sell-trigger' in params:
                if params['sell-trigger'] == 'sell-fast-previous':
                    conditions.append(dataframe['fastk-previous'] < dataframe['fastd-previous'])
                if params['sell-trigger'] == 'sell-ema':
                    conditions.append(dataframe['close'] > dataframe['ema5'])

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell'] = 1

            return dataframe

        return populate_sell_trend

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching sell strategy parameters.
        """
        return [
            Integer(0, 50, name='sell-slowadx-value'),
            Integer(50, 100, name='sell-fastk-value'),
            Integer(50, 100, name='sell-fastd-value'),
            Categorical([True, False], name='sell-slowadx-enabled'),
            Categorical([True, False], name='sell-fast-enabled'),
            Categorical(['sell-ema', 'sell-fast-previous'], name='sell-trigger')
        ]

    @staticmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:
        """
        Generate the ROI table that will be used by Hyperopt
        This implementation generates the default legacy Freqtrade ROI tables.
        Change it if you need different number of steps in the generated
        ROI tables or other structure of the ROI tables.
        Please keep it aligned with parameters in the 'roi' optimization
        hyperspace defined by the roi_space method.
        """
        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    @staticmethod
    def roi_space() -> List[Dimension]:
        """
        Values to search for each ROI steps
        Override it if you need some different ranges for the parameters in the
        'roi' optimization hyperspace.
        Please keep it aligned with the implementation of the
        generate_roi_table method.
        """
        return [
            Integer(10, 120, name='roi_t1'),
            Integer(10, 60, name='roi_t2'),
            Integer(10, 40, name='roi_t3'),
            Real(0.01, 0.04, name='roi_p1'),
            Real(0.01, 0.07, name='roi_p2'),
            Real(0.01, 0.20, name='roi_p3'),
        ]

    @staticmethod
    def stoploss_space() -> List[Dimension]:
        """
        Stoploss Value to search
        Override it if you need some different range for the parameter in the
        'stoploss' optimization hyperspace.
        """
        return [
            Real(-0.35, -0.02, name='stoploss'),
        ]

    @staticmethod
    def trailing_space() -> List[Dimension]:
        """
        Create a trailing stoploss space.
        You may override it in your custom Hyperopt class.
        """
        return [
            # It was decided to always set trailing_stop is to True if the 'trailing' hyperspace
            # is used. Otherwise hyperopt will vary other parameters that won't have effect if
            # trailing_stop is set False.
            # This parameter is included into the hyperspace dimensions rather than assigning
            # it explicitly in the code in order to have it printed in the results along with
            # other 'trailing' hyperspace parameters.
            Categorical([True], name='trailing_stop'),

            Real(0.01, 0.35, name='trailing_stop_positive'),

            # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
            # so this intermediate parameter is used as the value of the difference between
            # them. The value of the 'trailing_stop_positive_offset' is constructed in the
            # generate_trailing_params() method.
            # This is similar to the hyperspace dimensions used for constructing the ROI tables.
            Real(0.001, 0.1, name='trailing_stop_positive_offset_p1'),

            Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (
                    (dataframe['adx'] > 50) |
                    (dataframe['slowadx'] > 26)
                ) &
                (dataframe['cci'] < -100) &
                (
                    (dataframe['fastk-previous'] < 20) &
                    (dataframe['fastd-previous'] < 20)
                ) &
                (
                    (dataframe['slowfastk-previous'] < 30) &
                    (dataframe['slowfastd-previous'] < 30)
                ) &
                (dataframe['fastk-previous'] < dataframe['fastd-previous']) &
                (dataframe['fastk'] > dataframe['fastd']) &
                (dataframe['mean-volume'] > 0.75) &
                (dataframe['close'] > 0.00000100)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['slowadx'] < 25) &
                ((dataframe['fastk'] > 70) | (dataframe['fastd'] > 70)) &
                (dataframe['fastk-previous'] < dataframe['fastd-previous']) &
                (dataframe['close'] > dataframe['ema5'])
            ),
            'sell'] = 1
        return dataframe
