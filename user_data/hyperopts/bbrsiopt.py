# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy   # noqa
import pandas  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib


class BBRSIopt(IHyperOpt):
    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Bollinger Bands stds 1-4
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_middleband'] = bollinger1['mid']
        dataframe['bb_upperband1'] = bollinger1['upper']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_upperband3'] = bollinger3['upper']

        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband4'] = bollinger4['lower']
        dataframe['bb_upperband4'] = bollinger4['upper']

        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            conditions = []

            # GUARDS AND TRENDS
            if params.get('rsi-enabled'):
                conditions.append(dataframe['rsi'] > params['rsi-value'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'bb_lower1':
                    conditions.append(dataframe['close'] < dataframe['bb_lowerband1'])
                if params['trigger'] == 'bb_lower2':
                    conditions.append(dataframe['close'] < dataframe['bb_lowerband2'])
                if params['trigger'] == 'bb_lower3':
                    conditions.append(dataframe['close'] < dataframe['bb_lowerband3'])
                if params['trigger'] == 'bb_lower4':
                    conditions.append(dataframe['close'] < dataframe['bb_lowerband4'])

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

        return [
            Integer(5, 50, name='rsi-value'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical(['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4'], name='trigger')
        ]

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            conditions = []

            # GUARDS AND TRENDS
            if params.get('sell-rsi-enabled'):
                conditions.append(dataframe['rsi'] > params['sell-rsi-value'])

            # TRIGGERS
            if 'sell-trigger' in params:
                # All bollinger bands with stds 1-4
                if params['sell-trigger'] == 'sell-bb_lower3':
                    conditions.append(dataframe['close'] > dataframe['bb_lowerband3'])
                if params['sell-trigger'] == 'sell-bb_lower2':
                    conditions.append(dataframe['close'] > dataframe['bb_lowerband2'])
                if params['sell-trigger'] == 'sell-bb_lower1':
                    conditions.append(dataframe['close'] > dataframe['bb_lowerband1'])
                if params['sell-trigger'] == 'sell-bb_middle':
                    conditions.append(dataframe['close'] > dataframe['bb_middleband'])
                if params['sell-trigger'] == 'sell-bb_upper1':
                    conditions.append(dataframe['close'] > dataframe['bb_upperband1'])
                if params['sell-trigger'] == 'sell-bb_upper2':
                    conditions.append(dataframe['close'] > dataframe['bb_upperband2'])
                if params['sell-trigger'] == 'sell-bb_upper3':
                    conditions.append(dataframe['close'] > dataframe['bb_upperband3'])
                if params['sell-trigger'] == 'sell-bb_upper4':
                    conditions.append(dataframe['close'] > dataframe['bb_upperband4'])

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

        return [
            Integer(5, 100, name='sell-rsi-value'),
            Categorical([True, False], name='sell-rsi-enabled'),
            Categorical(['sell-bb_lower3',
                         'sell-bb_lower2',
                         'sell-bb_lower1',
                         'sell-bb_middle',
                         'sell-bb_upper1',
                         'sell-bb_upper2',
                         'sell-bb_upper3',
                         'sell-bb_upper4', ], name='sell-trigger')
        ]

    @staticmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:
        # did not touch
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
        # did not touch
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
        return [
            Real(-0.50, -0.02, name='stoploss'),
        ]

    @staticmethod
    def trailing_space() -> List[Dimension]:
        # did not touch
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
        # copy paste from bbrsi.py
        dataframe.loc[
            (
                (dataframe['rsi'] > 30) &  # RSI above 30
                (dataframe['close'] < dataframe['bb_lowerband'])  # close price under low bb
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # copy paste from bbrsi.py
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middleband'])  # close price above the middle bb
            ),
            'sell'] = 1
        return dataframe
