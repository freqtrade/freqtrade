# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy   # noqa
import pandas   # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Examplestrategy1opt(IHyperOpt):
    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']

        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            conditions = []

            # GUARDS AND TRENDS
            if params.get('ema20-enabled'):
                conditions.append(dataframe['ha_close'] > dataframe['ema20'])
            if params.get('ema50-enabled'):
                conditions.append(dataframe['ha_close'] > dataframe['ema50'])
            if params.get('greenbar-enabled'):
                conditions.append(dataframe['ha_close'] > dataframe['ha_open'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'ema2050_crossabove':
                    conditions.append(qtpylib.crossed_above(dataframe['ema20'], dataframe['ema50']))
                if params['trigger'] == 'ema50100_crossabove':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['ema50'], dataframe['ema100']))
                if params['trigger'] == 'ema20100_crossabove':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['ema20'], dataframe['ema100']))

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
            Categorical([True, False], name='ema20-enabled'),
            Categorical([True, False], name='ema50-enabled'),
            Categorical([True, False], name='greenbar-enabled'),
            Categorical(['ema2050_crossabove', 'ema50100_crossabove',
                         'ema20100_crossabove'], name='trigger')
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
            if params.get('sell-ema20-enabled'):
                conditions.append(dataframe['ha_close'] < dataframe['ema20'])
            if params.get('sell-ema50-enabled'):
                conditions.append(dataframe['ha_close'] < dataframe['ema50'])
            if params.get('sell-redbar-enabled'):
                conditions.append(dataframe['ha_close'] < dataframe['ha_open'])

            # TRIGGERS
            if 'sell-trigger' in params:
                if params['sell-trigger'] == 'sell-ema2050_crossbelow':
                    conditions.append(qtpylib.crossed_below(dataframe['ema20'], dataframe['ema50']))
                if params['sell-trigger'] == 'sell-ema50100_crossbelow':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['ema50'], dataframe['ema100']))
                if params['sell-trigger'] == 'sell-ema20100_crossbelow':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['ema20'], dataframe['ema100']))
                if params['sell-trigger'] == 'sell-ema50100_crossabove':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['ema50'], dataframe['ema100']))

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
            Categorical([True, False], name='sell-ema20-enabled'),
            Categorical([True, False], name='sell-ema50-enabled'),
            Categorical([True, False], name='sell-redbar-enabled'),
            Categorical(['sell-ema2050_crossbelow',
                         'sell-ema50100_crossbelow',
                         'sell-ema20100_crossbelow',
                         'sell-ema50100_crossabove'], name='sell-trigger')
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
            Real(-0.5, -0.02, name='stoploss'),
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
                qtpylib.crossed_above(dataframe['ema20'], dataframe['ema50']) &
                (dataframe['ha_close'] > dataframe['ema20']) &
                (dataframe['ha_open'] < dataframe['ha_close'])  # green bar
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
                qtpylib.crossed_above(dataframe['ema50'], dataframe['ema100']) &
                (dataframe['ha_close'] < dataframe['ema20']) &
                (dataframe['ha_open'] > dataframe['ha_close'])  # red bar
            ),
            'sell'] = 1

        return dataframe
