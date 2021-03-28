# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib


class hyper_ltcusdt_1h(IHyperOpt):
    """
    This is a Hyperopt template to get you started.

    More information in the documentation: https://www.freqtrade.io/en/latest/hyperopt/

    You should:
    - Add any lib you need to build your hyperopt.

    You must keep:
    - The prototypes for the methods: populate_indicators, indicator_space, buy_strategy_generator.

    The methods roi_space, generate_roi_table and stoploss_space are not required
    and are provided by default.
    However, you may override them if you need 'roi' and 'stoploss' spaces that
    differ from the defaults offered by Freqtrade.
    Sample implementation of these methods will be copied to `user_data/hyperopts` when
    creating the user-data directory using `freqtrade create-userdir --userdir user_data`,
    or is available online under the following URL:
    https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_advanced.py.
    """

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
            if params.get('ao-enabled'):
                conditions.append(dataframe['ao'] > params['ao-value'])
            if params.get('uo-enabled'):
                conditions.append(dataframe['uo'] > params['uo-value'])
            if params.get('sar_close-enabled'):
                conditions.append(dataframe['sar_close'] < params['sar_close-value'])
            if params.get('natr-enabled'):
                conditions.append(dataframe['natr'] > params['natr-value'])
            if params.get('angle_tsf_mid-enabled'):
                conditions.append(dataframe['angle_tsf_mid'] < params['angle_tsf_mid-value'])
            if params.get('angle_trend_mid-enabled'):
                conditions.append(dataframe['angle_trend_mid'] < params['angle_trend_mid-value'])
            if params.get('mfi-enabled'):
                conditions.append(params['mfi-value'] < dataframe['mfi'])
            if params.get('angle-enabled'):
                conditions.append(params['angle-value'] < dataframe['angle'])
            if params.get('angle_macdsignal-enabled'):
                conditions.append(params['angle_macdsignal-value'] < dataframe['angle_macdsignal'])
            if params.get('macdhist-enabled'):
                conditions.append(params['macdhist-value'] < dataframe['macdhist'])
            if params.get('macdsignal-enabled'):
                conditions.append(params['macdsignal-value'] < dataframe['macdsignal'])
            if params.get('macd-enabled'):
                conditions.append(params['macd-value'] < (dataframe['macd'] - dataframe['macdsignal']))

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'sar':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['sar'], dataframe['close']
                    ))
                if params['trigger'] == 'sine':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['leadsine'], dataframe['sine']
                    ))
                if params['trigger'] == 'phase':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['quadrature'], dataframe['inphase']
                    ))
                if params['trigger'] == 'angle_tsf_mid':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['angle_tsf_mid'], 0
                    ))

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
            Integer(-50, 35, name='ao-value'),
            Integer(-86, 70, name='sar_close-value'),
            Integer(-75, 72, name='angle_tsf_mid-value'),
            Integer(-73, 68, name='angle_trend_mid-value'),
            Integer(-2, 80, name='uo-value'),
            Real(0.36, 11, name='natr-value'),
            Integer(0, 100, name='mfi-value'),
            Integer(-81, 78, name='angle-value'),
            Integer(-38, 50, name='angle_macdsignal-value'),
            Real(-4, 5.3, name='macdhist-value'),
            Integer(-15, 11, name='macdsignal-value'),
            Real(-4, 5, name='macd-value'),
            Categorical([True, False], name='ao-enabled'),
            Categorical([True, False], name='angle_tsf_mid-enabled'),
            Categorical([True, False], name='sar_close-enabled'),
            Categorical([True, False], name='angle_trend_mid-enabled'),
            Categorical([True, False], name='uo-enabled'),
            Categorical([True, False], name='natr-enabled'),
            Categorical([True, False], name='mfi-enabled'),
            Categorical([True, False], name='angle-enabled'),
            Categorical([True, False], name='angle_macdsignal-enabled'),
            Categorical([True, False], name='macdhist-enabled'),
            Categorical([True, False], name='macdsignal-enabled'),
            Categorical([True, False], name='macd-enabled'),
            Categorical(['sar', 'sine', 'angle_tsf_mid', 'angle_trend_mid', 'phase'], name='trigger')
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
            if params.get('ao-enabled'):
                conditions.append(dataframe['ao'] > params['ao-value_sell'])
            if params.get('uo-enabled'):
                conditions.append(dataframe['uo'] > params['uo-value_sell'])
            if params.get('sar_close-enabled'):
                conditions.append(dataframe['sar_close'] < params['sar_close-value_sell'])
            if params.get('natr-enabled'):
                conditions.append(dataframe['natr'] > params['natr-value_sell'])
            if params.get('angle_tsf_mid-enabled'):
                conditions.append(dataframe['angle_tsf_mid'] < params['angle_tsf_mid-value_sell'])
            if params.get('angle_trend_mid-enabled'):
                conditions.append(dataframe['angle_trend_mid'] < params['angle_trend_mid-value_sell'])
            if params.get('mfi-enabled'):
                conditions.append(params['mfi-value_sell'] < dataframe['mfi'])
            if params.get('angle-enabled'):
                conditions.append(params['angle-value_sell'] < dataframe['angle'])
            if params.get('angle_macdsignal-enabled'):
                conditions.append(params['angle_macdsignal-value_sell'] < dataframe['angle_macdsignal'])
            if params.get('macdhist-enabled'):
                conditions.append(params['macdhist-value_sell'] < dataframe['macdhist'])
            if params.get('macdsignal-enabled'):
                conditions.append(params['macdsignal-value_sell'] < dataframe['macdsignal'])
            if params.get('macd-enabled'):
                conditions.append(params['macd-value_sell'] < (dataframe['macd'] - dataframe['macdsignal']))

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'sar':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['sar'], dataframe['close']
                    ))
                if params['trigger'] == 'sine':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['leadsine'], dataframe['sine']
                    ))
                if params['trigger'] == 'phase':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['quadrature'], dataframe['inphase']
                    ))
                if params['trigger'] == 'angle_trend_mid':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['angle_trend_mid'], 0
                    ))
                if params['trigger'] == 'angle_tsf_mid':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['angle_tsf_mid'], 0
                    ))


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
            Integer(-50, 35, name='ao-value_sell'),
            Integer(-86, 70, name='sar_close-value_sell'),
            Integer(-75, 72, name='angle_tsf_mid-value_sell'),
            Integer(-73, 68, name='angle_trend_mid-value_sell'),
            Integer(-2, 80, name='uo-value_sell'),
            Real(0.36, 11, name='natr-value_sell'),
            Integer(0, 100, name='mfi-value_sell'),
            Integer(-81, 78, name='angle-value_sell'),
            Integer(-38, 50, name='angle_macdsignal-value_sell'),
            Real(-4, 5.3, name='macdhist-value_sell'),
            Integer(-15, 11, name='macdsignal-value_sell'),
            Real(-4, 5, name='macd-value_sell'),
            Categorical([True, False], name='ao-enabled'),
            Categorical([True, False], name='angle_tsf_mid-enabled'),
            Categorical([True, False], name='sar_close-enabled'),
            Categorical([True, False], name='angle_trend_mid-enabled'),
            Categorical([True, False], name='uo-enabled'),
            Categorical([True, False], name='natr-enabled'),
            Categorical([True, False], name='mfi-enabled'),
            Categorical([True, False], name='angle-enabled'),
            Categorical([True, False], name='angle_macdsignal-enabled'),
            Categorical([True, False], name='macdhist-enabled'),
            Categorical([True, False], name='macdsignal-enabled'),
            Categorical([True, False], name='macd-enabled'),
            Categorical(['sar', 'sine', 'angle_tsf_mid', 'angle_trend_mid', 'phase'], name='trigger')
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
            Integer(60, 600, name='roi_t1'),
            Integer(300, 1000, name='roi_t2'),
            Integer(500, 1500, name='roi_t3'),
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
            Real(-0.3, -0.02, name='stoploss'),
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
