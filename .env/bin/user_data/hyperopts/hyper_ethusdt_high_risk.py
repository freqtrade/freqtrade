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


class hyper_ethusdt_high_risk(IHyperOpt):
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
            if params.get('vwap_short-enabled'):
                conditions.append(params['vwap_short-value'] > dataframe['vwap_short'])
            if params.get('correl_h_l_3-enabled'):
                conditions.append(params['correl_h_l_3-value'] < dataframe['correl_h_l_3'])
            if params.get('cci_45-enabled'):
                conditions.append(params['cci_45-value'] > dataframe['cci_45'])
            if params.get('cci_30-enabled'):
                conditions.append(params['cci_30-value'] > dataframe['cci_30'])
            if params.get('natr-enabled'):
                conditions.append(params['natr-value'] < dataframe['natr'])
            if params.get('trend_line-enabled'):
                conditions.append(params['trend_line-value'] < (dataframe['trend_line'] - dataframe['close']))

            # TRIGGERS
            if 'trigger' in params:

                if 'trigger' in params:
                    if params['trigger'] == 'ao_cross':
                        conditions.append(qtpylib.crossed_above(
                            dataframe['ao'], 0
                        ))
                    if params['trigger'] == 'macd_cross_signal':
                        conditions.append(qtpylib.crossed_above(
                            dataframe['macd'], dataframe['macdsignal']
                        ))
                    if params['trigger'] == 'sar':
                        conditions.append(qtpylib.crossed_below(
                            dataframe['sar'], dataframe['close']))
                    if params['trigger'] == 'cci_30':
                        conditions.append(qtpylib.crossed_above(
                            dataframe['cci_30'], -50))
                    if params['trigger'] == 'cci_45':
                        conditions.append(qtpylib.crossed_above(
                            dataframe['cci_45'], -50))

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
            Real(0.0, 4.2, name='ao-value'),
            Integer(-139, 105, name='trend_line-value'),
            Real(-1, 1, name='correl_h_l_3-value'),
            Real(-1, 1, name='correl_h_l_30-value'),
            Real(-1, 1, name='correl_h_l_10-value'),
            Integer(-200, 200, name='cci_30-value'),
            Integer(-200, 200, name='cci_45-value'),
            Categorical([True, False], name='ao-enabled'),
            Categorical([True, False], name='correl_h_l_30-enabled'),
            Categorical([True, False], name='correl_h_l_10-enabled'),
            Categorical([True, False], name='cci_30-enabled'),
            Categorical([True, False], name='cci_45-enabled'),
            Categorical([True, False], name='trend_line-enabled'),
            Categorical([True, False], name='natr-enabled'),

            Categorical(
                ['ao_cross', 'macd_cross_signal', 'correl_h_l_10' 'cci_45', 'cci_30'],
                name='trigger')
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
            if params.get('correl_h_l_30-enabled'):
                conditions.append(params['correl_h_l_30-value_sell'] < dataframe['correl_h_l_30'])
            if params.get('correl_h_l_10-enabled'):
                conditions.append(params['correl_h_l_10-value_sell'] < dataframe['correl_h_l_10'])
            if params.get('correl_h_l_3-enabled'):
                conditions.append(params['correl_h_l_3-value_sell'] < dataframe['correl_h_l_3'])
            if params.get('cci_45-enabled'):
                conditions.append(params['cci_45-value_sell'] > dataframe['cci_45'])
            if params.get('cci_30-enabled'):
                conditions.append(params['cci_30-value_sell'] > dataframe['cci_30'])
            if params.get('natr-enabled'):
                conditions.append(params['natr-value_sell'] < dataframe['natr'])
            if params.get('trend_line-enabled'):
                conditions.append(params['trend_line-value_sell'] < (dataframe['trend_line'] - dataframe['close']))

            # TRIGGERS
            if 'trigger' in params:

                if params['trigger'] == 'ao_cross':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['ao'], 0
                    ))
                if params['trigger'] == 'macd_cross_signal':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['macd'], dataframe['macdsignal']
                    ))
                if params['trigger'] == 'sar':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['sar'], dataframe['close']))
                if params['trigger'] == 'cci_30':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['cci_30'], -50))
                if params['trigger'] == 'cci_45':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['cci_45'], -50))

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
            Real(0.0, 4.2, name='natr-value_sell'),
            Integer(-139, 105, name='trend_line-value_sell'),
            Real(-1, 1, name='correl_h_l_3-value_sell'),
            Real(-1, 1, name='correl_h_l_30-value_sell'),
            Real(-1, 1, name='correl_h_l_10-value_sell'),
            Integer(-200, 200, name='cci_30-value_sell'),
            Integer(-200, 200, name='cci_45-value_sell'),
            Categorical([True, False], name='correl_h_l_3-enabled'),
            Categorical([True, False], name='correl_h_l_30-enabled'),
            Categorical([True, False], name='correl_h_l_10-enabled'),
            Categorical([True, False], name='cci_30-enabled'),
            Categorical([True, False], name='cci_45-enabled'),
            Categorical([True, False], name='trend_line-enabled'),
            Categorical([True, False], name='natr-enabled'),

            Categorical(
                ['sar', 'macd_cross_signal', 'correl_h_l_10' 'cci_45', 'cci_30'],
                name='trigger')
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
            Real(-0.25, -0.02, name='stoploss'),
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
