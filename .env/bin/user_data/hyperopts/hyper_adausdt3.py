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


class hyper_adausdt3(IHyperOpt):
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
            if params.get('aroon-enabled'):
                conditions.append((dataframe['aroon_up'] - dataframe['aroon_down']) > params['aroon-value'])
            if params.get('rsi-enabled'):
                conditions.append(dataframe['rsi'] > params['rsi-value'])
            if params.get('ul-enabled'):
                conditions.append(dataframe['ul'] > params['ul-value'])
            if params.get('stoch_ratio-enabled'):
                conditions.append((dataframe['slowk'] - dataframe['slowd']) > params['stoch_ratio-value'])
            if params.get('stoch-enabled'):
                conditions.append(dataframe['slowk'] > params['stoch-value'])
            if params.get('ht_phase-enabled'):
                conditions.append(dataframe['ht_phase'] < params['ht_phase-value'])
            if params.get('inphase-enabled'):
                conditions.append(dataframe['inphase'] > params['inphase-value'])
            if params.get('quadrature-enabled'):
                conditions.append(dataframe['quadrature'] > params['quadrature-value'])
            if params.get('ht_trendmode-enabled'):
                conditions.append(dataframe['ht_trendmode'] > params['ht_trendmode-value'])
            if params.get('correl_sine_trend-enabled'):
                conditions.append(params['correl_sine_trend-value'] < dataframe['correl_sine_trend'])
            if params.get('ht_trendline-enabled'):
                conditions.append(params['ht_trendline-value'] < (dataframe['ht_trendline'] - dataframe['close']))
            if params.get('ht_sine-enabled'):
                conditions.append(params['ht_sine-value'] < (dataframe['ht_sine'] - dataframe['leadsine']))
            if params.get('correl_sine_trend-enabled'):
                conditions.append(params['correl_sine_trend-value'] < dataframe['correl_sine_trend'])
            if params.get('correl_ht_sine_trend-enabled'):
                conditions.append(params['correl_ht_sine_trend-value'] < dataframe['correl_ht_sine_trend'])
            if params.get('correl_ht_sine_close-enabled'):
                conditions.append(params['correl_ht_sine_close-value'] < dataframe['correl_ht_sine_close'])
            if params.get('cci-enabled'):
                conditions.append(params['cci-value'] < dataframe['cci'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'macd_cross_signal':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['macd'], dataframe['macdsignal']
                    ))
                if params['trigger'] == 'phasor':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['quadrature'], dataframe['inphase']
                    ))
                if params['trigger'] == 'ht_trendmode':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['ht_trendmode'], 0
                    ))
                if params['trigger'] == 'ht_sine':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['leadsine'], dataframe['ht_sine']
                    ))
                if params['trigger'] == 'aroon_osc':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['aroonosc'], 0))

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
            Integer(-100, 100, name='aroon-value'),
            Real(2.4, 100, name='rsi-value'),
            Real(0, 100, name='ul-value'),
            Real(-42, 42, name='stoch_ratio-value'),
            Integer(0, 100, name='stoch-value'),
            Real(-0.025, 0.0238, name='inphase-value'),
            Real(-0.0741, 0.047, name='quadrature-value'),
            Integer(0, 1, name='ht_trendmode-value'),
            Real(-1, 1, name='correl_sine_trend-value'),
            Real(-1, 1, name='ht_trendline-value'),
            Real(-0.76, 0.76, name='ht_sine-value'),
            Real(-1, 1, name='correl_sine_trend-value'),
            Real(-1, 1, name='correl_ht_sine_trend-value'),
            Real(-1, 1, name='correl_ht_sine_close-value'),
            Integer(-1000, 1000, name='cci-value'),
            Integer(-44, 310, name='ht_phase-value'),

            Categorical([True, False], name='aroon-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical([True, False], name='ul-enabled'),
            Categorical([True, False], name='stoch_ratio-enabled'),
            Categorical([True, False], name='stoch-enabled'),
            Categorical([True, False], name='inphase-enabled'),
            Categorical([True, False], name='ht_phase-enabled'),
            Categorical([True, False], name='correl_sine_trend-enabled'),
            Categorical([True, False], name='quadrature-enabled'),
            Categorical([True, False], name='ht_trendline-enabled'),
            Categorical([True, False], name='correl_sine_trend-enabled'),
            Categorical([True, False], name='correl_ht_sine_trend-enabled'),
            Categorical([True, False], name='cci-enabled'),
            Categorical(['ht_sine', 'phasor', 'macd_cross_signal', 'aroon_osc', 'ht_trendmode'],
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
            if params.get('aroon-enabled'):
                conditions.append((dataframe['aroon_down'] - dataframe['aroon_up']) > params['aroon-value_sell'])
            if params.get('rsi-enabled'):
                conditions.append(dataframe['rsi'] < params['rsi-value_sell'])
            if params.get('ul-enabled'):
                conditions.append(dataframe['ul'] < params['ul-value_sell'])
            if params.get('stoch_ratio-enabled'):
                conditions.append((dataframe['slowd'] - dataframe['slowk']) > params['stoch_ratio-value_sell'])
            if params.get('stoch-enabled'):
                conditions.append(dataframe['slowk'] < params['stoch-value_sell'])
            if params.get('ht_phase-enabled'):
                conditions.append(dataframe['ht_phase'] < params['ht_phase-value_sell'])
            if params.get('inphase-enabled'):
                conditions.append(dataframe['inphase'] > params['inphase-value_sell'])
            if params.get('quadrature-enabled'):
                conditions.append(dataframe['quadrature'] > params['quadrature-value_sell'])
            if params.get('ht_trendmode-enabled'):
                conditions.append(dataframe['ht_trendmode'] < params['ht_trendmode-value'])
            if params.get('correl_sine_trend-enabled'):
                conditions.append(params['correl_sine_trend-value_sell'] < dataframe['correl_sine_trend'])
            if params.get('ht_trendline-enabled'):
                conditions.append(params['ht_trendline-value_sell'] > (dataframe['close'] - dataframe['ht_trendline']))
            if params.get('ht_sine-enabled'):
                conditions.append(params['ht_sine-value_sell'] < (dataframe['ht_trendline'] - dataframe['ht_sine']))
            if params.get('correl_sine_trend-enabled'):
                conditions.append(params['correl_sine_trend-value_sell'] < dataframe['correl_sine_trend'])
            if params.get('correl_ht_sine_trend-enabled'):
                conditions.append(params['correl_ht_sine_trend-value_sell'] < dataframe['correl_ht_sine_trend'])
            if params.get('correl_ht_sine_close-enabled'):
                conditions.append(params['correl_ht_sine_close-value_sell'] < dataframe['correl_ht_sine_close'])
            if params.get('cci-enabled'):
                conditions.append(params['cci-value'] < dataframe['cci'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'macd_cross_signal':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['macd'], dataframe['macdsignal']
                    ))
                if params['trigger'] == 'phasor':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['quadrature'], dataframe['inphase']
                    ))
                if params['trigger'] == 'ht_trendmode':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['ht_trendmode'], 1
                    ))
                if params['trigger'] == 'ht_sine':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['leadsine'], dataframe['ht_sine']
                    ))
                if params['trigger'] == 'aroon_osc':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['aroonosc'], 0))

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
            Integer(-100, 100, name='aroon-value_sell'),
            Real(2.4, 100, name='rsi-value_sell'),
            Real(0, 100, name='ul-value_sell'),
            Real(-42, 42, name='stoch_ratio-value_sell'),
            Integer(0, 100, name='stoch-value_sell'),
            Real(-0.025, 0.0238, name='inphase-value_sell'),
            Real(-0.0741, 0.047, name='quadrature-value_sell'),
            Real(-1, 1, name='correl_sine_trend-value_sell'),
            Real(-1, 1, name='ht_trendline-value_sell'),
            Real(-0.76, 0.76, name='ht_sine-value_sell'),
            Real(-1, 1, name='correl_sine_trend-value_sell'),
            Real(-1, 1, name='correl_ht_sine_trend-value_sell'),
            Real(-1, 1, name='correl_ht_sine_close-value_sell'),
            Integer(-1000, 1000, name='cci-value_sell'),
            Integer(-44, 310, name='ht_phase-value_sell'),
            Integer(0, 1, name='ht_trendmode-value_sell'),

            Categorical([True, False], name='aroon-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical([True, False], name='ul-enabled'),
            Categorical([True, False], name='stoch_ratio-enabled'),
            Categorical([True, False], name='stoch-enabled'),
            Categorical([True, False], name='inphase-enabled'),
            Categorical([True, False], name='ht_phase-enabled'),
            Categorical([True, False], name='correl_sine_trend-enabled'),
            Categorical([True, False], name='quadrature-enabled'),
            Categorical([True, False], name='ht_trendline-enabled'),
            Categorical([True, False], name='correl_sine_trend-enabled'),
            Categorical([True, False], name='correl_ht_sine_trend-enabled'),
            Categorical([True, False], name='cci-enabled'),
            Categorical(
                ['ht_sine', 'aroon_osc', 'ht_trendmode', 'phasor', 'macd_cross_signal'],
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
