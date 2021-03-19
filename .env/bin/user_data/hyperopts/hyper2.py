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


class hyper2(IHyperOpt):
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
            if params.get('uo-enabled'):
                conditions.append(dataframe['uo'] > params['uo_h'])
            if params.get('angle-enabled'):
                conditions.append(dataframe['angle'] > params['angle_h'])
            if params.get('sar-enabled'):
                conditions.append(params['sar_ratio'] < (dataframe['sar'] - dataframe['close']))
                conditions.append(params['sar_shift'] > (dataframe['sar'] - dataframe['sar'].shift(3)))
            if params.get('macd-enabled'):
                conditions.append(params['macd_value'] > dataframe['macd'])
                conditions.append(params['macdhist_value'] > dataframe['macdhist'])
                conditions.append(params['macdhist_shift'] < (dataframe['macdhist'] - dataframe['macdhist'].shift(3)))

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'macd_cross_signal':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['macd'], dataframe['macdsignal']
                    ))
                if params['trigger'] == 'sar':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['close'], dataframe['sar']
                    ))
                if params['trigger'] == 'angle':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['angle'], 0
                    ))

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)
            conditions.append(dataframe['angle'] > 0)

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
            Integer(0, 100, name='uo_h'),
            Real(-1, 1, name='angle_h'),
            Real(0, 1, name='sar_ratio'),
            Real(0, 1, name='sar_shift'),
            Real(-1, 1, name='macd_value'),
            Real(-1, 1, name='macdhist_value'),
            Real(-1, 1, name='macdhist_shift'),
            Categorical([True, False], name='angle-enabled'),
            Categorical([True, False], name='sar-enabled'),
            Categorical([True, False], name='macd-enabled'),
            Categorical(['sar'], name='trigger')
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
            if params.get('sell-uo-enabled'):
                conditions.append(dataframe['uo'] > params['uo_l_s'])
            if params.get('angle-enabled'):
                conditions.append(params['angle_h_s'] < dataframe['angle'])
            if params.get('sar-enabled'):
                conditions.append(params['sar_ratio_s'] < (dataframe['sar'] - dataframe['close']))
                conditions.append(params['sar_shift_s'] < (dataframe['sar'] - dataframe['sar'].shift(3)))
            if params.get('macd-enabled'):
                conditions.append(params['macd_value_s'] > dataframe['macd'])
                conditions.append(params['macdhist_value_s'] > dataframe['macdhist'])
                conditions.append(params['macdhist_shift_s'] > (dataframe['macdhist'] - dataframe['macdhist'].shift(3)))

            # TRIGGERS
            if 'sell-trigger' in params:
                if params['sell-trigger'] == 'sell-sar_reversal':
                    conditions.append(qtpylib.crossed_below(
                        dataframe['sar'], dataframe['close']
                    ))

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
            Integer(0, 100, name='uo_l_s'),
            Real(-1, 1, name='angle_h_s'),
            Real(0, 1, name='sar_ratio_s'),
            Real(0, 1, name='sar_shift_s'),
            Real(-1, 1, name='macd_value_s'),
            Real(-1, 1, name='macdhist_value_s'),
            Real(-1, 1, name='macdhist_shift_s'),
            Categorical([True, False], name='sell-uo-enabled'),
            Categorical([True, False], name='sell-angle-enabled'),
            Categorical([True, False], name='sell-sar-enabled'),
            Categorical([True, False], name='sell-macd-enabled'),
            Categorical(['sell-angle', 'sell-macd_cross_signal', 'sell-sar_reversal'], name='trigger')
        ]
