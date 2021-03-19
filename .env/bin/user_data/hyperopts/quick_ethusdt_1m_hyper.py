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


class quick_ethusdt_1m_hyper(IHyperOpt):
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
            if params.get('macd-enabled'):
                conditions.append((dataframe['macdhist']) > params['macdhist_diff_buy'])
            if params.get('macd_angle-enabled'):
                conditions.append((dataframe['macd_angle']) > params['macd_angle_buy'])
            if params.get('tema-enabled'):
                conditions.append(dataframe['tema'] < params['tema_value_buy'])
            if params.get('sr_fastd-enabled'):
                conditions.append(dataframe['sr_fastd'] > params['sr_fastd_value_buy'])
            if params.get('sr_fastd_angle-enabled'):
                conditions.append(dataframe['sr_fastd_angle'] > params['sr_fastd_angle_value_buy'])
            if params.get('slowk-enabled'):
                conditions.append(dataframe['slowk'] > params['slowk_value_buy'])
            if params.get('slowd_angle-enabled'):
                conditions.append(dataframe['slowd_angle'] > params['slowd_angle_value_buy'])
            if params.get('sf_fastk-enabled'):
                conditions.append(dataframe['sf_fastk'] > params['sf_fastk_value_buy'])
            if params.get('sf_fastd_angle-enabled'):
                conditions.append(dataframe['sf_fastd_angle'] > params['sf_fastd_angle_value_buy'])
            if params.get('rsi-enabled'):
                conditions.append(dataframe['rsi'] > params['rsi_value_buy'])
            if params.get('rsi_angle-enabled'):
                conditions.append(dataframe['rsi_angle'] > params['rsi_angle_value_buy'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'sr':
                    conditions.append(qtpylib.crossed_above(dataframe['sr_fastk'], dataframe['sr_fastd']))
                if params['trigger'] == 'macd_cross_signal':
                    conditions.append(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
                if params['trigger'] == 'macd_histogram':
                    conditions.append((dataframe['macdhist'] - dataframe['macdhist'].shift(1)) > 0)
                if params['trigger'] == 'slow':
                    conditions.append(qtpylib.crossed_above(dataframe['slowk'], dataframe['slowd']))
                if params['trigger'] == 'sf_fast':
                    conditions.append(qtpylib.crossed_above(dataframe['sf_fastk'], dataframe['sf_fastd']))
                if params['trigger'] == 'tema':
                    conditions.append(qtpylib.crossed_above(dataframe['close'], dataframe['tema']))

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
            Real(-0.00072, 0.0008, name='macdhist_diff_buy'),
            Real(-0.019, 0.018, name='macd_angle_buy'),
            Real(0.016, 0.13, name='tema_value_buy'),
            Integer(0, 100, name='sr_fastd_value_buy'),
            Integer(-90, 90, name='sr_fastd_angle_value_buy'),
            Integer(0, 100, name='slowk_value_buy'),
            Integer(-90, 90, name='slowd_angle_value_buy'),
            Integer(0, 100, name='sf_fastk_value_buy'),
            Integer(-90, 90, name='sf_fastd_angle_value_buy'),
            Integer(0, 100, name='rsi_value_buy'),
            Integer(-90, 90, name='rsi_angle_value_buy'),
            Categorical([True, False], name='macd-enabled'),
            Categorical([True, False], name='macd_angle-enabled'),
            Categorical([True, False], name='tema-enabled'),
            Categorical([True, False], name='sr_fastd-enabled'),
            Categorical([True, False], name='sr_fastd_angle-enabled'),
            Categorical([True, False], name='slowk-enabled'),
            Categorical([True, False], name='slowd_angle-enabled'),
            Categorical([True, False], name='sf_fastk-enabled'),
            Categorical([True, False], name='sf_fastk_angle-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical([True, False], name='rsi_angle-enabled'),
            Categorical(['sr', 'macd_cross_signal', 'macd_histogram', 'slow', 'sf_fast', 'tema'], name='trigger')
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
            if params.get('macd-enabled'):
                conditions.append((dataframe['macdhist']) < params['macdhist_diff_sell'])
            if params.get('macd_angle-enabled'):
                conditions.append((dataframe['macd_angle']) < params['macd_angle_sell'])
            if params.get('tema-enabled'):
                conditions.append(dataframe['tema'] < params['tema_value_sell'])
            if params.get('sr_fastd-enabled'):
                conditions.append(dataframe['sr_fastd'] < params['sr_fastd_value_sell'])
            if params.get('sr_fastd_angle-enabled'):
                conditions.append(dataframe['sr_fastd_angle'] > params['sr_fastd_angle_value_sell'])
            if params.get('slowk-enabled'):
                conditions.append(dataframe['slowk'] < params['slowk_value_sell'])
            if params.get('slowd_angle-enabled'):
                conditions.append(dataframe['slowd_angle'] < params['slowd_angle_value_sell'])
            if params.get('sf_fastk-enabled'):
                conditions.append(dataframe['sf_fastk'] > params['sf_fastk_value_sell'])
            if params.get('sf_fastd_angle-enabled'):
                conditions.append(dataframe['sf_fastd_angle'] < params['sf_fastd_angle_value_sell'])

            # TRIGGERS
            if 'sell-trigger' in params:
                if params['trigger'] == 'sr':
                    conditions.append(qtpylib.crossed_below(dataframe['sr_fastk'], dataframe['sr_fastd']))
                if params['trigger'] == 'macd_histogram':
                    conditions.append((dataframe['macdhist'] - dataframe['macdhist'].shift(1)) < 0)
                if params['trigger'] == 'slow':
                    conditions.append(qtpylib.crossed_below(dataframe['slowk'], dataframe['slowd']))
                if params['trigger'] == 'sf_fast':
                    conditions.append(qtpylib.crossed_below(dataframe['sf_fastk'], dataframe['sf_fastd']))
                if params['trigger'] == 'tema':
                    conditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['tema']))

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
            Real(-0.00072, 0.0008, name='macdhist_diff_sell'),
            Real(-0.019, 0.018, name='macd_angle_sell'),
            Real(0.016, 0.13, name='tema_value_sell'),
            Integer(0, 100, name='sr_fastd_value_sell'),
            Integer(-90, 90, name='sr_fastd_angle_value_sell'),
            Integer(0, 100, name='slowk_value_sell'),
            Integer(-90, 90, name='slowd_angle_value_sell'),
            Integer(0, 100, name='sf_fastk_value_sell'),
            Integer(-90, 90, name='sf_fastd_angle_value_sell'),
            Categorical([True, False], name='macd-enabled'),
            Categorical([True, False], name='macd_angle-enabled'),
            Categorical([True, False], name='tema-enabled'),
            Categorical([True, False], name='sr_fastd-enabled'),
            Categorical([True, False], name='sr_fastd_angle-enabled'),
            Categorical([True, False], name='slowk-enabled'),
            Categorical([True, False], name='slowd_angle-enabled'),
            Categorical([True, False], name='sf_fastk-enabled'),
            Categorical([True, False], name='sf_fastk_angle-enabled'),
            Categorical(['sr', 'macd_histogram', 'slow', 'sf_fast', 'tema'], name='trigger')

        ]
