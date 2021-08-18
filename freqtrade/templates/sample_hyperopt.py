# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file

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


class SampleHyperOpt(IHyperOpt):
    """
    This is a sample Hyperopt to inspire you.

    More information in the documentation: https://www.freqtrade.io/en/latest/hyperopt/

    You should:
    - Rename the class name to some unique name.
    - Add any methods you want to build your hyperopt.
    - Add any lib you need to build your hyperopt.

    An easier way to get a new hyperopt file is by using
    `freqtrade new-hyperopt --hyperopt MyCoolHyperopt`.

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
    def buy_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching buy strategy parameters.
        """
        return [
            Integer(10, 25, name='mfi-value'),
            Integer(15, 45, name='fastd-value'),
            Integer(20, 50, name='adx-value'),
            Integer(20, 40, name='rsi-value'),
            Integer(75, 90, name='short-mfi-value'),
            Integer(55, 85, name='short-fastd-value'),
            Integer(50, 80, name='short-adx-value'),
            Integer(60, 80, name='short-rsi-value'),
            Categorical([True, False], name='mfi-enabled'),
            Categorical([True, False], name='fastd-enabled'),
            Categorical([True, False], name='adx-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical(['boll', 'macd_cross_signal', 'sar_reversal'], name='trigger'),

        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use.
            """
            long_conditions = []
            short_conditions = []

            # GUARDS AND TRENDS
            if 'mfi-enabled' in params and params['mfi-enabled']:
                long_conditions.append(dataframe['mfi'] < params['mfi-value'])
                short_conditions.append(dataframe['mfi'] > params['short-mfi-value'])
            if 'fastd-enabled' in params and params['fastd-enabled']:
                long_conditions.append(dataframe['fastd'] < params['fastd-value'])
                short_conditions.append(dataframe['fastd'] > params['short-fastd-value'])
            if 'adx-enabled' in params and params['adx-enabled']:
                long_conditions.append(dataframe['adx'] > params['adx-value'])
                short_conditions.append(dataframe['adx'] < params['short-adx-value'])
            if 'rsi-enabled' in params and params['rsi-enabled']:
                long_conditions.append(dataframe['rsi'] < params['rsi-value'])
                short_conditions.append(dataframe['rsi'] > params['short-rsi-value'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'boll':
                    long_conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
                    short_conditions.append(dataframe['close'] > dataframe['bb_upperband'])
                if params['trigger'] == 'macd_cross_signal':
                    long_conditions.append(qtpylib.crossed_above(
                        dataframe['macd'],
                        dataframe['macdsignal']
                    ))
                    short_conditions.append(qtpylib.crossed_below(
                        dataframe['macd'],
                        dataframe['macdsignal']
                    ))
                if params['trigger'] == 'sar_reversal':
                    long_conditions.append(qtpylib.crossed_above(
                        dataframe['close'],
                        dataframe['sar']
                    ))
                    short_conditions.append(qtpylib.crossed_below(
                        dataframe['close'],
                        dataframe['sar']
                    ))

            # Check that volume is not 0
            long_conditions.append(dataframe['volume'] > 0)
            short_conditions.append(dataframe['volume'] > 0)

            if long_conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, long_conditions),
                    'buy'] = 1

            if short_conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, short_conditions),
                    'enter_short'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching sell strategy parameters.
        """
        return [
            Integer(75, 100, name='sell-mfi-value'),
            Integer(50, 100, name='sell-fastd-value'),
            Integer(50, 100, name='sell-adx-value'),
            Integer(60, 100, name='sell-rsi-value'),
            Integer(1, 25, name='exit-short-mfi-value'),
            Integer(1, 50, name='exit-short-fastd-value'),
            Integer(1, 50, name='exit-short-adx-value'),
            Integer(1, 40, name='exit-short-rsi-value'),
            Categorical([True, False], name='sell-mfi-enabled'),
            Categorical([True, False], name='sell-fastd-enabled'),
            Categorical([True, False], name='sell-adx-enabled'),
            Categorical([True, False], name='sell-rsi-enabled'),
            Categorical(['sell-boll',
                         'sell-macd_cross_signal',
                         'sell-sar_reversal'],
                        name='sell-trigger'
                        ),
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
            exit_long_conditions = []
            exit_short_conditions = []

            # GUARDS AND TRENDS
            if 'sell-mfi-enabled' in params and params['sell-mfi-enabled']:
                exit_long_conditions.append(dataframe['mfi'] > params['sell-mfi-value'])
                exit_short_conditions.append(dataframe['mfi'] < params['exit-short-mfi-value'])
            if 'sell-fastd-enabled' in params and params['sell-fastd-enabled']:
                exit_long_conditions.append(dataframe['fastd'] > params['sell-fastd-value'])
                exit_short_conditions.append(dataframe['fastd'] < params['exit-short-fastd-value'])
            if 'sell-adx-enabled' in params and params['sell-adx-enabled']:
                exit_long_conditions.append(dataframe['adx'] < params['sell-adx-value'])
                exit_short_conditions.append(dataframe['adx'] > params['exit-short-adx-value'])
            if 'sell-rsi-enabled' in params and params['sell-rsi-enabled']:
                exit_long_conditions.append(dataframe['rsi'] > params['sell-rsi-value'])
                exit_short_conditions.append(dataframe['rsi'] < params['exit-short-rsi-value'])

            # TRIGGERS
            if 'sell-trigger' in params:
                if params['sell-trigger'] == 'sell-boll':
                    exit_long_conditions.append(dataframe['close'] > dataframe['bb_upperband'])
                    exit_short_conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
                if params['sell-trigger'] == 'sell-macd_cross_signal':
                    exit_long_conditions.append(qtpylib.crossed_above(
                        dataframe['macdsignal'],
                        dataframe['macd']
                    ))
                    exit_short_conditions.append(qtpylib.crossed_below(
                        dataframe['macdsignal'],
                        dataframe['macd']
                    ))
                if params['sell-trigger'] == 'sell-sar_reversal':
                    exit_long_conditions.append(qtpylib.crossed_above(
                        dataframe['sar'],
                        dataframe['close']
                    ))
                    exit_short_conditions.append(qtpylib.crossed_below(
                        dataframe['sar'],
                        dataframe['close']
                    ))

            # Check that volume is not 0
            exit_long_conditions.append(dataframe['volume'] > 0)
            exit_short_conditions.append(dataframe['volume'] > 0)

            if exit_long_conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, exit_long_conditions),
                    'sell'] = 1

            if exit_short_conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, exit_short_conditions),
                    'exit_short'] = 1

            return dataframe

        return populate_sell_trend
