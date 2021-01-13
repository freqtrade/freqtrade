# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from functools import reduce
from typing import Any, Callable, Dict, List

import talib.abstract as ta
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.optimize.hyperopt_interface import IHyperOpt


class ReinforcedSmoothScalp(IHyperOpt):
    """
    Default hyperopt provided by the Freqtrade bot.
    You can override it with your own Hyperopt
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
            if 'mfi-enabled' in params and params['mfi-enabled']:
                conditions.append(dataframe['mfi'] < params['mfi-value'])
            if 'fastd-enabled' in params and params['fastd-enabled']:
                conditions.append(dataframe['fastd'] < params['fastd-value'])
            if 'adx-enabled' in params and params['adx-enabled']:
                conditions.append(dataframe['adx'] > params['adx-value'])
            # if 'rsi-enabled' in params and params['rsi-enabled']:
            #   conditions.append(dataframe['rsi'] < params['rsi-value'])
            if 'fastk-enabled' in params and params['fastk-enabled']:
                conditions.append(dataframe['fastk'] < params['fastk-value'])
            # TRIGGERS
            # if 'trigger' in params:
            #    if params['trigger'] == 'bb_lower':
            #        conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
            #    if params['trigger'] == 'macd_cross_signal':
            #        conditions.append(qtpylib.crossed_above(
            #            dataframe['macd'], dataframe['macdsignal']
            #        ))
            #    if params['trigger'] == 'sar_reversal':
            #        conditions.append(qtpylib.crossed_above(
            #            dataframe['close'], dataframe['sar']
            #        ))

            # Check that volume is not 0
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
            Integer(10, 25, name='mfi-value'),
            Integer(15, 45, name='fastd-value'),
            Integer(15, 45, name='fastk-value'),
            Integer(20, 50, name='adx-value'),
            # Integer(20, 40, name='rsi-value'),
            Categorical([True, False], name='mfi-enabled'),
            Categorical([True, False], name='fastd-enabled'),
            Categorical([True, False], name='adx-enabled'),
            Categorical([True, False], name='fastk-enabled'),
            # Categorical([True, False], name='rsi-enabled'),
            # Categorical(['bb_lower', 'macd_cross_signal', 'sar_reversal'], name='trigger')
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
            if 'sell-mfi-enabled' in params and params['sell-mfi-enabled']:
                conditions.append(dataframe['mfi'] > params['sell-mfi-value'])
            if 'sell-fastd-enabled' in params and params['sell-fastd-enabled']:
                conditions.append(dataframe['fastd'] > params['sell-fastd-value'])
            if 'sell-adx-enabled' in params and params['sell-adx-enabled']:
                conditions.append(dataframe['adx'] < params['sell-adx-value'])
            if 'sell-fastk-enabled' in params and params['sell-fastk-enabled']:
                conditions.append(dataframe['fastk'] > params['sell-fastk-value'])
            if 'sell-cci-enabled' in params and params['sell-cci-enabled']:
                conditions.append(dataframe['cci'] > params['sell-cci-value'])

            # TRIGGERS
            # if 'sell-trigger' in params:
                # if params['sell-trigger'] == 'sell-bb_upper':
                #    conditions.append(dataframe['close'] > dataframe['bb_upperband'])
                # if params['sell-trigger'] == 'sell-macd_cross_signal':
                #    conditions.append(qtpylib.crossed_above(
                #        dataframe['macdsignal'], dataframe['macd']
                #    ))
                # if params['sell-trigger'] == 'sell-sar_reversal':
                #    conditions.append(qtpylib.crossed_above(
                #        dataframe['sar'], dataframe['close']
                #    ))

            # Check that volume is not 0
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
            Integer(75, 100, name='sell-mfi-value'),
            Integer(50, 100, name='sell-fastd-value'),
            Integer(50, 100, name='sell-fastk-value'),
            Integer(50, 100, name='sell-adx-value'),
            Integer(100, 200, name='sell-cci-value'),
            Categorical([True, False], name='sell-mfi-enabled'),
            Categorical([True, False], name='sell-fastd-enabled'),
            Categorical([True, False], name='sell-adx-enabled'),
            Categorical([True, False], name='sell-cci-enabled'),
            Categorical([True, False], name='sell-fastk-enabled'),
            # Categorical(['sell-bb_upper',
            #             'sell-macd_cross_signal',
            #             'sell-sar_reversal'], name='sell-trigger')
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe['open'] < dataframe['ema_low']) &
                        (dataframe['adx'] > 30) &
                        (dataframe['mfi'] < 30) &
                        (
                                (dataframe['fastk'] < 30) &
                                (dataframe['fastd'] < 30) &
                                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
                        ) &
                        (dataframe['resample_sma'] < dataframe['close'])
                )
                # |
                # # try to get some sure things independent of resample
                # ((dataframe['rsi'] - dataframe['mfi']) < 10) &
                # (dataframe['mfi'] < 30) &
                # (dataframe['cci'] < -200)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (
                            (
                                (dataframe['open'] >= dataframe['ema_high'])

                            ) |
                            (
                                    (qtpylib.crossed_above(dataframe['fastk'], 70)) |
                                    (qtpylib.crossed_above(dataframe['fastd'], 70))

                            )
                    ) & (dataframe['cci'] > 100)
            )
            ,
            'sell'] = 1
        return dataframe
