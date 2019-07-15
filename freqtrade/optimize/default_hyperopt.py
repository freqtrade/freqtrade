# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from functools import reduce
from typing import Any, Callable, Dict, List

import talib.abstract as ta
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.optimize.hyperopt_interface import IHyperOpt

# set TARGET_TRADES to suit your number concurrent trades so its realistic
# to the number of days
TARGET_TRADES = 600
# This is assumed to be expected avg profit * expected trade count.
# For example, for 0.35% avg per trade (or 0.0035 as ratio) and 1100 trades,
# self.expected_max_profit = 3.85
# Check that the reported Σ% values do not exceed this!
# Note, this is ratio. 3.85 stated above means 385Σ%.
EXPECTED_MAX_PROFIT = 3.0

# max average trade duration in minutes
# if eval ends with higher value, we consider it a failed eval
MAX_ACCEPTED_TRADE_DURATION = 300


class DefaultHyperOpts(IHyperOpt):
    """
    Default hyperopt provided by freqtrade bot.
    You can override it with your own hyperopt
    """

    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['sar'] = ta.SAR(dataframe)
        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by hyperopt
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use
            """
            conditions = []
            # GUARDS AND TRENDS
            if 'mfi-enabled' in params and params['mfi-enabled']:
                conditions.append(dataframe['mfi'] < params['mfi-value'])
            if 'fastd-enabled' in params and params['fastd-enabled']:
                conditions.append(dataframe['fastd'] < params['fastd-value'])
            if 'adx-enabled' in params and params['adx-enabled']:
                conditions.append(dataframe['adx'] > params['adx-value'])
            if 'rsi-enabled' in params and params['rsi-enabled']:
                conditions.append(dataframe['rsi'] < params['rsi-value'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'bb_lower':
                    conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
                if params['trigger'] == 'macd_cross_signal':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['macd'], dataframe['macdsignal']
                    ))
                if params['trigger'] == 'sar_reversal':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['close'], dataframe['sar']
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
        Define your Hyperopt space for searching strategy parameters
        """
        return [
            Integer(10, 25, name='mfi-value'),
            Integer(15, 45, name='fastd-value'),
            Integer(20, 50, name='adx-value'),
            Integer(20, 40, name='rsi-value'),
            Categorical([True, False], name='mfi-enabled'),
            Categorical([True, False], name='fastd-enabled'),
            Categorical([True, False], name='adx-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical(['bb_lower', 'macd_cross_signal', 'sar_reversal'], name='trigger')
        ]

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the sell strategy parameters to be used by hyperopt
        """
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Sell strategy Hyperopt will build and use
            """
            # print(params)
            conditions = []
            # GUARDS AND TRENDS
            if 'sell-mfi-enabled' in params and params['sell-mfi-enabled']:
                conditions.append(dataframe['mfi'] > params['sell-mfi-value'])
            if 'sell-fastd-enabled' in params and params['sell-fastd-enabled']:
                conditions.append(dataframe['fastd'] > params['sell-fastd-value'])
            if 'sell-adx-enabled' in params and params['sell-adx-enabled']:
                conditions.append(dataframe['adx'] < params['sell-adx-value'])
            if 'sell-rsi-enabled' in params and params['sell-rsi-enabled']:
                conditions.append(dataframe['rsi'] > params['sell-rsi-value'])

            # TRIGGERS
            if 'sell-trigger' in params:
                if params['sell-trigger'] == 'sell-bb_upper':
                    conditions.append(dataframe['close'] > dataframe['bb_upperband'])
                if params['sell-trigger'] == 'sell-macd_cross_signal':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['macdsignal'], dataframe['macd']
                    ))
                if params['sell-trigger'] == 'sell-sar_reversal':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['sar'], dataframe['close']
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
        Define your Hyperopt space for searching sell strategy parameters
        """
        return [
            Integer(75, 100, name='sell-mfi-value'),
            Integer(50, 100, name='sell-fastd-value'),
            Integer(50, 100, name='sell-adx-value'),
            Integer(60, 100, name='sell-rsi-value'),
            Categorical([True, False], name='sell-mfi-enabled'),
            Categorical([True, False], name='sell-fastd-enabled'),
            Categorical([True, False], name='sell-adx-enabled'),
            Categorical([True, False], name='sell-rsi-enabled'),
            Categorical(['sell-bb_upper',
                         'sell-macd_cross_signal',
                         'sell-sar_reversal'], name='sell-trigger')
        ]

    @staticmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:
        """
        Generate the ROI table that will be used by Hyperopt
        """
        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    @staticmethod
    def stoploss_space() -> List[Dimension]:
        """
        Stoploss Value to search
        """
        return [
            Real(-0.5, -0.02, name='stoploss'),
        ]

    @staticmethod
    def roi_space() -> List[Dimension]:
        """
        Values to search for each ROI steps
        """
        return [
            Integer(10, 120, name='roi_t1'),
            Integer(10, 60, name='roi_t2'),
            Integer(10, 40, name='roi_t3'),
            Real(0.01, 0.04, name='roi_p1'),
            Real(0.01, 0.07, name='roi_p2'),
            Real(0.01, 0.20, name='roi_p3'),
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators. Should be a copy of from strategy
        must align to populate_indicators in this file
        Only used when --spaces does not include buy
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['mfi'] < 16) &
                (dataframe['adx'] > 25) &
                (dataframe['rsi'] < 21)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators. Should be a copy of from strategy
        must align to populate_indicators in this file
        Only used when --spaces does not include sell
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(
                    dataframe['macdsignal'], dataframe['macd']
                )) &
                (dataframe['fastd'] > 54)
            ),
            'sell'] = 1
        return dataframe
