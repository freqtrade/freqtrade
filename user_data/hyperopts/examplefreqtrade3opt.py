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


class Examplestrategy3opt(IHyperOpt):
    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        return dataframe

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
            if params.get('mfi-enabled'):
                conditions.append(dataframe['mfi'] < params['mfi-value'])
            if params.get('fastd-enabled'):
                conditions.append(dataframe['fastd'] > params['fastd-value'])
            if params.get('rsi_min-enabled'):
                conditions.append(dataframe['rsi'] > params['rsi_min-value'])
            if params.get('rsi_max-enabled'):
                conditions.append(dataframe['rsi'] < params['rsi_max-value'])
            if params.get('fisher_max-enabled'):
                conditions.append(dataframe['fisher_rsi'] < params['fisher_rsi_max-value'])
            if params.get('fisher_min-enabled'):
                conditions.append(dataframe['fisher_rsi'] > params['fisher_rsi_min-value'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'fastk':
                    conditions.append(dataframe['fastd'] > dataframe['fastk'])
                if params['trigger'] == 'sma_reversal':
                    conditions.append(dataframe['close'] < dataframe['sma'])
                if params['trigger'] == 'ema':
                    conditions.append((dataframe['ema50'] > dataframe['ema100']) |
                                      (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema10'])))

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy']=1

            return dataframe

        return populate_buy_trend

    @ staticmethod
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching buy strategy parameters.
        """
        return [
            Integer(5, 40, name='mfi-value'),
            Integer(0, 50, name='fastd-value'),
            Integer(0, 15, name='rsi_min-value'),
            Integer(10, 50, name='rsi_max-value'),
            Integer(-1, 0.5, name='fisher_rsi_min-value'),
            Integer(-0.5, 1, name='fisher_rsi_max-value'),
            Categorical([True, False], name='mfi-enabled'),
            Categorical([True, False], name='fastd-enabled'),
            Categorical([True, False], name='rsi_min-enabled'),
            Categorical([True, False], name='rsi_max-enabled'),
            Categorical([True, False], name='fisher_min-enabled'),
            Categorical([True, False], name='fisher_max-enabled'),
            Categorical(['fastk', 'sma_reversal', 'ema'], name='trigger')
        ]

    @ staticmethod
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
            if params.get('sell-fisher-enabled'):
                conditions.append(dataframe['fisher_rsi'] > params['sell-fisher-value'])

            # TRIGGERS
            if 'sell-trigger' in params:
                if params['sell-trigger'] == 'sell-sar':
                    conditions.append(dataframe['sar'] > dataframe['close'])

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell']=1

            return dataframe

        return populate_sell_trend

    @ staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching sell strategy parameters.
        """
        return [
            Integer(-1, 1, name='sell-fisher-value'),
            Categorical([True, False], name='sell-fisher-enabled'),
            Categorical(['sell-sar'], name='sell-trigger')
        ]

    @ staticmethod
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

    @ staticmethod
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

    @ staticmethod
    def stoploss_space() -> List[Dimension]:
        """
        Stoploss Value to search
        Override it if you need some different range for the parameter in the
        'stoploss' optimization hyperspace.
        """
        return [
            Real(-0.35, -0.02, name='stoploss'),
        ]

    @ staticmethod
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
        dataframe.loc[
            (
                (dataframe['rsi'] < 28) &
                (dataframe['rsi'] > 0) &
                (dataframe['close'] < dataframe['sma']) &
                (dataframe['fisher_rsi'] < -0.94) &
                (dataframe['mfi'] < 16.0) &
                (
                    (dataframe['ema50'] > dataframe['ema100']) |
                    (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema10']))
                ) &
                (dataframe['fastd'] > dataframe['fastk']) &
                (dataframe['fastd'] > 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators.
        Can be a copy of the corresponding method from the strategy,
        or will be loaded from the strategy.
        Must align to populate_indicators used (either from this File, or from the strategy)
        Only used when --spaces does not include sell
        """
        dataframe.loc[
            (
                (dataframe['sar'] > dataframe['close']) &
                (dataframe['fisher_rsi'] > 0.3)
            ),
            'sell'] = 1
        return dataframe
