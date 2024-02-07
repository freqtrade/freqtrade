# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import timeframe_to_minutes
from pandas import DataFrame
from technical.util import resample_to_interval, resampled_merge
from functools import reduce
import numpy  # noqa
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class StrategyScalpingFast2(IStrategy):
    INTERFACE_VERSION = 3
    '\n        Based on ReinforcedSmoothScalp\n        https://github.com/freqtrade/freqtrade-strategies/blob/master/user_data/strategies/berlinguyinca/ReinforcedSmoothScalp.py\n        this strategy is based around the idea of generating a lot of potentatils entrys and make tiny profits on each trade\n\n        we recommend to have at least 60 parallel trades at any time to cover non avoidable losses\n    '
    # Buy hyperspace params:
    entry_params = {'mfi-value': 19, 'fastd-value': 29, 'fastk-value': 19, 'adx-value': 30, 'mfi-enabled': False, 'fastd-enabled': False, 'adx-enabled': False, 'fastk-enabled': False}
    # Sell hyperspace params:
    exit_params = {'exit-mfi-value': 89, 'exit-fastd-value': 72, 'exit-fastk-value': 68, 'exit-adx-value': 86, 'exit-cci-value': 157, 'exit-mfi-enabled': True, 'exit-fastd-enabled': True, 'exit-adx-enabled': True, 'exit-cci-enabled': False, 'exit-fastk-enabled': False}
    # ROI table:
    minimal_roi = {'0': 0.082, '18': 0.06, '51': 0.012, '123': 0}
    use_exit_signal = False
    # Stoploss:
    stoploss = -0.326
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    #minimal_roi = {
    #    "0": 0.02
    #}
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    # should not be below 3% loss
    #stoploss = -0.1
    # Optimal timeframe for the strategy
    # the shorter the better
    timeframe = '1m'
    # resample factor to establish our general trend. Basically don't entry if a trend is not given
    resample_factor = 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tf_res = timeframe_to_minutes(self.timeframe) * self.resample_factor
        df_res = resample_to_interval(dataframe, tf_res)
        df_res['sma'] = ta.SMA(df_res, 50, price='close')
        dataframe = resampled_merge(dataframe, df_res, fill_na=True)
        dataframe['resample_sma'] = dataframe[f'resample_{tf_res}_sma']
        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe)
        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['open'] < dataframe['ema_low'])
        conditions.append(dataframe['resample_sma'] < dataframe['close'])
        if self.entry_params['adx-enabled']:
            conditions.append(dataframe['adx'] < self.entry_params['adx-value'])
        if self.entry_params['mfi-enabled']:
            conditions.append(dataframe['mfi'] < self.entry_params['mfi-value'])
        if self.entry_params['fastk-enabled']:
            conditions.append(dataframe['fastk'] < self.entry_params['fastk-value'])
        if self.entry_params['fastd-enabled']:
            conditions.append(dataframe['fastd'] < self.entry_params['fastd-value'])
        if self.entry_params['fastk-enabled'] == True & self.entry_params['fastd-enabled'] == True:
            conditions.append(qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
        # |
        # # try to get some sure things independent of resample
        # ((dataframe['rsi'] - dataframe['mfi']) < 10) &
        # (dataframe['mfi'] < 30) &
        # (dataframe['cci'] < -200)
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'entry'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['open'] >= dataframe['ema_high'])
        if self.exit_params['exit-fastd-enabled'] == True | self.exit_params['exit-fastk-enabled'] == True:
            conditions.append(qtpylib.crossed_above(dataframe['fastk'], self.exit_params['exit-fastk-value']) | qtpylib.crossed_above(dataframe['fastd'], self.exit_params['exit-fastd-value']))
        if self.exit_params['exit-cci-enabled'] == True:
            conditions.append(dataframe['cci'] > 100)
        if self.exit_params['exit-mfi-enabled'] == True:
            conditions.append(dataframe['mfi'] > self.exit_params['exit-mfi-value'])
        if self.exit_params['exit-adx-enabled'] == True:
            conditions.append(dataframe['adx'] < self.exit_params['exit-adx-value'])
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit'] = 1
        return dataframe