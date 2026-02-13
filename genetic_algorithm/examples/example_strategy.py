"""
Auto-generated strategy by Genetic Algorithm
Generation: 0
Individual: 0
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class GAStrategy_Gen0_Ind0(IStrategy):
    """Auto-generated GA strategy"""
    
    INTERFACE_VERSION = 3
    
    # Strategy parameters
    timeframe = '15m'
    stoploss = -0.06045211161528827
    minimal_roi = {0: 0.09799530973163299, 30: 0.0262573210437368, 60: 0.0489443994048185}
    trailing_stop = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators"""
        dataframe['sma_98'] = ta.SMA(dataframe, timeperiod=98)
        dataframe['atr_10'] = ta.ATR(dataframe, timeperiod=10)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=23, signalperiod=5)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals"""
        conditions = (
            ((dataframe['macd'] > dataframe['macdsignal']))
        )
        dataframe.loc[conditions, 'enter_long'] = 1

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals"""
        conditions = (
            ((dataframe['macd'] < dataframe['macdsignal']))
        )
        dataframe.loc[conditions, 'exit_long'] = 1

        return dataframe
