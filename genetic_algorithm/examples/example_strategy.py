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
    timeframe = '1h'
    stoploss = -0.07518603821081395
    minimal_roi = {'0': 0.09384811654257705, '30': 0.03588395946778237, '60': 0.042410791445870163}
    trailing_stop = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators"""
        dataframe['rsi_19'] = ta.RSI(dataframe, timeperiod=19)
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0878650025789773, nbdevdn=2.0878650025789773)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals"""
        conditions = (
            ((dataframe['rsi_19'] < 30))
        )
        dataframe.loc[conditions, 'enter_long'] = 1

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals"""
        conditions = (
            ((dataframe['rsi_19'] > 61))
        )
        dataframe.loc[conditions, 'exit_long'] = 1

        return dataframe
