# Freqtrade_backtest_validation_freqtrade1.py
# This script is 1 of a pair the other being freqtrade_backtest_validation_tradingview1
# These should be executed on their respective platforms for the same coin/period/resolution
# The purpose is to test Freqtrade backtest provides like results to a known industry platform.
#
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Freqtrade_backtest_validation_freqtrade1(IStrategy):
    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "40": 2.0,
        "30": 2.01,
        "20": 2.02,
        "0": 2.04
    }

    stoploss = -0.90
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # SMA - Simple Moving Average
        dataframe['fastMA'] = ta.SMA(dataframe, timeperiod=14)
        dataframe['slowMA'] = ta.SMA(dataframe, timeperiod=28)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['fastMA'] > dataframe['slowMA'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['fastMA'] < dataframe['slowMA'])
            ),
            'sell'] = 1
        return dataframe
