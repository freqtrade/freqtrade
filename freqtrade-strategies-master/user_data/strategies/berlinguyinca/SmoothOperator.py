# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa

# DO NOT USE, just playing with smooting and graphs!


class SmoothOperator(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:

    The concept is about combining several common indicators, with a heavily smoothing, while trying to detect
    a none completed peak shape.
    """

    # Minimal ROI designed for the strategy.
    # we only sell after 100%, unless our sell points are found before
    minimal_roi = {
        "0": 0.10
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    # should be converted to a trailing stop loss
    stoploss = -0.05

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ##################################################################################
        # required for entry and exit
        # CCI
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['mfi_smooth'] = ta.EMA(dataframe, timeperiod=11, price='mfi')
        dataframe['cci_smooth'] = ta.EMA(dataframe, timeperiod=11, price='cci')
        dataframe['rsi_smooth'] = ta.EMA(dataframe, timeperiod=11, price='rsi')

        ##################################################################################
        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        ##################################################################################
        # required for entry
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=1.6)
        dataframe['entry_bb_lowerband'] = bollinger['lower']
        dataframe['entry_bb_upperband'] = bollinger['upper']
        dataframe['entry_bb_middleband'] = bollinger['mid']

        dataframe['bpercent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (
                dataframe['bb_upperband'] - dataframe['bb_lowerband']) * 100

        dataframe['bsharp'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / (
            dataframe['bb_middleband'])

        # these seem to be kind useful to measure when bands widen
        # but than they are directly based on the moving average
        dataframe['bsharp_slow'] = ta.SMA(dataframe, price='bsharp', timeperiod=11)
        dataframe['bsharp_medium'] = ta.SMA(dataframe, price='bsharp', timeperiod=8)
        dataframe['bsharp_fast'] = ta.SMA(dataframe, price='bsharp', timeperiod=5)

        ##################################################################################
        # rsi and mfi are slightly weighted
        dataframe['mfi_rsi_cci_smooth'] = (dataframe['rsi_smooth'] * 1.125 + dataframe['mfi_smooth'] * 1.125 +
                                           dataframe[
                                               'cci_smooth']) / 3

        dataframe['mfi_rsi_cci_smooth'] = ta.TEMA(dataframe, timeperiod=21, price='mfi_rsi_cci_smooth')

        # playgound
        dataframe['candle_size'] = (dataframe['close'] - dataframe['open']) * (
                dataframe['close'] - dataframe['open']) / 2

        # helps with pattern recognition
        dataframe['average'] = (dataframe['close'] + dataframe['open'] + dataframe['high'] + dataframe['low']) / 4
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=200, price='close')
        dataframe['sma_medium'] = ta.SMA(dataframe, timeperiod=100, price='close')
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=50, price='close')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                # protection against pump and dump
                #     (dataframe['volume'] < (dataframe['volume'].rolling(window=30).mean().shift(1) * 20))
                #
                #     & (dataframe['macd'] < dataframe['macdsignal'])
                #     & (dataframe['macd'] > 0)

                # # spike below entry band for 3 consecutive ticks
                # & (dataframe['low'] < dataframe['entry_bb_lowerband'])
                # & (dataframe['low'].shift(1) < dataframe['bb_lowerband'].shift(1))
                # & (dataframe['low'].shift(2) < dataframe['bb_lowerband'].shift(2))
                # # pattern recognition
                # & (
                #         (dataframe['close'] > dataframe['open'])
                #         | (dataframe['CDLHAMMER'] == 100)
                #         | (dataframe['CDLINVERTEDHAMMER'] == 100)
                #         | (dataframe['CDLDRAGONFLYDOJI'] == 100)
                # )
                # bottom curve detection
                # & (dataframe['mfi_rsi_cci_smooth'] < 0)
                #
                # |

                (
                    # simple v bottom shape (lopsided to the left to increase reactivity)
                    # which has to be below a very slow average
                    # this pattern only catches a few, but normally very good buy points
                    (
                            (dataframe['average'].shift(5) > dataframe['average'].shift(4))
                            & (dataframe['average'].shift(4) > dataframe['average'].shift(3))
                            & (dataframe['average'].shift(3) > dataframe['average'].shift(2))
                            & (dataframe['average'].shift(2) > dataframe['average'].shift(1))
                            & (dataframe['average'].shift(1) < dataframe['average'].shift(0))
                            & (dataframe['low'].shift(1) < dataframe['bb_middleband'])
                            & (dataframe['cci'].shift(1) < -100)
                            & (dataframe['rsi'].shift(1) < 30)

                    )
                    |
                    # buy in very oversold conditions
                    (
                            (dataframe['low'] < dataframe['bb_middleband'])
                            & (dataframe['cci'] < -200)
                            & (dataframe['rsi'] < 30)
                            & (dataframe['mfi'] < 30)
                    )

                    |
                    # etc tends to trade like this
                    # over very long periods of slowly building up coins
                    # does not happen often, but once in a while
                    (
                            (dataframe['mfi'] < 10)
                            & (dataframe['cci'] < -150)
                            & (dataframe['rsi'] < dataframe['mfi'])
                    )

                )

                &
                # ensure we have an overall uptrend
                (dataframe['close'] > dataframe['close'].shift())
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # different strategy used for sell points, due to be able to duplicate it to 100%
        dataframe.loc[
            (
                (
                    #   This generates very nice sale points, and mostly sit's one stop behind
                    #   the top of the peak
                    (
                        (dataframe['mfi_rsi_cci_smooth'] > 100)
                        & (dataframe['mfi_rsi_cci_smooth'].shift(1) > dataframe['mfi_rsi_cci_smooth'])
                        & (dataframe['mfi_rsi_cci_smooth'].shift(2) < dataframe['mfi_rsi_cci_smooth'].shift(1))
                        & (dataframe['mfi_rsi_cci_smooth'].shift(3) < dataframe['mfi_rsi_cci_smooth'].shift(2))
                    )
                    |
                    #   This helps with very long, sideways trends, to get out of a market before
                    #   it dumps
                    (
                        StrategyHelper.eight_green_candles(dataframe)
                    )
                    |
                    # in case of very overbought market, like some one pumping
                    # sell
                    (
                        (dataframe['cci'] > 200)
                        & (dataframe['rsi'] > 70)
                    )
                )

            ),
            'sell'] = 1
        return dataframe


class StrategyHelper:
    """
        simple helper class to predefine a couple of patterns for our
        strategy
    """

    @staticmethod
    def seven_green_candles(dataframe):
        """
            evaluates if we are having 7 green candles in a row
        :param self:
        :param dataframe:
        :return:
        """
        return (
                (dataframe['open'] < dataframe['close']) &
                (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) < dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) < dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) < dataframe['close'].shift(4)) &
                (dataframe['open'].shift(5) < dataframe['close'].shift(5)) &
                (dataframe['open'].shift(6) < dataframe['close'].shift(6)) &
                (dataframe['open'].shift(7) < dataframe['close'].shift(7))
        )

    @staticmethod
    def eight_green_candles(dataframe):
        """
            evaluates if we are having 8 green candles in a row
        :param self:
        :param dataframe:
        :return:
        """
        return (
                (dataframe['open'] < dataframe['close']) &
                (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) < dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) < dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) < dataframe['close'].shift(4)) &
                (dataframe['open'].shift(5) < dataframe['close'].shift(5)) &
                (dataframe['open'].shift(6) < dataframe['close'].shift(6)) &
                (dataframe['open'].shift(7) < dataframe['close'].shift(7)) &
                (dataframe['open'].shift(8) < dataframe['close'].shift(8))
        )

    @staticmethod
    def eight_red_candles(dataframe, shift=0):
        """
            evaluates if we are having 8 red candles in a row
        :param self:
        :param dataframe:
        :param shift: shift the pattern by n
        :return:
        """
        return (
                (dataframe['open'].shift(shift) > dataframe['close'].shift(shift)) &
                (dataframe['open'].shift(1 + shift) > dataframe['close'].shift(1 + shift)) &
                (dataframe['open'].shift(2 + shift) > dataframe['close'].shift(2 + shift)) &
                (dataframe['open'].shift(3 + shift) > dataframe['close'].shift(3 + shift)) &
                (dataframe['open'].shift(4 + shift) > dataframe['close'].shift(4 + shift)) &
                (dataframe['open'].shift(5 + shift) > dataframe['close'].shift(5 + shift)) &
                (dataframe['open'].shift(6 + shift) > dataframe['close'].shift(6 + shift)) &
                (dataframe['open'].shift(7 + shift) > dataframe['close'].shift(7 + shift)) &
                (dataframe['open'].shift(8 + shift) > dataframe['close'].shift(8 + shift))
        )

    @staticmethod
    def four_green_one_red_candle(dataframe):
        """
            evaluates if we are having a red candle and 4 previous green
        :param self:
        :param dataframe:
        :return:
        """
        return (
                (dataframe['open'] > dataframe['close']) &
                (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) < dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) < dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) < dataframe['close'].shift(4))
        )

    @staticmethod
    def four_red_one_green_candle(dataframe):
        """
            evaluates if we are having a green candle and 4 previous red
        :param self:
        :param dataframe:
        :return:
        """
        return (
                (dataframe['open'] < dataframe['close']) &
                (dataframe['open'].shift(1) > dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) > dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) > dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) > dataframe['close'].shift(4))
        )
