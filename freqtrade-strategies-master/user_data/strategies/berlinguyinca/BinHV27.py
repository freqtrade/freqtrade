from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class BinHV27(IStrategy):
    """

        strategy sponsored by user BinH from slack

    """

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.50
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = numpy.nan_to_num(ta.RSI(dataframe, timeperiod=5))
        rsiframe = DataFrame(dataframe['rsi']).rename(columns={'rsi': 'close'})
        dataframe['emarsi'] = numpy.nan_to_num(ta.EMA(rsiframe, timeperiod=5))
        dataframe['adx'] = numpy.nan_to_num(ta.ADX(dataframe))
        dataframe['minusdi'] = numpy.nan_to_num(ta.MINUS_DI(dataframe))
        minusdiframe = DataFrame(dataframe['minusdi']).rename(columns={'minusdi': 'close'})
        dataframe['minusdiema'] = numpy.nan_to_num(ta.EMA(minusdiframe, timeperiod=25))
        dataframe['plusdi'] = numpy.nan_to_num(ta.PLUS_DI(dataframe))
        plusdiframe = DataFrame(dataframe['plusdi']).rename(columns={'plusdi': 'close'})
        dataframe['plusdiema'] = numpy.nan_to_num(ta.EMA(plusdiframe, timeperiod=5))
        dataframe['lowsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=60))
        dataframe['highsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=120))
        dataframe['fastsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=120))
        dataframe['slowsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=240))
        dataframe['bigup'] = dataframe['fastsma'].gt(dataframe['slowsma']) & ((dataframe['fastsma'] - dataframe['slowsma']) > dataframe['close'] / 300)
        dataframe['bigdown'] = ~dataframe['bigup']
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']
        dataframe['preparechangetrend'] = dataframe['trend'].gt(dataframe['trend'].shift())
        dataframe['preparechangetrendconfirm'] = dataframe['preparechangetrend'] & dataframe['trend'].shift().gt(dataframe['trend'].shift(2))
        dataframe['continueup'] = dataframe['slowsma'].gt(dataframe['slowsma'].shift()) & dataframe['slowsma'].shift().gt(dataframe['slowsma'].shift(2))
        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma'].shift()
        dataframe['slowingdown'] = dataframe['delta'].lt(dataframe['delta'].shift())
        return dataframe
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe['slowsma'].gt(0) &
            dataframe['close'].lt(dataframe['highsma']) &
            dataframe['close'].lt(dataframe['lowsma']) &
            dataframe['minusdi'].gt(dataframe['minusdiema']) &
            dataframe['rsi'].ge(dataframe['rsi'].shift()) &
            (
              (
                ~dataframe['preparechangetrend'] &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(25) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                ~dataframe['preparechangetrend'] &
                dataframe['continueup'] &
                dataframe['adx'].gt(30) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                ~dataframe['continueup'] &
                dataframe['adx'].gt(35) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                dataframe['continueup'] &
                dataframe['adx'].gt(30) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(25)
              )
            ),
            'buy'] = 1
        return dataframe
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
              (
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                (dataframe['close'].gt(dataframe['lowsma']) | dataframe['close'].gt(dataframe['highsma'])) &
                dataframe['highsma'].gt(0) &
                dataframe['bigdown']
              ) |
              (
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                (dataframe['emarsi'].ge(75) | dataframe['close'].gt(dataframe['slowsma'])) &
                dataframe['bigdown']
              ) |
              (
                ~dataframe['preparechangetrendconfirm'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                dataframe['adx'].gt(30) &
                dataframe['emarsi'].ge(80) &
                dataframe['bigup']
              ) |
              (
                dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['slowingdown'] &
                dataframe['emarsi'].ge(75) &
                dataframe['slowsma'].gt(0)
              ) |
              (
                dataframe['preparechangetrendconfirm'] &
                dataframe['minusdi'].lt(dataframe['plusdi']) &
                dataframe['close'].gt(dataframe['lowsma']) &
                dataframe['slowsma'].gt(0)
              )
            ),
            'sell'] = 1
        return dataframe
