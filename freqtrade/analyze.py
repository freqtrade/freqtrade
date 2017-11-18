"""
Functions to analyze ticker data with indicators and produce buy and sell signals
"""
from enum import Enum
import logging
from datetime import timedelta

import arrow
import talib.abstract as ta
from pandas import DataFrame, to_datetime

from freqtrade.exchange import get_ticker_history
from freqtrade.vendor.qtpylib.indicators import awesome_oscillator, crossed_above, crossed_below

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """ Enum to distinguish between buy and sell signals """
    BUY = "buy"
    SELL = "sell"


def parse_ticker_dataframe(ticker: list) -> DataFrame:
    """
    Analyses the trend for the given ticker history
    :param ticker: See exchange.get_ticker_history
    :return: DataFrame
    """
    columns = {'C': 'close', 'V': 'volume', 'O': 'open', 'H': 'high', 'L': 'low', 'T': 'date'}
    frame = DataFrame(ticker) \
        .drop('BV', 1) \
        .rename(columns=columns)
    frame['date'] = to_datetime(frame['date'], utc=True, infer_datetime_format=True)
    frame.sort_values('date', inplace=True)
    return frame


def populate_indicators(dataframe: DataFrame) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame
    """
    dataframe['sar'] = ta.SAR(dataframe)
    dataframe['adx'] = ta.ADX(dataframe)
    stoch = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch['fastd']
    dataframe['fastk'] = stoch['fastk']
    dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
    dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
    dataframe['mfi'] = ta.MFI(dataframe)
    dataframe['rsi'] = ta.RSI(dataframe)
    dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
    dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
    dataframe['ao'] = awesome_oscillator(dataframe)
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    hilbert = ta.HT_SINE(dataframe)
    dataframe['htsine'] = hilbert['sine']
    dataframe['htleadsine'] = hilbert['leadsine']
    return dataframe


def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the buy signal for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (dataframe['close'] < dataframe['sma']) &
        (dataframe['tema'] <= dataframe['blower']) &
        (dataframe['mfi'] < 25) &
        (dataframe['fastd'] < 25) &
        (dataframe['adx'] > 30),
        'buy'] = 1

    return dataframe

def populate_sell_trend(dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the sell signal for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (crossed_above(dataframe['rsi'], 70)),
        'sell'] = 1

    return dataframe


def analyze_ticker(pair: str) -> DataFrame:
    """
    Get ticker data for given currency pair, push it to a DataFrame and
    add several TA indicators and buy signal to it
    :return DataFrame with ticker data and indicator data
    """
    ticker_hist = get_ticker_history(pair)
    if not ticker_hist:
        logger.warning('Empty ticker history for pair %s', pair)
        return DataFrame()

    dataframe = parse_ticker_dataframe(ticker_hist)
    dataframe = populate_indicators(dataframe)
    dataframe = populate_buy_trend(dataframe)
    dataframe = populate_sell_trend(dataframe)
    # TODO: buy_price and sell_price are only used by the plotter, should probably be moved there
    dataframe.loc[dataframe['buy'] == 1, 'buy_price'] = dataframe['close']
    dataframe.loc[dataframe['sell'] == 1, 'sell_price'] = dataframe['close']
    return dataframe


def get_signal(pair: str, signal: SignalType) -> bool:
    """
    Calculates current signal based several technical analysis indicators
    :param pair: pair in format BTC_ANT or BTC-ANT
    :return: True if pair is good for buying, False otherwise
    """
    dataframe = analyze_ticker(pair)
    if dataframe.empty:
        return False

    latest = dataframe.iloc[-1]

    # Check if dataframe is out of date
    signal_date = arrow.get(latest['date'])
    if signal_date < arrow.now() - timedelta(minutes=10):
        return False

    result = latest[signal.value] == 1
    logger.debug('%s_trigger: %s (pair=%s, signal=%s)', signal.value, latest['date'], pair, result)
    return result
