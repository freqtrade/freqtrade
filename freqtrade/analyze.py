import logging
import time
from datetime import timedelta

import arrow
import talib.abstract as ta
from pandas import DataFrame, to_datetime

from freqtrade import exchange
from freqtrade.exchange import Bittrex, get_ticker_history
from freqtrade.vendor.qtpylib.indicators import awesome_oscillator

logger = logging.getLogger(__name__)


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
    dataframe['cci'] = ta.CCI(dataframe)
    dataframe['rsi'] = ta.RSI(dataframe)
    dataframe['mom'] = ta.MOM(dataframe)
    dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
    dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
    dataframe['ao'] = awesome_oscillator(dataframe)
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    return dataframe


def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the buy trend for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe.ix[
        (dataframe['close'] < dataframe['sma']) &
        (dataframe['tema'] <= dataframe['blower']) &
        (dataframe['mfi'] < 25) &
        (dataframe['fastd'] < 25) &
        (dataframe['adx'] > 30),
        'buy'] = 1
    dataframe.ix[dataframe['buy'] == 1, 'buy_price'] = dataframe['close']

    return dataframe


def analyze_ticker(pair: str) -> DataFrame:
    """
    Get ticker data for given currency pair, push it to a DataFrame and
    add several TA indicators and buy signal to it
    :return DataFrame with ticker data and indicator data
    """
    data = get_ticker_history(pair)
    dataframe = parse_ticker_dataframe(data)

    if dataframe.empty:
        logger.warning('Empty dataframe for pair %s', pair)
        return dataframe

    dataframe = populate_indicators(dataframe)
    dataframe = populate_buy_trend(dataframe)
    return dataframe


def get_buy_signal(pair: str) -> bool:
    """
    Calculates a buy signal based several technical analysis indicators
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

    signal = latest['buy'] == 1
    logger.debug('buy_trigger: %s (pair=%s, signal=%s)', latest['date'], pair, signal)
    return signal


def plot_analyzed_dataframe(pair: str) -> None:
    """
    Calls analyze() and plots the returned dataframe
    :param pair: pair as str
    :return: None
    """
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    # Init Bittrex to use public API
    exchange._API = Bittrex({'key': '', 'secret': ''})
    dataframe = analyze_ticker(pair)

    # Two subplots sharing x axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle(pair, fontsize=14, fontweight='bold')
    ax1.plot(dataframe.index.values, dataframe['close'], label='close')
    # ax1.plot(dataframe.index.values, dataframe['sell'], 'ro', label='sell')
    ax1.plot(dataframe.index.values, dataframe['sma'], '--', label='SMA')
    ax1.plot(dataframe.index.values, dataframe['tema'], ':', label='TEMA')
    ax1.plot(dataframe.index.values, dataframe['blower'], '-.', label='BB low')
    ax1.plot(dataframe.index.values, dataframe['buy_price'], 'bo', label='buy')
    ax1.legend()

    ax2.plot(dataframe.index.values, dataframe['adx'], label='ADX')
    ax2.plot(dataframe.index.values, dataframe['mfi'], label='MFI')
    # ax2.plot(dataframe.index.values, [25] * len(dataframe.index.values))
    ax2.legend()

    ax3.plot(dataframe.index.values, dataframe['fastk'], label='k')
    ax3.plot(dataframe.index.values, dataframe['fastd'], label='d')
    ax3.plot(dataframe.index.values, [20] * len(dataframe.index.values))
    ax3.legend()

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()


if __name__ == '__main__':
    # Install PYQT5==5.9 manually if you want to test this helper function
    while True:
        for p in ['BTC_ANT', 'BTC_ETH', 'BTC_GNT', 'BTC_ETC']:
            plot_analyzed_dataframe(p)
        time.sleep(60)
