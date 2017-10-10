import logging
import time
from datetime import timedelta

import arrow
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.exchange import get_ticker_history

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_ticker_dataframe(ticker: list, minimum_date: arrow.Arrow) -> DataFrame:
    """
    Analyses the trend for the given pair
    :param pair: pair as str in format BTC_ETH or BTC-ETH
    :return: DataFrame
    """
    df = DataFrame(ticker) \
        .drop('BV', 1) \
        .rename(columns={'C':'close', 'V':'volume', 'O':'open', 'H':'high', 'L':'low', 'T':'date'}) \
        .sort_values('date')
    return df[df['date'].map(arrow.get) > minimum_date]


def populate_indicators(dataframe: DataFrame) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame
    """
    dataframe['sar'] = ta.SAR(dataframe, 0.02, 0.22)
    dataframe['adx'] = ta.ADX(dataframe)
    stoch = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch['fastd']
    dataframe['fastk'] = stoch['fastk']
    dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
    dataframe['cci'] = ta.CCI(dataframe, timeperiod=5)
    dataframe['sma'] = ta.SMA(dataframe, timeperiod=100)
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=4)
    dataframe['mfi'] = ta.MFI(dataframe)
    return dataframe


def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the buy trend for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """

    dataframe.loc[
        (dataframe['close'] < dataframe['sma']) &
        (dataframe['cci'] < -100) &
        (dataframe['tema'] <= dataframe['blower']) &
        (dataframe['mfi'] < 30) &
        (dataframe['fastd'] < 20) &
        (dataframe['adx'] > 20),
        'buy'] = 1
    dataframe.loc[dataframe['buy'] == 1, 'buy_price'] = dataframe['close']

    return dataframe


def analyze_ticker(pair: str) -> DataFrame:
    """
    Get ticker data for given currency pair, push it to a DataFrame and
    add several TA indicators and buy signal to it
    :return DataFrame with ticker data and indicator data
    """
    minimum_date = arrow.utcnow().shift(hours=-24)
    data = get_ticker_history(pair, minimum_date)
    dataframe = parse_ticker_dataframe(data['result'], minimum_date)

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


def plot_dataframe(dataframe: DataFrame, pair: str) -> None:
    """
    Plots the given dataframe
    :param dataframe: DataFrame
    :param pair: pair as str
    :return: None
    """

    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    # Two subplots sharing x axis
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(pair, fontsize=14, fontweight='bold')
    ax1.plot(dataframe.index.values, dataframe['sar'], 'g_', label='pSAR')
    ax1.plot(dataframe.index.values, dataframe['close'], label='close')
    # ax1.plot(dataframe.index.values, dataframe['sell'], 'ro', label='sell')
    ax1.plot(dataframe.index.values, dataframe['sma'], '--', label='SMA')
    ax1.plot(dataframe.index.values, dataframe['buy_price'], 'bo', label='buy')
    ax1.legend()

    # ax2.plot(dataframe.index.values, dataframe['adx'], label='ADX')
    ax2.plot(dataframe.index.values, dataframe['mfi'], label='MFI')
    # ax2.plot(dataframe.index.values, [25] * len(dataframe.index.values))
    ax2.legend()

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()


if __name__ == '__main__':
    # Install PYQT5==5.9 manually if you want to test this helper function
    while True:
        test_pair = 'BTC_ETH'
        # for pair in ['BTC_ANT', 'BTC_ETH', 'BTC_GNT', 'BTC_ETC']:
        #     get_buy_signal(pair)
        plot_dataframe(analyze_ticker(test_pair), test_pair)
        time.sleep(60)
