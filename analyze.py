from datetime import timedelta
import time
import arrow
import matplotlib
import logging

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import requests
from pandas.io.json import json_normalize
from stockstats import StockDataFrame

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_ticker_dataframe(pair):
    """
    Analyses the trend for the given pair
    :param pair: pair as str in format BTC_ETH or BTC-ETH
    :return: StockDataFrame
    """
    minimum_date = arrow.now() - timedelta(hours=6)
    url = 'https://bittrex.com/Api/v2.0/pub/market/GetTicks'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    }
    params = {
        'marketName': pair.replace('_', '-'),
        'tickInterval': 'OneMin',
        '_': minimum_date.timestamp * 1000
    }
    data = requests.get(url, params=params, headers=headers).json()
    if not data['success']:
        raise RuntimeError('BITTREX: {}'.format(data['message']))

    data = [{
        'close': t['C'],
        'volume': t['V'],
        'open': t['O'],
        'high': t['H'],
        'low': t['L'],
        'date': t['T'],
    } for t in sorted(data['result'], key=lambda k: k['T']) if arrow.get(t['T']) > minimum_date]
    dataframe = StockDataFrame(json_normalize(data))

    # calculate StochRSI
    window = 14
    rsi = dataframe['rsi_{}'.format(window)]
    rolling = rsi.rolling(window=window, center=False)
    low = rolling.min()
    high = rolling.max()
    dataframe['stochrsi'] = (rsi - low) / (high - low)
    return dataframe


def populate_trends(dataframe):
    """
    Populates the trends for the given dataframe
    :param dataframe: StockDataFrame
    :return: StockDataFrame with populated trends
    """
    """
    dataframe.loc[
        (dataframe['stochrsi'] < 0.20)
        & (dataframe['close_30_ema'] > (1 + 0.0025) * dataframe['close_60_ema']),
        'underpriced'
    ] = 1
    """
    dataframe.loc[
        dataframe['stochrsi'] < 0.20,
        'underpriced'
    ] = 1
    dataframe.loc[dataframe['underpriced'] == 1, 'buy'] = dataframe['close']
    return dataframe


def get_buy_signal(pair):
    """
    Calculates a buy signal based on StochRSI indicator
    :param pair: pair in format BTC_ANT or BTC-ANT
    :return: True if pair is underpriced, False otherwise
    """
    dataframe = get_ticker_dataframe(pair)
    dataframe = populate_trends(dataframe)
    latest = dataframe.iloc[-1]

    # Check if dataframe is out of date
    signal_date = arrow.get(latest['date'])
    if signal_date < arrow.now() - timedelta(minutes=10):
        return False

    signal = latest['underpriced'] == 1
    logger.debug('buy_trigger: {} (pair={}, signal={})'.format(latest['date'], pair, signal))
    return signal


def plot_dataframe(dataframe, pair):
    """
    Plots the given dataframe
    :param dataframe: StockDataFrame
    :param pair: pair as str
    :return: None
    """

    # Three subplots sharing x axe
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    f.suptitle(pair, fontsize=14, fontweight='bold')
    ax1.plot(dataframe.index.values, dataframe['close'], label='close')
    ax1.plot(dataframe.index.values, dataframe['close_60_ema'], label='EMA(60)')
    ax1.plot(dataframe.index.values, dataframe['close_120_ema'], label='EMA(120)')
    # ax1.plot(dataframe.index.values, dataframe['sell'], 'ro', label='sell')
    ax1.plot(dataframe.index.values, dataframe['buy'], 'bo', label='buy')
    ax1.legend()

    #ax2.plot(dataframe.index.values, dataframe['macd'], label='MACD')
    #ax2.plot(dataframe.index.values, dataframe['macds'], label='MACDS')
    #ax2.plot(dataframe.index.values, dataframe['macdh'], label='MACD Histogram')
    #ax2.plot(dataframe.index.values, [0] * len(dataframe.index.values))
    #ax2.legend()

    ax2.plot(dataframe.index.values, dataframe['stochrsi'], label='StochRSI')
    ax2.plot(dataframe.index.values, [0.80] * len(dataframe.index.values))
    ax2.plot(dataframe.index.values, [0.20] * len(dataframe.index.values))
    ax2.legend()

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()


if __name__ == '__main__':
    while True:
        pair = 'BTC_ANT'
        for pair in ['BTC_ANT', 'BTC_ETH', 'BTC_GNT', 'BTC_ETC']:
            get_buy_signal(pair)
        #dataframe = get_ticker_dataframe(pair)
        #dataframe = populate_trends(dataframe)
        #plot_dataframe(dataframe, pair)
        #time.sleep(60)

