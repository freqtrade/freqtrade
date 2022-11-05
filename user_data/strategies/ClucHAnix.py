import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from wao.wao_strategy import WAOStrategy
from wao.brain_config2 import *


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class ClucHAnix(WAOStrategy):
    BRAIN = "Freq_ClucHAnix"

    def __init__(self, config: dict):
        super().__init__(config, 8, 0.15)
    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01965,
        'bbdelta-tail': 0.95089,
        'close-bblower': 0.00799,
        'closedelta-close': 0.00556,
        'rocr-1h': 0.54904
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-fisher': 0.38414,
        'sell-bbmiddle-close': 1.07634
    }

    # ROI table:
    minimal_roi = {
        "0": 0.02139,
        "36": 0.01666,
        "174": 0.01191,
        "463": 0.00874,
        "593": 0.00514,
        "604": 0.00031,
        "786": 0
    }

    # Stoploss:
    stoploss = -0.23144

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.3207
    trailing_stop_positive_offset = 0.3849
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """

    timeframe = '1m'

    # Make sure these match or are not overridden in config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(params['rocr-1h'])
            ) &
            ((
                     (dataframe['lower'].shift().gt(0)) &
                     (dataframe['bbdelta'].gt(dataframe['ha_close'] * params['bbdelta-close'])) &
                     (dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close'])) &
                     (dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail'])) &
                     (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                     (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
             ) |
             (
                     (dataframe['ha_close'] < dataframe['ema_slow']) &
                     (dataframe['ha_close'] < params['close-bblower'] * dataframe['bb_lowerband'])
             )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (dataframe['fisher'] > params['sell-fisher']) &
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] > dataframe['ha_close']) &
            ((dataframe['ha_close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        return dataframe


class ClucHAnix_ETH(ClucHAnix):
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01566,
        'bbdelta-tail': 0.8478,
        'close-bblower': 0.00998,
        'closedelta-close': 0.00614,
        'rocr-1h': 0.61579,
        'volume': 27
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 1.02894,
        'sell-fisher': 0.38414
    }

    # ROI table:
    minimal_roi = {
        "0": 0.14414,
        "13": 0.10123,
        "20": 0.03256,
        "47": 0.0177,
        "132": 0.01016,
        "177": 0.00328,
        "277": 0
    }

    # Stoploss:
    stoploss = -0.02

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.0116
    trailing_only_offset_is_reached = False


class ClucHAnix_BTC(ClucHAnix):
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01192,
        'bbdelta-tail': 0.96183,
        'close-bblower': 0.01212,
        'closedelta-close': 0.01039,
        'rocr-1h': 0.53422,
        'volume': 27
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.98016,
        'sell-fisher': 0.38414
    }

    # ROI table:
    minimal_roi = {
        "0": 0.19724,
        "15": 0.14323,
        "33": 0.07688,
        "52": 0.03011,
        "144": 0.01616,
        "307": 0.0063,
        "449": 0
    }

    # Stoploss:
    stoploss = -0.11356

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01544
    trailing_stop_positive_offset = 0.11438
    trailing_only_offset_is_reached = False


class ClucHAnix_USD(ClucHAnix):
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01806,
        'bbdelta-tail': 0.85912,
        'close-bblower': 0.01158,
        'closedelta-close': 0.01466,
        'rocr-1h': 0.51901,
        'volume': 26
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.96094,
        'sell-fisher': 0.38414
    }

    # ROI table:
    minimal_roi = {
        "0": 0.16139,
        "11": 0.12608,
        "54": 0.08335,
        "140": 0.03423,
        "197": 0.0123,
        "325": 0.00649,
        "417": 0
    }

    # Stoploss:
    stoploss = -0.17654

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0101
    trailing_stop_positive_offset = 0.02952
    trailing_only_offset_is_reached = False
