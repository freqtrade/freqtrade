import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame
from wao.wao_strategy import WAOStrategy


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


class Cluc6werk(WAOStrategy):
    brain = "Freq_Cluc6werk"

    def __init__(self, config: dict):
        super().__init__(config, self.brain, 8, 0.15)

    # Used for "informative pairs"
    stake = 'BTC'
    fiat = 'USD'

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00793,
        'bbdelta-tail': 0.83802,
        'close-bblower': 0.0034,
        'closedelta-close': 0.00613,
        'rocr-1h': 0.64081,
        'volume': 21
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.97703
    }

    # ROI table:
    minimal_roi = {
        "0": 0.0155,
        "109": 0.01075,
        "393": 0.00771,
        "587": 0.00643,
        "711": 0.00377,
        "770": 0.00114,
        "1039": 0
    }

    # Stoploss:
    stoploss = -0.31742

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.31289
    trailing_stop_positive_offset = 0.33275
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """

    timeframe = '1m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        """
        Idea is to have "STAKE/USD" and "COIN/USD" as informative pairs as they move inverse of COIN/STAKE.
        For example, stake currency is BTC, whitelist is */BTC
        Current pair being examined (metadata['pair']) is XLM/BTC
        Be able to have informative pairs BTC/USD and XLM/USD available for use with some indicators for all pairs in the whitelist.
        Ideally have this work gracefully with a change to the stake/whitelist in the config file.
        If a desired informative pair does not exist (e.g. if exchange doesnt trade XLM/USD in this example), simply ignore those indicators without errors.
        """

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']

        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe, timeperiod=28)

        inf_tf = '1h'

        """
        informative = self.dp.get_pair_dataframe(pair="ETH/USDT", timeframe="5m")
        # ETH/USDT RSI based on 5m candles
        informative['rsi'] = ta.RSI(informative, timeperiod=14)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '5m', ffill=True)
        """

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative['rocr'] = ta.ROCR(informative, timeperiod=168)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(params['rocr-1h'])
            ) &
            ((
                     dataframe['lower'].shift().gt(0) &
                     dataframe['bbdelta'].gt(dataframe['close'] * params['bbdelta-close']) &
                     dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close']) &
                     dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail']) &
                     dataframe['close'].lt(dataframe['lower'].shift()) &
                     dataframe['close'].le(dataframe['close'].shift())
             ) |
             (
                     (dataframe['close'] < dataframe['ema_slow']) &
                     (dataframe['close'] < params['close-bblower'] * dataframe['bb_lowerband']) &
                     (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
             )),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            # (dataframe['high'].le(dataframe['high'].shift(1))) &
            # (dataframe['close'] > dataframe['bb_middleband']) &
            (qtpylib.crossed_above((dataframe['close'] * params['sell-bbmiddle-close']), dataframe['bb_middleband'])) &
            # (qtpylib.crossed_above(dataframe['close'],dataframe['bb_middleband'])) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        return dataframe
