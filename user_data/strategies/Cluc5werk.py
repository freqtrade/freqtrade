import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame
from wao.wao_strategy import WAOStrategy
from wao.brain_config import *


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


class Cluc5werk(WAOStrategy):
    BRAIN = "Freq_Cluc5werk"

    def __init__(self, config: dict):
        super().__init__(config, 8, 0.15)
    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    # 989/1000:    331 trades. 305/9/17 Wins/Draws/Losses. Avg profit   1.54%. Median profit   2.13%. Total profit  0.00510181 BTC ( 509.36Σ%). Avg duration 367.3 min. Objective: -0.69786
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01853,
        'bbdelta-tail': 0.78758,
        'close-bblower': 0.00931,
        'closedelta-close': 0.00169,
        'rocr-1h': 0.8973,
        'volume': 35
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.97103
    }

    # ROI table:
    minimal_roi = {
        "0": 0.02134,
        "275": 0.01745,
        "559": 0.01618,
        "621": 0.0131,
        "791": 0.00843,
        "1048": 0.00443,
        "1074": 0
    }

    # Stoploss:
    stoploss = -0.22405

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.18622
    trailing_stop_positive_offset = 0.23091
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """

    timeframe = '1m'

    # Make sure these match or are not overridden in config
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
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
            'fake_buy'
        ] = 1

        dataframe.loc[
            (dataframe['fake_buy'].shift(1).eq(1)) &
            (dataframe['fake_buy'].eq(1)) &
            (dataframe['volume'] > 0)
            ,
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (dataframe['high'].le(dataframe['high'].shift(1))) &
            (dataframe['high'].shift(1).le(dataframe['high'].shift(2))) &
            (dataframe['close'].le(dataframe['close'].shift(1))) &
            ((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        return dataframe


class Cluc5werk_ETH(Cluc5werk):
    # hyperopt --config user_data/config-backtest-USD.json --hyperopt Cluc5werkHyperopt --hyperopt-loss OnlyProfitHyperOptLoss --strategy Cluc5werk_USD -e 1000 --spaces buy --timeframe 1m --timerange 20210101-
    # 677/1000:    618 trades. 581/20/17 Wins/Draws/Losses. Avg profit   1.23%. Median profit   1.65%. Total profit  379.36403713 USD ( 757.52Σ%). Avg duration 297.5 min. Objective: -1.52505
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00902,
        'bbdelta-tail': 0.91508,
        'close-bblower': 0.00603,
        'closedelta-close': 0.00424,
        'rocr-1h': 0.93725,
        'volume': 38
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.97181
    }

    # ROI table:
    minimal_roi = {
        "0": 0.01648,
        "38": 0.01484,
        "303": 0.01317,
        "597": 0.00952,
        "869": 0.00724,
        "896": 0.00253,
        "1062": 0
    }

    # Stoploss:
    stoploss = -0.33703

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.29564
    trailing_stop_positive_offset = 0.38855
    trailing_only_offset_is_reached = False


class Cluc5werk_BTC(Cluc5werk):
    # hyperopt --config user_data/config-backtest-BTC.json --hyperopt Cluc5werkHyperopt --hyperopt-loss OnlyProfitHyperOptLoss --strategy Cluc5werk_BTC -e 500 --spaces all --timeframe 1m --timerange 20210101-
    # 125/500:    422 trades. 369/14/39 Wins/Draws/Losses. Avg profit   0.97%. Median profit   2.18%. Total profit  0.00408737 BTC ( 408.13Σ%). Avg duration 307.8 min. Objective: -0.36043
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01511,
        'bbdelta-tail': 0.90705,
        'close-bblower': 0.01972,
        'closedelta-close': 0.00099,
        'rocr-1h': 0.97131,
        'volume': 27
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.97906
    }

    # ROI table:
    minimal_roi = {
        "0": 0.0218,
        "242": 0.02079,
        "308": 0.01803,
        "372": 0.01325,
        "390": 0.00905,
        "619": 0.00467,
        "737": 0
    }

    # Stoploss:
    stoploss = -0.14515

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.03046
    trailing_stop_positive_offset = 0.04631
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """


class Cluc5werk_USD(Cluc5werk):
    # hyperopt --config user_data/config-backtest-USD.json --hyperopt Cluc5werkHyperopt --hyperopt-loss OnlyProfitHyperOptLoss --strategy Cluc5werk_USD -e 1000 --spaces buy --timeframe 1m --timerange 20210101-
    # 677/1000:    618 trades. 581/20/17 Wins/Draws/Losses. Avg profit   1.23%. Median profit   1.65%. Total profit  379.36403713 USD ( 757.52Σ%). Avg duration 297.5 min. Objective: -1.52505
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00902,
        'bbdelta-tail': 0.91508,
        'close-bblower': 0.00603,
        'closedelta-close': 0.00424,
        'rocr-1h': 0.93725,
        'volume': 38
    }

    # hyperopt --config user_data/config-backtest-USD.json --hyperopt Cluc5werkHyperopt --hyperopt-loss OnlyProfitHyperOptLoss --strategy Cluc5werk_USD -e 250 --spaces sell --timeframe 1m --timerange 20210101- 
    # 38/250:    609 trades. 573/20/16 Wins/Draws/Losses. Avg profit   1.25%. Median profit   1.65%. Total profit  382.03235064 USD ( 762.84Σ%). Avg duration 304.3 min. Objective: -1.54281
    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.97008
    }

    # hyperopt --config user_data/config-backtest-USD.json --hyperopt Cluc5werkHyperopt --hyperopt-loss OnlyProfitHyperOptLoss --strategy Cluc5werk_USD -e 250 --spaces roi --timeframe 1m --timerange 20210101- 
    # 139/250:    575 trades. 531/28/16 Wins/Draws/Losses. Avg profit   1.38%. Median profit   1.88%. Total profit  396.08871240 USD ( 790.91Σ%). Avg duration 330.9 min. Objective: -1.63637
    # ROI table:
    minimal_roi = {
        "0": 0.01887,
        "150": 0.016,
        "243": 0.01193,
        "471": 0.0103,
        "475": 0.00687,
        "744": 0.00271,
        "793": 0
    }

    # Stoploss:
    stoploss = -0.33703

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.29564
    trailing_stop_positive_offset = 0.38855
    trailing_only_offset_is_reached = False
