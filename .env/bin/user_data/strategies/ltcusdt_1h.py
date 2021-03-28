# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, LINEARREG_ANGLE, TSF, MFI, EMA, MA, BBANDS, CORREL, MAX, MIN, SAR, CCI, \
    HT_TRENDLINE, HT_DCPERIOD, HT_TRENDMODE, HT_SINE, RSI, NATR, HT_PHASOR

from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib

"""

"""


# This class is a sample. Feel free to customize it.
class ltcusdt_1h(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.11465,
        "607": 0.08395,
        "1517": 0.01583,
        "1780": 0

    }
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.23987

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.29213
    trailing_stop_positive_offset = 0.3379
    trailing_only_offset_is_reached = True


    plot_config = {
        'main_plot': {
            'upperband': {'upperband': 'green'},
            'middleband': {'color': 'green'},
            'lowerband': {'color': 'green'},
            'tsf_mid': {'color': 'white'},
            'ema': {'color': 'white'},
        },
        'subplots': {
            "corr": {
                'correl_h_l': {'color': 'black'},
            },
            "correl_tsf_mid_close": {
                'correl_tsf_mid_close': {'color': 'grey'},
            },
            "correl_angle_short_close": {
                'correl_angle_short_close': {'color': 'blue'},
            },
            "correl_angle_long_close": {
                'correl_angle_long_close': {'color': 'red'},
            },
            "correl_mfi_close": {
                'correl_mfi_close': {'color': 'black'},
            },
            "correl_hist_close": {
                'correl_tsf_mid_close': {'color': 'red'},
            },
            "mfi": {
                'mfi': {'color': 'yellow'},
            },
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return [("ETH/USDT", "1d")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = MACD(dataframe['close'], fastperiod=12,
                                                                                 slowperiod=24, signalperiod=7)
        dataframe['mfi'] = MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'],
                               timeperiod=12)
        dataframe['uo'] = ULTOSC(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod1=7, timeperiod2=12,
                                 timeperiod3=24)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe, weighted=False, fast=5, slow=36)
        dataframe['tsf_mid'] = TSF(dataframe['close'], timeperiod=48)
        dataframe['sar'] = SAR(dataframe['high'], dataframe['low'])
        dataframe['sar_close'] = dataframe['sar'] - dataframe['close']
        dataframe['natr'] = NATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)

        dataframe['angle_tsf_mid'] = LINEARREG_ANGLE(dataframe['tsf_mid'], timeperiod=14)
        dataframe['sine'], dataframe['leadsine'] = HT_SINE(dataframe['close'])
        dataframe['trend'] = HT_TRENDLINE(dataframe['close'])
        dataframe['inphase'], dataframe['quadrature'] = HT_PHASOR(dataframe['close'])

        dataframe['angle_trend_mid'] = LINEARREG_ANGLE(dataframe['trend'], timeperiod=12)

        dataframe['angle'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=12)
        dataframe['angle_min'] = MIN(dataframe['angle_trend_mid'], timeperiod=7)
        dataframe['angle_min_lead'] = MIN(dataframe['angle_trend_mid'], timeperiod=3)
        dataframe['angle_max_lead'] = MAX(dataframe['angle_trend_mid'], timeperiod=3)
        dataframe['angle_max'] = MAX(dataframe['angle_trend_mid'], timeperiod=7)
        dataframe['angle_macdsignal'] = LINEARREG_ANGLE(dataframe['macdsignal'], timeperiod=12)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (qtpylib.crossed_above(dataframe['angle_trend_mid'], 0)) &
                    (dataframe['angle'] > -48) &
                    (-21 < dataframe['angle_macdsignal']) &
                    (dataframe['uo'] > 9) &
                    (-0.90098 < (dataframe['macd'] - dataframe['macdsignal'])) &
                    (-50 < dataframe['ao']) &

                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (qtpylib.crossed_below(dataframe['angle_trend_mid'], 0)) &
                    (dataframe['uo'] > 10) &
                    (dataframe['angle'] > 58) &
                    (10 < dataframe['ao']) &

                    (-3.60339 < (dataframe['macd'] - dataframe['macdsignal'])) &
                    (10 > dataframe['angle_trend_mid']) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe


"""

freqtrade hyperopt --config user_data/config_ltcusdt_1h.json --hyperopt hyper_ltcusdt_1h --hyperopt-loss OnlyProfitHyperOptLoss --strategy ltcusdt_1h -e 500 --spaces all --timerange 20200601-


+--------+---------+----------+------------------+--------------+-------------------------------+----------------+-------------+                                                
|   Best |   Epoch |   Trades |    Win Draw Loss |   Avg profit |                        Profit |   Avg duration |   Objective |
|--------+---------+----------+------------------+--------------+-------------------------------+----------------+-------------|
| * Best |   8/500 |        1 |      1    0    0 |       17.74% |   17.75689174 USDT   (17.74%) |        660.0 m |     0.94087 |
| * Best |  24/500 |       71 |     43   25    3 |        1.30% |   92.24310471 USDT   (92.15%) |      1,950.4 m |     0.69283 |                                                
|   Best |  98/500 |       52 |     34   15    3 |        2.56% |  133.14323340 USDT  (133.01%) |      2,576.5 m |     0.55663 |                                                
|   Best | 134/500 |      119 |     61   52    6 |        1.19% |  141.48585752 USDT  (141.34%) |      2,884.5 m |     0.52885 |                                                
|   Best | 149/500 |      120 |     63   48    9 |        1.22% |  146.96457805 USDT  (146.82%) |      3,091.5 m |     0.51061 |                                                
|   Best | 163/500 |       56 |     38   18    0 |        3.21% |  180.15476192 USDT  (179.97%) |      4,530.0 m |     0.40008 |                                                
 [Epoch 500 of 500 (100%)] ||                                                                                                          | [Time:  0:24:40, Elapsed Time: 0:24:40]
2021-03-20 23:04:51,078 - freqtrade.optimize.hyperopt - INFO - 500 epochs saved to '/home/crypto_rahino/freqtrade/user_data/hyperopt_results/strategy_ltcusdt_1h_hyperopt_results_2021-03-20_22-40-07.pickle'.

Best result:

   163/500:     56 trades. 38/18/0 Wins/Draws/Losses. Avg profit   3.21%. Median profit   1.58%. Total profit  180.15476192 USDT ( 179.97Σ%). Avg duration 4530.0 min. Objective: 0.40008


    # Buy hyperspace params:
    buy_params = {
        'angle-enabled': True,
        'angle-value': -48,
        'angle_macdsignal-enabled': True,
        'angle_macdsignal-value': -21,
        'angle_trend_mid-enabled': False,
        'angle_trend_mid-value': -9,
        'angle_tsf_mid-enabled': False,
        'angle_tsf_mid-value': -18,
        'ao-enabled': True,
        'ao-value': -50,
        'macd-enabled': True,
        'macd-value': -0.90098,
        'macdhist-enabled': False,
        'macdhist-value': 1.44126,
        'macdsignal-enabled': False,
        'macdsignal-value': 10,
        'mfi-enabled': False,
        'mfi-value': 63,
        'natr-enabled': False,
        'natr-value': 1.38067,
        'sar_close-enabled': False,
        'sar_close-value': -3,
        'trigger': 'angle_trend_mid',
        'uo-enabled': True,
        'uo-value': 9
    }

    # Sell hyperspace params:
    sell_params = {
        'angle-enabled': True,
        'angle-value_sell': 58,
        'angle_macdsignal-enabled': True,
        'angle_macdsignal-value_sell': -14,
        'angle_trend_mid-enabled': False,
        'angle_trend_mid-value_sell': -73,
        'angle_tsf_mid-enabled': False,
        'angle_tsf_mid-value_sell': -11,
        'ao-enabled': True,
        'ao-value_sell': 10,
        'macd-enabled': True,
        'macd-value_sell': -3.60339,
        'macdhist-enabled': False,
        'macdhist-value_sell': 4.06627,
        'macdsignal-enabled': False,
        'macdsignal-value_sell': -1,
        'mfi-enabled': False,
        'mfi-value_sell': 13,
        'natr-enabled': False,
        'natr-value_sell': 7.852,
        'sar_close-enabled': False,
        'sar_close-value_sell': 17,
        'trigger': 'angle_trend_mid',
        'uo-enabled': True,
        'uo-value_sell': 10
    }

    # ROI table:
    minimal_roi = {
        "0": 0.11465,
        "607": 0.08395,
        "1517": 0.01583,
        "1780": 0
    }

    # Stoploss:
    stoploss = -0.20987

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.29213
    trailing_stop_positive_offset = 0.3379
    trailing_only_offset_is_reached = True

"""
