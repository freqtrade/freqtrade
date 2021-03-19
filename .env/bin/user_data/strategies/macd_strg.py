# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, SAR, LINEARREG_ANGLE, TEMA, STOCHRSI, STOCH, STOCHF, RSI

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class macd_ethbtc_1m(IStrategy):
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
    INTERFACE_VERSION = 1

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.04025819697656752,
        "7": 0.015188707936204564,
        "18": 0.005472487470606337,
        "41": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.33515742514178193

    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.34089
    trailing_stop_positive_offset = 0.43254
    trailing_only_offset_is_reached = False

    # Optimal ticker interval for the strategy.
    timeframe = '1m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'close': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "OU": {
                'ou': {'color': 'red'},
            }
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
        return [("ETH/BTC", "1m")]

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

        # MACD
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = MACD(dataframe['close'], fastperiod=12,
                                                                                 slowperiod=26, signalperiod=9)
        dataframe['macd_angle'] = LINEARREG_ANGLE(dataframe['macd'], timeperiod=3)
        dataframe['macdhist_angle'] = LINEARREG_ANGLE(dataframe['macd'], timeperiod=3)


        # Parabolic SAR
        dataframe['sar'] = SAR(dataframe['high'], dataframe['low'], acceleration=0, maximum=0)
        dataframe['sar_angle'] = LINEARREG_ANGLE(dataframe['sar'], timeperiod=3)

        # Linear angle 
        dataframe['angle'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=14)

        dataframe['tema'] = TEMA(dataframe['close'], timeperiod=30)
        dataframe['sr_fastk'], dataframe['sr_fastd'] = STOCHRSI(dataframe['close'], timeperiod=14, fastk_period=5,
                                                                fastd_period=3, fastd_matype=0)
        dataframe['sr_fastd_angle'] = LINEARREG_ANGLE(dataframe['sr_fastd'], timeperiod=4)

        dataframe['slowk'], dataframe['slowd'] = STOCH(dataframe['high'], dataframe['low'], dataframe['close'],
                                                       fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
                                                       slowd_matype=0)
        dataframe['slowd_angle'] = LINEARREG_ANGLE(dataframe['slowd'], timeperiod=3)
        dataframe['sf_fastk'], dataframe['sf_fastd'] = STOCHF(dataframe['high'], dataframe['low'], dataframe['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
        dataframe['sf_fastd_angle'] = LINEARREG_ANGLE(dataframe['sf_fastd'], timeperiod=3)

        dataframe['rsi'] = RSI(dataframe['close'], timeperiod=14)
        dataframe['rsi_angle'] = LINEARREG_ANGLE(dataframe['rsi'], timeperiod=5)


        # # first check if dataprovider is available
        # if self.dp:
        #     if self.dp.runmode in ('live', 'dry_run'):
        #         ob = self.dp.orderbook(metadata['pair'], 1)
        #         dataframe['best_bid'] = ob['bids'][0][0]
        #         dataframe['best_ask'] = ob['asks'][0][0]
        #
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
                    (qtpylib.crossed_above(dataframe['close'], dataframe['sar'])) &
                    (0 > (dataframe['sar'] - dataframe['close'])) &
                    (1 > (dataframe['sar'] - dataframe['sar'].shift(3))) &
                    (1 > dataframe['macd']) &
                    (1 > dataframe['macdhist']) &
                    (0 < (dataframe['macdhist'] - dataframe['macdhist'].shift(3))) &
                    (0 < dataframe['angle']) &
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
                    (qtpylib.crossed_below(dataframe['sar'], dataframe['close'])) &
                    (dataframe['uo'] < 69) &
                    (0 > (dataframe['sar'] - dataframe['close'])) &
                    (1 > (dataframe['sar'] - dataframe['sar'].shift(3))) &
                    (0 < dataframe['macd']) &
                    (1 < dataframe['macdhist']) &
                    (0 > (dataframe['macdhist'] - dataframe['macdhist'].shift(3))) &
                    (0 > dataframe['angle']) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe


"""
buy1:
trigger = macd cross above macdsignal
guard = sar < close
guard = 70 > ou > 50
guard = lenear_angle > 0

buy2: 
trigger = sar < close
guard = macd > macdsignal
guard = 70 > ou > 50
buy3:
trigger = ou cross below 30 & close > max(close)

sell1:
trigger = macd cross below macdsignal
guard = sar > close
guard = 50 > ou > 20
sell2:
trigger = sar > close
guard = macd < macdsignal
guard = 50 > ou > 20
"""

"""
+--------+---------+----------+------------------+--------------+------------------------------+----------------+-------------+                                                 
|   Best |   Epoch |   Trades |    Win Draw Loss |   Avg profit |                       Profit |   Avg duration |   Objective |
|--------+---------+----------+------------------+--------------+------------------------------+----------------+-------------|
| * Best |   3/500 |     1194 |    523  654   17 |       -0.12% |   -0.06906927 BTC (-138.00%) |      1,559.8 m |     2.71708 |
| * Best |   6/500 |      100 |     20    0   80 |       -0.07% |   -0.00352831 BTC   (-7.05%) |         11.5 m |     1.87067 |                                                 
|   Best |  37/500 |       12 |      6    0    6 |        0.49% |    0.00294367 BTC    (5.88%) |         18.2 m |     1.86009 |                                                 
|   Best |  67/500 |       10 |      6    0    4 |        0.89% |    0.00443786 BTC    (8.87%) |         19.5 m |     1.85245 |                                                 
|   Best |  91/500 |       73 |     21    5   47 |        0.00% |    0.00007645 BTC    (0.15%) |         10.2 m |     1.85215 |                                                 
|   Best |  92/500 |       48 |     17    2   29 |        0.05% |    0.00116911 BTC    (2.34%) |         10.4 m |     1.85189 |                                                 
|   Best |  94/500 |       12 |      6    0    6 |        0.69% |    0.00416071 BTC    (8.31%) |         17.8 m |     1.85143 |                                                 
|   Best | 110/500 |       18 |      6    1   11 |        0.36% |    0.00327838 BTC    (6.55%) |         14.3 m |     1.85113 |                                                 
|   Best | 257/500 |       48 |     16    0   32 |        0.06% |    0.00154456 BTC    (3.09%) |         11.3 m |      1.8505 |                                                 
|   Best | 388/500 |       74 |     23   14   37 |        0.01% |    0.00045826 BTC    (0.92%) |          8.2 m |     1.84665 |                                                 
 [Epoch 500 of 500 (100%)] ||                                                                                                          | [Time:  1:04:49, Elapsed Time: 1:04:49]
2021-01-31 01:14:47,851 - freqtrade.optimize.hyperopt - INFO - 500 epochs saved to '/home/yakov/PycharmProjects/freqtrade/.env/bin/user_data/hyperopt_results/hyperopt_results_2021-01-31_00-07-23.pickle'.

Best result:

   388/500:     74 trades. 23/14/37 Wins/Draws/Losses. Avg profit   0.01%. Median profit  -0.02%. Total profit  0.00045826 BTC (   0.92Σ%). Avg duration   8.2 min. Objective: 1.84665


    # Buy hyperspace params:
    buy_params = {
        'angle-enabled': False,
        'macd-enabled': False,
        'macd_value': 0.73579,
        'macdhist_shift': 0.87895,
        'macdhist_value': 0.29935,
        'sar-enabled': False,
        'sar_ratio': 0.43819,
        'sar_shift': 0.98992,
        'trigger': 'sell-macd_cross_signal'
    }

    # Sell hyperspace params:
    sell_params = {
        'angle_h_s': 0.07105,
        'macd_value_s': 0.08408,
        'macdhist_shift_s': -0.72567,
        'macdhist_value_s': 0.77324,
        'sar_ratio_s': 0.58347,
        'sar_shift_s': 0.81212,
        'sell-angle-enabled': True,
        'sell-macd-enabled': False,
        'sell-sar-enabled': False,
        'sell-uo-enabled': True,
        'trigger': 'sell-macd_cross_signal',
        'uo_l_s': 14
    }

    # ROI table:
    minimal_roi = {
        "0": 0.07193,
        "3": 0.0382,
        "5": 0.01183,
        "7": 0
    }

    # Stoploss:
    stoploss = -0.25471

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.04697
    trailing_stop_positive_offset = 0.05329
    trailing_only_offset_is_reached = True

"""
