# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, SAR, LINEARREG_ANGLE, TEMA, TSF, CCI, ATR, CORREL, \
    BOP, WMA, KAMA, HT_DCPERIOD, HT_TRENDMODE, HT_SINE

from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class quick_btcusdt_1m(IStrategy):
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
        "0": 0.06443,
        "360": 0.06597,
        "1790": 0.0108,
        "2116": 0

    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # Stoploss:
    stoploss = -0.15825

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.3274
    trailing_stop_positive_offset = 0.38967
    trailing_only_offset_is_reached = True


    # Optimal ticker interval for the strategy.
    timeframe = '1m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

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
        'buy': 'fok',
        'sell': 'fok'
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
        return [("BTC/USDT", "1h")]

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

        dataframe['period'] = HT_DCPERIOD(dataframe['close'])

        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = MACD(dataframe['close'], fastperiod=12,
                                                                                 slowperiod=26, signalperiod=9)
        dataframe['cci'] = CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=30)
        dataframe['sar'] = SAR(dataframe['high'], dataframe['low'])
        dataframe['wma'] = WMA(dataframe['close'], timeperiod=30)
        dataframe['wma_ratio'] = (dataframe['close'] - dataframe['wma'])
        dataframe['kama'] = KAMA(dataframe['close'], timeperiod=30)
        dataframe['angle_kama'] = LINEARREG_ANGLE(dataframe['kama'], timeperiod=10)
        dataframe['tsf_mid'] = TSF(dataframe['close'], timeperiod=30)
        dataframe['angle_tsf_mid'] = LINEARREG_ANGLE(dataframe['tsf_mid'], timeperiod=10)
        dataframe['atr'] = ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=30)
        dataframe['uo'] = ULTOSC(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod1=7, timeperiod2=14,
                                 timeperiod3=28)
        dataframe['tema'] = TEMA(dataframe['close'], timeperiod=50)
        dataframe['macd_ratio'] = (dataframe['macd'] - dataframe['macdsignal'])
        dataframe['tsf_ratio'] = (dataframe['tsf_mid'] - dataframe['close'])
        dataframe['correl_h_l'] = CORREL(dataframe['high'], dataframe['low'], timeperiod=30)
        dataframe['correl_tsf_mid_close'] = CORREL(dataframe['tsf_mid'], dataframe['close'], timeperiod=12)
        dataframe['bop'] = BOP(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])

        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
        informative['bop'] = BOP(informative['open'], informative['high'], informative['low'], informative['close'])
        informative['period'] = HT_DCPERIOD(informative['close'])
        informative['mode'] = HT_TRENDMODE(informative['close'])
        informative['sine'], informative['leadsine'] = HT_SINE(informative['close'])

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '1h', ffill=True)
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
                (qtpylib.crossed_above(dataframe['low'], dataframe['tsf_mid']))

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
                (qtpylib.crossed_below(dataframe['high'], dataframe['tsf_mid']))

            ),
            'sell'] = 1
        return dataframe


"""
freqtrade hyperopt --config user_data/config_btcusdt_1m.json --hyperopt hyper_btcusdt_1m --hyperopt-loss OnlyProfitHyperOptLoss --strategy quick_btcusdt_1m -e 500 --spaces all

+--------+---------+----------+------------------+--------------+-------------------------------+----------------+-------------+
|   Best |   Epoch |   Trades |    Win Draw Loss |   Avg profit |                        Profit |   Avg duration |   Objective |
|--------+---------+----------+------------------+--------------+-------------------------------+----------------+-------------|
| * Best |   1/500 |        2 |      1    0    1 |       -3.39% |  -67.10930706 USDT   (-6.77%) |      4,154.0 m |     1.02257 |
| * Best |   2/500 |        2 |      1    0    1 |        0.08% |    1.60699817 USDT    (0.16%) |      1,156.5 m |     0.99946 |                                                
| * Best |   8/500 |        4 |      4    0    0 |        3.59% |  142.17731149 USDT   (14.35%) |        981.5 m |     0.95218 |                                                
|   Best |  39/500 |       13 |     11    0    2 |        3.80% |  489.61841267 USDT   (49.41%) |        736.2 m |     0.83531 |                                                
|   Best |  70/500 |       25 |     19    4    2 |        2.99% |  740.67168932 USDT   (74.74%) |      1,597.7 m |     0.75086 |                                                
|   Best | 106/500 |       52 |     35   12    5 |        1.82% |  937.66182441 USDT   (94.62%) |      1,666.9 m |      0.6846 |                                                
|   Best | 427/500 |       52 |     32   17    3 |        2.83% | 1,460.67464976 USDT  (147.40%) |      2,592.9 m |     0.50868 |                                               
|   Best | 439/500 |      179 |    101   57   21 |        0.83% | 1,480.44117660 USDT  (149.39%) |      3,457.9 m |     0.50203 |                                               
 [Epoch 500 of 500 (100%)] ||                                                                                                          | [Time:  1:22:09, Elapsed Time: 1:22:09]
2021-03-26 22:31:08,615 - freqtrade.optimize.hyperopt - INFO - 500 epochs saved to '/home/crypto_rahino/freqtrade/user_data/hyperopt_results/strategy_quick_btcusdt_1m_hyperopt_results_2021-03-26_21-08-16.pickle'.

Best result:

   439/500:    179 trades. 101/57/21 Wins/Draws/Losses. Avg profit   0.83%. Median profit   0.82%. Total profit  1480.44117660 USDT ( 149.39Σ%). Avg duration 3457.9 min. Objective: 0.50203


    # Buy hyperspace params:
    buy_params = {
        'angle_tsf_mid-enabled': False,
        'angle_tsf_mid-value': 9,
        'atr-enabled': False,
        'atr-value': 180,
        'bop-value': 0.7274,
        'cci-enabled': False,
        'cci-value': 57,
        'correl_h_l-enabled': True,
        'correl_h_l-value': -0.5389,
        'correl_tsf_mid_close-enabled': False,
        'correl_tsf_mid_close-value': -0.8364,
        'macd_ratio-enabled': False,
        'macd_ratio-value': 101,
        'macdhist-enabled': True,
        'macdhist-value': 22,
        'macdsignal-enabled': False,
        'macdsignal-value': -478,
        'trigger': 'macd',
        'tsf_ratio-enabled': False,
        'tsf_ratio-value': -1323,
        'uo-enabled': False,
        'uo-value': 33.5021
    }

    # Sell hyperspace params:
    sell_params = {
        'angle_tsf_mid-enabled': False,
        'angle_tsf_mid-value': 9,
        'atr-enabled': False,
        'atr-value': 180,
        'cci-enabled': False,
        'cci-value': 57,
        'correl_h_l-enabled': True,
        'correl_h_l-value': -0.5389,
        'correl_tsf_mid_close-enabled': False,
        'correl_tsf_mid_close-value': -0.8364,
        'macd_ratio-enabled': False,
        'macd_ratio-value': 101,
        'macdhist-enabled': True,
        'macdhist-value': 22,
        'macdsignal-enabled': False,
        'macdsignal-value': -478,
        'trigger': 'macd',
        'tsf_ratio-enabled': False,
        'tsf_ratio-value': -1323,
        'uo-enabled': False,
        'uo-value': 33.5021
    }

    # ROI table:
    minimal_roi = {
        "0": 0.16344,
        "793": 0.05931,
        "1121": 0.03143,
        "1474": 0
    }

    # Stoploss:
    stoploss = -0.2884

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.21554
    trailing_stop_positive_offset = 0.23749
    trailing_only_offset_is_reached = False


"""