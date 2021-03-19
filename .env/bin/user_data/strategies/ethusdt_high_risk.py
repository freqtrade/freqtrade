# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, LINEARREG_ANGLE, TSF, MFI, EMA, MA, BBANDS, CORREL, MAX, MIN, SAR, CCI, \
    HT_TRENDLINE, HT_DCPERIOD, HT_TRENDMODE, HT_SINE, RSI, NATR, STOCH, STOCHF, STOCHRSI

from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib

"""

"""


# This class is a sample. Feel free to customize it.
class ETHUSDT_1m_high_risk(IStrategy):
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
        "0": 0.22703036349783817,
        "30": 0.09085576426119433,
        "82": 0.029443202051755248,
        "164": 0
    }
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # Stoploss:
    stoploss = -0.22515

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.03428
    trailing_stop_positive_offset = 0.05094
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
        return [("ETH/BTC", "1h")]

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
        dataframe['trend'] = HT_TRENDLINE(dataframe['close'])
        dataframe['tsf_short'] = TSF(dataframe['close'], timeperiod=12)
        dataframe['tsf_mid'] = TSF(dataframe['close'], timeperiod=48)

        dataframe['vwap_short'] = qtpylib.rolling_vwap(dataframe, window=5)
        dataframe['vwap_mid'] = qtpylib.rolling_vwap(dataframe, window=90)
        dataframe['vwap_long'] = qtpylib.rolling_vwap(dataframe, window=1440)

        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = MACD(dataframe['close'], fastperiod=12,
                                                                                 slowperiod=26, signalperiod=7)
        dataframe['ao_mid'] = qtpylib.awesome_oscillator(dataframe, weighted=False, fast=5, slow=36)
        dataframe['ao_short'] = qtpylib.awesome_oscillator(dataframe, weighted=False, fast=5, slow=15)

        dataframe['cci'] = CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=30)
        dataframe['rsi'] = RSI(dataframe['close'], timeperiod=14)
        dataframe['slowk'], dataframe['slowd'] = STOCH(dataframe['high'], dataframe['low'], dataframe['close'],
                                                       fastk_period=5, slowk_period=3,
                                                       slowk_matype=0, slowd_period=3,
                                                       slowd_matype=0)
        dataframe['fastk'], dataframe['fastd'] = STOCHF(dataframe['high'], dataframe['low'], dataframe['close'],
                                                        fastk_period=5, fastd_period=3, fastd_matype=0)
        dataframe['fastk'], dataframe['fastd'] = STOCHRSI(dataframe['close'], timeperiod=14, fastk_period=5,
                                                          fastd_period=3, fastd_matype=0)


        dataframe['sar'] = SAR(dataframe['high'], dataframe['low'])

        dataframe['min_high'] = MIN(dataframe['high'], timeperiod=5)
        dataframe['max_low'] = MAX(dataframe['low'], timeperiod=5)
        dataframe['max_low_min_high_ratio'] = dataframe['max_low'] - dataframe['min_high'].shift(15)
        dataframe['correl_h_l_30'] = CORREL(dataframe['high'], dataframe['low'], timeperiod=30)


        dataframe['angle_tsf_short'] = LINEARREG_ANGLE(dataframe['tsf_short'], timeperiod=5)
        dataframe['angle_close'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=5)
        dataframe['sine'], dataframe['leadsine'] = HT_SINE(dataframe['close'])
        dataframe['mode'] = HT_TRENDMODE(dataframe['close'])
        dataframe['corel_mode'] = CORREL(dataframe['mode'], dataframe['close'], timeperiod=30)

        dataframe = dataframe.reset_index().dropna()

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
                    (qtpylib.crossed_below(dataframe['sar'], dataframe['close'])) &
                    (dataframe['angle_trend'] > 0) &
                    (dataframe['angle_tsf_mid'] > 0) &
                    (0.75227 < dataframe['correl_h_l_30']) &
                    (0.4 < dataframe['correl_h_l_30']) &

                    # (dataframe['sine_1h'] < dataframe['leadsine_1h']) &
                    # (dataframe['tsf_mid'] > dataframe['close']) &
                    # (dataframe['angle'] > 0) &
                    # (dataframe['rsi'] <50) &
                    # (dataframe['natr'] > 1.1) &

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
                    (qtpylib.crossed_above(dataframe['sar'], dataframe['close'])) &

                    (dataframe['max_low'] - dataframe['min_high'].shift(15) < 0) &
                    (dataframe['angle_trend'] < 2) &
                    (dataframe['angle_tsf_mid'] < 2) &
                    (dataframe['mode'] == 1) &
                    # (dataframe['close'] < dataframe['vwap']) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
