# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, BBANDS, CORREL, MAX, AROON, HT_PHASOR, HT_SINE, HT_DCPHASE, HT_TRENDMODE, \
    HT_TRENDLINE, CCI, AROONOSC, \
    RSI, MFI, LINEARREG_ANGLE, TSF

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib

"""

"""


# This class is a sample. Feel free to customize it.
class strg3_ADAUSDT_1m_test(IStrategy):
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
        "0": 0.2,
        "18": 0.07762,
        "65": 0.02142,
        "158": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # Stoploss:
    stoploss = -0.17

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    # Optimal ticker interval for the strategy.
    timeframe = '1m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 130

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
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
        },
        'subplots': {
            "sine": {
                'leadsine': {'color': 'green'},
                'ht_sine': {'color': 'orange'},
            },
            "aroon": {
                'aroon_down': {'color': 'brown'},
                'aroon_up': {'color': 'blue'},
            },
            "cci": {
                'cci': {'color': 'yellow'},
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
        return [("ADA/USDT", "1m")]

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

        dataframe['aroon_down'], dataframe['aroon_up'] = AROON(dataframe['high'], dataframe['low'], timeperiod=120)
        dataframe['aroonosc'] = AROONOSC(dataframe['high'], dataframe['low'], timeperiod=120)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = MACD(dataframe['close'], fastperiod=30,
                                                                                 slowperiod=60, signalperiod=15)
        dataframe['upperband'], dataframe['middleband'], dataframe['lowerband'] = BBANDS(dataframe['close'],
                                                                                         timeperiod=30, nbdevup=2,
                                                                                         nbdevdn=2, matype=0)
        dataframe['rsi'] = RSI(dataframe['close'], timeperiod=30)
        dataframe['ul'] = ULTOSC(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod1=5, timeperiod2=15,
                                 timeperiod3=30)
        # dataframe['slowk'], dataframe['slowd'] = STOCH(dataframe['high'], dataframe['low'], dataframe['close'],
        #                                                fastk_period=15, slowk_period=7, slowk_matype=0, slowd_period=7,
        #                                                slowd_matype=0)
        dataframe['mfi'] = MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'],
                               timeperiod=30)
        dataframe['angle_short'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=30)
        dataframe['tsf_short'] = TSF(dataframe['close'], timeperiod=10)

        dataframe['correl_h_l'] = CORREL(dataframe['high'], dataframe['low'], timeperiod=30)
        dataframe['ht_phase'] = HT_DCPHASE(dataframe['close'])
        dataframe['ht_trendmode'] = HT_TRENDMODE(dataframe['close'])
        dataframe['ht_trendline'] = HT_TRENDLINE(dataframe['close'])
        dataframe['inphase'], dataframe['quadrature'] = HT_PHASOR(dataframe['close'])
        dataframe['ht_sine'], dataframe['leadsine'] = HT_SINE(dataframe['close'])
        dataframe['correl_sine_trend'] = CORREL(dataframe['leadsine'], dataframe['ht_trendmode'], timeperiod=10)
        dataframe['correl_ht_sine_trend'] = CORREL(dataframe['ht_sine'], dataframe['ht_trendmode'], timeperiod=10)
        dataframe['correl_ht_sine_close'] = CORREL(dataframe['ht_sine'], dataframe['close'], timeperiod=10)
        dataframe['cci'] = CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=120)
        # dataframe['ht_phase_min'] = MIN(dataframe['ht_phase'], timeperiod=30)
        # dataframe['ht_phase_max'] = MAX(dataframe['ht_phase'], timeperiod=60)
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
                    (qtpylib.crossed_above(dataframe['leadsine'], dataframe['ht_sine'])) &
                    (dataframe['cci'] < 0) &
                    (dataframe['angle_short'] > 0) &
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
                    (qtpylib.crossed_below(dataframe['aroon_up'], 83)) &
                    (dataframe['angle_short'] < 0) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                      # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
