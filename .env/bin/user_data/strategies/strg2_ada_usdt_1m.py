# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, LINEARREG_ANGLE, TSF, MFI, EMA, MA, BBANDS, CORREL, SAR, HT_TRENDLINE, RSI, NATR

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib

"""

"""


# This class is a sample. Feel free to customize it.
class strg2_ADAUSDT_1m(IStrategy):
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
        "0": 0.0816,
        "30": 0.04762,
        "60": 0.01142,
        "150": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # Stoploss:
    stoploss = -0.20102

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.10581
    trailing_stop_positive_offset = 0.11224
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
            'correl_h_l': {'color': 'black'},
            'upperband': {'color': 'green'},
            'middleband': {'color': 'green'},
            'lowerband': {'color': 'green'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "mfi": {
                'mfi': {'color': 'red'},
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
        dataframe['tsf_short'] = TSF(dataframe['close'], timeperiod=5)
        dataframe['tsf_mid'] = TSF(dataframe['close'], timeperiod=30)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe, weighted=False, fast=5, slow=34)
        dataframe['rsi'] = RSI(dataframe['close'], timeperiod=14)


        dataframe['angle_tsf_mid'] = LINEARREG_ANGLE(dataframe['tsf_mid'], timeperiod=5)
        dataframe['angle_tsf_mid'] = LINEARREG_ANGLE(dataframe['tsf_short'], timeperiod=5)

        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = MACD(dataframe['close'], fastperiod=10,
                                                                                 slowperiod=24, signalperiod=7)
        dataframe['sar'] = SAR(dataframe['high'], dataframe['low'])
        dataframe['mfi'] = MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'],
                               timeperiod=14)
        dataframe['correl_h_l'] = CORREL(dataframe['high'], dataframe['low'], timeperiod=30)
        dataframe['upperband'], dataframe['middleband'], dataframe['lowerband'] = BBANDS(dataframe['close'],
                                                                                         timeperiod=30, nbdevup=2,
                                                                                         nbdevdn=2, matype=0)
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
                    (60 > dataframe['mfi']) &
                    # (0.15 < dataframe['natr']) &
                    (dataframe['angle_tsf_mid'] > 0) &

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
                    # (-0.00354 > dataframe['macdhist']) &
                    (50.23156 > dataframe['mfi']) &
                    (dataframe['angle_tsf_mid'] < 0) &


                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
