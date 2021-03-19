# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, LINEARREG_ANGLE, TSF, MFI, EMA, MA, BBANDS, CORREL

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib

"""

"""


# This class is a sample. Feel free to customize it.
class strg2_ETHUSDT_1m_prod(IStrategy):
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
        "0": 0.14918562374415067,
        "34": 0.058385372968253246,
        "81": 0.02650295809826818,
        "164": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.22617753002701768

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.32322177120674106
    trailing_stop_positive_offset = 0.3600439363018107
    trailing_only_offset_is_reached = False

    # Optimal ticker interval for the strategy.
    timeframe = '1m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

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
        return [("ETH/USDT", "1m")]

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
                                                                                 slowperiod=26, signalperiod=7)

        dataframe['mfi'] = MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'],
                               timeperiod=30)
        dataframe['angle_short'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=30)
        dataframe['angle_mid'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=720)
        dataframe['angle_long'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=1440)

        dataframe['tsf_long'] = TSF(dataframe['close'], timeperiod=1440)
        dataframe['tsf_mid'] = TSF(dataframe['close'], timeperiod=720)
        dataframe['tsf_short'] = TSF(dataframe['close'], timeperiod=30)

        dataframe['correl_h_l'] = CORREL(dataframe['high'], dataframe['low'], timeperiod=30)
        dataframe['correl_close_last_close'] = CORREL(dataframe['close'].shift(1), dataframe['close'], timeperiod=30)

        dataframe['correl_tsf_long_close'] = CORREL(dataframe['tsf_long'], dataframe['close'], timeperiod=1440)
        dataframe['correl_tsf_mid_close'] = CORREL(dataframe['tsf_mid'], dataframe['close'], timeperiod=720)
        dataframe['correl_tsf_short_close'] = CORREL(dataframe['tsf_short'], dataframe['close'], timeperiod=30)

        dataframe['correl_angle_short_close'] = CORREL(dataframe['angle_short'], dataframe['close'], timeperiod=30)
        dataframe['correl_angle_mid_close'] = CORREL(dataframe['angle_mid'], dataframe['close'], timeperiod=720)
        dataframe['correl_angle_long_close'] = CORREL(dataframe['angle_long'], dataframe['close'], timeperiod=1440)

        dataframe['correl_hist_close'] = CORREL(dataframe['macdhist'], dataframe['close'], timeperiod=30)
        dataframe['correl_mfi_close'] = CORREL(dataframe['mfi'], dataframe['close'], timeperiod=30)




        dataframe['ema'] = EMA(dataframe['close'], timeperiod=15)
        dataframe['ma'] = MA(dataframe['close'], timeperiod=15, matype=0)
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
                    (qtpylib.crossed_above(dataframe['close'], dataframe['middleband'])) &
                    ((dataframe['tsf_mid'] - dataframe['close']) > 10) &
                    ((dataframe['tsf_mid'] - dataframe['close']) > -23) &
                    (dataframe['angle_mid'] > -31) &
                    (0.6120614546605481 < dataframe['correl_angle_short_close']) &
                    (0.4128274947005438 < dataframe['correl_angle_mid_close']) &
                    (18 < dataframe['mfi']) &
                    (-18 < (dataframe['ema'] - dataframe['middleband'])) &

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
                    (qtpylib.crossed_below(dataframe['close'], dataframe['middleband'])) &
                    ((dataframe['tsf_mid'] - dataframe['close']) > 29) &
                    ((dataframe['tsf_short'] - dataframe['close']) > -94) &
                    (dataframe['angle_mid'] > -10) &
                    (0.2602396802858502 < dataframe['correl_tsf_mid_close']) &
                    (-0.8240439026300765 < dataframe['correl_angle_short_close']) &
                    (-0.9736515042784262 < dataframe['correl_angle_mid_close']) &
                    (80 < dataframe['mfi']) &
                    (26 < (dataframe['middleband'] - dataframe['ema'])) &
                    (18 > dataframe['macdhist']) &


                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
