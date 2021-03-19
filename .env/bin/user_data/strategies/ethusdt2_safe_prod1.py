# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, LINEARREG_ANGLE, TSF, MFI, EMA, MA, BBANDS, CORREL, MAX, MIN, SAR, CCI, \
    HT_TRENDLINE, HT_DCPERIOD, HT_TRENDMODE, HT_SINE, RSI

from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib

"""

"""


# This class is a sample. Feel free to customize it.
class strg2_ETHUSDT_1h_prod1(IStrategy):
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
        "0": 0.07371,
        "9817": 0.0461,
        "14487": 0.0254,
        "15960": 0
    }
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.23371

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.11193
    trailing_stop_positive_offset = 0.20381
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

        # dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = MACD(dataframe['close'], fastperiod=12,
        #                                                                          slowperiod=26, signalperiod=7)

        # dataframe['cci'] = CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=30)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe, weighted=False, fast=5, slow=34)
        # dataframe['vwap'] = qtpylib.vwap(dataframe)
        # dataframe['vwap'] = qtpylib.rolling_vwap(dataframe)
        dataframe['rsi'] = RSI(dataframe['close'], timeperiod=14)
        # dataframe['tsf_long'] = TSF(dataframe['close'], timeperiod=200)
        # dataframe['tsf_short'] = TSF(dataframe['close'], timeperiod=7)
        dataframe['tsf_mid'] = TSF(dataframe['close'], timeperiod=50)

        dataframe['angle_tsf_mid'] = LINEARREG_ANGLE(dataframe['tsf_mid'], timeperiod=10)
        # dataframe['angle'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=50)

        # dataframe['tsf_max'] = MAX(dataframe['tsf_mid'], timeperiod=30)
        # dataframe['tsf_min'] = MIN(dataframe['tsf_mid'], timeperiod=30)
        # dataframe['sar'] = SAR(dataframe['high'], dataframe['low'])
        # dataframe['sine'], dataframe['leadsine'] = HT_SINE(dataframe['close'])
        # dataframe['trend'] = HT_TRENDLINE(dataframe['close'])
        # dataframe['mfi'] = MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'],
        #                        timeperiod=5)
        # dataframe['angle_trend_mid'] = LINEARREG_ANGLE(dataframe['trend'], timeperiod=10)

        # dataframe['upperband'], dataframe['middleband'], dataframe['lowerband'] = BBANDS(dataframe['close'],
        #                                                                                  timeperiod=30, nbdevup=2,
        #                                                                                  nbdevdn=2, matype=0)
        # if not self.dp:
        #     # Don't do anything if DataProvider is not available.
        #     return dataframe
        # # Get the informative pair
        # informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1d')
        # informative['trend'] = HT_TRENDLINE(informative['close'])
        # informative['period'] = HT_DCPERIOD(informative['close'])
        # informative['mode'] = HT_TRENDMODE(informative['close'])
        # # informative['sine'], informative['leadsine'] = HT_SINE(informative['close'])
        # informative['angle_trend'] = LINEARREG_ANGLE(informative['trend'], timeperiod=5)
        # # informative['sar'] = SAR(informative['high'], informative['low'])
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '1d', ffill=True)
        # return dataframe.set_index('date')['2021-01-01':].reset_index()
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
                    (qtpylib.crossed_above(dataframe['angle_tsf_mid'], -50)) &

                    # (dataframe['sine_1h'] < dataframe['leadsine_1h']) &
                    # (dataframe['tsf_mid'] > dataframe['close']) &
                    # (dataframe['ao'] > -5) &
                    # (dataframe['rsi'] < 36) &
                    (dataframe['rsi'] > 14) &

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
                    (qtpylib.crossed_below(dataframe['angle_tsf_mid'], 50)) &
                    # (dataframe['sine_1h'] > dataframe['leadsine_1h']) &
                    # (dataframe['sar_1d'] < dataframe['close']) &
                    # (dataframe['tsf_mid'] < dataframe['close']) &
                    (dataframe['rsi'] > 50) &
                    (dataframe['ao'] < -10) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
