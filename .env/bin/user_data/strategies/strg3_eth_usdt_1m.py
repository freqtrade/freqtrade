# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from talib._ta_lib import ULTOSC, MACD, LINEARREG_ANGLE, TSF, MFI, EMA, MA, BBANDS, CORREL, CCI, HT_DCPHASE, \
    HT_TRENDMODE, HT_TRENDLINE, MIN, MAX, HT_PHASOR, HT_SINE, LINEARREG, ATR, NATR, SAR

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib

"""

"""


# This class is a sample. Feel free to customize it.
class strg3_ETHUSDT_1m(IStrategy):
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
        "30": 0.1085576426119433,
        "82": 0.029443202051755248,
        "164": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1963679962572551

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.2294395227514193
    trailing_stop_positive_offset = 0.3040424465654783
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
            'lr': {'color': 'green'},
            # 'angle_mid': {'color': 'blue'},
            'tsf_mid': {'color': 'orange'},
        },
        'subplots': {
            "correl_h_l": {
                'correl_h_l': {'color': 'black'}
            },
            "cci": {
                'cci': {'color': 'red'}
            },
            "natr": {
                'natr': {'color': 'red'}
            },
            "ht_phase": {
                'ht_phase': {'color': 'red'}
            },
            "ht_trendmode": {
                'ht_trendmode': {'color': 'red'}
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


        # dataframe['mfi'] = MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'],
        #                        timeperiod=15)
        dataframe['cci_30'] = CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=30)
        dataframe['cci_45'] = CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=45)
        dataframe['trend_line'] = HT_TRENDLINE(dataframe['close'])

        dataframe['angle_short'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=5)
        dataframe['angle_mid'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=10)
        dataframe['angle_long'] = LINEARREG_ANGLE(dataframe['close'], timeperiod=30)
        dataframe['angle_trend_45'] = LINEARREG_ANGLE(dataframe['trend_line'], timeperiod=45)
        dataframe['angle_trend_30'] = LINEARREG_ANGLE(dataframe['trend_line'], timeperiod=30)
        # dataframe['atr'] = ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=30)
        dataframe['natr'] = NATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['tsf_short'] = TSF(dataframe['close'], timeperiod=30)

        dataframe['correl_h_l_30'] = CORREL(dataframe['high'], dataframe['low'], timeperiod=30)
        dataframe['correl_h_l_3'] = CORREL(dataframe['high'], dataframe['low'], timeperiod=3)
        dataframe['correl_h_l_10'] = CORREL(dataframe['high'], dataframe['low'], timeperiod=10)
        # dataframe['correl_close_last_close'] = CORREL(dataframe['close'].shift(1), dataframe['close'], timeperiod=30)
        #
        # dataframe['correl_tsf_long_close'] = CORREL(dataframe['tsf_long'], dataframe['close'], timeperiod=45)
        # dataframe['correl_tsf_mid_close'] = CORREL(dataframe['tsf_mid'], dataframe['close'], timeperiod=45)
        # dataframe['correl_tsf_short_close'] = CORREL(dataframe['tsf_short'], dataframe['close'], timeperiod=5)
        #
        # dataframe['correl_angle_short_close'] = CORREL(dataframe['angle_short'], dataframe['close'], timeperiod=45)
        # dataframe['correl_angle_mid_close'] = CORREL(dataframe['angle_mid'], dataframe['close'], timeperiod=45)
        # dataframe['correl_angle_long_close'] = CORREL(dataframe['angle_long'], dataframe['close'], timeperiod=45)
        #
        # dataframe['correl_mfi_close'] = CORREL(dataframe['mfi'], dataframe['close'], timeperiod=45)
        #
        # dataframe['ema'] = EMA(dataframe['close'], timeperiod=7)
        dataframe['upperband'], dataframe['middleband'], dataframe['lowerband'] = BBANDS(dataframe['close'],
                                                                                         timeperiod=28, nbdevup=2,
                                                                                         nbdevdn=2, matype=0)

        # dataframe['correl_ht_sine_close'] = CORREL(dataframe['ht_sine'], dataframe['close'], timeperiod=30)
        dataframe['sar'] = SAR(dataframe['high'], dataframe['low'], acceleration=0.02, maximum=0.2)



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
                    (0 > dataframe['cci_30']) &
                    (0.15 < dataframe['natr']) &
                    (0 < (dataframe['trend_line'] - dataframe['close'])) &
                    (0.75227 < dataframe['correl_h_l_30']) &
                    # (dataframe['angle_long'] > 0) &
                    # (0.44494 < dataframe['correl_angle_mid_close']) &
                    # (-0.97597 < dataframe['correl_mfi_close']) & #
                    # (57 > dataframe['mfi']) &
                    # ((dataframe['close'] - dataframe['tsf_long']) > -3) &
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
                    (50 > dataframe['cci_30']) &
                    # (100 < dataframe['cci_30']) &
                    (0 < (dataframe['close'] - dataframe['trend_line'])) &
                    (0.2 < dataframe['natr']) &
                    (0.4 < dataframe['correl_h_l_30']) &

                    # (1 == dataframe['ht_trendmode']) &
                    # ((dataframe['tsf_long'] - dataframe['close']) < -3) &
                    # (dataframe['angle_mid'] < 50) &
                    # (dataframe['angle_long'] < 0) &
                    # (dataframe['angle_short'] > -9) &
                    # (dataframe['correl_mfi_close'] > -0.4065) & #
                    # (0.2552 < dataframe['correl_angle_long_close']) &
                    # (0.46789 < dataframe['correl_angle_mid_close']) &
                    # (57 < dataframe['mfi']) &
                    # (0.00585 < (dataframe['quadrature'] - dataframe['inphase'])) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
