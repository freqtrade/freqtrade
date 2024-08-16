# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import datetime
import logging
from typing import Tuple, Optional

import numpy as np  # noqa
import pandas as pd  # noqa

from freqtrade.enums import SignalDirection
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade

pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from freqtrade.strategy import IStrategy, stoploss_from_absolute

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque
logger = logging.getLogger(__name__)

class PlotConfig():

    def __init__(self):
        self.config = {
            'main_plot': {
                resample('bollinger_upperband'): {'color': 'rgba(4,137,122,0.7)'},
                resample('kc_upperband'): {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_middleband'): {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_lowerband'): {'color': 'rgba(4,146,250,0.7)'},
                resample('bollinger_lowerband'): {
                    'color': 'rgba(4,137,122,0.7)',
                    'fill_to': resample('bollinger_upperband'),
                    'fill_color': 'rgba(4,137,122,0.07)'
                },
                resample(ema_fast): {'color': 'purple'},
                resample(ema_middle): {'color': 'yellow'},
                resample(ema_slow): {'color': 'red'},
            },
            'subplots': {
                "ATR": {
                    resample('atr'): {'color': 'firebrick'}
                }
            }
        }

    def add_pivots_in_config(self):
        self.config['main_plot']["pivot_lows"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'olive'
                }
            }
        }
        self.config['main_plot']["pivot_highs"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'violet'
                }
            }
        }
        self.config['main_plot']["pivot_highs"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'violet'
                }
            }
        }
        return self

    def add_divergence_in_config(self, indicator: str):
        # self.config['main_plot']["bullish_divergence_" + indicator + "_occurence"] = {
        #     "plotly": {
        #         'mode': 'markers',
        #         'marker': {
        #             'symbol': 'diamond',
        #             'size': 11,
        #             'line': {
        #                 'width': 2
        #             },
        #             'color': 'orange'
        #         }
        #     }
        # }
        # self.config['main_plot']["bearish_divergence_" + indicator + "_occurence"] = {
        #     "plotly": {
        #         'mode': 'markers',
        #         'marker': {
        #             'symbol': 'diamond',
        #             'size': 11,
        #             'line': {
        #                 'width': 2
        #             },
        #             'color': 'purple'
        #         }
        #     }
        # }
        for i in range(3):
            self.config['main_plot']["bullish_divergence_" + indicator + "_line_" + str(i)] = {
                "plotly": {
                    'mode': 'lines',
                    'line': {
                        'color': 'green',
                        'dash': 'dash'
                    }
                }
            }
            self.config['main_plot']["bearish_divergence_" + indicator + "_line_" + str(i)] = {
                "plotly": {
                    'mode': 'lines',
                    'line': {
                        "color": 'crimson',
                        'dash': 'dash'
                    }
                }
            }
        return self

    def add_total_divergences_in_config(self, dataframe):
        total_bullish_divergences_count = dataframe[resample("total_bullish_divergences_count")]
        total_bullish_divergences_names = dataframe[resample("total_bullish_divergences_names")]
        self.config['main_plot'][resample("total_bullish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': total_bullish_divergences_count,
                'hovertext': total_bullish_divergences_names,
                'textfont': {'size': 11, 'color': 'green'},
                'textposition': 'bottom center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'green'
                }
            }
        }
        total_bearish_divergences_count = dataframe[resample("total_bearish_divergences_count")]
        total_bearish_divergences_names = dataframe[resample("total_bearish_divergences_names")]
        self.config['main_plot'][resample("total_bearish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': total_bearish_divergences_count,
                'hovertext': total_bearish_divergences_names,
                'textfont': {'size': 11, 'color': 'crimson'},
                'textposition': 'top center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'crimson'
                }
            }
        }
        return self


fast = 7
middle = 25
slow = 99
ema_fast = f"ema_{fast}"
ema_middle = f"ema_{middle}"
ema_slow = f"ema_{slow}"


class HarmonicDivergence(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".

    # Max Open Trades:
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "exit_pricing" section in the config.
    can_short: bool = True
    use_custom_stoploss = True
    # use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable: bool = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = None

    timeframe = "15m"

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    # @informative(timeframe=f"{timeframe}")
    # def populate_indicators_adjust(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     if "enter_adjust_trade_position" not in dataframe:
    #         dataframe["enter_adjust_trade_position"] = False
    #     return dataframe

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

        # Get the informative pair
        # informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='15m')
        # informative = resample_to_interval(dataframe, self.get_ticker_indicator() * 15)
        informative = dataframe
        # Momentum Indicators
        # ------------------------------------

        # RSI
        informative['rsi'] = ta.RSI(informative)
        # # Stochastic Slow
        # informative['stoch'] = ta.STOCH(informative)['slowk']
        # # ROC
        # informative['roc'] = ta.ROC(informative)
        # # Ultimate Oscillator
        # informative['uo'] = ta.ULTOSC(informative)
        # Awesome Oscillator
        # informative['ao'] = qtpylib.awesome_oscillator(informative)
        # MACD
        informative['macd'] = ta.MACD(informative)['macd']
        # # Commodity Channel Index
        informative['cci'] = ta.CCI(informative)
        # CMF
        informative['cmf'] = chaikin_money_flow(informative, 20)
        # 止损
        informative['sar'] = ta.SAR(dataframe)

        # # OBV
        # informative['obv'] = ta.OBV(informative)
        # # MFI
        # informative['mfi'] = ta.MFI(informative)
        # # ADX
        # informative['adx'] = ta.ADX(informative)

        # ATR
        informative['atr'] = qtpylib.atr(informative, window=14, exp=False)

        # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe, window=20, atrs=1)
        keltner = emaKeltner(informative)
        informative["kc_upperband"] = keltner["upper"]
        informative["kc_middleband"] = keltner["mid"]
        informative["kc_lowerband"] = keltner["lower"]

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bollinger_upperband'] = bollinger['upper']
        informative['bollinger_lowerband'] = bollinger['lower']

        # EMA - Exponential Moving Average
        informative[ema_fast] = ta.EMA(informative, timeperiod=fast)
        informative[ema_middle] = ta.EMA(informative, timeperiod=middle)
        informative[ema_slow] = ta.EMA(informative, timeperiod=slow)

        pivots = pivot_points(informative)
        informative['pivot_lows'] = pivots['pivot_lows']
        informative['pivot_highs'] = pivots['pivot_highs']

        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below

        initialize_divergences_lists(informative)
        # divergence_finder_dataframe(dataframe,indicator)
        divergence_finder_dataframe(informative, 'rsi')
        # divergence_finder_dataframe(informative, 'stoch')
        # divergence_finder_dataframe(informative, 'roc')
        # divergence_finder_dataframe(informative, 'uo')
        # divergence_finder_dataframe(informative, 'ao')
        divergence_finder_dataframe(informative, 'macd')
        divergence_finder_dataframe(informative, 'cci')
        divergence_finder_dataframe(informative, 'cmf')
        # divergence_finder_dataframe(informative, 'obv')
        # divergence_finder_dataframe(informative, 'mfi')
        # divergence_finder_dataframe(informative, 'adx')

        init_ema_crossed(informative, ema_fast, ema_middle)
        init_ema_crossed(informative, ema_fast, ema_slow)
        init_ema_crossed(informative, ema_middle, ema_slow)

        # print("-------------------informative-------------------")
        # print(informative)
        # print("-------------------dataframe-------------------")
        # print(dataframe)
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '15m', ffill=True)

        # dataframe = resampled_merge(dataframe, informative)
        # print(dataframe[resample("total_bullish_divergences_count")])
        # for index, value in enumerate(dataframe[resample("total_bullish_divergences_count")]):
        #     if value < 0.5:
        #         dataframe[resample("total_bullish_divergences_count")][index] = None
        #         dataframe[resample("total_bullish_divergences")][index] = None
        #         dataframe[resample("total_bullish_divergences_names")][index] = None
        #     else:
        #         print(value)
        #         print(dataframe[resample("total_bullish_divergences")][index])
        #         print(dataframe[resample("total_bullish_divergences_names")][index])
        HarmonicDivergence.plot_config = (
            PlotConfig()
            # .add_pivots_in_config()
            # .add_divergence_in_config('rsi')
            # .add_divergence_in_config('stoch')
            # .add_divergence_in_config('roc')
            # .add_divergence_in_config('uo')
            # .add_divergence_in_config('ao')
            # .add_divergence_in_config('macd')
            # .add_divergence_in_config('cci')
            # .add_divergence_in_config('cmf')
            # .add_divergence_in_config('obv')
            # .add_divergence_in_config('mfi')
            # .add_divergence_in_config('adx')
            .add_total_divergences_in_config(dataframe)
            .config)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        self.enter_long(dataframe)
        self.enter_short(dataframe)
        return dataframe

    def enter_long(self, dataframe):
        dataframe.loc[
            (
                    (dataframe[resample('total_bullish_divergences')].shift() > 0)
                    # & (dataframe[resample(f'{ema_fast}_{ema_middle}_cross_above')])
                    & (dataframe[resample(f'{ema_fast}_{ema_middle}_adhesion')].shift())
                    # # & (dataframe['high'] > dataframe['high'].shift())
                    # & (
                    #     (keltner_middleband_check(dataframe) & (ema_check(dataframe)) & (green_candle(dataframe)))
                    #     # (keltner_middleband_check(dataframe) & (green_candle(dataframe)))
                    #     | (keltner_lowerband_check(dataframe) & (ema_check(dataframe)))
                    #     # | keltner_lowerband_check(dataframe)
                    #     # | (keltner_lowerband_check(dataframe) & (green_candle(dataframe)))
                    #     | (bollinger_lowerband_check(dataframe) & (ema_check(dataframe)))
                    # )
                    & two_bands_check(dataframe)
                    # # & bollinger_keltner_check(dataframe)
                    & ema_crossed_below_check(dataframe).shift()
                    & (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

    def enter_short(self, dataframe):
        dataframe.loc[
            (
                    (dataframe[resample('total_bearish_divergences')].shift() > 0)
                    # & (dataframe[resample(f'{ema_fast}_{ema_middle}_crossed_below')])
                    & (dataframe[resample(f'{ema_fast}_{ema_middle}_adhesion')].shift())
                    # # & (dataframe['high'] > dataframe['high'].shift())
                    # & (
                    #     (keltner_middleband_check(dataframe) & (ema_check(dataframe)) & (green_candle(dataframe)))
                    #     # (keltner_middleband_check(dataframe) & (green_candle(dataframe)))
                    #     | (keltner_lowerband_check(dataframe) & (ema_check(dataframe)))
                    #     # | keltner_lowerband_check(dataframe)
                    #     # | (keltner_lowerband_check(dataframe) & (green_candle(dataframe)))
                    #     | (bollinger_lowerband_check(dataframe) & (ema_check(dataframe)))
                    # )
                    & two_bands_check(dataframe)
                    # # & bollinger_keltner_check(dataframe)
                    & ema_crossed_above_check(dataframe).shift()
                    & (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # 均线名称
        _fast_middle_cross_above_name = f'{ema_fast}_{ema_middle}_cross_above'
        _fast_slow__cross_above_name = f'{ema_fast}_{ema_slow}_cross_above'
        _middle_slow__cross_above_name = f'{ema_middle}_{ema_slow}_cross_above_original'

        _fast_middle_crossed_below_name = f'{ema_fast}_{ema_middle}_crossed_below'
        _fast_slow_crossed_below_name = f'{ema_fast}_{ema_slow}_crossed_below'
        _middle_slow_crossed_below_name = f'{ema_middle}_{ema_slow}_crossed_below_original'

        exit_long = dataframe[_fast_middle_crossed_below_name] & dataframe[_fast_slow_crossed_below_name] & dataframe[
            _middle_slow_crossed_below_name].shift()
        exit_short = dataframe[_fast_middle_cross_above_name] & dataframe[_fast_slow__cross_above_name] & dataframe[
            _middle_slow__cross_above_name].shift()

        dataframe.loc[
            (
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                    & exit_long
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                    & exit_short
            ),
            'exit_short'] = 1
        return dataframe

    def leverage(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_leverage: float,
            max_leverage: float,
            entry_tag: Optional[str],
            side: str,
            **kwargs,
    ) -> float:
        # Return 3.0 in all cases.
        # Bot-logic must make sure it's an allowed leverage and eventually adjust accordingly.
        stabilize_coin = ["BTC", "ETH", "XRP", "LINK", "BNB"]
        is_stabilize_coin = any(string in pair for string in stabilize_coin)
        if (is_stabilize_coin):
            return 15.0
        else:
            return 5.0

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # 使用抛物线SAR作为绝对止损价
        stoploss_price = last_candle['sar']

        # 将绝对价格转换为相对于当前汇率的百分比
        if stoploss_price < current_rate:
            return stoploss_from_absolute(stoploss_price, current_rate, is_short=trade.is_short)
        # 返回最大的止损值，保持当前止损价不变
        return None

    # 保证当前信号在上一次买入的时间间隔两根时间区间以外
    def is_not_nearest_timeframe(self, current_time, date_last_filled_utc, timeframe):
        # 调仓获取时间是当前两个时间区间都会取到入场信号、为避免第二根柱子接著买，所以乘以2
        timeframe_minutes = timeframe_to_minutes(timeframe)
        offset = self.config.get("exchange", {}).get("outdated_offset", 5)
        return date_last_filled_utc < (current_time - datetime.timedelta(minutes=timeframe_minutes * 2 + offset))

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        try:
            if trade.is_open and (self.is_not_nearest_timeframe(current_time, trade.date_last_filled_utc,self.timeframe)):
                dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                # dataframe.loc[dataframe.iloc[-1:].index, ] = True
                # latest["enter_adjust_trade_position"] = True
                enter_signal, _ = self.get_entry_signal(trade.pair, self.timeframe, dataframe)
                enter = True if (trade.is_short and (enter_signal == SignalDirection.SHORT)) or (
                        (not trade.is_short) and (enter_signal == SignalDirection.LONG)) else False
                if enter:
                    filled_entries = trade.select_filled_orders(trade.entry_side)
                    count_of_entries = trade.nr_of_successful_entries
                    stake_amount = filled_entries[0].stake_amount
                    return stake_amount * (count_of_entries + 1)
        except Exception as e:
            logger.error("An error occurred: ", exc_info=True)
            pass
        return None


def resample(indicator):
    # return "resample_15_" + indicator
    return indicator


def two_bands_check(dataframe):
    check = (
        # ((dataframe['low'] < dataframe['bollinger_lowerband']) & (dataframe['high'] > dataframe['kc_lowerband'])) |
        ((dataframe[resample('low')] < dataframe[resample('kc_lowerband')]) & (
                dataframe[resample('high')] > dataframe[resample('kc_upperband')]))  # 1
        #  ((dataframe['low'] < dataframe['kc_lowerband']) & (dataframe['high'] > dataframe['kc_middleband'])) # 2
        # | ((dataframe['low'] < dataframe['kc_middleband']) & (dataframe['high'] > dataframe['kc_upperband'])) # 2
    )
    return ~check


# 均线上线穿判断
def init_ema_crossed(dataframe, fast_row_name, slow_row_name):
    # 上线穿
    _cross_above_name = f'{fast_row_name}_{slow_row_name}_cross_above'
    _crossed_below_name = f'{fast_row_name}_{slow_row_name}_crossed_below'
    # 上穿
    _cross_above_series = dataframe[f"{_cross_above_name}_original"] = qtpylib.crossed_above(
        dataframe[resample(fast_row_name)],
        dataframe[resample(slow_row_name)])
    # 下穿
    _crossed_below_series = dataframe[f"{_crossed_below_name}_original"] = qtpylib.crossed_below(
        dataframe[resample(fast_row_name)],
        dataframe[resample(slow_row_name)])

    for index in range(len(_cross_above_series)):
        _cross_above = _cross_above_series.iloc[index]
        _crossed_below = _crossed_below_series.iloc[index]
        if _cross_above | _cross_above_series.iloc[index - 1]:
            _cross_above_series.iloc[index] = (~_crossed_below) & True
        if _crossed_below | _crossed_below_series.iloc[index - 1]:
            _crossed_below_series.iloc[index] = (~_cross_above) & True

    dataframe[resample(_cross_above_name)] = _cross_above_series
    dataframe[resample(_crossed_below_name)] = _crossed_below_series

    # 判断均线粘连
    distance = dataframe[resample(fast_row_name)] - dataframe[resample(slow_row_name)].abs()
    threshold = distance.mean() * 0.5
    result = (distance - threshold) > 0
    dataframe[resample(f'{fast_row_name}_{slow_row_name}_adhesion')] = result
    return (dataframe[resample(_cross_above_name)], dataframe[resample(_crossed_below_name)])


# 判断EMA均线是否存在下穿,存在一个则返回false
def ema_crossed_below_check(dataframe):
    dataframe[f'{ema_fast}_{ema_middle}_cross'] = qtpylib.crossed_below(dataframe[resample(ema_fast)],
                                                                        dataframe[resample(ema_middle)])
    dataframe[f'{ema_fast}_{ema_slow}_cross'] = qtpylib.crossed_below(dataframe[resample(ema_fast)],
                                                                      dataframe[resample(ema_slow)])
    dataframe[f'{ema_middle}_{ema_slow}_cross'] = qtpylib.crossed_below(dataframe[resample(ema_middle)],
                                                                        dataframe[resample(ema_slow)])
    return ~(
            dataframe[f'{ema_fast}_{ema_middle}_cross']
            | dataframe[f'{ema_fast}_{ema_slow}_cross']
            | dataframe[f'{ema_middle}_{ema_slow}_cross']
    )


# 判断EMA均线是否存在上穿,存在一个则返回false
def ema_crossed_above_check(dataframe):
    dataframe[f'{ema_fast}_{ema_middle}_cross'] = qtpylib.crossed_above(dataframe[resample(ema_fast)],
                                                                        dataframe[resample(ema_middle)])
    dataframe[f'{ema_fast}_{ema_slow}_cross'] = qtpylib.crossed_above(dataframe[resample(ema_fast)],
                                                                      dataframe[resample(ema_slow)])
    dataframe[f'{ema_middle}_{ema_slow}_cross'] = qtpylib.crossed_above(dataframe[resample(ema_middle)],
                                                                        dataframe[resample(ema_slow)])
    return ~(
            dataframe[f'{ema_fast}_{ema_middle}_cross']
            | dataframe[f'{ema_fast}_{ema_slow}_cross']
            | dataframe[f'{ema_middle}_{ema_slow}_cross']
    )


def green_candle(dataframe):
    return dataframe[resample('open')] < dataframe[resample('close')]


def keltner_middleband_check(dataframe):
    return (dataframe[resample('low')] < dataframe[resample('kc_middleband')]) & (
            dataframe[resample('high')] > dataframe[resample('kc_middleband')])


def keltner_lowerband_check(dataframe):
    return (dataframe[resample('low')] < dataframe[resample('kc_lowerband')]) & (
            dataframe[resample('high')] > dataframe[resample('kc_lowerband')])


def bollinger_lowerband_check(dataframe):
    return (dataframe[resample('low')] < dataframe[resample('bollinger_lowerband')]) & (
            dataframe[resample('high')] > dataframe[resample('bollinger_lowerband')])


def bollinger_keltner_check(dataframe):
    return (dataframe[resample('bollinger_lowerband')] < dataframe[resample('kc_lowerband')]) & (
            dataframe[resample('bollinger_upperband')] > dataframe[resample('kc_upperband')])


def initialize_divergences_lists(dataframe: DataFrame):
    dataframe["total_bullish_divergences"] = np.empty(len(dataframe['close'])) * np.nan
    dataframe["total_bullish_divergences_count"] = np.empty(len(dataframe['close'])) * np.nan
    dataframe["total_bullish_divergences_count"] = [0 if x != x else x for x in
                                                    dataframe["total_bullish_divergences_count"]]
    dataframe["total_bullish_divergences_names"] = np.empty(len(dataframe['close'])) * np.nan
    dataframe["total_bullish_divergences_names"] = ['' if x != x else x for x in
                                                    dataframe["total_bullish_divergences_names"]]
    dataframe["total_bearish_divergences"] = np.empty(len(dataframe['close'])) * np.nan
    dataframe["total_bearish_divergences_count"] = np.empty(len(dataframe['close'])) * np.nan
    dataframe["total_bearish_divergences_count"] = [0 if x != x else x for x in
                                                    dataframe["total_bearish_divergences_count"]]
    dataframe["total_bearish_divergences_names"] = np.empty(len(dataframe['close'])) * np.nan
    dataframe["total_bearish_divergences_names"] = ['' if x != x else x for x in
                                                    dataframe["total_bearish_divergences_names"]]


def add_divergences(dataframe: DataFrame, indicator: str):
    (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = divergence_finder_dataframe(dataframe,
                                                                                                           indicator)
    dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences
    # for index, bearish_line in enumerate(bearish_lines):
    #     dataframe['bearish_divergence_' + indicator + '_line_'+ str(index)] = bearish_line
    dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences
    # for index, bullish_line in enumerate(bullish_lines):
    #     dataframe['bullish_divergence_' + indicator + '_line_'+ str(index)] = bullish_line


def divergence_finder_dataframe(dataframe: pd.DataFrame, indicator_source: str) -> tuple[pd.Series, pd.Series]:
    # 定义处理数据框和指标源以查找分歧的函数，并指定返回值为元组

    # 获取数据框中 'pivot_lows' 不为 NaN 的行的索引
    low_indices = dataframe[dataframe['pivot_lows'].notna()].index
    # 获取数据框中 'pivot_highs' 不为 NaN 的行的索引
    high_indices = dataframe[dataframe['pivot_highs'].notna()].index

    # 初始化熊势线条列表，初始值为全 NaN 的数组
    bearish_lines = [np.nan * np.empty(len(dataframe['close']))]
    # 初始化熊势分歧数组，初始值为全 NaN
    bearish_divergences = np.nan * np.empty(len(dataframe['close']))
    # 初始化牛势线条列表，初始值为全 NaN 的数组
    bullish_lines = [np.nan * np.empty(len(dataframe['close']))]
    # 初始化牛势分歧数组，初始值为全 NaN
    bullish_divergences = np.nan * np.empty(len(dataframe['close']))

    for index in range(len(dataframe)):
        # 遍历数据框的索引

        # 通过索引直接获取当前行的数据
        row = dataframe.loc[index]

        # 调用函数查找熊势分歧
        bearish_occurence = bearish_divergence_finder(dataframe, indicator_source, high_indices, index)
        if bearish_occurence is not None:
            # 如果找到熊势分歧

            # 提取分歧的两个枢轴索引
            prev_pivot, current_pivot = bearish_occurence
            # 获取对应枢轴的 'close' 值
            bearish_prev_pivot = dataframe.loc[prev_pivot, 'close']
            bearish_current_pivot = dataframe.loc[current_pivot, 'close']
            # 获取对应枢轴的指标值
            bearish_ind_prev_pivot = dataframe.loc[prev_pivot, indicator_source]
            bearish_ind_current_pivot = dataframe.loc[current_pivot, indicator_source]
            # 计算两个枢轴之间的距离
            length = current_pivot - prev_pivot

            can_exist = True
            found_drawable_line = False
            for bearish_lines_index in range(len(bearish_lines)):
                # 遍历熊势线条列表

                actual_bearish_lines = bearish_lines[bearish_lines_index]
                can_draw = True
                for i in range(length + 1):
                    # 计算线条上的点和对应指标点
                    point = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                    indicator_point = bearish_ind_prev_pivot + (
                            bearish_ind_current_pivot - bearish_ind_prev_pivot) * i / length
                    if i != 0 and i != length:
                        # 检查条件，如果不满足存在性条件
                        if (point <= dataframe.loc[prev_pivot + i, 'close'] or indicator_point <= dataframe.loc[
                            prev_pivot + i, indicator_source]):
                            can_exist = False
                    if not np.isnan(actual_bearish_lines[prev_pivot + i]):
                        # 检查是否可绘制
                        can_draw = False
                if not can_exist:
                    # 如果不存在，退出内层循环
                    break
                if can_draw:
                    # for i in range(length + 1):
                    #     # 绘制线条
                    #     actual_bearish_lines[prev_pivot + i] = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                    found_drawable_line = True
                    break
            if can_exist and found_drawable_line:
                # 如果条件满足，记录熊势分歧相关数据
                bearish_divergences[index] = row['close']
                dataframe.loc[index, "total_bearish_divergences"] = row['close']
                # if index > 30:
                #     dataframe.loc[index - 30, "total_bearish_divergences_count"] += 1
                #     dataframe.loc[index - 30, "total_bearish_divergences_names"] += indicator_source.upper() + '<br>'

        # 类似地处理牛势分歧
        bullish_occurence = bullish_divergence_finder(dataframe, indicator_source, low_indices, index)
        if bullish_occurence is not None:
            prev_pivot, current_pivot = bullish_occurence
            bullish_prev_pivot = dataframe.loc[prev_pivot, 'close']
            bullish_current_pivot = dataframe.loc[current_pivot, 'close']
            bullish_ind_prev_pivot = dataframe.loc[prev_pivot, indicator_source]
            bullish_ind_current_pivot = dataframe.loc[current_pivot, indicator_source]
            length = current_pivot - prev_pivot

            can_exist = True
            found_drawable_line = False
            for bullish_lines_index in range(len(bullish_lines)):
                actual_bullish_lines = bullish_lines[bullish_lines_index]
                can_draw = True
                for i in range(length + 1):
                    point = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                    indicator_point = bullish_ind_prev_pivot + (
                            bullish_ind_current_pivot - bullish_ind_prev_pivot) * i / length
                    if i != 0 and i != length:
                        if (point >= dataframe.loc[prev_pivot + i, 'close'] or indicator_point >= dataframe.loc[
                            prev_pivot + i, indicator_source]):
                            can_exist = False
                    if not np.isnan(actual_bullish_lines[prev_pivot + i]):
                        can_draw = False
                if not can_exist:
                    break
                if can_draw:
                    # for i in range(length + 1):
                    #     actual_bullish_lines[prev_pivot + i] = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                    found_drawable_line = True
                    break
            if can_exist and found_drawable_line:
                bullish_divergences[index] = row['close']
                dataframe.loc[index, "total_bullish_divergences"] = row['close']
                # if index > 30:
                #     dataframe.loc[index - 30, "total_bullish_divergences_count"] += 1
                #     dataframe.loc[index - 30, "total_bullish_divergences_names"] += indicator_source.upper() + '<br>'

    return bearish_divergences, bearish_lines, bullish_divergences, bullish_lines
    # 返回熊势和牛势的分歧及线条数据


def bearish_divergence_finder(dataframe, indicator_source, high_indices, index):
    # 定义查找熊势分歧的函数

    if index in high_indices:
        # 如果当前索引在高枢轴索引列表中

        current_pivot = index
        occurences = list(dict.fromkeys(high_indices))
        current_index = occurences.index(current_pivot)
        for i in range(current_index - 1, max(current_index - 6, -1), -1):
            # 在之前的几个索引中查找

            prev_pivot = occurences[i]
            if pd.isnull(prev_pivot):
                # 如果前一个枢轴为 NaN，返回
                return
            if ((dataframe.loc[current_pivot, 'pivot_highs'] < dataframe.loc[prev_pivot, 'pivot_highs'] and
                 dataframe.loc[current_pivot, indicator_source] > dataframe.loc[prev_pivot, indicator_source])
                    or (dataframe.loc[current_pivot, 'pivot_highs'] > dataframe.loc[prev_pivot, 'pivot_highs'] and
                        dataframe.loc[current_pivot, indicator_source] < dataframe.loc[prev_pivot, indicator_source])):
                # 如果满足熊势分歧条件，返回两个枢轴
                return prev_pivot, current_pivot
    return None
    # 如果未找到，返回 None


def bullish_divergence_finder(dataframe, indicator_source, low_indices, index):
    # 定义查找牛势分歧的函数，逻辑与熊势类似

    if index in low_indices:
        current_pivot = index
        occurences = list(dict.fromkeys(low_indices))
        current_index = occurences.index(current_pivot)
        for i in range(current_index - 1, max(current_index - 6, -1), -1):
            prev_pivot = occurences[i]
            if pd.isnull(prev_pivot):
                return
            if ((dataframe.loc[current_pivot, 'pivot_lows'] < dataframe.loc[prev_pivot, 'pivot_lows'] and dataframe.loc[
                current_pivot, indicator_source] > dataframe.loc[prev_pivot, indicator_source])
                    or (dataframe.loc[current_pivot, 'pivot_lows'] > dataframe.loc[prev_pivot, 'pivot_lows'] and
                        dataframe.loc[current_pivot, indicator_source] < dataframe.loc[prev_pivot, indicator_source])):
                return prev_pivot, current_pivot
    return None
    # 如果未找到，返回 None


from enum import Enum


class PivotSource(Enum):
    HighLow = 0
    Close = 1


def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source: PivotSource = PivotSource.Close) -> DataFrame:
    high_source = None
    low_source = None

    if pivot_source == PivotSource.Close:
        high_source = 'close'
        low_source = 'close'
    elif pivot_source == PivotSource.HighLow:
        high_source = 'high'
        low_source = 'low'

    pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
    pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
    last_values = deque()

    # find pivot points
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True
            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source,
                                                                                    low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            if is_greater:
                pivot_points_highs[index - window] = getattr(current_value, high_source)
            if is_less:
                pivot_points_lows[index - window] = getattr(current_value, low_source)
            last_values.popleft()

    # find last one
    if len(last_values) >= window + 2:
        current_value = last_values[-2]
        is_greater = True
        is_less = True
        for window_index in range(0, window):
            left = last_values[-2 - window_index - 1]
            right = last_values[-1]
            local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source,
                                                                                left, right)
            is_greater &= local_is_greater
            is_less &= local_is_less
        if is_greater:
            pivot_points_highs[index - 1] = getattr(current_value, high_source)
        if is_less:
            pivot_points_lows[index - 1] = getattr(current_value, low_source)

    return pd.DataFrame(index=dataframe.index, data={
        'pivot_lows': pivot_points_lows,
        'pivot_highs': pivot_points_highs
    })


def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[
    bool, bool]:
    is_greater = True
    is_less = True
    if (getattr(current_value, high_source) < getattr(left, high_source) or
            getattr(current_value, high_source) < getattr(right, high_source)):
        is_greater = False

    if (getattr(current_value, low_source) > getattr(left, low_source) or
            getattr(current_value, low_source) > getattr(right, low_source)):
        is_less = False
    return (is_greater, is_less)


def emaKeltner(dataframe):
    keltner = {}
    atr = qtpylib.atr(dataframe, window=7)
    ema25 = ta.EMA(dataframe, timeperiod=25)
    keltner['upper'] = ema25 + atr
    keltner['mid'] = ema25
    keltner['lower'] = ema25 - atr
    return keltner


def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')