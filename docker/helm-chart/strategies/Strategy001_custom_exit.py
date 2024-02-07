# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Strategy001_custom_exit(IStrategy):
    INTERFACE_VERSION = 3
    '\n    Strategy 001_custom_exit\n    author@: Gerald Lonlas, froggleston\n    github@: https://github.com/freqtrade/freqtrade-strategies\n\n    How to use it?\n    > python3 ./freqtrade/main.py -s Strategy001_custom_exit\n    '
    INTERFACE_VERSION: int = 3
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {'60': 0.01, '30': 0.03, '20': 0.04, '0': 0.05}
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.1
    # Optimal timeframe for the strategy
    timeframe = '5m'
    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    # run "populate_indicators" only for new candle
    process_only_new_candles = True
    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    # Optional order type mapping
    order_types = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}

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
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['rsi'] = ta.RSI(dataframe, 14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with entry column
        """  # green bar
        dataframe.loc[qtpylib.crossed_above(dataframe['ema20'], dataframe['ema50']) & (dataframe['ha_close'] > dataframe['ema20']) & (dataframe['ha_open'] < dataframe['ha_close']), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with entry column
        """  # red bar
        dataframe.loc[qtpylib.crossed_above(dataframe['ema50'], dataframe['ema100']) & (dataframe['ha_close'] < dataframe['ema20']) & (dataframe['ha_open'] > dataframe['ha_close']), 'exit_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        """
        Sell only when matching some criteria other than those used to generate the exit signal
        :return: str exit_reason, if any, otherwise None
        """
        # get dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # get the current candle
        current_candle = dataframe.iloc[-1].squeeze()
        # if RSI greater than 70 and profit is positive, then exit
        if current_candle['rsi'] > 70 and current_profit > 0:
            return 'rsi_profit_exit'
        # else, hold
        return None