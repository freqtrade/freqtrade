# --- Do not remove these libs ---
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa
from wao.wao_strategy import WAOStrategy


class bbrsi_scalp(WAOStrategy):
    brain = "Freq_bbrsi_scalp"

    def __init__(self, config: dict):
        super().__init__(config, self.brain, 8, 0.15)
        self.coin = str(config.get('pairs')[0]).split('/')[0]

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    if self.coin == "BTC":
        # ROI table:
        minimal_roi = {
            "0": 0.266,
            "40": 0.081,
            "96": 0.033,
            "191": 0
        }

        # Stoploss:
        stoploss = -0.309

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.322
        trailing_stop_positive_offset = 0.345
        trailing_only_offset_is_reached = True

    if self.coin == "XMR":
        # ROI table:
        minimal_roi = {
            "0": 0.249,
            "39": 0.062,
            "95": 0.013,
            "197": 0
        }

        # Stoploss:
        stoploss = -0.303

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.089
        trailing_stop_positive_offset = 0.11
        trailing_only_offset_is_reached = False

    if self.coin == "XLM":
        # ROI table:
        minimal_roi = {
            "0": 0.22,
            "40": 0.101,
            "85": 0.032,
            "205": 0
        }

        # Stoploss:
        stoploss = -0.325

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.328
        trailing_stop_positive_offset = 0.385
        trailing_only_offset_is_reached = True

    if self.coin == "LTC":
        # ROI table:
        minimal_roi = {
            "0": 0.249,
            "30": 0.075,
            "90": 0.022,
            "209": 0
        }

        # Stoploss:
        stoploss = -0.292

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.341
        trailing_stop_positive_offset = 0.362
        trailing_only_offset_is_reached = True

    if self.coin == "XRP":
        # ROI table:
        minimal_roi = {
            "0": 0.169,
            "40": 0.056,
            "88": 0.038,
            "204": 0
        }

        # Stoploss:
        stoploss = -0.347

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.158
        trailing_stop_positive_offset = 0.255
        trailing_only_offset_is_reached = False

    if self.coin == "ADA":
        minimal_roi = {
            "0": 0.226,
            "40": 0.054,
            "94": 0.028,
            "213": 0
        }

        # Stoploss:
        stoploss = -0.347

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.35
        trailing_stop_positive_offset = 0.422
        trailing_only_offset_is_reached = False

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

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

        # Stoch
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowk'] = stoch['slowk']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # SAR Parabol
        # dataframe['sar'] = ta.SAR(dataframe)

        # Hammer: values [0, 100]
        # dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (dataframe['close'].shift(1) < dataframe['bb_lowerband']) &
                    (dataframe['close'] > dataframe['bb_lowerband']) &
                    (dataframe['rsi'] < 50)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['bb_upperband']) |
                    (dataframe['rsi'] > 60)
            ),
            'sell'] = 1
        return dataframe
