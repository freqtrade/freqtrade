# --- Do not remove these libs ---
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

# --------------------------------


class ProfessionalScalpingStrategy(IStrategy):
    """
    Professional Scalping Strategy
    """

    # Strategy interface version - attribute needed by backtesting tools
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.02
    # trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Hyperopt parameters
    buy_rsi = IntParameter(20, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 80, default=70, space="sell")
    buy_atr_mult = DecimalParameter(0.5, 2.0, default=1.0, space="buy")
    sell_atr_mult = DecimalParameter(0.5, 2.0, default=1.0, space="sell")
    ema_period = IntParameter(50, 250, default=200, space="buy")

    # Bollinger Bands parameters (for future use)
    bb_window = IntParameter(10, 30, default=20, space="buy")
    bb_stddev = DecimalParameter(1.5, 2.5, default=2.0, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance, TA libraries should be used sparingly.
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # EMA
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe),
                                            window=self.bb_window.value,
                                            stds=self.bb_stddev.value)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        """
        dataframe.loc[
            (
                # Placeholder for buy logic
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        """
        dataframe.loc[
            (
                # Placeholder for sell logic
            ),
            'sell'] = 1
        return dataframe
