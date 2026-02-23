# --- Do not remove these libs ---
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


# --------------------------------


class ProfessionalScalpingStrategy(IStrategy):
    """
    Professional Scalping Strategy

    Fixed stake amount is required. Set stake_amount in config.json.
    """

    # Strategy interface version - attribute needed by backtesting tools
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # Fixed ROI at 1.0% (0.010) as per task requirements
    minimal_roi = {"0": 0.010}

    # Stoploss: Fixed 1.5% stoploss as per task requirements
    stoploss = -0.015

    # Trailing stop: Activated after +0.6% profit
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.006
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Hyperopt parameters
    buy_rsi = IntParameter(20, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 80, default=70, space="sell")
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
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # EMA
        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=self.bb_window.value, stds=self.bb_stddev.value
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        # Volatility Filter: Bollinger Band Width percentage
        bb_delta = dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        dataframe["bb_width"] = bb_delta / dataframe["bb_middleband"]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        """
        dataframe.loc[
            (
                # Strong entry trigger: RSI < 30 OR RSI crossing 35 upwards
                ((dataframe["rsi"] < 30) | (qtpylib.crossed_above(dataframe["rsi"], 35)))
                &
                # Trend filter: Price must be above EMA 200
                (dataframe["close"] > dataframe["ema"])
                &
                # Volatility filter: BB width > 1% to avoid flat markets
                (dataframe["bb_width"] > 0.01)
                &
                # Volume filter
                (dataframe["volume"] > 0)
            ),
            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        """
        dataframe.loc[
            (
                # Exit trigger: RSI in 65-75 zone
                (dataframe["rsi"] >= 65) & (dataframe["rsi"] <= 75) & (dataframe["volume"] > 0)
            ),
            "sell",
        ] = 1

        return dataframe
