# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
from pandas import DataFrame
from technical.indicators import ichimoku

from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import CategoricalParameter


class strategy_test_v3_with_lookahead_bias(IStrategy):
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy
    minimal_roi = {"40": 0.0, "30": 0.01, "20": 0.02, "0": 0.04}

    # Optimal stoploss designed for the strategy
    stoploss = -0.10

    # Optimal timeframe for the strategy
    timeframe = "5m"
    scenario = CategoricalParameter(["no_bias", "bias1"], default="bias1", space="buy")

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # bias is introduced here
        if self.scenario.value != "no_bias":
            ichi = ichimoku(
                dataframe,
                conversion_line_period=20,
                base_line_periods=60,
                laggin_span=120,
                displacement=30,
            )
            dataframe["chikou_span"] = ichi["chikou_span"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.scenario.value == "no_bias":
            dataframe.loc[dataframe["close"].shift(10) < dataframe["close"], "enter_long"] = 1
        else:
            dataframe.loc[dataframe["close"].shift(-10) > dataframe["close"], "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.scenario.value == "no_bias":
            dataframe.loc[dataframe["close"].shift(10) < dataframe["close"], "exit"] = 1
        else:
            dataframe.loc[dataframe["close"].shift(-10) > dataframe["close"], "exit"] = 1

        return dataframe
