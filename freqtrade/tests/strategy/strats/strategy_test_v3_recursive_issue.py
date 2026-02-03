# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import CategoricalParameter


class strategy_test_v3_recursive_issue(IStrategy):
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy
    minimal_roi = {"0": 0.04}

    # Optimal stoploss designed for the strategy
    stoploss = -0.10

    # Optimal timeframe for the strategy
    timeframe = "5m"
    scenario = CategoricalParameter(["no_bias", "bias1", "bias2"], default="bias1", space="buy")

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # bias is introduced here
        if self.scenario.value == "no_bias":
            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        else:
            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=50)

        if self.scenario.value == "bias2":
            # Has both bias1 and bias2
            dataframe["rsi_lookahead"] = ta.RSI(dataframe, timeperiod=50).shift(-1)

        # String columns shouldn't cause issues
        dataframe["test_string_column"] = f"a{len(dataframe)}"

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
