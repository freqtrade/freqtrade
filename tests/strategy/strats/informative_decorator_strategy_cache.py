# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from collections import Counter

from pandas import DataFrame

from freqtrade.strategy import IStrategy, informative


class InformativeDecoratorCacheTest(IStrategy):
    """
    Strategy used by tests freqtrade bot.
    Please do not modify this strategy, it's  intended for internal use only.
    Please look at the SampleStrategy in the user_data/strategy directory
    or strategy repository https://github.com/freqtrade/freqtrade-strategies
    for samples and inspiration.
    """

    INTERFACE_VERSION = 3
    stoploss = -0.10
    timeframe = "5m"
    startup_candle_count: int = 20
    # Counts populate_indicators calls per (pair, timeframe).
    informative_counter: Counter[tuple[str, str]] = Counter()

    # Decorator stacking test.
    @informative("30m")
    @informative("1h", cache=False)
    @informative("30m", "ETH/USDT")
    @informative("1h", "ETH/USDT", cache=False)
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.informative_counter[metadata["pair"], metadata["timeframe"]] += 1

        return dataframe

    @informative("30m", "ETH/USDT", "{column}_{base}_{timeframe}")
    def populate_indicators_30m_eth(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.informative_counter[metadata["pair"], metadata["timeframe"]] += 1

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
