# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

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
    informative_counter: dict[str, dict[str, int]] = {}

    # Decorator stacking test.
    @informative("30m")
    @informative("1h", cache=False)
    @informative("30m", "ETH/USDT")
    @informative("1h", "ETH/USDT", cache=False)
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        timeframe = metadata["timeframe"]
        if pair not in self.informative_counter:
            self.informative_counter[pair] = {}
        if timeframe not in self.informative_counter[pair]:
            self.informative_counter[pair][timeframe] = 0
        self.informative_counter[pair][timeframe] += 1

        return dataframe

    @informative("30m", "ETH/USDT", "{column}_{base}_{timeframe}")
    def populate_indicators_30m_eth(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        timeframe = metadata["timeframe"]
        if pair not in self.informative_counter:
            self.informative_counter[pair] = {}
        if timeframe not in self.informative_counter[pair]:
            self.informative_counter[pair][timeframe] = 0
        self.informative_counter[pair][timeframe] += 1

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
