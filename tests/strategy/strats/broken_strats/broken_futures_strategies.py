# The strategy which fails to load due to non-existent dependency

from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


class TestStrategyNoImplements(IStrategy):

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_indicators(dataframe, metadata)


class TestStrategyNoImplementSell(TestStrategyNoImplements):
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_entry_trend(dataframe, metadata)
