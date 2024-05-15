"""
The strategies here are minimal strategies designed to fail loading in certain conditions.
They are not operational, and don't aim to be.
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.persistence.trade_model import Order
from freqtrade.strategy.interface import IStrategy


class TestStrategyNoImplements(IStrategy):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_indicators(dataframe, metadata)


class TestStrategyNoImplementSell(TestStrategyNoImplements):
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_entry_trend(dataframe, metadata)


class TestStrategyImplementCustomSell(TestStrategyNoImplementSell):
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_exit_trend(dataframe, metadata)

    def custom_sell(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        return False


class TestStrategyImplementBuyTimeout(TestStrategyNoImplementSell):
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_exit_trend(dataframe, metadata)

    def check_buy_timeout(
        self, pair: str, trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        return False


class TestStrategyImplementSellTimeout(TestStrategyNoImplementSell):
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_exit_trend(dataframe, metadata)

    def check_sell_timeout(
        self, pair: str, trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        return False
