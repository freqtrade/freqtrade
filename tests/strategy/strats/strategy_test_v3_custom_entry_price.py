# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from datetime import datetime
from typing import Optional

from pandas import DataFrame
from strategy_test_v3 import StrategyTestV3

from freqtrade.persistence import Trade


class StrategyTestV3CustomEntryPrice(StrategyTestV3):
    """
    Strategy used by tests freqtrade bot.
    Please do not modify this strategy, it's  intended for internal use only.
    Please look at the SampleStrategy in the user_data/strategy directory
    or strategy repository https://github.com/freqtrade/freqtrade-strategies
    for samples and inspiration.
    """
    new_entry_price: float = 0.001

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            dataframe['volume'] > 0,
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        return self.new_entry_price
