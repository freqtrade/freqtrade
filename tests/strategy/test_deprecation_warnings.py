import logging
import pytest
from pandas import DataFrame
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy
from tests.conftest import log_has

class StrategyWithLegacyMethods(IStrategy):
    INTERFACE_VERSION = 2
    minimal_roi = {"0": 1}
    stoploss = -0.1
    timeframe = "5m"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_sell(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        return None

def test_deprecation_warnings_legacy_methods(default_conf, caplog):
    caplog.set_level(logging.WARNING)

    strategy = StrategyWithLegacyMethods(default_conf)

    # We call validate_strategy directly to check for warnings
    StrategyResolver.validate_strategy(strategy)

    assert log_has("DEPRECATED: Class overrides 'populate_buy_trend'. This method is deprecated and will be removed in a future version. Please use 'populate_entry_trend' instead.", caplog)
    assert log_has("DEPRECATED: Class overrides 'populate_sell_trend'. This method is deprecated and will be removed in a future version. Please use 'populate_exit_trend' instead.", caplog)
    assert log_has("DEPRECATED: Class overrides 'custom_sell'. This method is deprecated and will be removed in a future version. Please use 'custom_exit' instead.", caplog)
