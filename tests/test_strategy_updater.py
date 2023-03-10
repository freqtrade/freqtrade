# pragma pylint: disable=missing-docstring, protected-access, invalid-name

import sys

import pytest

from freqtrade.strategy.strategyupdater import StrategyUpdater


def test_strategy_updater(default_conf, caplog) -> None:
    if sys.version_info < (3, 9):
        pytest.skip("StrategyUpdater is not compatible with Python 3.8", allow_module_level=True)

    instance_strategy_updater = StrategyUpdater()
    modified_code1 = instance_strategy_updater.update_code("""
class testClass(IStrategy):
    def populate_buy_trend():
        pass
    def populate_sell_trend():
        pass
    def check_buy_timeout():
        pass
    def check_sell_timeout():
        pass
    def custom_sell():
        pass
""")
    modified_code2 = instance_strategy_updater.update_code("""
ticker_interval = '15m'
buy_some_parameter = IntParameter(space='buy')
sell_some_parameter = IntParameter(space='sell')
""")
    modified_code3 = instance_strategy_updater.update_code("""
use_sell_signal = True
sell_profit_only = True
sell_profit_offset = True
ignore_roi_if_buy_signal = True
forcebuy_enable = True
""")
    modified_code4 = instance_strategy_updater.update_code("""
dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_1")
dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
""")
    modified_code5 = instance_strategy_updater.update_code("""
def confirm_trade_exit(sell_reason: str):
    pass
    """)
    modified_code6 = instance_strategy_updater.update_code("""
order_time_in_force = {
    'buy': 'gtc',
    'sell': 'ioc'
}
order_types = {
    'buy': 'limit',
    'sell': 'market',
    'stoploss': 'market',
    'stoploss_on_exchange': False
}
unfilledtimeout = {
    'buy': 1,
    'sell': 2
}
""")

    modified_code7 = instance_strategy_updater.update_code("""
def confirm_trade_exit(sell_reason):
    if (sell_reason == 'stop_loss'):
        pass
""")
    modified_code8 = instance_strategy_updater.update_code("""
sell_reason == 'sell_signal'
sell_reason == 'force_sell'
sell_reason == 'emergency_sell'
""")
    modified_code9 = instance_strategy_updater.update_code("""
# This is the 1st comment
import talib.abstract as ta
# This is the 2nd comment
import freqtrade.vendor.qtpylib.indicators as qtpylib


class someStrategy(IStrategy):
    # This is the 3rd comment
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.50
    }

    # This is the 4th comment
    stoploss = -0.1
""")
    # currently still missing:
    # Webhook terminology, Telegram notification settings, Strategy/Config settings

    assert "populate_entry_trend" in modified_code1
    assert "populate_exit_trend" in modified_code1
    assert "check_entry_timeout" in modified_code1
    assert "check_exit_timeout" in modified_code1
    assert "custom_exit" in modified_code1
    assert "INTERFACE_VERSION = 3" in modified_code1

    assert "timeframe" in modified_code2
    # check for not editing hyperopt spaces
    assert "space='buy'" in modified_code2
    assert "space='sell'" in modified_code2

    assert "use_exit_signal" in modified_code3
    assert "exit_profit_only" in modified_code3
    assert "exit_profit_offset" in modified_code3
    assert "ignore_roi_if_entry_signal" in modified_code3
    assert "force_entry_enable" in modified_code3

    assert "enter_long" in modified_code4
    assert "exit_long" in modified_code4
    assert "enter_tag" in modified_code4

    assert "exit_reason" in modified_code5

    assert "'entry': 'gtc'" in modified_code6
    assert "'exit': 'ioc'" in modified_code6
    assert "'entry': 'limit'" in modified_code6
    assert "'exit': 'market'" in modified_code6
    assert "'entry': 1" in modified_code6
    assert "'exit': 2" in modified_code6

    assert "exit_reason" in modified_code7
    assert "exit_reason == 'stop_loss'" in modified_code7

    # those tests currently don't work, next in line.
    assert "exit_signal" in modified_code8
    assert "exit_reason" in modified_code8
    assert "force_exit" in modified_code8
    assert "emergency_exit" in modified_code8

    assert "This is the 1st comment" in modified_code9
    assert "This is the 2nd comment" in modified_code9
    assert "This is the 3rd comment" in modified_code9
