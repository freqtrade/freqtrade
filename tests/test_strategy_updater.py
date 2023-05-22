# pragma pylint: disable=missing-docstring, protected-access, invalid-name

import re
import shutil
import sys
from pathlib import Path

import pytest

from freqtrade.commands.strategy_utils_commands import start_strategy_update
from freqtrade.strategy.strategyupdater import StrategyUpdater
from tests.conftest import get_args


if sys.version_info < (3, 9):
    pytest.skip("StrategyUpdater is not compatible with Python 3.8", allow_module_level=True)


def test_strategy_updater_start(user_dir, capsys) -> None:
    # Effective test without mocks.
    teststrats = Path(__file__).parent / 'strategy/strats'
    tmpdirp = Path(user_dir) / "strategies"
    tmpdirp.mkdir(parents=True, exist_ok=True)
    shutil.copy(teststrats / 'strategy_test_v2.py', tmpdirp)
    old_code = (teststrats / 'strategy_test_v2.py').read_text()

    args = [
        "strategy-updater",
        "--userdir",
        str(user_dir),
        "--strategy-list",
        "StrategyTestV2"
         ]
    pargs = get_args(args)
    pargs['config'] = None

    start_strategy_update(pargs)

    assert Path(user_dir / "strategies_orig_updater").exists()
    # Backup file exists
    assert Path(user_dir / "strategies_orig_updater" / 'strategy_test_v2.py').exists()
    # updated file exists
    new_file = Path(tmpdirp / 'strategy_test_v2.py')
    assert new_file.exists()
    new_code = new_file.read_text()
    assert 'INTERFACE_VERSION = 3' in new_code
    assert 'INTERFACE_VERSION = 2' in old_code
    captured = capsys.readouterr()

    assert 'Conversion of strategy_test_v2.py started.' in captured.out
    assert re.search(r'Conversion of strategy_test_v2\.py took .* seconds', captured.out)


def test_strategy_updater_methods(default_conf, caplog) -> None:

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

    assert "populate_entry_trend" in modified_code1
    assert "populate_exit_trend" in modified_code1
    assert "check_entry_timeout" in modified_code1
    assert "check_exit_timeout" in modified_code1
    assert "custom_exit" in modified_code1
    assert "INTERFACE_VERSION = 3" in modified_code1


def test_strategy_updater_params(default_conf, caplog) -> None:
    instance_strategy_updater = StrategyUpdater()

    modified_code2 = instance_strategy_updater.update_code("""
ticker_interval = '15m'
buy_some_parameter = IntParameter(space='buy')
sell_some_parameter = IntParameter(space='sell')
""")

    assert "timeframe" in modified_code2
    # check for not editing hyperopt spaces
    assert "space='buy'" in modified_code2
    assert "space='sell'" in modified_code2


def test_strategy_updater_constants(default_conf, caplog) -> None:
    instance_strategy_updater = StrategyUpdater()
    modified_code3 = instance_strategy_updater.update_code("""
use_sell_signal = True
sell_profit_only = True
sell_profit_offset = True
ignore_roi_if_buy_signal = True
forcebuy_enable = True
""")

    assert "use_exit_signal" in modified_code3
    assert "exit_profit_only" in modified_code3
    assert "exit_profit_offset" in modified_code3
    assert "ignore_roi_if_entry_signal" in modified_code3
    assert "force_entry_enable" in modified_code3


def test_strategy_updater_df_columns(default_conf, caplog) -> None:
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code("""
dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_1")
dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
""")

    assert "enter_long" in modified_code
    assert "exit_long" in modified_code
    assert "enter_tag" in modified_code


def test_strategy_updater_method_params(default_conf, caplog) -> None:
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code("""
def confirm_trade_exit(sell_reason: str):
    nr_orders = trade.nr_of_successful_buys
    pass
    """)
    assert "exit_reason" in modified_code
    assert "nr_orders = trade.nr_of_successful_entries" in modified_code


def test_strategy_updater_dicts(default_conf, caplog) -> None:
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code("""
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

    assert "'entry': 'gtc'" in modified_code
    assert "'exit': 'ioc'" in modified_code
    assert "'entry': 'limit'" in modified_code
    assert "'exit': 'market'" in modified_code
    assert "'entry': 1" in modified_code
    assert "'exit': 2" in modified_code


def test_strategy_updater_comparisons(default_conf, caplog) -> None:
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code("""
def confirm_trade_exit(sell_reason):
    if (sell_reason == 'stop_loss'):
        pass
""")
    assert "exit_reason" in modified_code
    assert "exit_reason == 'stop_loss'" in modified_code


def test_strategy_updater_strings(default_conf, caplog) -> None:
    instance_strategy_updater = StrategyUpdater()

    modified_code = instance_strategy_updater.update_code("""
sell_reason == 'sell_signal'
sell_reason == 'force_sell'
sell_reason == 'emergency_sell'
""")

    # those tests currently don't work, next in line.
    assert "exit_signal" in modified_code
    assert "exit_reason" in modified_code
    assert "force_exit" in modified_code
    assert "emergency_exit" in modified_code


def test_strategy_updater_comments(default_conf, caplog) -> None:
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code("""
# This is the 1st comment
import talib.abstract as ta
# This is the 2nd comment
import freqtrade.vendor.qtpylib.indicators as qtpylib


class someStrategy(IStrategy):
    INTERFACE_VERSION = 2
    # This is the 3rd comment
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.50
    }

    # This is the 4th comment
    stoploss = -0.1
""")

    assert "This is the 1st comment" in modified_code
    assert "This is the 2nd comment" in modified_code
    assert "This is the 3rd comment" in modified_code
    assert "INTERFACE_VERSION = 3" in modified_code
    # currently still missing:
    # Webhook terminology, Telegram notification settings, Strategy/Config settings
