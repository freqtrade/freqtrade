from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import PropertyMock

from freqtrade.commands.optimize_commands import setup_optimize_configuration
from freqtrade.enums import RunMode
from freqtrade.optimize.backtesting import Backtesting
from tests.conftest import (CURRENT_TEST_STRATEGY, get_args, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)


def test_freqai_backtest_start_backtest_list(freqai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('freqtrade.optimize.backtesting.history.load_data')
    mocker.patch('freqtrade.optimize.backtesting.history.get_timerange', return_value=(now, now))

    patched_configuration_load_config_file(mocker, freqai_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '1m',
        '--strategy-list', CURRENT_TEST_STRATEGY
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)
    Backtesting(bt_config)
    assert log_has_re('Using --strategy-list with FreqAI REQUIRES all strategies to have identical '
                      'populate_any_indicators.', caplog)
    Backtesting.cleanup()


def test_freqai_backtest_load_data(freqai_conf, mocker, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('freqtrade.optimize.backtesting.history.load_data')
    mocker.patch('freqtrade.optimize.backtesting.history.get_timerange', return_value=(now, now))
    backtesting = Backtesting(deepcopy(freqai_conf))
    backtesting.load_bt_data()

    assert log_has_re('Increasing startup_candle_count for freqai to.*', caplog)

    Backtesting.cleanup()
