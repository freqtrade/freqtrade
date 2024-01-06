from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import PropertyMock

import pytest

from freqtrade.commands.optimize_commands import setup_optimize_configuration
from freqtrade.configuration.timerange import TimeRange
from freqtrade.data import history
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RunMode
from freqtrade.enums.candletype import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.optimize.backtesting import Backtesting
from tests.conftest import (CURRENT_TEST_STRATEGY, get_args, get_patched_exchange, log_has_re,
                            patch_exchange, patched_configuration_load_config_file)
from tests.freqai.conftest import get_patched_freqai_strategy


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
    assert log_has_re('Using --strategy-list with FreqAI REQUIRES all strategies to have identical',
                      caplog)
    Backtesting.cleanup()


@pytest.mark.parametrize(
    "timeframe, expected_startup_candle_count",
    [
        ("5m", 876),
        ("15m", 492),
        ("1d", 302),
    ],
)
def test_freqai_backtest_load_data(freqai_conf, mocker, caplog,
                                   timeframe, expected_startup_candle_count):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('freqtrade.optimize.backtesting.history.load_data')
    mocker.patch('freqtrade.optimize.backtesting.history.get_timerange', return_value=(now, now))
    freqai_conf['timeframe'] = timeframe
    freqai_conf.get('freqai', {}).get('feature_parameters', {}).update({'include_timeframes': []})
    backtesting = Backtesting(deepcopy(freqai_conf))
    backtesting.load_bt_data()

    assert log_has_re(f'Increasing startup_candle_count for freqai on {timeframe} '
                      f'to {expected_startup_candle_count}', caplog)
    assert history.load_data.call_args[1]['startup_candles'] == expected_startup_candle_count

    Backtesting.cleanup()


def test_freqai_backtest_live_models_model_not_found(freqai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('freqtrade.optimize.backtesting.history.load_data')
    mocker.patch('freqtrade.optimize.backtesting.history.get_timerange', return_value=(now, now))
    freqai_conf["timerange"] = ""
    freqai_conf.get("freqai", {}).update({"backtest_using_historic_predictions": False})

    patched_configuration_load_config_file(mocker, freqai_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '5m',
        '--freqai-backtest-live-models'
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)

    with pytest.raises(OperationalException,
                       match=r".* Historic predictions data is required to run backtest .*"):
        Backtesting(bt_config)

    Backtesting.cleanup()


def test_freqai_backtest_consistent_timerange(mocker, freqai_conf):
    freqai_conf['runmode'] = 'backtest'
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['XRP/USDT:USDT']))

    gbs = mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')

    freqai_conf['candle_type_def'] = CandleType.FUTURES
    freqai_conf.get('exchange', {}).update({'pair_whitelist': ['XRP/USDT:USDT']})
    freqai_conf.get('freqai', {}).get('feature_parameters', {}).update(
        {'include_timeframes': ['5m', '1h'], 'include_corr_pairlist': []})
    freqai_conf['timerange'] = '20211120-20211121'

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)

    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.dk = FreqaiDataKitchen(freqai_conf)

    timerange = TimeRange.parse_timerange("20211115-20211122")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    backtesting = Backtesting(deepcopy(freqai_conf))
    backtesting.start()

    gbs.call_args[1]['min_date'] == datetime(2021, 11, 20, 0, 0, tzinfo=timezone.utc)
    gbs.call_args[1]['max_date'] == datetime(2021, 11, 21, 0, 0, tzinfo=timezone.utc)
    Backtesting.cleanup()
