# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.commands.optimize_commands import start_lookahead_analysis
from freqtrade.data.history import get_timerange
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.analysis.lookahead import Analysis, LookaheadAnalysis
from freqtrade.optimize.analysis.lookahead_helpers import LookaheadAnalysisSubFunctions
from tests.conftest import EXMS, get_args, log_has_re, patch_exchange


@pytest.fixture
def lookahead_conf(default_conf_usdt):
    default_conf_usdt['minimum_trade_amount'] = 10
    default_conf_usdt['targeted_trade_amount'] = 20
    default_conf_usdt['timerange'] = '20220101-20220501'

    default_conf_usdt['strategy_path'] = str(
        Path(__file__).parent.parent / "strategy/strats/lookahead_bias")
    default_conf_usdt['strategy'] = 'strategy_test_v3_with_lookahead_bias'
    default_conf_usdt['max_open_trades'] = 1
    default_conf_usdt['dry_run_wallet'] = 1000000000
    default_conf_usdt['pairs'] = ['UNITTEST/USDT']
    return default_conf_usdt


def test_start_lookahead_analysis(mocker):
    single_mock = MagicMock()
    text_table_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.optimize.analysis.lookahead_helpers.LookaheadAnalysisSubFunctions',
        initialize_single_lookahead_analysis=single_mock,
        text_table_lookahead_analysis_instances=text_table_mock,
    )
    args = [
        "lookahead-analysis",
        "--strategy",
        "strategy_test_v3_with_lookahead_bias",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy/strats/lookahead_bias"),
        "--pairs",
        "UNITTEST/BTC",
        "--max-open-trades",
        "1",
        "--timerange",
        "20220101-20220201"
    ]
    pargs = get_args(args)
    pargs['config'] = None

    start_lookahead_analysis(pargs)
    assert single_mock.call_count == 1
    assert text_table_mock.call_count == 1

    single_mock.reset_mock()

    # Test invalid config
    args = [
        "lookahead-analysis",
        "--strategy",
        "strategy_test_v3_with_lookahead_bias",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy/strats/lookahead_bias"),
        "--targeted-trade-amount",
        "10",
        "--minimum-trade-amount",
        "20",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"Targeted trade amount can't be smaller than minimum trade amount.*"):
        start_lookahead_analysis(pargs)

    # Missing timerange
    args = [
        "lookahead-analysis",
        "--strategy",
        "strategy_test_v3_with_lookahead_bias",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy/strats/lookahead_bias"),
        "--pairs",
        "UNITTEST/BTC",
        "--max-open-trades",
        "1",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"Please set a timerange\..*"):
        start_lookahead_analysis(pargs)


def test_lookahead_helper_invalid_config(lookahead_conf) -> None:
    conf = deepcopy(lookahead_conf)
    conf['targeted_trade_amount'] = 10
    conf['minimum_trade_amount'] = 40
    with pytest.raises(OperationalException,
                       match=r"Targeted trade amount can't be smaller than minimum trade amount.*"):
        LookaheadAnalysisSubFunctions.start(conf)


def test_lookahead_helper_no_strategy_defined(lookahead_conf):
    conf = deepcopy(lookahead_conf)
    conf['pairs'] = ['UNITTEST/USDT']
    del conf['strategy']
    with pytest.raises(OperationalException,
                       match=r"No Strategy specified"):
        LookaheadAnalysisSubFunctions.start(conf)


def test_lookahead_helper_start(lookahead_conf, mocker) -> None:
    single_mock = MagicMock()
    text_table_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.optimize.analysis.lookahead_helpers.LookaheadAnalysisSubFunctions',
        initialize_single_lookahead_analysis=single_mock,
        text_table_lookahead_analysis_instances=text_table_mock,
    )
    LookaheadAnalysisSubFunctions.start(lookahead_conf)
    assert single_mock.call_count == 1
    assert text_table_mock.call_count == 1

    single_mock.reset_mock()
    text_table_mock.reset_mock()


def test_lookahead_helper_text_table_lookahead_analysis_instances(lookahead_conf):
    analysis = Analysis()
    analysis.has_bias = True
    analysis.total_signals = 5
    analysis.false_entry_signals = 4
    analysis.false_exit_signals = 3

    strategy_obj = {
        'name': "strategy_test_v3_with_lookahead_bias",
        'location': Path(lookahead_conf['strategy_path'], f"{lookahead_conf['strategy']}.py")
    }

    instance = LookaheadAnalysis(lookahead_conf, strategy_obj)
    instance.current_analysis = analysis
    table, headers, data = (LookaheadAnalysisSubFunctions.
                            text_table_lookahead_analysis_instances(lookahead_conf, [instance]))

    # check row contents for a try that has too few signals
    assert data[0][0] == 'strategy_test_v3_with_lookahead_bias.py'
    assert data[0][1] == 'strategy_test_v3_with_lookahead_bias'
    assert data[0][2].__contains__('too few trades')
    assert len(data[0]) == 3

    # now check for an error which occured after enough trades
    analysis.total_signals = 12
    analysis.false_entry_signals = 11
    analysis.false_exit_signals = 10
    instance = LookaheadAnalysis(lookahead_conf, strategy_obj)
    instance.current_analysis = analysis
    table, headers, data = (LookaheadAnalysisSubFunctions.
                            text_table_lookahead_analysis_instances(lookahead_conf, [instance]))
    assert data[0][2].__contains__("error")

    # edit it into not showing an error
    instance.failed_bias_check = False
    table, headers, data = (LookaheadAnalysisSubFunctions.
                            text_table_lookahead_analysis_instances(lookahead_conf, [instance]))
    assert data[0][0] == 'strategy_test_v3_with_lookahead_bias.py'
    assert data[0][1] == 'strategy_test_v3_with_lookahead_bias'
    assert data[0][2]  # True
    assert data[0][3] == 12
    assert data[0][4] == 11
    assert data[0][5] == 10
    assert data[0][6] == ''

    analysis.false_indicators.append('falseIndicator1')
    analysis.false_indicators.append('falseIndicator2')
    table, headers, data = (LookaheadAnalysisSubFunctions.
                            text_table_lookahead_analysis_instances(lookahead_conf, [instance]))

    assert data[0][6] == 'falseIndicator1, falseIndicator2'

    # check amount of returning rows
    assert len(data) == 1

    # check amount of multiple rows
    table, headers, data = (LookaheadAnalysisSubFunctions.text_table_lookahead_analysis_instances(
        lookahead_conf, [instance, instance, instance]))
    assert len(data) == 3


def test_lookahead_helper_export_to_csv(lookahead_conf):
    import pandas as pd
    lookahead_conf['lookahead_analysis_exportfilename'] = "temp_csv_lookahead_analysis.csv"

    # just to be sure the test won't fail: remove file if exists for some reason
    # (repeat this at the end once again to clean up)
    if Path(lookahead_conf['lookahead_analysis_exportfilename']).exists():
        Path(lookahead_conf['lookahead_analysis_exportfilename']).unlink()

    # before we can start we have to delete the

    # 1st check: create a new file and verify its contents
    analysis1 = Analysis()
    analysis1.has_bias = True
    analysis1.total_signals = 12
    analysis1.false_entry_signals = 11
    analysis1.false_exit_signals = 10
    analysis1.false_indicators.append('falseIndicator1')
    analysis1.false_indicators.append('falseIndicator2')
    lookahead_conf['lookahead_analysis_exportfilename'] = "temp_csv_lookahead_analysis.csv"

    strategy_obj1 = {
        'name': "strat1",
        'location': Path("file1.py"),
    }

    instance1 = LookaheadAnalysis(lookahead_conf, strategy_obj1)
    instance1.failed_bias_check = False
    instance1.current_analysis = analysis1

    LookaheadAnalysisSubFunctions.export_to_csv(lookahead_conf, [instance1])
    saved_data1 = pd.read_csv(lookahead_conf['lookahead_analysis_exportfilename'])

    expected_values1 = [
        [
            'file1.py', 'strat1', True,
            12, 11, 10,
            "falseIndicator1,falseIndicator2"
        ],
    ]
    expected_columns = ['filename', 'strategy', 'has_bias',
                        'total_signals', 'biased_entry_signals', 'biased_exit_signals',
                        'biased_indicators']
    expected_data1 = pd.DataFrame(expected_values1, columns=expected_columns)

    assert Path(lookahead_conf['lookahead_analysis_exportfilename']).exists()
    assert expected_data1.equals(saved_data1)

    # 2nd check: update the same strategy (which internally changed or is being retested)
    expected_values2 = [
        [
            'file1.py', 'strat1', False,
            22, 21, 20,
            "falseIndicator3,falseIndicator4"
        ],
    ]
    expected_data2 = pd.DataFrame(expected_values2, columns=expected_columns)

    analysis2 = Analysis()
    analysis2.has_bias = False
    analysis2.total_signals = 22
    analysis2.false_entry_signals = 21
    analysis2.false_exit_signals = 20
    analysis2.false_indicators.append('falseIndicator3')
    analysis2.false_indicators.append('falseIndicator4')

    strategy_obj2 = {
        'name': "strat1",
        'location': Path("file1.py"),
    }

    instance2 = LookaheadAnalysis(lookahead_conf, strategy_obj2)
    instance2.failed_bias_check = False
    instance2.current_analysis = analysis2

    LookaheadAnalysisSubFunctions.export_to_csv(lookahead_conf, [instance2])
    saved_data2 = pd.read_csv(lookahead_conf['lookahead_analysis_exportfilename'])

    assert expected_data2.equals(saved_data2)

    # 3rd check: now we add a new row to an already existing file
    expected_values3 = [
        [
            'file1.py', 'strat1', False,
            22, 21, 20,
            "falseIndicator3,falseIndicator4"
        ],
        [
            'file3.py', 'strat3', True,
            32, 31, 30, "falseIndicator5,falseIndicator6"
        ],
    ]

    expected_data3 = pd.DataFrame(expected_values3, columns=expected_columns)

    analysis3 = Analysis()
    analysis3.has_bias = True
    analysis3.total_signals = 32
    analysis3.false_entry_signals = 31
    analysis3.false_exit_signals = 30
    analysis3.false_indicators.append('falseIndicator5')
    analysis3.false_indicators.append('falseIndicator6')
    lookahead_conf['lookahead_analysis_exportfilename'] = "temp_csv_lookahead_analysis.csv"

    strategy_obj3 = {
        'name': "strat3",
        'location': Path("file3.py"),
    }

    instance3 = LookaheadAnalysis(lookahead_conf, strategy_obj3)
    instance3.failed_bias_check = False
    instance3.current_analysis = analysis3

    LookaheadAnalysisSubFunctions.export_to_csv(lookahead_conf, [instance3])
    saved_data3 = pd.read_csv(lookahead_conf['lookahead_analysis_exportfilename'])
    assert expected_data3.equals(saved_data3)

    # remove csv file after the test is done
    if Path(lookahead_conf['lookahead_analysis_exportfilename']).exists():
        Path(lookahead_conf['lookahead_analysis_exportfilename']).unlink()


def test_initialize_single_lookahead_analysis(lookahead_conf, mocker, caplog):
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    mocker.patch(f'{EXMS}.get_fee', return_value=0.0)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=0.00001)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    patch_exchange(mocker)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))
    lookahead_conf['pairs'] = ['UNITTEST/USDT']

    lookahead_conf['timeframe'] = '5m'
    lookahead_conf['timerange'] = '20180119-20180122'
    start_mock = mocker.patch('freqtrade.optimize.analysis.lookahead.LookaheadAnalysis.start')
    strategy_obj = {
        'name': "strategy_test_v3_with_lookahead_bias",
        'location': Path(lookahead_conf['strategy_path'], f"{lookahead_conf['strategy']}.py")
    }

    instance = LookaheadAnalysisSubFunctions.initialize_single_lookahead_analysis(
        lookahead_conf, strategy_obj)
    assert log_has_re(r"Bias test of .* started\.", caplog)
    assert start_mock.call_count == 1

    assert instance.strategy_obj['name'] == "strategy_test_v3_with_lookahead_bias"


@pytest.mark.parametrize('scenario', [
    'no_bias', 'bias1'
])
def test_biased_strategy(lookahead_conf, mocker, caplog, scenario) -> None:
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    mocker.patch(f'{EXMS}.get_fee', return_value=0.0)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=0.00001)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    patch_exchange(mocker)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))
    lookahead_conf['pairs'] = ['UNITTEST/USDT']

    lookahead_conf['timeframe'] = '5m'
    lookahead_conf['timerange'] = '20180119-20180122'

    # Patch scenario Parameter to allow for easy selection
    mocker.patch('freqtrade.strategy.hyper.HyperStrategyMixin.load_params_from_file',
                 return_value={
                     'params': {
                         "buy": {
                             "scenario": scenario
                         }
                     }
                 })

    strategy_obj = {'name': "strategy_test_v3_with_lookahead_bias"}
    instance = LookaheadAnalysis(lookahead_conf, strategy_obj)
    instance.start()
    # Assert init correct
    assert log_has_re(f"Strategy Parameter: scenario = {scenario}", caplog)

    # check non-biased strategy
    if scenario == "no_bias":
        assert not instance.current_analysis.has_bias
    # check biased strategy
    elif scenario == "bias1":
        assert instance.current_analysis.has_bias


def test_config_overrides(lookahead_conf):
    lookahead_conf['max_open_trades'] = 0
    lookahead_conf['dry_run_wallet'] = 1
    lookahead_conf['pairs'] = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    lookahead_conf = LookaheadAnalysisSubFunctions.calculate_config_overrides(lookahead_conf)

    assert lookahead_conf['dry_run_wallet'] == 1000000000
    assert lookahead_conf['max_open_trades'] == 3
