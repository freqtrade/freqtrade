# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.commands.optimize_commands import start_recursive_analysis
from freqtrade.data.history import get_timerange
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.recursive_analysis import Analysis, RecursiveAnalysis
from freqtrade.optimize.recursive_analysis_helpers import RecursiveAnalysisSubFunctions
from tests.conftest import EXMS, get_args, log_has_re, patch_exchange


@pytest.fixture
def recursive_conf(default_conf_usdt):
    default_conf_usdt['timerange'] = '20220101-20220501'

    default_conf_usdt['strategy_path'] = str(
        Path(__file__).parent.parent / "strategy/strats")
    default_conf_usdt['strategy'] = 'strategy_test_v3_recursive_issue'
    default_conf_usdt['pairs'] = ['UNITTEST/USDT']
    return default_conf_usdt


# def test_start_recursive_analysis(mocker):
#     single_mock = MagicMock()
#     text_table_mock = MagicMock()
#     mocker.patch.multiple(
#         'freqtrade.optimize.recursive_analysis_helpers.RecursiveAnalysisSubFunctions',
#         initialize_single_recursive_analysis=single_mock,
#         text_table_recursive_analysis_instances=text_table_mock,
#     )
#     args = [
#         "recursive-analysis",
#         "--strategy",
#         "strategy_test_v3_recursive_issue",
#         "--strategy-path",
#         str(Path(__file__).parent.parent / "strategy/strats"),
#         "--pairs",
#         "UNITTEST/BTC",
#         "--timerange",
#         "20220101-20220201"
#     ]
#     pargs = get_args(args)
#     pargs['config'] = None

#     start_recursive_analysis(pargs)
#     assert single_mock.call_count == 1
#     assert text_table_mock.call_count == 1

#     single_mock.reset_mock()

#     # Test invalid config
#     args = [
#         "lookahead-analysis",
#         "--strategy",
#         "strategy_test_v3_with_lookahead_bias",
#         "--strategy-path",
#         str(Path(__file__).parent.parent / "strategy/strats/lookahead_bias"),
#         "--targeted-trade-amount",
#         "10",
#         "--minimum-trade-amount",
#         "20",
#     ]
#     pargs = get_args(args)
#     pargs['config'] = None
#     with pytest.raises(OperationalException,
#                        match=r"Targeted trade amount can't be smaller than minimum trade amount.*"):
#         start_lookahead_analysis(pargs)

#     # Missing timerange
#     args = [
#         "lookahead-analysis",
#         "--strategy",
#         "strategy_test_v3_with_lookahead_bias",
#         "--strategy-path",
#         str(Path(__file__).parent.parent / "strategy/strats/lookahead_bias"),
#         "--pairs",
#         "UNITTEST/BTC",
#         "--max-open-trades",
#         "1",
#     ]
#     pargs = get_args(args)
#     pargs['config'] = None
#     with pytest.raises(OperationalException,
#                        match=r"Please set a timerange\..*"):
#         start_lookahead_analysis(pargs)


# def test_lookahead_helper_invalid_config(recursive_conf) -> None:
#     conf = deepcopy(recursive_conf)
#     conf['targeted_trade_amount'] = 10
#     conf['minimum_trade_amount'] = 40
#     with pytest.raises(OperationalException,
#                        match=r"Targeted trade amount can't be smaller than minimum trade amount.*"):
#         RecursiveAnalysisSubFunctions.start(conf)


# def test_lookahead_helper_no_strategy_defined(recursive_conf):
#     conf = deepcopy(recursive_conf)
#     conf['pairs'] = ['UNITTEST/USDT']
#     del conf['strategy']
#     with pytest.raises(OperationalException,
#                        match=r"No Strategy specified"):
#         RecursiveAnalysisSubFunctions.start(conf)


# def test_lookahead_helper_start(recursive_conf, mocker) -> None:
#     single_mock = MagicMock()
#     text_table_mock = MagicMock()
#     mocker.patch.multiple(
#         'freqtrade.optimize.lookahead_analysis_helpers.RecursiveAnalysisSubFunctions',
#         initialize_single_lookahead_analysis=single_mock,
#         text_table_lookahead_analysis_instances=text_table_mock,
#     )
#     RecursiveAnalysisSubFunctions.start(recursive_conf)
#     assert single_mock.call_count == 1
#     assert text_table_mock.call_count == 1

#     single_mock.reset_mock()
#     text_table_mock.reset_mock()


# def test_lookahead_helper_text_table_lookahead_analysis_instances(recursive_conf):
#     analysis = Analysis()
#     analysis.has_bias = True
#     analysis.total_signals = 5
#     analysis.false_entry_signals = 4
#     analysis.false_exit_signals = 3

#     strategy_obj = {
#         'name': "strategy_test_v3_with_lookahead_bias",
#         'location': Path(recursive_conf['strategy_path'], f"{recursive_conf['strategy']}.py")
#     }

#     instance = LookaheadAnalysis(recursive_conf, strategy_obj)
#     instance.current_analysis = analysis
#     table, headers, data = (RecursiveAnalysisSubFunctions.
#                             text_table_lookahead_analysis_instances(recursive_conf, [instance]))

#     # check row contents for a try that has too few signals
#     assert data[0][0] == 'strategy_test_v3_with_lookahead_bias.py'
#     assert data[0][1] == 'strategy_test_v3_with_lookahead_bias'
#     assert data[0][2].__contains__('too few trades')
#     assert len(data[0]) == 3

#     # now check for an error which occured after enough trades
#     analysis.total_signals = 12
#     analysis.false_entry_signals = 11
#     analysis.false_exit_signals = 10
#     instance = LookaheadAnalysis(recursive_conf, strategy_obj)
#     instance.current_analysis = analysis
#     table, headers, data = (RecursiveAnalysisSubFunctions.
#                             text_table_lookahead_analysis_instances(recursive_conf, [instance]))
#     assert data[0][2].__contains__("error")

#     # edit it into not showing an error
#     instance.failed_bias_check = False
#     table, headers, data = (RecursiveAnalysisSubFunctions.
#                             text_table_lookahead_analysis_instances(recursive_conf, [instance]))
#     assert data[0][0] == 'strategy_test_v3_with_lookahead_bias.py'
#     assert data[0][1] == 'strategy_test_v3_with_lookahead_bias'
#     assert data[0][2]  # True
#     assert data[0][3] == 12
#     assert data[0][4] == 11
#     assert data[0][5] == 10
#     assert data[0][6] == ''

#     analysis.false_indicators.append('falseIndicator1')
#     analysis.false_indicators.append('falseIndicator2')
#     table, headers, data = (RecursiveAnalysisSubFunctions.
#                             text_table_lookahead_analysis_instances(recursive_conf, [instance]))

#     assert data[0][6] == 'falseIndicator1, falseIndicator2'

#     # check amount of returning rows
#     assert len(data) == 1

#     # check amount of multiple rows
#     table, headers, data = (RecursiveAnalysisSubFunctions.text_table_lookahead_analysis_instances(
#         recursive_conf, [instance, instance, instance]))
#     assert len(data) == 3


# def test_lookahead_helper_export_to_csv(recursive_conf):
#     import pandas as pd
#     recursive_conf['lookahead_analysis_exportfilename'] = "temp_csv_lookahead_analysis.csv"

#     # just to be sure the test won't fail: remove file if exists for some reason
#     # (repeat this at the end once again to clean up)
#     if Path(recursive_conf['lookahead_analysis_exportfilename']).exists():
#         Path(recursive_conf['lookahead_analysis_exportfilename']).unlink()

#     # before we can start we have to delete the

#     # 1st check: create a new file and verify its contents
#     analysis1 = Analysis()
#     analysis1.has_bias = True
#     analysis1.total_signals = 12
#     analysis1.false_entry_signals = 11
#     analysis1.false_exit_signals = 10
#     analysis1.false_indicators.append('falseIndicator1')
#     analysis1.false_indicators.append('falseIndicator2')
#     recursive_conf['lookahead_analysis_exportfilename'] = "temp_csv_lookahead_analysis.csv"

#     strategy_obj1 = {
#         'name': "strat1",
#         'location': Path("file1.py"),
#     }

#     instance1 = LookaheadAnalysis(recursive_conf, strategy_obj1)
#     instance1.failed_bias_check = False
#     instance1.current_analysis = analysis1

#     RecursiveAnalysisSubFunctions.export_to_csv(recursive_conf, [instance1])
#     saved_data1 = pd.read_csv(recursive_conf['lookahead_analysis_exportfilename'])

#     expected_values1 = [
#         [
#             'file1.py', 'strat1', True,
#             12, 11, 10,
#             "falseIndicator1,falseIndicator2"
#         ],
#     ]
#     expected_columns = ['filename', 'strategy', 'has_bias',
#                         'total_signals', 'biased_entry_signals', 'biased_exit_signals',
#                         'biased_indicators']
#     expected_data1 = pd.DataFrame(expected_values1, columns=expected_columns)

#     assert Path(recursive_conf['lookahead_analysis_exportfilename']).exists()
#     assert expected_data1.equals(saved_data1)

#     # 2nd check: update the same strategy (which internally changed or is being retested)
#     expected_values2 = [
#         [
#             'file1.py', 'strat1', False,
#             22, 21, 20,
#             "falseIndicator3,falseIndicator4"
#         ],
#     ]
#     expected_data2 = pd.DataFrame(expected_values2, columns=expected_columns)

#     analysis2 = Analysis()
#     analysis2.has_bias = False
#     analysis2.total_signals = 22
#     analysis2.false_entry_signals = 21
#     analysis2.false_exit_signals = 20
#     analysis2.false_indicators.append('falseIndicator3')
#     analysis2.false_indicators.append('falseIndicator4')

#     strategy_obj2 = {
#         'name': "strat1",
#         'location': Path("file1.py"),
#     }

#     instance2 = LookaheadAnalysis(recursive_conf, strategy_obj2)
#     instance2.failed_bias_check = False
#     instance2.current_analysis = analysis2

#     RecursiveAnalysisSubFunctions.export_to_csv(recursive_conf, [instance2])
#     saved_data2 = pd.read_csv(recursive_conf['lookahead_analysis_exportfilename'])

#     assert expected_data2.equals(saved_data2)

#     # 3rd check: now we add a new row to an already existing file
#     expected_values3 = [
#         [
#             'file1.py', 'strat1', False,
#             22, 21, 20,
#             "falseIndicator3,falseIndicator4"
#         ],
#         [
#             'file3.py', 'strat3', True,
#             32, 31, 30, "falseIndicator5,falseIndicator6"
#         ],
#     ]

#     expected_data3 = pd.DataFrame(expected_values3, columns=expected_columns)

#     analysis3 = Analysis()
#     analysis3.has_bias = True
#     analysis3.total_signals = 32
#     analysis3.false_entry_signals = 31
#     analysis3.false_exit_signals = 30
#     analysis3.false_indicators.append('falseIndicator5')
#     analysis3.false_indicators.append('falseIndicator6')
#     recursive_conf['lookahead_analysis_exportfilename'] = "temp_csv_lookahead_analysis.csv"

#     strategy_obj3 = {
#         'name': "strat3",
#         'location': Path("file3.py"),
#     }

#     instance3 = LookaheadAnalysis(recursive_conf, strategy_obj3)
#     instance3.failed_bias_check = False
#     instance3.current_analysis = analysis3

#     RecursiveAnalysisSubFunctions.export_to_csv(recursive_conf, [instance3])
#     saved_data3 = pd.read_csv(recursive_conf['lookahead_analysis_exportfilename'])
#     assert expected_data3.equals(saved_data3)

#     # remove csv file after the test is done
#     if Path(recursive_conf['lookahead_analysis_exportfilename']).exists():
#         Path(recursive_conf['lookahead_analysis_exportfilename']).unlink()


# def test_initialize_single_lookahead_analysis(recursive_conf, mocker, caplog):
#     mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
#     mocker.patch(f'{EXMS}.get_fee', return_value=0.0)
#     mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=0.00001)
#     mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
#     patch_exchange(mocker)
#     mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
#                  PropertyMock(return_value=['UNITTEST/BTC']))
#     recursive_conf['pairs'] = ['UNITTEST/USDT']

#     recursive_conf['timeframe'] = '5m'
#     recursive_conf['timerange'] = '20180119-20180122'
#     start_mock = mocker.patch('freqtrade.optimize.lookahead_analysis.LookaheadAnalysis.start')
#     strategy_obj = {
#         'name': "strategy_test_v3_with_lookahead_bias",
#         'location': Path(recursive_conf['strategy_path'], f"{recursive_conf['strategy']}.py")
#     }

#     instance = RecursiveAnalysisSubFunctions.initialize_single_lookahead_analysis(
#         recursive_conf, strategy_obj)
#     assert log_has_re(r"Bias test of .* started\.", caplog)
#     assert start_mock.call_count == 1

#     assert instance.strategy_obj['name'] == "strategy_test_v3_with_lookahead_bias"


@pytest.mark.parametrize('scenario', [
    'no_bias', 'bias1'
])
def test_biased_strategy(recursive_conf, mocker, caplog, scenario) -> None:
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))
    recursive_conf['pairs'] = ['UNITTEST/USDT']

    recursive_conf['timeframe'] = '5m'
    recursive_conf['timerange'] = '20180119-20180122'
    recursive_conf['startup_candle'] = [100]

    # Patch scenario Parameter to allow for easy selection
    mocker.patch('freqtrade.strategy.hyper.HyperStrategyMixin.load_params_from_file',
                 return_value={
                     'params': {
                         "buy": {
                             "scenario": scenario
                         }
                     }
                 })

    strategy_obj = {'name': "strategy_test_v3_recursive_issue"}
    instance = RecursiveAnalysis(recursive_conf, strategy_obj)
    instance.start()
    # Assert init correct
    assert log_has_re(f"Strategy Parameter: scenario = {scenario}", caplog)

    diff_pct = float(instance.dict_recursive['rsi'][100].replace("%", ""))
    # check non-biased strategy
    if scenario == "no_bias":
        assert diff_pct < 0.01
    # check biased strategy
    elif scenario == "bias1":
        assert diff_pct >= 0.01
