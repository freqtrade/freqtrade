from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pandas as pd

from freqtrade.commands.analyze_commands import start_analysis_entries_exits
from freqtrade.commands.optimize_commands import start_backtesting
from freqtrade.enums import ExitType
from tests.conftest import get_args, patch_exchange, patched_configuration_load_config_file


def test_backtest_analysis_nomock(default_conf, mocker, caplog, testdatadir, capsys):
    default_conf.update({
        "use_exit_signal": True,
        "exit_profit_only": False,
        "exit_profit_offset": 0.0,
        "ignore_roi_if_entry_signal": False,
        'analysis_groups': "0",
        'enter_reason_list': "all",
        'exit_reason_list': "all",
        'indicator_list': "bb_upperband,ema_10"
    })
    patch_exchange(mocker)
    result1 = pd.DataFrame({'pair': ['ETH/BTC', 'LTC/BTC'],
                            'profit_ratio': [0.0, 0.0],
                            'profit_abs': [0.0, 0.0],
                            'open_date': pd.to_datetime(['2018-01-29 18:40:00',
                                                         '2018-01-30 03:30:00', ], utc=True
                                                        ),
                            'close_date': pd.to_datetime(['2018-01-29 20:45:00',
                                                          '2018-01-30 05:35:00', ], utc=True),
                            'trade_duration': [235, 40],
                            'is_open': [False, False],
                            'stake_amount': [0.01, 0.01],
                            'open_rate': [0.104445, 0.10302485],
                            'close_rate': [0.104969, 0.103541],
                            "is_short": [False, False],
                            'enter_tag': ["enter_tag_long", "enter_tag_long"],
                            'exit_reason': [ExitType.ROI, ExitType.ROI]
                            })

    backtestmock = MagicMock(side_effect=[
        {
            'results': result1,
            'config': default_conf,
            'locks': [],
            'rejected_signals': 20,
            'timedout_entry_orders': 0,
            'timedout_exit_orders': 0,
            'canceled_trade_entries': 0,
            'canceled_entry_orders': 0,
            'replaced_entry_orders': 0,
            'final_balance': 1000,
        }
    ])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['ETH/BTC', 'LTC/BTC', 'DASH/BTC']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '5m',
        '--timerange', '1515560100-1517287800',
        '--export', 'signals',
        '--cache', 'none',
        '--strategy-list',
        'StrategyTestV3',
    ]
    args = get_args(args)
    start_backtesting(args)

    captured = capsys.readouterr()
    assert 'BACKTESTING REPORT' in captured.out
    assert 'EXIT REASON STATS' in captured.out
    assert 'LEFT OPEN TRADES REPORT' in captured.out

    args = [
        'analysis',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--analysis_groups', '0',
        '--strategy',
        'StrategyTestV3',
    ]
    args = get_args(args)
    start_analysis_entries_exits(args)

    captured = capsys.readouterr()
    assert 'enter_tag_long' in captured.out
