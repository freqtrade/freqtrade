# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

import freqtrade.commands.arguments
import freqtrade.optimize.lookahead_analysis
from freqtrade.commands.optimize_commands import start_lookahead_analysis
from freqtrade.data.history import get_timerange
from freqtrade.exceptions import OperationalException
from tests.conftest import CURRENT_TEST_STRATEGY, get_args, patch_exchange


@pytest.fixture
def lookahead_conf(default_conf_usdt):
    default_conf_usdt['minimum_trade_amount'] = 10
    default_conf_usdt['targeted_trade_amount'] = 20

    return default_conf_usdt


def test_start_start_lookahead_analysis(mocker):
    single_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.optimize.lookahead_analysis.LookaheadAnalysisSubFunctions',
        initialize_single_lookahead_analysis=single_mock,
        text_table_lookahead_analysis_instances=MagicMock(),
        )
    args = [
        "lookahead-analysis",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
    ]
    pargs = get_args(args)
    pargs['config'] = None

    start_lookahead_analysis(pargs)
    assert single_mock.call_count == 1

    single_mock.reset_mock()

    # Test invalid config
    args = [
        "lookahead-analysis",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
        "--targeted-trade-amount",
        "10",
        "--minimum-trade-amount",
        "20",
        ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"targeted trade amount can't be smaller than .*"):
        start_lookahead_analysis(pargs)



def test_biased_strategy(lookahead_conf, mocker, caplog) -> None:

    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))

    lookahead_conf['timeframe'] = '5m'
    lookahead_conf['timerange'] = '-1510694220'
    lookahead_conf['strategy'] = 'strategy_test_v3_with_lookahead_bias'

    strategy_obj = {}
    strategy_obj['name'] = "strategy_test_v3_with_lookahead_bias"
    freqtrade.optimize.lookahead_analysis.LookaheadAnalysis(lookahead_conf, strategy_obj)
    pass
