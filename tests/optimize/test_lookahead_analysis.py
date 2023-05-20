# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

from unittest.mock import PropertyMock

import numpy as np
import pytest

import freqtrade.commands.arguments
import freqtrade.optimize.lookahead_analysis
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.converter import clean_ohlcv_dataframe
from freqtrade.data.history import get_timerange
from tests.conftest import generate_test_data, patch_exchange


@pytest.fixture
def lookahead_conf(default_conf_usdt):
    default_conf_usdt['minimum_trade_amount'] = 10
    default_conf_usdt['targeted_trade_amount'] = 20

    return default_conf_usdt


def trim_dictlist(dict_list, num):
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new


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
