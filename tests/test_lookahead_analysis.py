# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

from unittest.mock import PropertyMock

import numpy as np

import freqtrade.commands.arguments
import freqtrade.optimize.lookahead_analysis
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.converter import clean_ohlcv_dataframe
from freqtrade.data.history import get_timerange
from tests.conftest import patch_exchange


def trim_dictlist(dict_list, num):
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new


def load_data_test(what, testdatadir):
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = history.load_pair_history(pair='UNITTEST/BTC', datadir=testdatadir,
                                     timeframe='1m', timerange=timerange,
                                     drop_incomplete=False,
                                     fill_up_missing=False)

    base = 0.001
    if what == 'raise':
        data.loc[:, 'open'] = data.index * base
        data.loc[:, 'high'] = data.index * base + 0.0001
        data.loc[:, 'low'] = data.index * base - 0.0001
        data.loc[:, 'close'] = data.index * base

    if what == 'lower':
        data.loc[:, 'open'] = 1 - data.index * base
        data.loc[:, 'high'] = 1 - data.index * base + 0.0001
        data.loc[:, 'low'] = 1 - data.index * base - 0.0001
        data.loc[:, 'close'] = 1 - data.index * base

    if what == 'sine':
        hz = 0.1  # frequency
        data.loc[:, 'open'] = np.sin(data.index * hz) / 1000 + base
        data.loc[:, 'high'] = np.sin(data.index * hz) / 1000 + base + 0.0001
        data.loc[:, 'low'] = np.sin(data.index * hz) / 1000 + base - 0.0001
        data.loc[:, 'close'] = np.sin(data.index * hz) / 1000 + base

    return {'UNITTEST/BTC': clean_ohlcv_dataframe(data, timeframe='1m', pair='UNITTEST/BTC',
                                                  fill_missing=True, drop_incomplete=True)}


def test_biased_strategy(default_conf, mocker, caplog) -> None:

    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))

    default_conf['timeframe'] = '5m'
    default_conf['timerange'] = '-1510694220'
    default_conf['strategy'] = 'strategy_test_v3_with_lookahead_bias'
    default_conf['strategy_path'] = 'tests/strategy/strats'

    strategy_obj = {}
    strategy_obj['name'] = "strategy_test_v3_with_lookahead_bias"
    freqtrade.optimize.lookahead_analysis.LookaheadAnalysis(default_conf, strategy_obj, {})
    pass
