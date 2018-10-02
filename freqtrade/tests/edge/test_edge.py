# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

import json
import math
import random
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from arrow import Arrow

from freqtrade import DependencyException, constants, optimize
from freqtrade.arguments import Arguments, TimeRange
from freqtrade.optimize.backtesting import (Backtesting, setup_configuration,
                                            start)
from freqtrade.tests.conftest import log_has, patch_exchange, get_patched_exchange
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.default_strategy import DefaultStrategy

from freqtrade.exchange import Exchange
from freqtrade.freqtradebot import FreqtradeBot

from freqtrade.edge import Edge


# Cases to be tested:
# 1) Three complete trades within dataframe (with sell or buy hit for all)
# 2) Two open trades but one without sell/buy hit
# 3) Two complete trades and one which should not be considered as it happend while
#    there was an already open trade on pair
# 4) Three complete trades with buy=1 on the last frame
# 5) Candle drops 8%, stoploss at 1%: Trade closed, 1% loss
# 6) Candle drops 4% but recovers to 1% loss, stoploss at 3%: Trade closed, 3% loss
# 7) Candle drops 4% recovers to 1% entry criteria are met candle drops
#    20%, stoploss at 2%: Trade 1 closed, Loss 2%, Trade 2 opened, Trade 2 closed, Loss 2%


def test_filter(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value=[
            ['E/F', -0.01, 0.66, 3.71, 0.50, 1.71],
            ['C/D', -0.01, 0.66, 3.71, 0.50, 1.71],
            ['N/O', -0.01, 0.66, 3.71, 0.50, 1.71]
        ]
    ))

    pairs = ['A/B', 'C/D', 'E/F', 'G/H']
    assert(edge.filter(pairs) == ['E/F', 'C/D'])
