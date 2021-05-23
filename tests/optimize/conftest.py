from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.state import RunMode
from freqtrade.strategy.interface import SellType
from tests.conftest import patch_exchange


@pytest.fixture(scope='function')
def hyperopt_conf(default_conf):
    hyperconf = deepcopy(default_conf)
    hyperconf.update({
        'datadir': Path(default_conf['datadir']),
        'runmode': RunMode.HYPEROPT,
        'hyperopt': 'DefaultHyperOpt',
        'hyperopt_loss': 'ShortTradeDurHyperOptLoss',
                         'hyperopt_path': str(Path(__file__).parent / 'hyperopts'),
                         'epochs': 1,
                         'timerange': None,
                         'spaces': ['default'],
                         'hyperopt_jobs': 1,
        'hyperopt_min_trades': 1,
    })
    return hyperconf


@pytest.fixture(scope='function')
def hyperopt(hyperopt_conf, mocker):

    patch_exchange(mocker)
    return Hyperopt(hyperopt_conf)


@pytest.fixture(scope='function')
def hyperopt_results():
    return pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_ratio': [-0.1, 0.2, 0.3],
            'profit_abs': [-0.2, 0.4, 0.6],
            'trade_duration': [10, 30, 10],
            'sell_reason': [SellType.STOP_LOSS, SellType.ROI, SellType.ROI],
            'close_date':
            [
                datetime(2019, 1, 1, 9, 26, 3, 478039),
                datetime(2019, 2, 1, 9, 26, 3, 478039),
                datetime(2019, 3, 1, 9, 26, 3, 478039)
            ]
        }
    )
