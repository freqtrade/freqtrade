from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from freqtrade.enums import ExitType, RunMode
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.hyperopt import Hyperopt
from tests.conftest import patch_exchange


@pytest.fixture(scope="function")
def hyperopt_conf(default_conf):
    hyperconf = deepcopy(default_conf)
    hyperconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "runmode": RunMode.HYPEROPT,
            "strategy": "HyperoptableStrategy",
            "hyperopt_loss": "ShortTradeDurHyperOptLoss",
            "hyperopt_path": str(Path(__file__).parent / "hyperopts"),
            "epochs": 1,
            "timerange": None,
            "spaces": ["default"],
            "hyperopt_jobs": 1,
            "hyperopt_min_trades": 1,
        }
    )
    return hyperconf


@pytest.fixture(autouse=True)
def backtesting_cleanup():
    yield None

    Backtesting.cleanup()


@pytest.fixture(scope="function")
def hyperopt(hyperopt_conf, mocker):
    patch_exchange(mocker)
    return Hyperopt(hyperopt_conf)


@pytest.fixture(scope="function")
def hyperopt_results():
    return pd.DataFrame(
        {
            "pair": ["ETH/USDT", "ETH/USDT", "ETH/USDT", "ETH/USDT"],
            "profit_ratio": [-0.1, 0.2, -0.12, 0.3],
            "profit_abs": [-0.2, 0.4, -0.21, 0.6],
            "trade_duration": [10, 30, 10, 10],
            "amount": [0.1, 0.1, 0.1, 0.1],
            "exit_reason": [ExitType.STOP_LOSS, ExitType.ROI, ExitType.STOP_LOSS, ExitType.ROI],
            "open_date": [
                datetime(2019, 1, 1, 9, 15, 0),
                datetime(2019, 1, 2, 8, 55, 0),
                datetime(2019, 1, 3, 9, 15, 0),
                datetime(2019, 1, 4, 9, 15, 0),
            ],
            "close_date": [
                datetime(2019, 1, 1, 9, 25, 0),
                datetime(2019, 1, 2, 9, 25, 0),
                datetime(2019, 1, 3, 9, 25, 0),
                datetime(2019, 1, 4, 9, 25, 0),
            ],
        }
    )
