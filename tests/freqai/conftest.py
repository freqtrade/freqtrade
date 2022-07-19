from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.resolvers import StrategyResolver
from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver


# @pytest.fixture(scope="function")
def freqai_conf(default_conf):
    freqaiconf = deepcopy(default_conf)
    freqaiconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "strategy": "FreqaiExampleStrategy",
            "strategy-path": "freqtrade/templates",
            "freqaimodel": "LightGBMPredictionModel",
            "freqaimodel_path": "freqai/prediction_models",
            "timerange": "20180110-20180115",
            "freqai": {
                "startup_candles": 10000,
                "purge_old_models": True,
                "train_period_days": 15,
                "backtest_period_days": 7,
                "live_retrain_hours": 0,
                "identifier": "uniqe-id7",
                "live_trained_timestamp": 0,
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "include_corr_pairlist": ["ADA/BTC", "DASH/BTC"],
                    "label_period_candles": 20,
                    "include_shifted_candles": 2,
                    "DI_threshold": 0.9,
                    "weight_factor": 0.9,
                    "principal_component_analysis": False,
                    "use_SVM_to_remove_outliers": True,
                    "stratify_training_data": 0,
                    "indicator_max_period_candles": 10,
                    "indicator_periods_candles": [10],
                },
                "data_split_parameters": {"test_size": 0.33, "random_state": 1},
                "model_training_parameters": {"n_estimators": 1000, "task_type": "CPU"},
            },
            "config_files": [Path('config_examples', 'config_freqai_futures.example.json')]
        }
    )
    freqaiconf['exchange'].update({'pair_whitelist': ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC']})
    return freqaiconf


def get_patched_data_kitchen(mocker, freqaiconf):
    dd = mocker.patch('freqtrade.freqai.data_drawer', MagicMock())
    dk = FreqaiDataKitchen(freqaiconf, dd)
    return dk


def get_patched_strategy(mocker, freqaiconf):
    strategy = StrategyResolver.load_strategy(freqaiconf)
    strategy.bot_start()

    return strategy


def get_patched_freqaimodel(mocker, freqaiconf):
    freqaimodel = FreqaiModelResolver.load_freqaimodel(freqaiconf)

    return freqaimodel
