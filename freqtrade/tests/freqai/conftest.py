import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.resolvers import StrategyResolver
from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver
from tests.conftest import get_patched_exchange


def is_py12() -> bool:
    return sys.version_info >= (3, 12)


@pytest.fixture(scope="function")
def freqai_conf(default_conf, tmp_path):
    freqaiconf = deepcopy(default_conf)
    freqaiconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "runmode": "backtest",
            "strategy": "freqai_test_strat",
            "user_data_dir": tmp_path,
            "strategy-path": "freqtrade/tests/strategy/strats",
            "freqaimodel": "LightGBMRegressor",
            "freqaimodel_path": "freqai/prediction_models",
            "timerange": "20180110-20180115",
            "freqai": {
                "enabled": True,
                "purge_old_models": 2,
                "train_period_days": 2,
                "backtest_period_days": 10,
                "live_retrain_hours": 0,
                "expiration_hours": 1,
                "identifier": "unique-id100",
                "live_trained_timestamp": 0,
                "data_kitchen_thread_count": 2,
                "activate_tensorboard": False,
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "include_corr_pairlist": ["ADA/BTC"],
                    "label_period_candles": 20,
                    "include_shifted_candles": 1,
                    "DI_threshold": 0.9,
                    "weight_factor": 0.9,
                    "principal_component_analysis": False,
                    "use_SVM_to_remove_outliers": True,
                    "stratify_training_data": 0,
                    "indicator_periods_candles": [10],
                    "shuffle_after_split": False,
                    "buffer_train_data_candles": 0,
                },
                "data_split_parameters": {"test_size": 0.33, "shuffle": False},
                "model_training_parameters": {"n_estimators": 100},
            },
            "config_files": [Path("config_examples", "config_freqai.example.json")],
        }
    )
    freqaiconf["exchange"].update({"pair_whitelist": ["ADA/BTC", "DASH/BTC", "ETH/BTC", "LTC/BTC"]})
    return freqaiconf


def make_rl_config(conf):
    conf.update({"strategy": "freqai_rl_test_strat"})
    conf["freqai"].update(
        {"model_training_parameters": {"learning_rate": 0.00025, "gamma": 0.9, "verbose": 1}}
    )
    conf["freqai"]["rl_config"] = {
        "train_cycles": 1,
        "thread_count": 2,
        "max_trade_duration_candles": 300,
        "model_type": "PPO",
        "policy_type": "MlpPolicy",
        "max_training_drawdown_pct": 0.5,
        "net_arch": [32, 32],
        "model_reward_parameters": {"rr": 1, "profit_aim": 0.02, "win_reward_factor": 2},
        "drop_ohlc_from_features": False,
    }

    return conf


def mock_pytorch_mlp_model_training_parameters() -> dict[str, Any]:
    return {
        "learning_rate": 3e-4,
        "trainer_kwargs": {
            "n_steps": None,
            "batch_size": 64,
            "n_epochs": 1,
        },
        "model_kwargs": {
            "hidden_dim": 32,
            "dropout_percent": 0.2,
            "n_layer": 1,
        },
    }


def get_patched_data_kitchen(mocker, freqaiconf):
    dk = FreqaiDataKitchen(freqaiconf)
    return dk


def get_patched_data_drawer(mocker, freqaiconf):
    # dd = mocker.patch('freqtrade.freqai.data_drawer', MagicMock())
    dd = FreqaiDataDrawer(freqaiconf)
    return dd


def get_patched_freqai_strategy(mocker, freqaiconf):
    strategy = StrategyResolver.load_strategy(freqaiconf)
    strategy.ft_bot_start()

    return strategy


def get_patched_freqaimodel(mocker, freqaiconf):
    freqaimodel = FreqaiModelResolver.load_freqaimodel(freqaiconf)

    return freqaimodel


def make_unfiltered_dataframe(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
    freqai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(data_load_timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = freqai.dd.get_base_and_corr_dataframes(
        data_load_timerange, freqai.dk.pair, freqai.dk
    )

    unfiltered_dataframe = freqai.dk.use_strategy_to_populate_indicators(
        strategy, corr_dataframes, base_dataframes, freqai.dk.pair
    )
    for i in range(5):
        unfiltered_dataframe[f"constant_{i}"] = i

    unfiltered_dataframe = freqai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    return freqai, unfiltered_dataframe


def make_data_dictionary(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
    freqai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(data_load_timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = freqai.dd.get_base_and_corr_dataframes(
        data_load_timerange, freqai.dk.pair, freqai.dk
    )

    unfiltered_dataframe = freqai.dk.use_strategy_to_populate_indicators(
        strategy, corr_dataframes, base_dataframes, freqai.dk.pair
    )

    unfiltered_dataframe = freqai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    freqai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = freqai.dk.filter_features(
        unfiltered_dataframe,
        freqai.dk.training_features_list,
        freqai.dk.label_list,
        training_filter=True,
    )

    data_dictionary = freqai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    data_dictionary = freqai.dk.normalize_data(data_dictionary)

    return freqai


def get_freqai_live_analyzed_dataframe(mocker, freqaiconf):
    strategy = get_patched_freqai_strategy(mocker, freqaiconf)
    exchange = get_patched_exchange(mocker, freqaiconf)
    strategy.dp = DataProvider(freqaiconf, exchange)
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dk.load_all_pair_histories(timerange)

    strategy.analyze_pair("ADA/BTC", "5m")
    return strategy.dp.get_analyzed_dataframe("ADA/BTC", "5m")


def get_freqai_analyzed_dataframe(mocker, freqaiconf):
    strategy = get_patched_freqai_strategy(mocker, freqaiconf)
    exchange = get_patched_exchange(mocker, freqaiconf)
    strategy.dp = DataProvider(freqaiconf, exchange)
    strategy.freqai_info = freqaiconf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = freqai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    return freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")


def get_ready_to_train(mocker, freqaiconf):
    strategy = get_patched_freqai_strategy(mocker, freqaiconf)
    exchange = get_patched_exchange(mocker, freqaiconf)
    strategy.dp = DataProvider(freqaiconf, exchange)
    strategy.freqai_info = freqaiconf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = freqai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")
    return corr_df, base_df, freqai, strategy
