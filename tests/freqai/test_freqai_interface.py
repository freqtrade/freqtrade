import platform
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from tests.conftest import get_patched_exchange, log_has_re
from tests.freqai.conftest import get_patched_freqai_strategy


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


def test_train_model_in_series_LightGBM(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").is_file()

    shutil.rmtree(Path(freqai.dk.full_path))


def test_train_model_in_series_LightGBMMultiModel(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"strategy": "freqai_test_multimodel_strat"})
    freqai_conf.update({"freqaimodel": "LightGBMRegressorMultiTarget"})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert len(freqai.dk.label_list) == 2
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").is_file()
    assert len(freqai.dk.data['training_features_list']) == 26

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.skipif(is_arm(), reason="no ARM for Catboost ...")
def test_train_model_in_series_Catboost(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"freqaimodel": "CatboostRegressor"})
    # freqai_conf.get('freqai', {}).update(
    #     {'model_training_parameters': {"n_estimators": 100, "verbose": 0}})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)

    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC",
                                 strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").exists()

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.skipif(is_arm(), reason="no ARM for Catboost ...")
def test_train_model_in_series_CatboostClassifier(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"freqaimodel": "CatboostClassifier"})
    freqai_conf.update({"strategy": "freqai_test_classifier"})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)

    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC",
                                 strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").exists()

    shutil.rmtree(Path(freqai.dk.full_path))


def test_train_model_in_series_LightGBMClassifier(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"freqaimodel": "LightGBMClassifier"})
    freqai_conf.update({"strategy": "freqai_test_classifier"})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)

    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC",
                                 strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").exists()

    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)

    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")

    metadata = {"pair": "LTC/BTC"}
    freqai.start_backtesting(df, metadata, freqai.dk)
    model_folders = [x for x in freqai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 5

    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180120-20180124"})
    freqai_conf.get("freqai", {}).update({"backtest_period_days": 0.5})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)

    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")

    metadata = {"pair": "LTC/BTC"}
    freqai.start_backtesting(df, metadata, freqai.dk)
    model_folders = [x for x in freqai.dd.full_path.iterdir() if x.is_dir()]
    assert len(model_folders) == 8

    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, freqai_conf, caplog):
    freqai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)

    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")

    metadata = {"pair": "ADA/BTC"}
    freqai.start_backtesting(df, metadata, freqai.dk)
    model_folders = [x for x in freqai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 5

    # without deleting the exiting folder structure, re-run

    freqai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)

    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")
    freqai.start_backtesting(df, metadata, freqai.dk)

    assert log_has_re(
        "Found model at ",
        caplog,
    )

    shutil.rmtree(Path(freqai.dk.full_path))


def test_follow_mode(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    metadata = {"pair": "ADA/BTC"}
    freqai.dd.set_pair_dict_info(metadata)

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").is_file()

    # start the follower and ask it to predict on existing files

    freqai_conf.get("freqai", {}).update({"follow_mode": "true"})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.live)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    df = strategy.dp.get_pair_dataframe('ADA/BTC', '5m')
    freqai.start_live(df, metadata, strategy, freqai.dk)

    assert len(freqai.dk.return_dataframe.index) == 5702

    shutil.rmtree(Path(freqai.dk.full_path))


def test_principal_component_analysis(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.get("freqai", {}).get("feature_parameters", {}).update(
        {"princpial_component_analysis": "true"})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_pca_object.pkl")

    shutil.rmtree(Path(freqai.dk.full_path))
