import platform
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import get_patched_exchange, log_has_re
from tests.freqai.conftest import get_patched_freqai_strategy


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


@pytest.mark.parametrize('model', [
    'LightGBMRegressor',
    'XGBoostRegressor',
    'CatboostRegressor',
    ])
def test_extract_data_and_train_model_Regressors(mocker, freqai_conf, model):
    if is_arm() and model == 'CatboostRegressor':
        pytest.skip("CatBoost is not supported on ARM")

    freqai_conf.update({"freqaimodel": model})
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"strategy": "freqai_test_strat"})

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

    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").is_file()

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.parametrize('model', [
    'LightGBMRegressorMultiTarget',
    'XGBoostRegressorMultiTarget',
    'CatboostRegressorMultiTarget',
    ])
def test_extract_data_and_train_model_MultiTargets(mocker, freqai_conf, model):
    if is_arm() and model == 'CatboostRegressorMultiTarget':
        pytest.skip("CatBoost is not supported on ARM")

    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"strategy": "freqai_test_multimodel_strat"})
    freqai_conf.update({"freqaimodel": model})
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

    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert len(freqai.dk.label_list) == 2
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").is_file()
    assert len(freqai.dk.data['training_features_list']) == 26

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.parametrize('model', [
    'LightGBMClassifier',
    'CatboostClassifier',
    'XGBoostClassifier',
    ])
def test_extract_data_and_train_model_Classifiers(mocker, freqai_conf, model):
    if is_arm() and model == 'CatboostClassifier':
        pytest.skip("CatBoost is not supported on ARM")

    freqai_conf.update({"freqaimodel": model})
    freqai_conf.update({"strategy": "freqai_test_classifier"})
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

    freqai.extract_data_and_train_model(new_timerange, "ADA/BTC",
                                        strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").exists()

    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180120-20180130"})
    freqai_conf.get("freqai", {}).update({"save_backtest_models": True})
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

    assert len(model_folders) == 6

    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180120-20180124"})
    freqai_conf.get("freqai", {}).update({"backtest_period_days": 0.5})
    freqai_conf.get("freqai", {}).update({"save_backtest_models": True})
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
    assert len(model_folders) == 9

    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, freqai_conf, caplog):
    freqai_conf.update({"timerange": "20180120-20180130"})
    freqai_conf.get("freqai", {}).update({"save_backtest_models": True})
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

    assert len(model_folders) == 6

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
        "Found backtesting prediction file ",
        caplog,
    )

    path = (freqai.dd.full_path / freqai.dk.backtest_predictions_folder)
    prediction_files = [x for x in path.iterdir() if x.is_file()]
    assert len(prediction_files) == 5

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

    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

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

    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_pca_object.pkl")

    shutil.rmtree(Path(freqai.dk.full_path))


def test_plot_feature_importance(mocker, freqai_conf):

    from freqtrade.freqai.utils import plot_feature_importance

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

    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    model = freqai.dd.load_data("ADA/BTC", freqai.dk)

    plot_feature_importance(model, "ADA/BTC", freqai.dk)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}.html")

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.parametrize('timeframes,corr_pairs', [
    (['5m'], ['ADA/BTC', 'DASH/BTC']),
    (['5m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']),
    (['5m', '15m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']),
])
def test_freqai_informative_pairs(mocker, freqai_conf, timeframes, corr_pairs):
    freqai_conf['freqai']['feature_parameters'].update({
        'include_timeframes': timeframes,
        'include_corr_pairlist': corr_pairs,

    })
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    pairlists = PairListManager(exchange, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange, pairlists)
    pairlist = strategy.dp.current_whitelist()

    pairs_a = strategy.informative_pairs()
    assert len(pairs_a) == 0
    pairs_b = strategy.gather_informative_pairs()
    # we expect unique pairs * timeframes
    assert len(pairs_b) == len(set(pairlist + corr_pairs)) * len(timeframes)
