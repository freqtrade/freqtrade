import platform
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RunMode
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.utils import download_all_data_for_training, get_required_data_timerange
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, create_mock_trades, get_patched_exchange, log_has_re
from tests.freqai.conftest import (get_patched_freqai_strategy, is_mac, make_rl_config,
                                   mock_pytorch_mlp_model_training_parameters)


def is_py11() -> bool:
    return sys.version_info >= (3, 11)


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


def can_run_model(model: str) -> None:
    if is_arm() and "Catboost" in model:
        pytest.skip("CatBoost is not supported on ARM.")

    is_pytorch_model = 'Reinforcement' in model or 'PyTorch' in model
    if is_pytorch_model and is_mac() and not is_arm():
        pytest.skip("Reinforcement learning / PyTorch module not available on intel based Mac OS.")


@pytest.mark.parametrize('model, pca, dbscan, float32, can_short, shuffle, buffer, noise', [
    ('LightGBMRegressor', True, False, True, True, False, 0, 0),
    ('XGBoostRegressor', False, True, False, True, False, 10, 0.05),
    ('XGBoostRFRegressor', False, False, False, True, False, 0, 0),
    ('CatboostRegressor', False, False, False, True, True, 0, 0),
    ('PyTorchMLPRegressor', False, False, False, False, False, 0, 0),
    ('PyTorchTransformerRegressor', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner', False, True, False, True, False, 0, 0),
    ('ReinforcementLearner_multiproc', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_4ac', False, False, False, True, False, 0, 0),
    ])
def test_extract_data_and_train_model_Standard(mocker, freqai_conf, model, pca,
                                               dbscan, float32, can_short, shuffle,
                                               buffer, noise):

    can_run_model(model)

    test_tb = True
    if is_mac():
        test_tb = False

    model_save_ext = 'joblib'
    freqai_conf.update({"freqaimodel": model})
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"strategy": "freqai_test_strat"})
    freqai_conf['freqai']['feature_parameters'].update({"principal_component_analysis": pca})
    freqai_conf['freqai']['feature_parameters'].update({"use_DBSCAN_to_remove_outliers": dbscan})
    freqai_conf.update({"reduce_df_footprint": float32})
    freqai_conf['freqai']['feature_parameters'].update({"shuffle_after_split": shuffle})
    freqai_conf['freqai']['feature_parameters'].update({"buffer_train_data_candles": buffer})
    freqai_conf['freqai']['feature_parameters'].update({"noise_standard_deviation": noise})

    if 'ReinforcementLearner' in model:
        model_save_ext = 'zip'
        freqai_conf = make_rl_config(freqai_conf)
        # test the RL guardrails
        freqai_conf['freqai']['feature_parameters'].update({"use_SVM_to_remove_outliers": True})
        freqai_conf['freqai']['feature_parameters'].update({"DI_threshold": 2})
        freqai_conf['freqai']['data_split_parameters'].update({'shuffle': True})

    if 'test_3ac' in model or 'test_4ac' in model:
        freqai_conf["freqaimodel_path"] = str(Path(__file__).parents[1] / "freqai" / "test_models")
        freqai_conf["freqai"]["rl_config"]["drop_ohlc_from_features"] = True

    if 'PyTorch' in model:
        model_save_ext = 'zip'
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        freqai_conf['freqai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            # transformer model takes a window, unlike the MLP regressor
            freqai_conf.update({"conv_width": 10})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.activate_tensorboard = test_tb
    freqai.can_short = can_short
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
    freqai.dk.set_paths('ADA/BTC', 10000)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180125-20180130")
    new_timerange = TimeRange.parse_timerange("20180127-20180130")
    freqai.dk.set_paths('ADA/BTC', None)

    freqai.train_timer("start", "ADA/BTC")
    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)
    freqai.train_timer("stop", "ADA/BTC")
    freqai.dd.save_metric_tracker_to_disk()
    freqai.dd.save_drawer_to_disk()

    assert Path(freqai.dk.full_path / "metric_tracker.json").is_file()
    assert Path(freqai.dk.full_path / "pair_dictionary.json").is_file()
    assert Path(freqai.dk.data_path /
                f"{freqai.dk.model_filename}_model.{model_save_ext}").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").is_file()

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.parametrize('model, strat', [
    ('LightGBMRegressorMultiTarget', "freqai_test_multimodel_strat"),
    ('XGBoostRegressorMultiTarget', "freqai_test_multimodel_strat"),
    ('CatboostRegressorMultiTarget', "freqai_test_multimodel_strat"),
    ('LightGBMClassifierMultiTarget', "freqai_test_multimodel_classifier_strat"),
    ('CatboostClassifierMultiTarget', "freqai_test_multimodel_classifier_strat")
    ])
def test_extract_data_and_train_model_MultiTargets(mocker, freqai_conf, model, strat):
    can_run_model(model)

    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"strategy": strat})
    freqai_conf.update({"freqaimodel": model})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    freqai.dk.set_paths('ADA/BTC', None)

    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert len(freqai.dk.label_list) == 2
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").is_file()
    assert len(freqai.dk.data['training_features_list']) == 14

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.parametrize('model', [
    'LightGBMClassifier',
    'CatboostClassifier',
    'XGBoostClassifier',
    'XGBoostRFClassifier',
    'PyTorchMLPClassifier',
    ])
def test_extract_data_and_train_model_Classifiers(mocker, freqai_conf, model):
    can_run_model(model)

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
    freqai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    freqai.dk.set_paths('ADA/BTC', None)

    freqai.extract_data_and_train_model(new_timerange, "ADA/BTC",
                                        strategy, freqai.dk, data_load_timerange)

    if 'PyTorchMLPClassifier':
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        freqai_conf['freqai']['model_training_parameters'].update(pytorch_mlp_mtp)

    if freqai.dd.model_type == 'joblib':
        model_file_extension = ".joblib"
    elif freqai.dd.model_type == "pytorch":
        model_file_extension = ".zip"
    else:
        raise Exception(f"Unsupported model type: {freqai.dd.model_type},"
                        f" can't assign model_file_extension")

    assert Path(freqai.dk.data_path /
                f"{freqai.dk.model_filename}_model{model_file_extension}").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").exists()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").exists()

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.parametrize(
    "model, num_files, strat",
    [
        ("LightGBMRegressor", 2, "freqai_test_strat"),
        ("XGBoostRegressor", 2, "freqai_test_strat"),
        ("CatboostRegressor", 2, "freqai_test_strat"),
        ("PyTorchMLPRegressor", 2, "freqai_test_strat"),
        ("PyTorchTransformerRegressor", 2, "freqai_test_strat"),
        ("ReinforcementLearner", 3, "freqai_rl_test_strat"),
        ("XGBoostClassifier", 2, "freqai_test_classifier"),
        ("LightGBMClassifier", 2, "freqai_test_classifier"),
        ("CatboostClassifier", 2, "freqai_test_classifier"),
        ("PyTorchMLPClassifier", 2, "freqai_test_classifier")
    ],
    )
def test_start_backtesting(mocker, freqai_conf, model, num_files, strat, caplog):
    can_run_model(model)
    test_tb = True
    if is_mac():
        test_tb = False

    freqai_conf.get("freqai", {}).update({"save_backtest_models": True})
    freqai_conf['runmode'] = RunMode.BACKTEST

    Trade.use_db = False

    freqai_conf.update({"freqaimodel": model})
    freqai_conf.update({"timerange": "20180120-20180130"})
    freqai_conf.update({"strategy": strat})

    if 'ReinforcementLearner' in model:
        freqai_conf = make_rl_config(freqai_conf)

    if 'test_4ac' in model:
        freqai_conf["freqaimodel_path"] = str(Path(__file__).parents[1] / "freqai" / "test_models")

    if 'PyTorch' in model:
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        freqai_conf['freqai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            # transformer model takes a window, unlike the MLP regressor
            freqai_conf.update({"conv_width": 10})

    freqai_conf.get("freqai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.activate_tensorboard = test_tb
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)
    df = base_df[freqai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    freqai.dk.set_paths('LTC/BTC', None)
    freqai.start_backtesting(df, metadata, freqai.dk, strategy)
    model_folders = [x for x in freqai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == num_files
    Trade.use_db = True
    Backtesting.cleanup()
    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180120-20180124"})
    freqai_conf.get("freqai", {}).update({"backtest_period_days": 0.5})
    freqai_conf.get("freqai", {}).update({"save_backtest_models": True})
    freqai_conf.get("freqai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})
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
    _, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)
    df = base_df[freqai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    freqai.start_backtesting(df, metadata, freqai.dk, strategy)
    model_folders = [x for x in freqai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 9

    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, freqai_conf, caplog):
    freqai_conf.update({"timerange": "20180120-20180130"})
    freqai_conf.get("freqai", {}).update({"save_backtest_models": True})
    freqai_conf.get("freqai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180101-20180130")
    _, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)
    df = base_df[freqai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    freqai.dk.pair = pair
    freqai.start_backtesting(df, metadata, freqai.dk, strategy)
    model_folders = [x for x in freqai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 2

    # without deleting the existing folder structure, re-run

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
    _, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)
    df = base_df[freqai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    freqai.dk.pair = pair
    freqai.start_backtesting(df, metadata, freqai.dk, strategy)

    assert log_has_re(
        "Found backtesting prediction file ",
        caplog,
    )

    pair = "ETH/BTC"
    metadata = {"pair": pair}
    freqai.dk.pair = pair
    freqai.start_backtesting(df, metadata, freqai.dk, strategy)

    path = (freqai.dd.full_path / freqai.dk.backtest_predictions_folder)
    prediction_files = [x for x in path.iterdir() if x.is_file()]
    assert len(prediction_files) == 2

    shutil.rmtree(Path(freqai.dk.full_path))


def test_backtesting_fit_live_predictions(mocker, freqai_conf, caplog):
    freqai_conf.get("freqai", {}).update({"fit_live_predictions_candles": 10})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180128-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180129-20180130")
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)
    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")
    df = strategy.set_freqai_targets(df.copy(), metadata={"pair": "LTC/BTC"})
    df = freqai.dk.remove_special_chars_from_feature_names(df)
    freqai.dk.get_unique_classes_from_labels(df)
    freqai.dk.pair = "ADA/BTC"
    freqai.dk.full_df = df.fillna(0)
    freqai.dk.full_df
    assert "&-s_close_mean" not in freqai.dk.full_df.columns
    assert "&-s_close_std" not in freqai.dk.full_df.columns
    freqai.backtesting_fit_live_predictions(freqai.dk)
    assert "&-s_close_mean" in freqai.dk.full_df.columns
    assert "&-s_close_std" in freqai.dk.full_df.columns
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
    freqai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = {"ADA/BTC": {"model_filename": "fake_name",
                                       "trained_timestamp": 1, "data_path": "", "extras": {}}}

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    freqai.dk.set_paths('ADA/BTC', None)

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


def test_start_set_train_queue(mocker, freqai_conf, caplog):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    pairlist = PairListManager(exchange, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange, pairlist)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False

    freqai.train_queue = freqai._set_train_queue()

    assert log_has_re(
        "Set fresh train queue from whitelist.",
        caplog,
    )


def test_get_required_data_timerange(mocker, freqai_conf):
    time_range = get_required_data_timerange(freqai_conf)
    assert (time_range.stopts - time_range.startts) == 177300


def test_download_all_data_for_training(mocker, freqai_conf, caplog, tmpdir):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    pairlist = PairListManager(exchange, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange, pairlist)
    freqai_conf['pairs'] = freqai_conf['exchange']['pair_whitelist']
    freqai_conf['datadir'] = Path(tmpdir)
    download_all_data_for_training(strategy.dp, freqai_conf)

    assert log_has_re(
        "Downloading",
        caplog,
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize('dp_exists', [(False), (True)])
def test_get_state_info(mocker, freqai_conf, dp_exists, caplog, tickers):

    if is_mac():
        pytest.skip("Reinforcement learning module not available on intel based Mac OS")
    if is_py11():
        pytest.skip("Reinforcement learning currently not available on python 3.11.")

    freqai_conf.update({"freqaimodel": "ReinforcementLearner"})
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"strategy": "freqai_rl_test_strat"})
    freqai_conf = make_rl_config(freqai_conf)
    freqai_conf['entry_pricing']['price_side'] = 'same'
    freqai_conf['exit_pricing']['price_side'] = 'same'

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    ticker_mock = MagicMock(return_value=tickers()['ETH/BTC'])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    strategy.dp = DataProvider(freqai_conf, exchange)

    if not dp_exists:
        strategy.dp._exchange = None

    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.data_provider = strategy.dp
    freqai.live = True

    Trade.use_db = True
    create_mock_trades(MagicMock(return_value=0.0025), False, True)
    freqai.get_state_info("ADA/BTC")
    freqai.get_state_info("ETH/BTC")

    if not dp_exists:
        assert log_has_re(
            "No exchange available",
            caplog,
        )
