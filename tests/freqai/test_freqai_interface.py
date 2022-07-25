# from unittest.mock import MagicMock
# from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_edge
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


def test_train_model_in_series_LightGBM(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dk.load_all_pair_histories(timerange)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_model.joblib").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_metadata.json").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_trained_df.pkl").is_file()
    assert Path(freqai.dk.data_path / f"{freqai.dk.model_filename}_svm_model.joblib").is_file()

    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.skipif("arm" in platform.uname()[-1], reason="no ARM for Catboost ...")
def test_train_model_in_series_Catboost(mocker, freqai_conf):
    freqai_conf.update({"timerange": "20180110-20180130"})
    freqai_conf.update({"freqaimodel": "CatboostPredictionModel"})
    del freqai_conf['freqai']['model_training_parameters']['verbosity']
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)

    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dk.load_all_pair_histories(timerange)

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
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    corr_df, base_df = freqai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")

    metadata = {"pair": "ADA/BTC"}
    freqai.start_backtesting(df, metadata, freqai.dk)
    model_folders = [x for x in freqai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 5

    shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, freqai_conf, caplog):
    freqai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    corr_df, base_df = freqai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

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
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    corr_df, base_df = freqai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

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
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dk.load_all_pair_histories(timerange)

    metadata = {"pair": "ADA/BTC"}
    freqai.dd.set_pair_dict_info(metadata)
    # freqai.dd.pair_dict = MagicMock()

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
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd, freqai.live)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dk.load_all_pair_histories(timerange)

    df = strategy.dp.get_pair_dataframe('ADA/BTC', '5m')
    freqai.start_live(df, metadata, strategy, freqai.dk)

    assert len(freqai.dk.return_dataframe.index) == 5702

    shutil.rmtree(Path(freqai.dk.full_path))
