# from unittest.mock import MagicMock
# from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_edge
import copy
# import platform
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from tests.conftest import get_patched_exchange, log_has_re
from tests.freqai.conftest import freqai_conf, get_patched_freqai_strategy


def test_train_model_in_series_LightGBM(mocker, default_conf):
    freqaiconf = freqai_conf(copy.deepcopy(default_conf))
    freqaiconf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_freqai_strategy(mocker, freqaiconf)
    exchange = get_patched_exchange(mocker, freqaiconf)
    strategy.dp = DataProvider(freqaiconf, exchange)
    strategy.freqai_info = freqaiconf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    freqai.dk.load_all_pair_histories(timerange)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    freqai.train_model_in_series(new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    assert (
        Path(freqai.dk.data_path / str(freqai.dk.model_filename + "_model.joblib"))
        .resolve()
        .exists()
    )
    assert (
        Path(freqai.dk.data_path / str(freqai.dk.model_filename + "_metadata.json"))
        .resolve()
        .exists()
    )
    assert (
        Path(freqai.dk.data_path / str(freqai.dk.model_filename + "_trained_df.pkl"))
        .resolve()
        .exists()
    )
    assert (
        Path(freqai.dk.data_path / str(freqai.dk.model_filename + "_svm_model.joblib"))
        .resolve()
        .exists()
    )

    shutil.rmtree(Path(freqai.dk.full_path))


# FIXME: hits segfault
# @pytest.mark.skipif("arm" in platform.uname()[-1], reason="no ARM..")
# def test_train_model_in_series_Catboost(mocker, default_conf):
#     freqaiconf = freqai_conf(copy.deepcopy(default_conf))
#     freqaiconf.update({"timerange": "20180110-20180130"})
#     freqaiconf.update({"freqaimodel": "CatboostPredictionModel"})
#     strategy = get_patched_freqai_strategy(mocker, freqaiconf)
#     exchange = get_patched_exchange(mocker, freqaiconf)
#     strategy.dp = DataProvider(freqaiconf, exchange)
#     strategy.freqai_info = freqaiconf.get("freqai", {})
#     freqai = strategy.model.bridge
#     freqai.live = True
#     freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
#     timerange = TimeRange.parse_timerange("20180110-20180130")
#     freqai.dk.load_all_pair_histories(timerange)

#     freqai.dd.pair_dict = MagicMock()

#     data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
#     new_timerange = TimeRange.parse_timerange("20180120-20180130")

#     freqai.train_model_in_series(new_timerange, "ADA/BTC",
#                                  strategy, freqai.dk, data_load_timerange)

#     assert (
#         Path(freqai.dk.data_path / str(freqai.dk.model_filename + "_model.joblib"))
#         .resolve()
#         .exists()
#     )
#     assert (
#         Path(freqai.dk.data_path / str(freqai.dk.model_filename + "_metadata.json"))
#         .resolve()
#         .exists()
#     )
#     assert (
#         Path(freqai.dk.data_path / str(freqai.dk.model_filename + "_trained_df.pkl"))
#         .resolve()
#         .exists()
#     )
#     assert (
#         Path(freqai.dk.data_path / str(freqai.dk.model_filename + "_svm_model.joblib"))
#         .resolve()
#         .exists()
#     )

#     shutil.rmtree(Path(freqai.dk.full_path))


def test_start_backtesting(mocker, default_conf):
    freqaiconf = freqai_conf(copy.deepcopy(default_conf))
    freqaiconf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_freqai_strategy(mocker, freqaiconf)
    exchange = get_patched_exchange(mocker, freqaiconf)
    strategy.dp = DataProvider(freqaiconf, exchange)
    strategy.freqai_info = freqaiconf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
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


def test_start_backtesting_from_existing_folder(mocker, default_conf, caplog):
    freqaiconf = freqai_conf(copy.deepcopy(default_conf))
    freqaiconf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_freqai_strategy(mocker, freqaiconf)
    exchange = get_patched_exchange(mocker, freqaiconf)
    strategy.dp = DataProvider(freqaiconf, exchange)
    strategy.freqai_info = freqaiconf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
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

    freqaiconf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_freqai_strategy(mocker, freqaiconf)
    exchange = get_patched_exchange(mocker, freqaiconf)
    strategy.dp = DataProvider(freqaiconf, exchange)
    strategy.freqai_info = freqaiconf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
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
