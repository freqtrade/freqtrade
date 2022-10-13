import platform
from unittest.mock import MagicMock

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_util import (get_assets_timestamps_training_from_ready_models,
                                          get_full_models_path,
                                          get_timerange_and_assets_end_dates_from_ready_models,
                                          get_timerange_backtest_live_models)
from tests.conftest import get_patched_exchange
from tests.freqai.conftest import get_patched_freqai_strategy


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


@pytest.mark.parametrize('model', [
    'LightGBMRegressor'
    ])
def test_get_full_model_path(mocker, freqai_conf, model):
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

    model_path = get_full_models_path(freqai_conf)
    assert model_path.is_dir() is True


def test_get_pairs_timestamp_validation(mocker, freqai_conf):
    model_path = get_full_models_path(freqai_conf)
    with pytest.raises(
            OperationalException,
            match=r'.*required to run backtest with the freqai-backtest-live-models.*'
            ):
        get_assets_timestamps_training_from_ready_models(model_path)


@pytest.mark.parametrize('model', [
    'LightGBMRegressor'
    ])
def test_get_timerange_from_ready_models(mocker, freqai_conf, model):
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
    timerange = TimeRange.parse_timerange("20180101-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

    freqai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180101-20180130")

    # 1516233600 (2018-01-18 00:00) - Start Training 1
    # 1516406400 (2018-01-20 00:00) - End Training 1 (Backtest slice 1)
    # 1516579200 (2018-01-22 00:00) - End Training 2 (Backtest slice 2)
    # 1516838400 (2018-01-25 00:00) - End Timerange

    new_timerange = TimeRange("date", "date", 1516233600, 1516406400)
    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    new_timerange = TimeRange("date", "date", 1516406400, 1516579200)
    freqai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, freqai.dk, data_load_timerange)

    model_path = get_full_models_path(freqai_conf)
    (backtesting_timerange,
     pairs_end_dates) = get_timerange_and_assets_end_dates_from_ready_models(models_path=model_path)

    assert len(pairs_end_dates["ADA"]) == 2
    assert backtesting_timerange.startts == 1516406400
    assert backtesting_timerange.stopts == 1516838400

    backtesting_string_timerange = get_timerange_backtest_live_models(freqai_conf)
    assert backtesting_string_timerange == '20180120-20180125'
