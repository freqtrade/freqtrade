import datetime
import shutil
from pathlib import Path

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
# from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.freqai.conftest import get_patched_data_kitchen, get_patched_freqai_strategy


@pytest.mark.parametrize(
    "timerange, train_period_days, expected_result",
    [
        ("20220101-20220201", 30, "20211202-20220201"),
        ("20220301-20220401", 15, "20220214-20220401"),
    ],
)
def test_create_fulltimerange(
    timerange, train_period_days, expected_result, freqai_conf, mocker, caplog
):
    dk = get_patched_data_kitchen(mocker, freqai_conf)
    assert dk.create_fulltimerange(timerange, train_period_days) == expected_result
    shutil.rmtree(Path(dk.full_path))


def test_create_fulltimerange_incorrect_backtest_period(mocker, freqai_conf):
    dk = get_patched_data_kitchen(mocker, freqai_conf)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be an integer"):
        dk.create_fulltimerange("20220101-20220201", 0.5)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be positive"):
        dk.create_fulltimerange("20220101-20220201", -1)
    shutil.rmtree(Path(dk.full_path))


@pytest.mark.parametrize(
    "timerange, train_period_days, backtest_period_days, expected_result",
    [
        ("20220101-20220201", 30, 7, 9),
        ("20220101-20220201", 30, 0.5, 120),
        ("20220101-20220201", 10, 1, 80),
    ],
)
def test_split_timerange(
    mocker, freqai_conf, timerange, train_period_days, backtest_period_days, expected_result
):
    freqai_conf.update({"timerange": "20220101-20220401"})
    dk = get_patched_data_kitchen(mocker, freqai_conf)
    tr_list, bt_list = dk.split_timerange(timerange, train_period_days, backtest_period_days)
    assert len(tr_list) == len(bt_list) == expected_result

    with pytest.raises(
        OperationalException, match=r"train_period_days must be an integer greater than 0."
    ):
        dk.split_timerange("20220101-20220201", -1, 0.5)
    shutil.rmtree(Path(dk.full_path))


def test_update_historic_data(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")

    freqai.dk.load_all_pair_histories(timerange)
    historic_candles = len(freqai.dd.historic_data["ADA/BTC"]["5m"])
    dp_candles = len(strategy.dp.get_pair_dataframe("ADA/BTC", "5m"))
    candle_difference = dp_candles - historic_candles
    freqai.dk.update_historic_data(strategy)

    updated_historic_candles = len(freqai.dd.historic_data["ADA/BTC"]["5m"])

    assert updated_historic_candles - historic_candles == candle_difference
    shutil.rmtree(Path(freqai.dk.full_path))


@pytest.mark.parametrize(
    "timestamp, expected",
    [
        (datetime.datetime.now(tz=datetime.timezone.utc).timestamp() - 7200, True),
        (datetime.datetime.now(tz=datetime.timezone.utc).timestamp(), False),
    ],
)
def test_check_if_model_expired(mocker, freqai_conf, timestamp, expected):
    dk = get_patched_data_kitchen(mocker, freqai_conf)
    assert dk.check_if_model_expired(timestamp) == expected
    shutil.rmtree(Path(dk.full_path))


def test_load_all_pairs_histories(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dk.load_all_pair_histories(timerange)

    assert len(freqai.dd.historic_data.keys()) == len(
        freqai_conf.get("exchange", {}).get("pair_whitelist")
    )
    assert len(freqai.dd.historic_data["ADA/BTC"]) == len(
        freqai_conf.get("freqai", {}).get("feature_parameters", {}).get("include_timeframes")
    )
    shutil.rmtree(Path(freqai.dk.full_path))


def test_get_base_and_corr_dataframes(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = freqai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    num_tfs = len(
        freqai_conf.get("freqai", {}).get("feature_parameters", {}).get("include_timeframes")
    )

    assert len(base_df.keys()) == num_tfs

    assert len(corr_df.keys()) == len(
        freqai_conf.get("freqai", {}).get("feature_parameters", {}).get("include_corr_pairlist")
    )

    assert len(corr_df["ADA/BTC"].keys()) == num_tfs
    shutil.rmtree(Path(freqai.dk.full_path))


def test_use_strategy_to_populate_indicators(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get("freqai", {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = freqai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, 'LTC/BTC')

    assert len(df.columns) == 45
    shutil.rmtree(Path(freqai.dk.full_path))
