
import shutil
from pathlib import Path

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.freqai.conftest import get_patched_freqai_strategy


def test_update_historic_data(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180114")

    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    historic_candles = len(freqai.dd.historic_data["ADA/BTC"]["5m"])
    dp_candles = len(strategy.dp.get_pair_dataframe("ADA/BTC", "5m"))
    candle_difference = dp_candles - historic_candles
    freqai.dk.pair = "ADA/BTC"
    freqai.dd.update_historic_data(strategy, freqai.dk)

    updated_historic_candles = len(freqai.dd.historic_data["ADA/BTC"]["5m"])

    assert updated_historic_candles - historic_candles == candle_difference
    shutil.rmtree(Path(freqai.dk.full_path))


def test_load_all_pairs_histories(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)

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
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)

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
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)

    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, 'LTC/BTC')

    assert len(df.columns) == 33
    shutil.rmtree(Path(freqai.dk.full_path))


def test_get_timerange_from_live_historic_predictions(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange = TimeRange.parse_timerange("20180126-20180130")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180128-20180130")
    _, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "ADA/BTC", freqai.dk)
    base_df["5m"]["date_pred"] = base_df["5m"]["date"]
    freqai.dd.historic_predictions = {}
    freqai.dd.historic_predictions["ADA/USDT"] = base_df["5m"]
    freqai.dd.save_historic_predictions_to_disk()
    freqai.dd.save_global_metadata_to_disk({"start_dry_live_date": 1516406400})

    timerange = freqai.dd.get_timerange_from_live_historic_predictions()
    assert timerange.startts == 1516406400
    assert timerange.stopts == 1517356500


def test_get_timerange_from_backtesting_live_df_pred_not_found(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    with pytest.raises(
            OperationalException,
            match=r'Historic predictions not found.*'
            ):
        freqai.dd.get_timerange_from_live_historic_predictions()
