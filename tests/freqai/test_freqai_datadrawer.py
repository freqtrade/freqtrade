import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from tests.conftest import get_patched_exchange, log_has_re
from tests.freqai.conftest import get_patched_freqai_strategy


def test_update_historic_data(mocker, freqai_conf):
    freqai_conf["runmode"] = "backtest"
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
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
    freqai.dk.live = True
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
    freqai.dk.live = True
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
    freqai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", freqai.dk)

    df = freqai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")

    assert len(df.columns) == 33
    shutil.rmtree(Path(freqai.dk.full_path))


def test_get_timerange_from_live_historic_predictions(mocker, freqai_conf):
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = False
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
    with pytest.raises(OperationalException, match=r"Historic predictions not found.*"):
        freqai.dd.get_timerange_from_live_historic_predictions()


def test_load_historic_predictions_repairs(mocker, freqai_conf, caplog):
    """
    Prediction files written by older versions can carry duplicated candles, numeric
    columns as object dtype, or dates as object dtype.
    """
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.dk = FreqaiDataKitchen(freqai_conf)

    pair = "BTC/USD"
    dates = pd.date_range(start="2025-09-01", periods=5, freq="D", tz="UTC").astype(
        "datetime64[ms, UTC]"
    )
    # 2025-09-03 was predicted twice, the second prediction is the valid one
    dupe_dates = dates.insert(3, dates[2])

    freqai.dd.historic_predictions[pair] = pd.DataFrame(
        {
            "date_pred": dupe_dates,
            # numpy scalars written into an object column stay numpy scalars
            "&-s_close": pd.Series([1.0, 2, np.float64(3.0), 3.5, np.int64(4), 5.0], dtype=object),
            "do_predict": pd.Series([np.int64(1)] * 6, dtype=object),
            # class labels are legitimately non-numeric and must survive untouched
            "&s-up_or_down": pd.Series(["up", "down", "up", "down", "up", "down"], dtype=object),
        }
    )
    # an old file may also carry object dtype dates
    freqai.dd.historic_predictions["ETH/USD"] = pd.DataFrame(
        {"date_pred": dates.astype(object), "&-s_close": range(5)}
    )
    freqai.dd.save_historic_predictions_to_disk()
    freqai.dd.historic_predictions = {}

    assert freqai.dd.load_historic_predictions_from_disk()

    hist_pred_df = freqai.dd.historic_predictions[pair]
    assert not hist_pred_df["date_pred"].duplicated().any()
    assert list(hist_pred_df["date_pred"]) == list(dates)
    # the most recent prediction of the duplicated candle survived
    assert hist_pred_df["&-s_close"].iloc[2] == 3.5
    assert list(hist_pred_df.index) == list(range(5))
    assert log_has_re(f"Found 1 duplicated candle.*{pair}.*", caplog)

    # numeric columns are converted back, class labels are left alone
    assert hist_pred_df["&-s_close"].dtype == "float64"
    assert hist_pred_df["do_predict"].dtype == "int64"
    assert hist_pred_df["&s-up_or_down"].dtype == "object"
    assert list(hist_pred_df["&s-up_or_down"]) == ["up", "down", "down", "up", "down"]
    assert log_has_re(f"Converting 2 object dtype column.*{pair}.*", caplog)

    # no numpy scalars survive - they'd break the API serialization
    assert not any(
        isinstance(value, np.generic) for row in hist_pred_df.values.tolist() for value in row
    )

    # untouched pairs keep working, and get their dtype fixed
    assert freqai.dd.historic_predictions["ETH/USD"]["date_pred"].dtype == "datetime64[ms, UTC]"
    assert len(freqai.dd.historic_predictions["ETH/USD"]) == 5

    shutil.rmtree(Path(freqai.dk.full_path))


def test_set_initial_return_values(mocker, freqai_conf):
    """
    Simple test of the set initial return values that ensures
    we are concatenating and ffilling values properly.
    """

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    # Setup
    pair = "BTC/USD"
    end_x = "2023-08-31"
    start_x_plus_1 = "2023-08-30"
    end_x_plus_5 = "2023-09-03"

    historic_data = {"date_pred": pd.date_range(end=end_x, periods=5), "value": range(1, 6)}
    new_data = {
        "date": pd.date_range(start=start_x_plus_1, end=end_x_plus_5),
        "value": range(6, 11),
    }

    freqai.dd.historic_predictions[pair] = pd.DataFrame(historic_data)

    new_pred_df = pd.DataFrame(new_data)
    dataframe = pd.DataFrame(new_data)

    # Action
    with patch("logging.Logger.warning") as mock_logger_warning:
        freqai.dd.set_initial_return_values(pair, new_pred_df, dataframe)

    # Assertions
    hist_pred_df = freqai.dd.historic_predictions[pair]
    model_return_df = freqai.dd.model_return_values[pair]

    assert hist_pred_df["date_pred"].iloc[-1] == pd.Timestamp(end_x_plus_5)
    assert "date_pred" in hist_pred_df.columns
    assert hist_pred_df.shape[0] == 8

    # compare values in model_return_df with hist_pred_df
    assert (
        model_return_df["value"].values == hist_pred_df.tail(len(dataframe))["value"].values
    ).all()
    assert model_return_df.shape[0] == len(dataframe)

    # Ensure logger error is not called
    mock_logger_warning.assert_not_called()


def test_set_initial_return_values_warning(mocker, freqai_conf):
    """
    Simple test of set_initial_return_values that hits the warning
    associated with leaving a FreqAI bot offline so long that the
    exchange candles have no common date with the historic predictions
    """

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    # Setup
    pair = "BTC/USD"
    end_x = "2023-08-31"
    start_x_plus_1 = "2023-09-01"
    end_x_plus_5 = "2023-09-05"

    historic_data = {"date_pred": pd.date_range(end=end_x, periods=5), "value": range(1, 6)}
    new_data = {
        "date": pd.date_range(start=start_x_plus_1, end=end_x_plus_5),
        "value": range(6, 11),
    }

    freqai.dd.historic_predictions[pair] = pd.DataFrame(historic_data)

    new_pred_df = pd.DataFrame(new_data)
    dataframe = pd.DataFrame(new_data)

    # Action
    with patch("logging.Logger.warning") as mock_logger_warning:
        freqai.dd.set_initial_return_values(pair, new_pred_df, dataframe)

    # Assertions
    hist_pred_df = freqai.dd.historic_predictions[pair]
    model_return_df = freqai.dd.model_return_values[pair]

    assert hist_pred_df["date_pred"].iloc[-1] == pd.Timestamp(end_x_plus_5)
    assert "date_pred" in hist_pred_df.columns
    assert hist_pred_df.shape[0] == 10

    # compare values in model_return_df with hist_pred_df
    assert (
        model_return_df["value"].values == hist_pred_df.tail(len(dataframe))["value"].values
    ).all()
    assert model_return_df.shape[0] == len(dataframe)

    # Ensure logger error is not called
    mock_logger_warning.assert_called()


def test_set_initial_return_values_shifted_index(mocker, freqai_conf, caplog):
    """
    Test that date_pred is assigned positionally. The prediction dataframe is always 0-indexed,
    while a strategy dropping rows without reset_index hands over a shifted index - assigning
    by index would null out every date and lose all predictions.
    """

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    # Setup
    pair = "BTC/USD"
    historic_dates = pd.date_range(end="2023-08-31", periods=5, tz="UTC")
    new_dates = pd.date_range(start="2023-08-30", periods=5, tz="UTC")

    freqai.dd.historic_predictions[pair] = pd.DataFrame(
        {"date_pred": historic_dates, "value": range(1, 6)}
    )

    # predictions always come back 0-indexed from IFreqaiModel.predict()
    new_pred_df = pd.DataFrame({"value": range(6, 11)})
    # strategy dataframe carrying a non-0-based index
    dataframe = pd.DataFrame({"date": new_dates, "value": range(6, 11)}, index=range(301, 306))

    # Action
    freqai.dd.set_initial_return_values(pair, new_pred_df, dataframe)

    # Assertions
    model_return_df = freqai.dd.model_return_values[pair]

    assert model_return_df.shape[0] == len(dataframe)
    assert list(model_return_df["date_pred"]) == list(new_dates)
    # dates do line up - the "instance was offline" warning must not trigger
    assert not log_has_re("No common dates found between new predictions and historic.*", caplog)

    # values must find their way back to the strategy, aligned by date
    result = freqai.dd.attach_return_values_to_return_dataframe(pair, dataframe)

    assert len(result) == len(dataframe)
    assert not result["date_pred"].isnull().any()
    assert list(result["date"]) == list(result["date_pred"])


def test_set_initial_return_values_gap(mocker, freqai_conf, caplog):
    """
    Historic predictions can have gaps (pair left the whitelist, candle skipped, model not
    ready yet). Cutting the overlap by row count would then re-append already known dates
    and duplicate them - the overlap must be cut by date instead.
    """

    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.live = False
    freqai.dk = FreqaiDataKitchen(freqai_conf)

    pair = "BTC/USD"
    all_dates = pd.date_range(start="2023-09-01", periods=10, freq="5min", tz="UTC").astype(
        "datetime64[ms, UTC]"
    )
    # historic predictions covering the first 7 candles, with 3 of them missing in the middle
    hist_dates = all_dates[:7].delete([2, 3, 4])

    freqai.dd.historic_predictions[pair] = pd.DataFrame(
        {"date_pred": hist_dates, "value": range(1, len(hist_dates) + 1)}
    )

    new_pred_df = pd.DataFrame({"value": range(11, 21)})
    dataframe = pd.DataFrame({"date": all_dates, "value": range(11, 21)})

    freqai.dd.set_initial_return_values(pair, new_pred_df, dataframe)

    hist_pred_df = freqai.dd.historic_predictions[pair]

    # only the 3 candles after the last known date got appended - the gap is not re-filled
    assert not hist_pred_df["date_pred"].duplicated().any()
    assert hist_pred_df.shape[0] == len(hist_dates) + 3
    assert hist_pred_df["date_pred"].iloc[-1] == all_dates[-1]
    assert not log_has_re("No common dates found between new predictions and historic.*", caplog)

    # and the (gapped) predictions still merge back onto the strategy dataframe
    result = freqai.dd.attach_return_values_to_return_dataframe(pair, dataframe)
    assert len(result) == len(dataframe)


def test_append_model_predictions_keeps_date_dtype(mocker, freqai_conf):
    """
    Appending a new candle must not degrade date_pred to object dtype - merging the
    predictions back into the strategy dataframe fails on a non-datetime key.
    """
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    dk = freqai.dk
    dk.data["labels_mean"] = {"&-s_close": 0.5}
    dk.data["labels_std"] = {"&-s_close": 0.1}
    dk.data["extra_returns_per_train"] = {}
    dk.DI_values = [0.4]

    pair = "BTC/USD"
    # ms resolution, as delivered by the strategy dataframe
    dates = pd.date_range(start="2023-09-01", periods=5, freq="D", tz="UTC").astype(
        "datetime64[ms, UTC]"
    )

    freqai.dd.historic_predictions[pair] = pd.DataFrame(
        {
            "&-s_close": [1.0] * 4,
            "&-s_close_mean": [0.5] * 4,
            "&-s_close_std": [0.1] * 4,
            "do_predict": [1] * 4,
            "DI_values": [0.2] * 4,
            "high_price": [2.0] * 4,
            "low_price": [1.0] * 4,
            "close_price": [1.5] * 4,
            "date_pred": dates[:-1],
        }
    )

    dataframe = pd.DataFrame(
        {
            "date": dates,
            "high": range(1, 6),
            "low": range(1, 6),
            "close": range(1, 6),
            "&-s_close": [None] * 5,
        }
    )
    predictions = pd.DataFrame({"&-s_close": [2.0] * 5})

    freqai.dd.append_model_predictions(pair, predictions, np.array([1] * 5), dk, dataframe)

    assert freqai.dd.historic_predictions[pair]["date_pred"].dtype.kind == "M"
    assert freqai.dd.model_return_values[pair]["date_pred"].dtype.kind == "M"

    # the merge back into the strategy dataframe works, and lines up
    result = freqai.dd.attach_return_values_to_return_dataframe(pair, dataframe)

    assert len(result) == len(dataframe)
    assert list(result["date"]) == list(result["date_pred"])
    assert not result["&-s_close"].isnull().any()
    assert result["date_pred"].dtype == "datetime64[ms, UTC]"
    assert result["date"].dtype == "datetime64[ms, UTC]"


@pytest.mark.parametrize(
    "hist_idx,strat_idx,calls,expected_idx,expected_preds,expected_warning",
    [
        # the regular case - one new candle gets appended and filled with the prediction
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], 1, [0, 1, 2, 3, 4, 5], [1.0] * 5 + [2.0], None),
        # freqtrade only marks a candle as seen once the full strategy callback chain
        # succeeded - an exception raised after FreqAI ran re-analyzes the same candle, which
        # must overwrite its row instead of appending a duplicate date
        ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 3, [0, 1, 2, 3, 4, 5], [1.0] * 5 + [2.0], None),
        # candles the bot skipped (bot not running, slow iteration) get a zeroed placeholder
        # row each and a warning, so no hole is left to merge back as NaN unnoticed
        (
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5, 6, 7],
            1,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1.0] * 5 + [0.0, 0.0, 2.0],
            r"FreqAI did not predict on 2 candle\(s\) of BTC/USD.*",
        ),
        # an empty prediction frame simply starts from the current candle
        ([], [0, 1, 2], 1, [2], [2.0], None),
        # a strategy dataframe ending before the last stored prediction must not append
        # (duplicating a date) nor overwrite (moving a date backwards)
        ([0, 1, 2, 3], [0, 1], 1, [0, 1, 2, 3], [1.0] * 4, r".*predates the last stored.*"),
    ],
    ids=["new candle", "repeated candle", "skipped candles", "empty history", "stale candle"],
)
def test_append_model_predictions_candle_alignment(
    mocker,
    freqai_conf,
    caplog,
    hist_idx,
    strat_idx,
    calls,
    expected_idx,
    expected_preds,
    expected_warning,
):
    """
    historic_predictions must stay aligned with the strategy dataframe: one row per candle,
    no date twice. Duplicated dates break the merge back into the strategy dataframe, holes
    come back as NaN predictions and drift the two apart on the next restart.
    """
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    dk = freqai.dk
    dk.data["labels_mean"] = {"&-s_close": 0.5}
    dk.data["labels_std"] = {"&-s_close": 0.1}
    dk.data["extra_returns_per_train"] = {}
    dk.DI_values = [0.4]

    pair = "BTC/USD"
    dates = pd.date_range(start="2023-09-01", periods=8, freq="5min", tz="UTC").astype(
        "datetime64[ms, UTC]"
    )

    hist_len = len(hist_idx)
    freqai.dd.historic_predictions[pair] = pd.DataFrame(
        {
            "&-s_close": [1.0] * hist_len,
            "&-s_close_mean": [0.5] * hist_len,
            "&-s_close_std": [0.1] * hist_len,
            "do_predict": [1] * hist_len,
            "DI_values": [0.2] * hist_len,
            "high_price": [2.0] * hist_len,
            "low_price": [1.0] * hist_len,
            "close_price": [1.5] * hist_len,
            "date_pred": dates[hist_idx],
        }
    )

    dataframe = pd.DataFrame(
        {
            "date": dates[strat_idx],
            "high": range(1, len(strat_idx) + 1),
            "low": range(1, len(strat_idx) + 1),
            "close": range(1, len(strat_idx) + 1),
            "&-s_close": [None] * len(strat_idx),
        }
    )
    predictions = pd.DataFrame({"&-s_close": [2.0] * len(strat_idx)})

    for _ in range(calls):
        freqai.dd.append_model_predictions(
            pair, predictions, np.array([1] * len(strat_idx)), dk, dataframe
        )

    hist_pred_df = freqai.dd.historic_predictions[pair]

    assert list(hist_pred_df["date_pred"]) == list(dates[expected_idx])
    assert list(hist_pred_df["&-s_close"]) == expected_preds
    assert not hist_pred_df["date_pred"].duplicated().any()
    # a numeric 0 placeholder would degrade the date column to object dtype, which no
    # longer merges onto the strategy dataframe
    assert hist_pred_df["date_pred"].dtype.kind == "M"
    assert freqai.dd.model_return_values[pair]["date_pred"].dtype.kind == "M"
    if expected_warning:
        assert log_has_re(expected_warning, caplog)
    else:
        assert not [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and r.name == "freqtrade.freqai.data_drawer"
        ]


def test_attach_return_values_object_date_dtype(mocker, freqai_conf):
    """
    Predictions restored from disk (written by older versions) can carry an object dtype
    date column - it must still merge onto the strategy dataframe.
    """
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.dk = FreqaiDataKitchen(freqai_conf)

    pair = "BTC/USD"
    dates = pd.date_range(start="2023-09-01", periods=5, freq="D", tz="UTC")

    freqai.dd.model_return_values[pair] = pd.DataFrame(
        {
            "date_pred": dates.astype(object),
            "&-s_close": range(6, 11),
            "do_predict": [1] * 5,
        }
    )
    dataframe = pd.DataFrame(
        {
            "date": dates.astype("datetime64[ms, UTC]"),
            "close": range(1, 6),
            "&-s_close": [None] * 5,
        }
    )

    result = freqai.dd.attach_return_values_to_return_dataframe(pair, dataframe)

    assert len(result) == len(dataframe)
    assert list(result["&-s_close"]) == list(range(6, 11))
    assert not result["do_predict"].isnull().any()
    assert result["date_pred"].dtype == "datetime64[ms, UTC]"
    assert result["date"].dtype == "datetime64[ms, UTC]"


def test_attach_return_values_to_return_dataframe(mocker, freqai_conf):
    """
    Test that the prediction buffer is always 0-indexed, so attaching it
    to a strategy dataframe that carries a non-0-based index must align by candle date and
    must not silently produce NaN predictions or change the dataframe length.
    """
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    freqai = strategy.freqai
    freqai.dk = FreqaiDataKitchen(freqai_conf)

    pair = "BTC/USD"
    dates = pd.date_range(start="2023-09-01", periods=5, freq="D")

    # Prediction buffer: always 0-indexed (as produced by reset_index in the drawer)
    freqai.dd.model_return_values[pair] = pd.DataFrame(
        {"date_pred": dates, "&-s_close": range(6, 11), "do_predict": [1] * 5}
    )

    # Strategy dataframe with a shifted (non-0-based) index, as could arrive from a strategy
    # that drops/filters rows without reset_index.
    dataframe = pd.DataFrame(
        {"date": dates, "close": range(1, 6), "&-s_close": [None] * 5},
        index=range(301, 306),
    )

    result = freqai.dd.attach_return_values_to_return_dataframe(pair, dataframe)

    # length is preserved (would double under the old index-concat)
    assert len(result) == len(dataframe)
    # predictions are attached to the correct candles, not NaN
    assert not result["&-s_close"].isnull().any()
    assert not result["do_predict"].isnull().any()
    assert list(result["&-s_close"]) == list(range(6, 11))
    # and the original close prices line up with their dates.
    assert list(result["close"]) == list(range(1, 6))
