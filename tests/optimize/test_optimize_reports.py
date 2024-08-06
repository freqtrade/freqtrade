import re
from datetime import timedelta
from pathlib import Path
from shutil import copyfile

import joblib
import pandas as pd
import pytest

from freqtrade.configuration import TimeRange
from freqtrade.constants import BACKTEST_BREAKDOWNS, DATETIME_PRINT_FORMAT, LAST_BT_RESULT_FN
from freqtrade.data import history
from freqtrade.data.btanalysis import (
    get_latest_backtest_filename,
    load_backtest_data,
    load_backtest_stats,
)
from freqtrade.edge import PairInfo
from freqtrade.enums import ExitType
from freqtrade.optimize.optimize_reports import (
    generate_backtest_stats,
    generate_daily_stats,
    generate_edge_table,
    generate_pair_metrics,
    generate_periodic_breakdown_stats,
    generate_strategy_comparison,
    generate_trading_stats,
    show_sorted_pairlist,
    store_backtest_analysis_results,
    store_backtest_stats,
    text_table_bt_results,
    text_table_strategy,
)
from freqtrade.optimize.optimize_reports.bt_output import text_table_tags
from freqtrade.optimize.optimize_reports.optimize_reports import (
    _get_resample_from_period,
    calc_streak,
    generate_tag_metrics,
)
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.util import dt_ts
from freqtrade.util.datetime_helpers import dt_from_ts, dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY
from tests.data.test_history import _clean_test_file


def _backup_file(file: Path, copy_file: bool = False) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :param copy_file: keep file in place too.
    :return: None
    """
    file_swp = str(file) + ".swp"
    if file.is_file():
        file.rename(file_swp)

        if copy_file:
            copyfile(file_swp, file)


def test_text_table_bt_results(capsys):
    results = pd.DataFrame(
        {
            "pair": ["ETH/BTC", "ETH/BTC", "ETH/BTC"],
            "profit_ratio": [0.1, 0.2, -0.05],
            "profit_abs": [0.2, 0.4, -0.1],
            "trade_duration": [10, 30, 20],
        }
    )

    pair_results = generate_pair_metrics(
        ["ETH/BTC"], stake_currency="BTC", starting_balance=4, results=results
    )
    text_table_bt_results(pair_results, stake_currency="BTC", title="title")
    text = capsys.readouterr().out
    re.search(
        r".* Pair .* Trades .* Avg Profit % .* Tot Profit BTC .* Tot Profit % .* "
        r"Avg Duration .* Win  Draw  Loss  Win% .*",
        text,
    )
    re.search(
        r".* ETH/BTC .* 3 .* 8.33 .* 0.50000000 .* 12.50 .* 0:20:00 .* 2     0     1  66.7 .*",
        text,
    )
    re.search(
        r".* TOTAL .* 3 .* 8.33 .* 0.50000000 .* 12.50 .* 0:20:00 .* 2     0     1  66.7 .*", text
    )


def test_generate_backtest_stats(default_conf, testdatadir, tmp_path):
    default_conf.update({"strategy": CURRENT_TEST_STRATEGY})
    StrategyResolver.load_strategy(default_conf)

    results = {
        "DefStrat": {
            "results": pd.DataFrame(
                {
                    "pair": ["UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC"],
                    "profit_ratio": [0.003312, 0.010801, 0.013803, 0.002780],
                    "profit_abs": [0.000003, 0.000011, 0.000014, 0.000003],
                    "open_date": [
                        dt_utc(2017, 11, 14, 19, 32, 00),
                        dt_utc(2017, 11, 14, 21, 36, 00),
                        dt_utc(2017, 11, 14, 22, 12, 00),
                        dt_utc(2017, 11, 14, 22, 44, 00),
                    ],
                    "close_date": [
                        dt_utc(2017, 11, 14, 21, 35, 00),
                        dt_utc(2017, 11, 14, 22, 10, 00),
                        dt_utc(2017, 11, 14, 22, 43, 00),
                        dt_utc(2017, 11, 14, 22, 58, 00),
                    ],
                    "open_rate": [0.002543, 0.003003, 0.003089, 0.003214],
                    "close_rate": [0.002546, 0.003014, 0.003103, 0.003217],
                    "trade_duration": [123, 34, 31, 14],
                    "is_open": [False, False, False, True],
                    "is_short": [False, False, False, False],
                    "stake_amount": [0.01, 0.01, 0.01, 0.01],
                    "exit_reason": [
                        ExitType.ROI.value,
                        ExitType.STOP_LOSS.value,
                        ExitType.ROI.value,
                        ExitType.FORCE_EXIT.value,
                    ],
                }
            ),
            "config": default_conf,
            "locks": [],
            "final_balance": 1000.02,
            "rejected_signals": 20,
            "timedout_entry_orders": 0,
            "timedout_exit_orders": 0,
            "canceled_trade_entries": 0,
            "canceled_entry_orders": 0,
            "replaced_entry_orders": 0,
            "backtest_start_time": dt_ts() // 1000,
            "backtest_end_time": dt_ts() // 1000,
            "run_id": "123",
        }
    }
    timerange = TimeRange.parse_timerange("1510688220-1510700340")
    min_date = dt_from_ts(1510688220)
    max_date = dt_from_ts(1510700340)
    btdata = history.load_data(
        testdatadir, "1m", ["UNITTEST/BTC"], timerange=timerange, fill_up_missing=True
    )

    stats = generate_backtest_stats(btdata, results, min_date, max_date)
    assert isinstance(stats, dict)
    assert "strategy" in stats
    assert "DefStrat" in stats["strategy"]
    assert "strategy_comparison" in stats
    strat_stats = stats["strategy"]["DefStrat"]
    assert strat_stats["backtest_start"] == min_date.strftime(DATETIME_PRINT_FORMAT)
    assert strat_stats["backtest_end"] == max_date.strftime(DATETIME_PRINT_FORMAT)
    assert strat_stats["total_trades"] == len(results["DefStrat"]["results"])
    # Above sample had no losing trade
    assert strat_stats["max_drawdown_account"] == 0.0

    # Retry with losing trade
    results = {
        "DefStrat": {
            "results": pd.DataFrame(
                {
                    "pair": ["UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC"],
                    "profit_ratio": [0.003312, 0.010801, -0.013803, 0.002780],
                    "profit_abs": [0.000003, 0.000011, -0.000014, 0.000003],
                    "open_date": [
                        dt_utc(2017, 11, 14, 19, 32, 00),
                        dt_utc(2017, 11, 14, 21, 36, 00),
                        dt_utc(2017, 11, 14, 22, 12, 00),
                        dt_utc(2017, 11, 14, 22, 44, 00),
                    ],
                    "close_date": [
                        dt_utc(2017, 11, 14, 21, 35, 00),
                        dt_utc(2017, 11, 14, 22, 10, 00),
                        dt_utc(2017, 11, 14, 22, 43, 00),
                        dt_utc(2017, 11, 14, 22, 58, 00),
                    ],
                    "open_rate": [0.002543, 0.003003, 0.003089, 0.003214],
                    "close_rate": [0.002546, 0.003014, 0.0032903, 0.003217],
                    "trade_duration": [123, 34, 31, 14],
                    "is_open": [False, False, False, True],
                    "is_short": [False, False, False, False],
                    "stake_amount": [0.01, 0.01, 0.01, 0.01],
                    "exit_reason": [
                        ExitType.ROI.value,
                        ExitType.ROI.value,
                        ExitType.STOP_LOSS.value,
                        ExitType.FORCE_EXIT.value,
                    ],
                }
            ),
            "config": default_conf,
            "locks": [],
            "final_balance": 1000.02,
            "rejected_signals": 20,
            "timedout_entry_orders": 0,
            "timedout_exit_orders": 0,
            "canceled_trade_entries": 0,
            "canceled_entry_orders": 0,
            "replaced_entry_orders": 0,
            "backtest_start_time": dt_ts() // 1000,
            "backtest_end_time": dt_ts() // 1000,
            "run_id": "124",
        }
    }

    stats = generate_backtest_stats(btdata, results, min_date, max_date)
    assert isinstance(stats, dict)
    assert "strategy" in stats
    assert "DefStrat" in stats["strategy"]
    assert "strategy_comparison" in stats
    strat_stats = stats["strategy"]["DefStrat"]

    assert pytest.approx(strat_stats["max_drawdown_account"]) == 1.399999e-08
    assert strat_stats["drawdown_start"] == "2017-11-14 22:10:00"
    assert strat_stats["drawdown_end"] == "2017-11-14 22:43:00"
    assert strat_stats["drawdown_end_ts"] == 1510699380000
    assert strat_stats["drawdown_start_ts"] == 1510697400000
    assert strat_stats["pairlist"] == ["UNITTEST/BTC"]

    # Test storing stats
    filename = tmp_path / "btresult.json"
    filename_last = tmp_path / LAST_BT_RESULT_FN
    _backup_file(filename_last, copy_file=True)
    assert not filename.is_file()

    store_backtest_stats(filename, stats, "2022_01_01_15_05_13")

    # get real Filename (it's btresult-<date>.json)
    last_fn = get_latest_backtest_filename(filename_last.parent)
    assert re.match(r"btresult-.*\.json", last_fn)

    filename1 = tmp_path / last_fn
    assert filename1.is_file()
    content = filename1.read_text()
    assert "max_drawdown_account" in content
    assert "strategy" in content
    assert "pairlist" in content

    assert filename_last.is_file()

    _clean_test_file(filename_last)
    filename1.unlink()


def test_store_backtest_stats(testdatadir, mocker):
    dump_mock = mocker.patch("freqtrade.optimize.optimize_reports.bt_storage.file_dump_json")

    data = {"metadata": {}, "strategy": {}, "strategy_comparison": []}
    store_backtest_stats(testdatadir, data, "2022_01_01_15_05_13")

    assert dump_mock.call_count == 3
    assert isinstance(dump_mock.call_args_list[0][0][0], Path)
    assert str(dump_mock.call_args_list[0][0][0]).startswith(str(testdatadir / "backtest-result"))

    dump_mock.reset_mock()
    filename = testdatadir / "testresult.json"
    store_backtest_stats(filename, data, "2022_01_01_15_05_13")
    assert dump_mock.call_count == 3
    assert isinstance(dump_mock.call_args_list[0][0][0], Path)
    # result will be testdatadir / testresult-<timestamp>.json
    assert str(dump_mock.call_args_list[0][0][0]).startswith(str(testdatadir / "testresult"))


def test_store_backtest_stats_real(tmp_path):
    data = {"metadata": {}, "strategy": {}, "strategy_comparison": []}
    store_backtest_stats(tmp_path, data, "2022_01_01_15_05_13")

    assert (tmp_path / "backtest-result-2022_01_01_15_05_13.json").is_file()
    assert (tmp_path / "backtest-result-2022_01_01_15_05_13.meta.json").is_file()
    assert not (tmp_path / "backtest-result-2022_01_01_15_05_13_market_change.feather").is_file()
    assert (tmp_path / LAST_BT_RESULT_FN).is_file()
    fn = get_latest_backtest_filename(tmp_path)
    assert fn == "backtest-result-2022_01_01_15_05_13.json"

    store_backtest_stats(tmp_path, data, "2024_01_01_15_05_25", market_change_data=pd.DataFrame())
    assert (tmp_path / "backtest-result-2024_01_01_15_05_25.json").is_file()
    assert (tmp_path / "backtest-result-2024_01_01_15_05_25.meta.json").is_file()
    assert (tmp_path / "backtest-result-2024_01_01_15_05_25_market_change.feather").is_file()
    assert (tmp_path / LAST_BT_RESULT_FN).is_file()

    # Last file reference should be updated
    fn = get_latest_backtest_filename(tmp_path)
    assert fn == "backtest-result-2024_01_01_15_05_25.json"


def test_store_backtest_candles(testdatadir, mocker):
    dump_mock = mocker.patch("freqtrade.optimize.optimize_reports.bt_storage.file_dump_joblib")

    candle_dict = {"DefStrat": {"UNITTEST/BTC": pd.DataFrame()}}

    # mock directory exporting
    store_backtest_analysis_results(testdatadir, candle_dict, {}, {}, "2022_01_01_15_05_13")

    assert dump_mock.call_count == 3
    assert isinstance(dump_mock.call_args_list[0][0][0], Path)
    assert str(dump_mock.call_args_list[0][0][0]).endswith("_signals.pkl")
    assert str(dump_mock.call_args_list[1][0][0]).endswith("_rejected.pkl")
    assert str(dump_mock.call_args_list[2][0][0]).endswith("_exited.pkl")

    dump_mock.reset_mock()
    # mock file exporting
    filename = Path(testdatadir / "testresult")
    store_backtest_analysis_results(filename, candle_dict, {}, {}, "2022_01_01_15_05_13")
    assert dump_mock.call_count == 3
    assert isinstance(dump_mock.call_args_list[0][0][0], Path)
    # result will be testdatadir / testresult-<timestamp>_signals.pkl
    assert str(dump_mock.call_args_list[0][0][0]).endswith("_signals.pkl")
    assert str(dump_mock.call_args_list[1][0][0]).endswith("_rejected.pkl")
    assert str(dump_mock.call_args_list[2][0][0]).endswith("_exited.pkl")

    dump_mock.reset_mock()


def test_write_read_backtest_candles(tmp_path):
    candle_dict = {"DefStrat": {"UNITTEST/BTC": pd.DataFrame()}}

    # test directory exporting
    sample_date = "2022_01_01_15_05_13"
    store_backtest_analysis_results(tmp_path, candle_dict, {}, {}, sample_date)
    stored_file = tmp_path / f"backtest-result-{sample_date}_signals.pkl"
    with stored_file.open("rb") as scp:
        pickled_signal_candles = joblib.load(scp)

    assert pickled_signal_candles.keys() == candle_dict.keys()
    assert pickled_signal_candles["DefStrat"].keys() == pickled_signal_candles["DefStrat"].keys()
    assert pickled_signal_candles["DefStrat"]["UNITTEST/BTC"].equals(
        pickled_signal_candles["DefStrat"]["UNITTEST/BTC"]
    )

    _clean_test_file(stored_file)

    # test file exporting
    filename = tmp_path / "testresult"
    store_backtest_analysis_results(filename, candle_dict, {}, {}, sample_date)
    stored_file = tmp_path / f"testresult-{sample_date}_signals.pkl"
    with stored_file.open("rb") as scp:
        pickled_signal_candles = joblib.load(scp)

    assert pickled_signal_candles.keys() == candle_dict.keys()
    assert pickled_signal_candles["DefStrat"].keys() == pickled_signal_candles["DefStrat"].keys()
    assert pickled_signal_candles["DefStrat"]["UNITTEST/BTC"].equals(
        pickled_signal_candles["DefStrat"]["UNITTEST/BTC"]
    )

    _clean_test_file(stored_file)


def test_generate_pair_metrics():
    results = pd.DataFrame(
        {
            "pair": ["ETH/BTC", "ETH/BTC"],
            "profit_ratio": [0.1, 0.2],
            "profit_abs": [0.2, 0.4],
            "trade_duration": [10, 30],
            "wins": [2, 0],
            "draws": [0, 0],
            "losses": [0, 0],
        }
    )

    pair_results = generate_pair_metrics(
        ["ETH/BTC"], stake_currency="BTC", starting_balance=2, results=results
    )
    assert isinstance(pair_results, list)
    assert len(pair_results) == 2
    assert pair_results[-1]["key"] == "TOTAL"
    assert (
        pytest.approx(pair_results[-1]["profit_mean_pct"]) == pair_results[-1]["profit_mean"] * 100
    )
    assert pytest.approx(pair_results[-1]["profit_sum_pct"]) == pair_results[-1]["profit_sum"] * 100


def test_generate_daily_stats(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    res = generate_daily_stats(bt_data)
    assert isinstance(res, dict)
    assert round(res["backtest_best_day"], 4) == 0.1796
    assert round(res["backtest_worst_day"], 4) == -0.1468
    assert res["winning_days"] == 19
    assert res["draw_days"] == 0
    assert res["losing_days"] == 2

    # Select empty dataframe!
    res = generate_daily_stats(bt_data.loc[bt_data["open_date"] == "2000-01-01", :])
    assert isinstance(res, dict)
    assert round(res["backtest_best_day"], 4) == 0.0
    assert res["winning_days"] == 0
    assert res["draw_days"] == 0
    assert res["losing_days"] == 0


def test_generate_trading_stats(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    res = generate_trading_stats(bt_data)
    assert isinstance(res, dict)
    assert res["winner_holding_avg"] == timedelta(seconds=1440)
    assert res["loser_holding_avg"] == timedelta(days=1, seconds=21420)
    assert "wins" in res
    assert "losses" in res
    assert "draws" in res

    # Select empty dataframe!
    res = generate_trading_stats(bt_data.loc[bt_data["open_date"] == "2000-01-01", :])
    assert res["wins"] == 0
    assert res["losses"] == 0


def test_calc_streak(testdatadir):
    df = pd.DataFrame(
        {
            "profit_ratio": [0.05, -0.02, -0.03, -0.05, 0.01, 0.02, 0.03, 0.04, -0.02, -0.03],
        }
    )
    # 4 consecutive wins, 3 consecutive losses
    res = calc_streak(df)
    assert res == (4, 3)
    assert isinstance(res[0], int)
    assert isinstance(res[1], int)

    # invert situation
    df1 = df.copy()
    df1["profit_ratio"] = df1["profit_ratio"] * -1
    assert calc_streak(df1) == (3, 4)

    df_empty = pd.DataFrame(
        {
            "profit_ratio": [],
        }
    )
    assert df_empty.empty
    assert calc_streak(df_empty) == (0, 0)

    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    assert calc_streak(bt_data) == (7, 18)


def test_text_table_exit_reason(capsys):
    results = pd.DataFrame(
        {
            "pair": ["ETH/BTC", "ETH/BTC", "ETH/BTC"],
            "profit_ratio": [0.1, 0.2, -0.1],
            "profit_abs": [0.2, 0.4, -0.2],
            "trade_duration": [10, 30, 10],
            "wins": [2, 0, 0],
            "draws": [0, 0, 0],
            "losses": [0, 0, 1],
            "exit_reason": [ExitType.ROI.value, ExitType.ROI.value, ExitType.STOP_LOSS.value],
        }
    )

    exit_reason_stats = generate_tag_metrics(
        "exit_reason", starting_balance=22, results=results, skip_nan=False
    )
    text_table_tags("exit_tag", exit_reason_stats, "BTC")
    text = capsys.readouterr().out

    assert re.search(
        r".* Exit Reason .* Exits .* Avg Profit % .* Tot Profit BTC .* Tot Profit % .* "
        r"Avg Duration .* Win  Draw  Loss  Win% .*",
        text,
    )
    assert re.search(
        r".* roi .* 2 .* 15.0 .* 0.60000000 .* 2.73 .* 0:20:00 .* 2     0     0   100 .*",
        text,
    )
    assert re.search(
        r".* stop_loss .* 1 .* -10.0 .* -0.20000000 .* -0.91 .* 0:10:00 .* 0     0     1     0 .*",
        text,
    )
    assert re.search(
        r".* TOTAL .* 3 .* 6.67 .* 0.40000000 .* 1.82 .* 0:17:00 .* 2     0     1  66.7 .*", text
    )


def test_generate_sell_reason_stats():
    results = pd.DataFrame(
        {
            "pair": ["ETH/BTC", "ETH/BTC", "ETH/BTC"],
            "profit_ratio": [0.1, 0.2, -0.1],
            "profit_abs": [0.2, 0.4, -0.2],
            "trade_duration": [10, 30, 10],
            "wins": [2, 0, 0],
            "draws": [0, 0, 0],
            "losses": [0, 0, 1],
            "exit_reason": [ExitType.ROI.value, ExitType.ROI.value, ExitType.STOP_LOSS.value],
        }
    )

    exit_reason_stats = generate_tag_metrics(
        "exit_reason", starting_balance=22, results=results, skip_nan=False
    )
    roi_result = exit_reason_stats[0]
    assert roi_result["key"] == "roi"
    assert roi_result["trades"] == 2
    assert pytest.approx(roi_result["profit_mean"]) == 0.15
    assert roi_result["profit_mean_pct"] == round(roi_result["profit_mean"] * 100, 2)
    assert pytest.approx(roi_result["profit_mean"]) == 0.15
    assert roi_result["profit_mean_pct"] == round(roi_result["profit_mean"] * 100, 2)

    stop_result = exit_reason_stats[1]

    assert stop_result["key"] == "stop_loss"
    assert stop_result["trades"] == 1
    assert pytest.approx(stop_result["profit_mean"]) == -0.1
    assert stop_result["profit_mean_pct"] == round(stop_result["profit_mean"] * 100, 2)
    assert pytest.approx(stop_result["profit_mean"]) == -0.1
    assert stop_result["profit_mean_pct"] == round(stop_result["profit_mean"] * 100, 2)


def test_text_table_strategy(testdatadir, capsys):
    filename = testdatadir / "backtest_results/backtest-result_multistrat.json"
    bt_res_data = load_backtest_stats(filename)

    bt_res_data_comparison = bt_res_data.pop("strategy_comparison")

    strategy_results = generate_strategy_comparison(bt_stats=bt_res_data["strategy"])
    assert strategy_results == bt_res_data_comparison
    text_table_strategy(strategy_results, "BTC", "STRATEGY SUMMARY")

    captured = capsys.readouterr()
    text = captured.out
    assert re.search(
        r".* Strategy .* Trades .* Avg Profit % .* Tot Profit BTC .* Tot Profit % .* "
        r"Avg Duration .* Win  Draw  Loss  Win% .* Drawdown .*",
        text,
    )
    assert re.search(
        r".*StrategyTestV2 .* 179 .* 0.08 .* 0.02608550 .* "
        r"260.85 .* 3:40:00 .* 170     0     9  95.0 .* 0.00308222 BTC  8.67%.*",
        text,
    )
    assert re.search(
        r".*TestStrategy .* 179 .* 0.08 .* 0.02608550 .* "
        r"260.85 .* 3:40:00 .* 170     0     9  95.0 .* 0.00308222 BTC  8.67%.*",
        text,
    )


def test_generate_edge_table(capsys):
    results = {}
    results["ETH/BTC"] = PairInfo(-0.01, 0.60, 2, 1, 3, 10, 60)
    generate_edge_table(results)
    text = capsys.readouterr().out
    assert re.search(r".* ETH/BTC .*", text)
    assert re.search(r".* Risk Reward Ratio .* Required Risk Reward .* Expectancy .*", text)


def test_generate_periodic_breakdown_stats(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename).to_dict(orient="records")

    res = generate_periodic_breakdown_stats(bt_data, "day")
    assert isinstance(res, list)
    assert len(res) == 21
    day = res[0]
    assert "date" in day
    assert "draws" in day
    assert "loses" in day
    assert "wins" in day
    assert "profit_abs" in day

    # Select empty dataframe!
    res = generate_periodic_breakdown_stats([], "day")
    assert res == []


def test__get_resample_from_period():
    assert _get_resample_from_period("day") == "1d"
    assert _get_resample_from_period("week") == "1W-MON"
    assert _get_resample_from_period("month") == "1ME"
    with pytest.raises(ValueError, match=r"Period noooo is not supported."):
        _get_resample_from_period("noooo")

    for period in BACKTEST_BREAKDOWNS:
        assert isinstance(_get_resample_from_period(period), str)


def test_show_sorted_pairlist(testdatadir, default_conf, capsys):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_stats(filename)
    default_conf["backtest_show_pair_list"] = True

    show_sorted_pairlist(default_conf, bt_data)

    out, _err = capsys.readouterr()
    assert "Pairs for Strategy StrategyTestV3: \n[" in out
    assert "TOTAL" not in out
    assert '"ETH/BTC",  // ' in out
