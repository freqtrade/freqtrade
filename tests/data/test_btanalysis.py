from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest
from pandas import DataFrame, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.data.btanalysis import (
    BT_DATA_COLUMNS,
    extract_trades_of_period,
    get_backtest_market_change,
    get_backtest_wallet_change,
    get_latest_backtest_filename,
    get_latest_hyperopt_file,
    load_backtest_data,
    load_backtest_metadata,
    load_file_from_zip,
    load_trades,
    load_trades_from_db,
)
from freqtrade.data.history import load_pair_history
from freqtrade.exceptions import OperationalException
from freqtrade.util import dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY, create_mock_trades
from tests.conftest_trades import MOCK_TRADE_COUNT


def test_get_latest_backtest_filename(testdatadir, mocker):
    with pytest.raises(ValueError, match=r"Directory .* does not exist\."):
        get_latest_backtest_filename(testdatadir / "does_not_exist")

    with pytest.raises(ValueError, match=r"Directory .* does not seem to contain .*"):
        get_latest_backtest_filename(testdatadir)

    testdir_bt = testdatadir / "backtest_results"
    res = get_latest_backtest_filename(testdir_bt)
    assert res == "backtest-result.json"

    res = get_latest_backtest_filename(str(testdir_bt))
    assert res == "backtest-result.json"

    mocker.patch("freqtrade.data.btanalysis.bt_fileutils.json_load", return_value={})

    with pytest.raises(ValueError, match=r"Invalid '.last_result.json' format."):
        get_latest_backtest_filename(testdir_bt)


def test_get_latest_hyperopt_file(testdatadir):
    res = get_latest_hyperopt_file(testdatadir / "does_not_exist", "testfile.pickle")
    assert res == testdatadir / "does_not_exist/testfile.pickle"

    res = get_latest_hyperopt_file(testdatadir.parent)
    assert res == testdatadir.parent / "hyperopt_results.pickle"

    res = get_latest_hyperopt_file(str(testdatadir.parent))
    assert res == testdatadir.parent / "hyperopt_results.pickle"

    # Test with absolute path
    with pytest.raises(
        OperationalException,
        match=r"--hyperopt-filename expects only the filename, not an absolute path\.",
    ):
        get_latest_hyperopt_file(str(testdatadir.parent), str(testdatadir.parent))


def test_load_backtest_metadata(mocker, testdatadir):
    res = load_backtest_metadata(testdatadir / "nonexistent.file.json")
    assert res == {}

    mocker.patch("freqtrade.data.btanalysis.bt_fileutils.get_backtest_metadata_filename")
    mocker.patch("freqtrade.data.btanalysis.bt_fileutils.json_load", side_effect=Exception())
    with pytest.raises(
        OperationalException, match=r"Unexpected error.*loading backtest metadata\."
    ):
        load_backtest_metadata(testdatadir / "nonexistent.file.json")


def test_load_backtest_data_old_format(testdatadir, mocker):
    filename = testdatadir / "backtest-result_test222.json"
    mocker.patch("freqtrade.data.btanalysis.bt_fileutils.load_backtest_stats", return_value=[])

    with pytest.raises(
        OperationalException,
        match=r"Backtest-results with only trades data are no longer supported.",
    ):
        load_backtest_data(filename)


def test_load_backtest_data_new_format(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    assert isinstance(bt_data, DataFrame)
    assert set(bt_data.columns) == set(BT_DATA_COLUMNS)
    assert len(bt_data) == 179

    # Test loading from string (must yield same result)
    bt_data2 = load_backtest_data(str(filename))
    assert bt_data.equals(bt_data2)

    # Test loading from folder (must yield same result)
    bt_data3 = load_backtest_data(testdatadir / "backtest_results")
    assert bt_data.equals(bt_data3)

    with pytest.raises(ValueError, match=r"File .* does not exist\."):
        load_backtest_data("filename" + "nofile")

    with pytest.raises(ValueError, match=r"Unknown dataformat."):
        load_backtest_data(testdatadir / "backtest_results" / LAST_BT_RESULT_FN)


def test_load_backtest_data_multi(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result_multistrat.json"
    for strategy in ("StrategyTestV2", "TestStrategy"):
        bt_data = load_backtest_data(filename, strategy=strategy)
        assert isinstance(bt_data, DataFrame)
        assert set(bt_data.columns) == set(BT_DATA_COLUMNS)
        assert len(bt_data) == 179

        # Test loading from string (must yield same result)
        bt_data2 = load_backtest_data(str(filename), strategy=strategy)
        assert bt_data.equals(bt_data2)

    with pytest.raises(ValueError, match=r"Strategy XYZ not available in the backtest result\."):
        load_backtest_data(filename, strategy="XYZ")

    with pytest.raises(ValueError, match=r"Detected backtest result with more than one strategy.*"):
        load_backtest_data(filename)


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_load_trades_from_db(default_conf, fee, is_short, mocker):
    create_mock_trades(fee, is_short)
    # remove init so it does not init again
    init_mock = mocker.patch("freqtrade.data.btanalysis.bt_fileutils.init_db", MagicMock())

    trades = load_trades_from_db(db_url=default_conf["db_url"])
    assert init_mock.call_count == 1
    assert len(trades) == MOCK_TRADE_COUNT
    assert isinstance(trades, DataFrame)
    assert "pair" in trades.columns
    assert "open_date" in trades.columns
    assert "profit_ratio" in trades.columns

    for col in BT_DATA_COLUMNS:
        if col not in ["index", "open_at_end"]:
            assert col in trades.columns
    trades = load_trades_from_db(db_url=default_conf["db_url"], strategy=CURRENT_TEST_STRATEGY)
    assert len(trades) == 4
    trades = load_trades_from_db(db_url=default_conf["db_url"], strategy="NoneStrategy")
    assert len(trades) == 0


def test_extract_trades_of_period(testdatadir):
    pair = "UNITTEST/BTC"
    # 2018-11-14 06:07:00
    timerange = TimeRange("date", None, 1510639620, 0)

    data = load_pair_history(pair=pair, timeframe="1m", datadir=testdatadir, timerange=timerange)

    trades = DataFrame(
        {
            "pair": [pair, pair, pair, pair],
            "profit_ratio": [0.0, 0.1, -0.2, -0.5],
            "profit_abs": [0.0, 1, -2, -5],
            "open_date": to_datetime(
                [
                    datetime(2017, 11, 13, 15, 40, 0, tzinfo=UTC),
                    datetime(2017, 11, 14, 9, 41, 0, tzinfo=UTC),
                    datetime(2017, 11, 14, 14, 20, 0, tzinfo=UTC),
                    datetime(2017, 11, 15, 3, 40, 0, tzinfo=UTC),
                ],
                utc=True,
            ),
            "close_date": to_datetime(
                [
                    datetime(2017, 11, 13, 16, 40, 0, tzinfo=UTC),
                    datetime(2017, 11, 14, 10, 41, 0, tzinfo=UTC),
                    datetime(2017, 11, 14, 15, 25, 0, tzinfo=UTC),
                    datetime(2017, 11, 15, 3, 55, 0, tzinfo=UTC),
                ],
                utc=True,
            ),
        }
    )
    trades1 = extract_trades_of_period(data, trades)
    # First and last trade are dropped as they are out of range
    assert len(trades1) == 2
    assert trades1.iloc[0].open_date == datetime(2017, 11, 14, 9, 41, 0, tzinfo=UTC)
    assert trades1.iloc[0].close_date == datetime(2017, 11, 14, 10, 41, 0, tzinfo=UTC)
    assert trades1.iloc[-1].open_date == datetime(2017, 11, 14, 14, 20, 0, tzinfo=UTC)
    assert trades1.iloc[-1].close_date == datetime(2017, 11, 14, 15, 25, 0, tzinfo=UTC)


def test_load_trades(default_conf, mocker):
    db_mock = mocker.patch(
        "freqtrade.data.btanalysis.bt_fileutils.load_trades_from_db", MagicMock()
    )
    bt_mock = mocker.patch("freqtrade.data.btanalysis.bt_fileutils.load_backtest_data", MagicMock())

    load_trades(
        "DB",
        db_url=default_conf.get("db_url"),
        exportfilename=default_conf.get("exportfilename"),
        no_trades=False,
        strategy=CURRENT_TEST_STRATEGY,
    )

    assert db_mock.call_count == 1
    assert bt_mock.call_count == 0

    db_mock.reset_mock()
    bt_mock.reset_mock()
    default_conf["exportfilename"] = Path("testfile.json")
    load_trades(
        "file",
        db_url=default_conf.get("db_url"),
        exportfilename=default_conf.get("exportfilename"),
    )

    assert db_mock.call_count == 0
    assert bt_mock.call_count == 1

    db_mock.reset_mock()
    bt_mock.reset_mock()
    default_conf["exportfilename"] = "testfile.json"
    load_trades(
        "file",
        db_url=default_conf.get("db_url"),
        exportfilename=default_conf.get("exportfilename"),
        no_trades=True,
    )

    assert db_mock.call_count == 0
    assert bt_mock.call_count == 0


def test_load_file_from_zip(tmp_path):
    with pytest.raises(ValueError, match=r"Zip file .* not found\."):
        load_file_from_zip(tmp_path / "test.zip", "testfile.txt")

    (tmp_path / "testfile.zip").touch()
    with pytest.raises(ValueError, match=r"Bad zip file.*"):
        load_file_from_zip(tmp_path / "testfile.zip", "testfile.txt")

    zip_file = tmp_path / "testfile2.zip"
    with ZipFile(zip_file, "w") as zipf:
        zipf.writestr("testfile.txt", "testfile content")

    content = load_file_from_zip(zip_file, "testfile.txt")
    assert content.decode("utf-8") == "testfile content"

    with pytest.raises(ValueError, match=r"File .* not found in zip.*"):
        load_file_from_zip(zip_file, "testfile55.txt")


def test_get_backtest_market_change(tmp_path):
    df = DataFrame(
        {
            "date": [dt_utc(2020, 1, 1), dt_utc(2020, 1, 2)],
            "price": [100.0, 110.0],
        }
    )
    feather_file = tmp_path / "backtest-result_market_change.feather"
    df.to_feather(feather_file)

    direct_df = get_backtest_market_change(feather_file)
    assert isinstance(direct_df, DataFrame)
    assert "__date_ts" in direct_df.columns
    assert direct_df.loc[0, "__date_ts"] == int(df.loc[0, "date"].timestamp() * 1000)

    no_ts_df = get_backtest_market_change(feather_file, include_ts=False)
    assert "__date_ts" not in no_ts_df.columns

    zip_file = tmp_path / "backtest-result.zip"
    with ZipFile(zip_file, "w") as zipf:
        zipf.write(feather_file, arcname=f"{zip_file.stem}_market_change.feather")

    zipped_df = get_backtest_market_change(zip_file)
    assert isinstance(zipped_df, DataFrame)
    assert zipped_df.loc[0, "__date_ts"] == int(df.loc[0, "date"].timestamp() * 1000)
    assert list(zipped_df["price"]) == [100.0, 110.0]


def test_get_backtest_wallet_change(tmp_path):
    df = DataFrame(
        {
            "date": [dt_utc(2020, 1, 1), dt_utc(2020, 1, 2)],
            "balance": [1.0, 1.1],
            "rate": [1.0, 1.1],
        }
    )
    wallet_feather = tmp_path / "backtest-result_TestStrategy_wallet.feather"
    df.to_feather(wallet_feather)

    zip_file = tmp_path / "backtest-result.zip"
    with ZipFile(zip_file, "w") as zipf:
        zipf.write(wallet_feather, arcname=wallet_feather.name)

    wallet_df = get_backtest_wallet_change(zip_file, "TestStrategy")
    assert isinstance(wallet_df, DataFrame)
    assert "__date_ts" in wallet_df.columns
    assert wallet_df.loc[0, "__date_ts"] == int(df.loc[0, "date"].timestamp() * 1000)
    assert list(wallet_df["balance"]) == [1.0, 1.1]

    assert get_backtest_wallet_change(tmp_path / "backtest-result.feather", "TestStrategy") is None
    assert get_backtest_wallet_change(zip_file, "UnknownStrategy") is None
