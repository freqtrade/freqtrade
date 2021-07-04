from math import isclose
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from arrow import Arrow
from pandas import DataFrame, DateOffset, Timestamp, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.data.btanalysis import (BT_DATA_COLUMNS, BT_DATA_COLUMNS_MID, BT_DATA_COLUMNS_OLD,
                                       analyze_trade_parallelism, calculate_csum,
                                       calculate_market_change, calculate_max_drawdown,
                                       combine_dataframes_with_mean, create_cum_profit,
                                       extract_trades_of_period, get_latest_backtest_filename,
                                       get_latest_hyperopt_file, load_backtest_data, load_trades,
                                       load_trades_from_db)
from freqtrade.data.history import load_data, load_pair_history
from tests.conftest import create_mock_trades
from tests.conftest_trades import MOCK_TRADE_COUNT


def test_get_latest_backtest_filename(testdatadir, mocker):
    with pytest.raises(ValueError, match=r"Directory .* does not exist\."):
        get_latest_backtest_filename(testdatadir / 'does_not_exist')

    with pytest.raises(ValueError,
                       match=r"Directory .* does not seem to contain .*"):
        get_latest_backtest_filename(testdatadir.parent)

    res = get_latest_backtest_filename(testdatadir)
    assert res == 'backtest-result_new.json'

    res = get_latest_backtest_filename(str(testdatadir))
    assert res == 'backtest-result_new.json'

    mocker.patch("freqtrade.data.btanalysis.json_load", return_value={})

    with pytest.raises(ValueError, match=r"Invalid '.last_result.json' format."):
        get_latest_backtest_filename(testdatadir)


def test_get_latest_hyperopt_file(testdatadir, mocker):
    res = get_latest_hyperopt_file(testdatadir / 'does_not_exist', 'testfile.pickle')
    assert res == testdatadir / 'does_not_exist/testfile.pickle'

    res = get_latest_hyperopt_file(testdatadir.parent)
    assert res == testdatadir.parent / "hyperopt_results.pickle"

    res = get_latest_hyperopt_file(str(testdatadir.parent))
    assert res == testdatadir.parent / "hyperopt_results.pickle"


def test_load_backtest_data_old_format(testdatadir):

    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    assert isinstance(bt_data, DataFrame)
    assert list(bt_data.columns) == BT_DATA_COLUMNS_OLD + ['profit_abs', 'profit_ratio']
    assert len(bt_data) == 179

    # Test loading from string (must yield same result)
    bt_data2 = load_backtest_data(str(filename))
    assert bt_data.equals(bt_data2)

    with pytest.raises(ValueError, match=r"File .* does not exist\."):
        load_backtest_data(str("filename") + "nofile")


def test_load_backtest_data_new_format(testdatadir):

    filename = testdatadir / "backtest-result_new.json"
    bt_data = load_backtest_data(filename)
    assert isinstance(bt_data, DataFrame)
    assert set(bt_data.columns) == set(BT_DATA_COLUMNS_MID)
    assert len(bt_data) == 179

    # Test loading from string (must yield same result)
    bt_data2 = load_backtest_data(str(filename))
    assert bt_data.equals(bt_data2)

    # Test loading from folder (must yield same result)
    bt_data3 = load_backtest_data(testdatadir)
    assert bt_data.equals(bt_data3)

    with pytest.raises(ValueError, match=r"File .* does not exist\."):
        load_backtest_data(str("filename") + "nofile")

    with pytest.raises(ValueError, match=r"Unknown dataformat."):
        load_backtest_data(testdatadir / LAST_BT_RESULT_FN)


def test_load_backtest_data_multi(testdatadir):

    filename = testdatadir / "backtest-result_multistrat.json"
    for strategy in ('DefaultStrategy', 'TestStrategy'):
        bt_data = load_backtest_data(filename, strategy=strategy)
        assert isinstance(bt_data, DataFrame)
        assert set(bt_data.columns) == set(BT_DATA_COLUMNS_MID)
        assert len(bt_data) == 179

        # Test loading from string (must yield same result)
        bt_data2 = load_backtest_data(str(filename), strategy=strategy)
        assert bt_data.equals(bt_data2)

    with pytest.raises(ValueError, match=r"Strategy XYZ not available in the backtest result\."):
        load_backtest_data(filename, strategy='XYZ')

    with pytest.raises(ValueError, match=r"Detected backtest result with more than one strategy.*"):
        load_backtest_data(filename)


@pytest.mark.usefixtures("init_persistence")
def test_load_trades_from_db(default_conf, fee, mocker):

    create_mock_trades(fee)
    # remove init so it does not init again
    init_mock = mocker.patch('freqtrade.data.btanalysis.init_db', MagicMock())

    trades = load_trades_from_db(db_url=default_conf['db_url'])
    assert init_mock.call_count == 1
    assert len(trades) == MOCK_TRADE_COUNT
    assert isinstance(trades, DataFrame)
    assert "pair" in trades.columns
    assert "open_date" in trades.columns
    assert "profit_ratio" in trades.columns

    for col in BT_DATA_COLUMNS:
        if col not in ['index', 'open_at_end']:
            assert col in trades.columns
    trades = load_trades_from_db(db_url=default_conf['db_url'], strategy='DefaultStrategy')
    assert len(trades) == 4
    trades = load_trades_from_db(db_url=default_conf['db_url'], strategy='NoneStrategy')
    assert len(trades) == 0


def test_extract_trades_of_period(testdatadir):
    pair = "UNITTEST/BTC"
    # 2018-11-14 06:07:00
    timerange = TimeRange('date', None, 1510639620, 0)

    data = load_pair_history(pair=pair, timeframe='1m',
                             datadir=testdatadir, timerange=timerange)

    trades = DataFrame(
        {'pair': [pair, pair, pair, pair],
         'profit_ratio': [0.0, 0.1, -0.2, -0.5],
         'profit_abs': [0.0, 1, -2, -5],
         'open_date': to_datetime([Arrow(2017, 11, 13, 15, 40, 0).datetime,
                                   Arrow(2017, 11, 14, 9, 41, 0).datetime,
                                   Arrow(2017, 11, 14, 14, 20, 0).datetime,
                                   Arrow(2017, 11, 15, 3, 40, 0).datetime,
                                   ], utc=True
                                  ),
         'close_date': to_datetime([Arrow(2017, 11, 13, 16, 40, 0).datetime,
                                    Arrow(2017, 11, 14, 10, 41, 0).datetime,
                                    Arrow(2017, 11, 14, 15, 25, 0).datetime,
                                    Arrow(2017, 11, 15, 3, 55, 0).datetime,
                                    ], utc=True)
         })
    trades1 = extract_trades_of_period(data, trades)
    # First and last trade are dropped as they are out of range
    assert len(trades1) == 2
    assert trades1.iloc[0].open_date == Arrow(2017, 11, 14, 9, 41, 0).datetime
    assert trades1.iloc[0].close_date == Arrow(2017, 11, 14, 10, 41, 0).datetime
    assert trades1.iloc[-1].open_date == Arrow(2017, 11, 14, 14, 20, 0).datetime
    assert trades1.iloc[-1].close_date == Arrow(2017, 11, 14, 15, 25, 0).datetime


def test_analyze_trade_parallelism(default_conf, mocker, testdatadir):
    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)

    res = analyze_trade_parallelism(bt_data, "5m")
    assert isinstance(res, DataFrame)
    assert 'open_trades' in res.columns
    assert res['open_trades'].max() == 3
    assert res['open_trades'].min() == 0


def test_load_trades(default_conf, mocker):
    db_mock = mocker.patch("freqtrade.data.btanalysis.load_trades_from_db", MagicMock())
    bt_mock = mocker.patch("freqtrade.data.btanalysis.load_backtest_data", MagicMock())

    load_trades("DB",
                db_url=default_conf.get('db_url'),
                exportfilename=default_conf.get('exportfilename'),
                no_trades=False,
                strategy="DefaultStrategy",
                )

    assert db_mock.call_count == 1
    assert bt_mock.call_count == 0

    db_mock.reset_mock()
    bt_mock.reset_mock()
    default_conf['exportfilename'] = Path("testfile.json")
    load_trades("file",
                db_url=default_conf.get('db_url'),
                exportfilename=default_conf.get('exportfilename'),
                )

    assert db_mock.call_count == 0
    assert bt_mock.call_count == 1

    db_mock.reset_mock()
    bt_mock.reset_mock()
    default_conf['exportfilename'] = "testfile.json"
    load_trades("file",
                db_url=default_conf.get('db_url'),
                exportfilename=default_conf.get('exportfilename'),
                no_trades=True
                )

    assert db_mock.call_count == 0
    assert bt_mock.call_count == 0


def test_calculate_market_change(testdatadir):
    pairs = ["ETH/BTC", "ADA/BTC"]
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe='5m')
    result = calculate_market_change(data)
    assert isinstance(result, float)
    assert pytest.approx(result) == 0.00955514


def test_combine_dataframes_with_mean(testdatadir):
    pairs = ["ETH/BTC", "ADA/BTC"]
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe='5m')
    df = combine_dataframes_with_mean(data)
    assert isinstance(df, DataFrame)
    assert "ETH/BTC" in df.columns
    assert "ADA/BTC" in df.columns
    assert "mean" in df.columns


def test_create_cum_profit(testdatadir):
    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange("20180110-20180112")

    df = load_pair_history(pair="TRX/BTC", timeframe='5m',
                           datadir=testdatadir, timerange=timerange)

    cum_profits = create_cum_profit(df.set_index('date'),
                                    bt_data[bt_data["pair"] == 'TRX/BTC'],
                                    "cum_profits", timeframe="5m")
    assert "cum_profits" in cum_profits.columns
    assert cum_profits.iloc[0]['cum_profits'] == 0
    assert isclose(cum_profits.iloc[-1]['cum_profits'], 8.723007518796964e-06)


def test_create_cum_profit1(testdatadir):
    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    # Move close-time to "off" the candle, to make sure the logic still works
    bt_data.loc[:, 'close_date'] = bt_data.loc[:, 'close_date'] + DateOffset(seconds=20)
    timerange = TimeRange.parse_timerange("20180110-20180112")

    df = load_pair_history(pair="TRX/BTC", timeframe='5m',
                           datadir=testdatadir, timerange=timerange)

    cum_profits = create_cum_profit(df.set_index('date'),
                                    bt_data[bt_data["pair"] == 'TRX/BTC'],
                                    "cum_profits", timeframe="5m")
    assert "cum_profits" in cum_profits.columns
    assert cum_profits.iloc[0]['cum_profits'] == 0
    assert isclose(cum_profits.iloc[-1]['cum_profits'], 8.723007518796964e-06)

    with pytest.raises(ValueError, match='Trade dataframe empty.'):
        create_cum_profit(df.set_index('date'), bt_data[bt_data["pair"] == 'NOTAPAIR'],
                          "cum_profits", timeframe="5m")


def test_calculate_max_drawdown(testdatadir):
    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    drawdown, hdate, lowdate, hval, lval = calculate_max_drawdown(bt_data)
    assert isinstance(drawdown, float)
    assert pytest.approx(drawdown) == 0.21142322
    assert isinstance(hdate, Timestamp)
    assert isinstance(lowdate, Timestamp)
    assert isinstance(hval, float)
    assert isinstance(lval, float)
    assert hdate == Timestamp('2018-01-24 14:25:00', tz='UTC')
    assert lowdate == Timestamp('2018-01-30 04:45:00', tz='UTC')
    with pytest.raises(ValueError, match='Trade dataframe empty.'):
        drawdown, hdate, lowdate, hval, lval = calculate_max_drawdown(DataFrame())


def test_calculate_csum(testdatadir):
    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    csum_min, csum_max = calculate_csum(bt_data)

    assert isinstance(csum_min, float)
    assert isinstance(csum_max, float)
    assert csum_min < 0.01
    assert csum_max > 0.02
    csum_min1, csum_max1 = calculate_csum(bt_data, 5)

    assert csum_min1 == csum_min + 5
    assert csum_max1 == csum_max + 5

    with pytest.raises(ValueError, match='Trade dataframe empty.'):
        csum_min, csum_max = calculate_csum(DataFrame())


def test_calculate_max_drawdown2():
    values = [0.011580, 0.010048, 0.011340, 0.012161, 0.010416, 0.010009, 0.020024,
              -0.024662, -0.022350, 0.020496, -0.029859, -0.030511, 0.010041, 0.010872,
              -0.025782, 0.010400, 0.012374, 0.012467, 0.114741, 0.010303, 0.010088,
              -0.033961, 0.010680, 0.010886, -0.029274, 0.011178, 0.010693, 0.010711]

    dates = [Arrow(2020, 1, 1).shift(days=i) for i in range(len(values))]
    df = DataFrame(zip(values, dates), columns=['profit', 'open_date'])
    # sort by profit and reset index
    df = df.sort_values('profit').reset_index(drop=True)
    df1 = df.copy()
    drawdown, hdate, ldate, hval, lval = calculate_max_drawdown(
        df, date_col='open_date', value_col='profit')
    # Ensure df has not been altered.
    assert df.equals(df1)

    assert isinstance(drawdown, float)
    # High must be before low
    assert hdate < ldate
    # High value must be higher than low value
    assert hval > lval
    assert drawdown == 0.091755

    df = DataFrame(zip(values[:5], dates[:5]), columns=['profit', 'open_date'])
    with pytest.raises(ValueError, match='No losing trade, therefore no drawdown.'):
        calculate_max_drawdown(df, date_col='open_date', value_col='profit')
