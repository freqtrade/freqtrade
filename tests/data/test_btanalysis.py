from unittest.mock import MagicMock

import pytest
from arrow import Arrow
from pandas import DataFrame, DateOffset, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.data.btanalysis import (BT_DATA_COLUMNS,
                                       combine_tickers_with_mean,
                                       create_cum_profit,
                                       extract_trades_of_period,
                                       load_backtest_data, load_trades,
                                       load_trades_from_db, analyze_trade_parallelism)
from freqtrade.data.history import load_data, load_pair_history
from tests.test_persistence import create_mock_trades


def test_load_backtest_data(testdatadir):

    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    assert isinstance(bt_data, DataFrame)
    assert list(bt_data.columns) == BT_DATA_COLUMNS + ["profit"]
    assert len(bt_data) == 179

    # Test loading from string (must yield same result)
    bt_data2 = load_backtest_data(str(filename))
    assert bt_data.equals(bt_data2)

    with pytest.raises(ValueError, match=r"File .* does not exist\."):
        load_backtest_data(str("filename") + "nofile")


@pytest.mark.usefixtures("init_persistence")
def test_load_trades_from_db(default_conf, fee, mocker):

    create_mock_trades(fee)
    # remove init so it does not init again
    init_mock = mocker.patch('freqtrade.persistence.init', MagicMock())

    trades = load_trades_from_db(db_url=default_conf['db_url'])
    assert init_mock.call_count == 1
    assert len(trades) == 3
    assert isinstance(trades, DataFrame)
    assert "pair" in trades.columns
    assert "open_time" in trades.columns
    assert "profitperc" in trades.columns

    for col in BT_DATA_COLUMNS:
        if col not in ['index', 'open_at_end']:
            assert col in trades.columns


def test_extract_trades_of_period(testdatadir):
    pair = "UNITTEST/BTC"
    # 2018-11-14 06:07:00
    timerange = TimeRange('date', None, 1510639620, 0)

    data = load_pair_history(pair=pair, timeframe='1m',
                             datadir=testdatadir, timerange=timerange)

    trades = DataFrame(
        {'pair': [pair, pair, pair, pair],
         'profit_percent': [0.0, 0.1, -0.2, -0.5],
         'profit_abs': [0.0, 1, -2, -5],
         'open_time': to_datetime([Arrow(2017, 11, 13, 15, 40, 0).datetime,
                                   Arrow(2017, 11, 14, 9, 41, 0).datetime,
                                   Arrow(2017, 11, 14, 14, 20, 0).datetime,
                                   Arrow(2017, 11, 15, 3, 40, 0).datetime,
                                   ], utc=True
                                  ),
         'close_time': to_datetime([Arrow(2017, 11, 13, 16, 40, 0).datetime,
                                    Arrow(2017, 11, 14, 10, 41, 0).datetime,
                                    Arrow(2017, 11, 14, 15, 25, 0).datetime,
                                    Arrow(2017, 11, 15, 3, 55, 0).datetime,
                                    ], utc=True)
         })
    trades1 = extract_trades_of_period(data, trades)
    # First and last trade are dropped as they are out of range
    assert len(trades1) == 2
    assert trades1.iloc[0].open_time == Arrow(2017, 11, 14, 9, 41, 0).datetime
    assert trades1.iloc[0].close_time == Arrow(2017, 11, 14, 10, 41, 0).datetime
    assert trades1.iloc[-1].open_time == Arrow(2017, 11, 14, 14, 20, 0).datetime
    assert trades1.iloc[-1].close_time == Arrow(2017, 11, 14, 15, 25, 0).datetime


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
                )

    assert db_mock.call_count == 1
    assert bt_mock.call_count == 0

    db_mock.reset_mock()
    bt_mock.reset_mock()
    default_conf['exportfilename'] = "testfile.json"
    load_trades("file",
                db_url=default_conf.get('db_url'),
                exportfilename=default_conf.get('exportfilename'),)

    assert db_mock.call_count == 0
    assert bt_mock.call_count == 1


def test_combine_tickers_with_mean(testdatadir):
    pairs = ["ETH/BTC", "ADA/BTC"]
    tickers = load_data(datadir=testdatadir,
                        pairs=pairs,
                        timeframe='5m'
                        )
    df = combine_tickers_with_mean(tickers)
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
    assert cum_profits.iloc[-1]['cum_profits'] == 0.0798005


def test_create_cum_profit1(testdatadir):
    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    # Move close-time to "off" the candle, to make sure the logic still works
    bt_data.loc[:, 'close_time'] = bt_data.loc[:, 'close_time'] + DateOffset(seconds=20)
    timerange = TimeRange.parse_timerange("20180110-20180112")

    df = load_pair_history(pair="TRX/BTC", timeframe='5m',
                           datadir=testdatadir, timerange=timerange)

    cum_profits = create_cum_profit(df.set_index('date'),
                                    bt_data[bt_data["pair"] == 'TRX/BTC'],
                                    "cum_profits", timeframe="5m")
    assert "cum_profits" in cum_profits.columns
    assert cum_profits.iloc[0]['cum_profits'] == 0
    assert cum_profits.iloc[-1]['cum_profits'] == 0.0798005
