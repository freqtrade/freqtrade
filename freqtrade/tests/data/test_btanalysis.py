from unittest.mock import MagicMock

from arrow import Arrow
import pytest
from pandas import DataFrame, to_datetime

from freqtrade.arguments import Arguments, TimeRange
from freqtrade.data.btanalysis import (BT_DATA_COLUMNS,
                                       extract_trades_of_period,
                                       load_backtest_data, load_trades)
from freqtrade.data.history import load_pair_history, make_testdata_path
from freqtrade.persistence import Trade, init
from freqtrade.strategy.interface import SellType
from freqtrade.tests.test_persistence import (create_mock_trades,
                                              init_persistence)


def test_load_backtest_data():

    filename = make_testdata_path(None) / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    assert isinstance(bt_data, DataFrame)
    assert list(bt_data.columns) == BT_DATA_COLUMNS + ["profitabs"]
    assert len(bt_data) == 179

    # Test loading from string (must yield same result)
    bt_data2 = load_backtest_data(str(filename))
    assert bt_data.equals(bt_data2)

    with pytest.raises(ValueError, match=r"File .* does not exist\."):
        load_backtest_data(str("filename") + "nofile")


def test_load_trades_file(default_conf, fee, mocker):
    # Real testing of load_backtest_data is done in test_load_backtest_data
    lbt = mocker.patch("freqtrade.data.btanalysis.load_backtest_data", MagicMock())
    filename = make_testdata_path(None) / "backtest-result_test.json"
    load_trades(db_url=None, exportfilename=filename)
    assert lbt.call_count == 1


@pytest.mark.usefixtures("init_persistence")
def test_load_trades_db(default_conf, fee, mocker):

    create_mock_trades(fee)
    # remove init so it does not init again
    init_mock = mocker.patch('freqtrade.persistence.init', MagicMock())

    trades = load_trades(db_url=default_conf['db_url'], exportfilename=None)
    assert init_mock.call_count == 1
    assert len(trades) == 3
    assert isinstance(trades, DataFrame)
    assert "pair" in trades.columns
    assert "open_time" in trades.columns


def test_extract_trades_of_period():
    pair = "UNITTEST/BTC"
    timerange = TimeRange(None, 'line', 0, -1000)

    data = load_pair_history(pair=pair, ticker_interval='1m',
                             datadir=None, timerange=timerange)

    # timerange = 2017-11-14 06:07 - 2017-11-14 22:58:00
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

