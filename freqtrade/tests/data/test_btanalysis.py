import pytest
from unittest.mock import MagicMock
from pandas import DataFrame

from freqtrade.data.btanalysis import BT_DATA_COLUMNS, load_backtest_data, load_trades
from freqtrade.data.history import make_testdata_path
from freqtrade.persistence import init, Trade
from freqtrade.tests.test_persistence import init_persistence, create_mock_trades


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
