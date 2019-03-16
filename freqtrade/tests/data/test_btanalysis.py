import pytest
from pandas import DataFrame

from freqtrade.data.btanalysis import BT_DATA_COLUMNS, load_backtest_data
from freqtrade.data.history import make_testdata_path


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
