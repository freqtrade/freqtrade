# pragma pylint: disable=missing-docstring, C0103

from datetime import UTC

import pandas as pd
from numpy import nan
from pandas import DataFrame, Timestamp

from freqtrade.data.btanalysis.historic_precision import get_tick_size_over_time


def test_get_tick_size_over_time():
    """
    Test the get_tick_size_over_time function with predefined data
    """
    # Create test dataframe with different levels of precision
    data = {
        "date": [
            Timestamp("2020-01-01 00:00:00", tz=UTC),
            Timestamp("2020-01-02 00:00:00", tz=UTC),
            Timestamp("2020-01-03 00:00:00", tz=UTC),
            Timestamp("2020-01-15 00:00:00", tz=UTC),
            Timestamp("2020-01-16 00:00:00", tz=UTC),
            Timestamp("2020-01-31 00:00:00", tz=UTC),
            Timestamp("2020-02-01 00:00:00", tz=UTC),
            Timestamp("2020-02-15 00:00:00", tz=UTC),
            Timestamp("2020-03-15 00:00:00", tz=UTC),
        ],
        "open": [1.23456, 1.234, 1.23, 1.2, 1.23456, 1.234, 2.3456, 2.34, 2.34],
        "high": [1.23457, 1.235, 1.24, 1.3, 1.23456, 1.235, 2.3457, 2.34, 2.34],
        "low": [1.23455, 1.233, 1.22, 1.1, 1.23456, 1.233, 2.3455, 2.34, 2.34],
        "close": [1.23456, 1.234, 1.23, 1.2, 1.23456, 1.234, 2.3456, 2.34, 2.34],
        "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900],
    }

    candles = DataFrame(data)

    # Calculate significant digits
    result = get_tick_size_over_time(candles)

    # Check that the result is a pandas Series
    assert isinstance(result, pd.Series)

    # Check that we have three months of data (Jan, Feb and March 2020 )
    assert len(result) == 3

    # Before
    assert result.asof("2019-01-01 00:00:00+00:00") is nan
    # January should have 5 significant digits (based on 1.23456789 being the most precise value)
    # which should be converted to 0.00001

    assert result.asof("2020-01-01 00:00:00+00:00") == 0.00001
    assert result.asof("2020-01-01 00:00:00+00:00") == 0.00001
    assert result.asof("2020-02-25 00:00:00+00:00") == 0.0001
    assert result.asof("2020-03-25 00:00:00+00:00") == 0.01
    assert result.asof("2020-04-01 00:00:00+00:00") == 0.01
    # Value far past the last date should be the last value
    assert result.asof("2025-04-01 00:00:00+00:00") == 0.01

    assert result.iloc[0] == 0.00001


def test_get_tick_size_over_time_real_data(testdatadir):
    """
    Test the get_tick_size_over_time function with real data from the testdatadir
    """
    from freqtrade.data.history import load_pair_history

    # Load some test data from the testdata directory
    pair = "UNITTEST/BTC"
    timeframe = "1m"

    candles = load_pair_history(
        datadir=testdatadir,
        pair=pair,
        timeframe=timeframe,
    )

    # Make sure we have test data
    assert not candles.empty, "No test data found, cannot run test"

    # Calculate significant digits
    result = get_tick_size_over_time(candles)

    assert isinstance(result, pd.Series)

    # Verify that all values are between 0 and 1 (valid precision values)
    assert all(result > 0)
    assert all(result < 1)

    assert all(result <= 0.0001)
    assert all(result >= 0.00000001)


def test_get_tick_size_over_time_small_numbers():
    """
    Test the get_tick_size_over_time function with predefined data
    """
    # Create test dataframe with different levels of precision
    data = {
        "date": [
            Timestamp("2020-01-01 00:00:00", tz=UTC),
            Timestamp("2020-01-02 00:00:00", tz=UTC),
            Timestamp("2020-01-03 00:00:00", tz=UTC),
            Timestamp("2020-01-15 00:00:00", tz=UTC),
            Timestamp("2020-01-16 00:00:00", tz=UTC),
            Timestamp("2020-01-31 00:00:00", tz=UTC),
            Timestamp("2020-02-01 00:00:00", tz=UTC),
            Timestamp("2020-02-15 00:00:00", tz=UTC),
            Timestamp("2020-03-15 00:00:00", tz=UTC),
        ],
        "open": [
            0.000000123456,
            0.0000001234,
            0.000000123,
            0.00000012,
            0.000000123456,
            0.0000001234,
            0.00000023456,
            0.000000234,
            0.000000234,
        ],
        "high": [
            0.000000123457,
            0.0000001235,
            0.000000124,
            0.00000013,
            0.000000123456,
            0.0000001235,
            0.00000023457,
            0.000000234,
            0.000000234,
        ],
        "low": [
            0.000000123455,
            0.0000001233,
            0.000000122,
            0.00000011,
            0.000000123456,
            0.0000001233,
            0.00000023455,
            0.000000234,
            0.000000234,
        ],
        "close": [
            0.000000123456,
            0.0000001234,
            0.000000123,
            0.00000012,
            0.000000123456,
            0.0000001234,
            0.00000023456,
            0.000000234,
            0.000000234,
        ],
        "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900],
    }

    candles = DataFrame(data)

    # Calculate significant digits
    result = get_tick_size_over_time(candles)

    # Check that the result is a pandas Series
    assert isinstance(result, pd.Series)

    # Check that we have three months of data (Jan, Feb and March 2020 )
    assert len(result) == 3

    # Before
    assert result.asof("2019-01-01 00:00:00+00:00") is nan
    # January should have 5 significant digits (based on 1.23456789 being the most precise value)
    # which should be converted to 0.00001

    assert result.asof("2020-01-01 00:00:00+00:00") == 0.000000000001
    assert result.asof("2020-01-01 00:00:00+00:00") == 0.000000000001
    assert result.asof("2020-02-25 00:00:00+00:00") == 0.00000000001
    assert result.asof("2020-03-25 00:00:00+00:00") == 0.000000001
    assert result.asof("2020-04-01 00:00:00+00:00") == 0.000000001
    # Value far past the last date should be the last value
    assert result.asof("2025-04-01 00:00:00+00:00") == 0.000000001

    assert result.iloc[0] == 0.000000000001
