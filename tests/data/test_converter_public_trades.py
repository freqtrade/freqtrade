import numpy as np
import pandas as pd
import pytest

from freqtrade.constants import DEFAULT_TRADES_COLUMNS
from freqtrade.data.converter import populate_dataframe_with_trades
from freqtrade.data.converter.orderflow import trades_to_volumeprofile_with_total_delta_bid_ask
from freqtrade.data.converter.trade_converter import trades_list_to_df


BIN_SIZE_SCALE = 0.5


def read_csv(filename, converter_columns: list = ["side", "type"]):
    return pd.read_csv(
        filename,
        skipinitialspace=True,
        infer_datetime_format=True,
        index_col=0,
        parse_dates=True,
        converters={col: str.strip for col in converter_columns},
    )


@pytest.fixture
def populate_dataframe_with_trades_dataframe(testdatadir):
    return pd.read_feather(testdatadir / "orderflow/populate_dataframe_with_trades_DF.feather")


@pytest.fixture
def populate_dataframe_with_trades_trades(testdatadir):
    return pd.read_feather(testdatadir / "orderflow/populate_dataframe_with_trades_TRADES.feather")


@pytest.fixture
def candles(testdatadir):
    return pd.read_json(testdatadir / "orderflow/candles.json").copy()


@pytest.fixture
def public_trades_list(testdatadir):
    return read_csv(testdatadir / "orderflow/public_trades_list.csv").copy()


@pytest.fixture
def public_trades_list_simple(testdatadir):
    return read_csv(testdatadir / "orderflow/public_trades_list_simple_example.csv").copy()


def test_public_trades_columns_before_change(
    populate_dataframe_with_trades_dataframe, populate_dataframe_with_trades_trades
):
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert populate_dataframe_with_trades_trades.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]


def test_public_trades_mock_populate_dataframe_with_trades__check_orderflow(
    populate_dataframe_with_trades_dataframe, populate_dataframe_with_trades_trades
):
    """
    Tests the `populate_dataframe_with_trades` function's order flow calculation.

    This test checks the generated data frame and order flow for specific properties
    based on the provided configuration and sample data.
    """
    # Create copies of the input data to avoid modifying the originals
    dataframe = populate_dataframe_with_trades_dataframe.copy()
    trades = populate_dataframe_with_trades_trades.copy()
    # Convert the 'date' column to datetime format with milliseconds
    dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms")
    # Select the last rows and reset the index (optional, depends on usage)
    dataframe = dataframe.copy().tail().reset_index(drop=True)
    # Define the configuration for order flow calculation
    config = {
        "timeframe": "5m",
        "orderflow": {
            "scale": 0.005,
            "imbalance_volume": 0,
            "imbalance_ratio": 3,
            "stacked_imbalance_range": 3,
        },
    }
    # Apply the function to populate the data frame with order flow data
    df = populate_dataframe_with_trades(config, dataframe, trades)
    # Extract results from the first row of the DataFrame
    results = df.iloc[0]
    t = results["trades"]
    of = results["orderflow"]

    # Assert basic properties of the results
    assert 0 != len(results)
    assert 151 == len(t)

    # --- Order Flow Analysis ---
    # Assert number of order flow data points
    assert 23 == len(of)  # Assert expected number of data points

    # Assert specific order flow values at the beginning of the DataFrame
    assert [0.0, 1.0, 4.999, 0.0, 4.999, 4.999, 1.0] == of.iloc[0].values.tolist()

    # Assert specific order flow values at the end of the DataFrame (excluding last row)
    assert [0.0, 1.0, 0.103, 0.0, 0.103, 0.103, 1.0] == of.iloc[-1].values.tolist()

    # Extract order flow from the last row of the DataFrame
    of = df.iloc[-1]["orderflow"]

    # Assert number of order flow data points in the last row
    assert 19 == len(of)  # Assert expected number of data points

    # Assert specific order flow values at the beginning of the last row
    assert [1.0, 0.0, -12.536, 12.536, 0.0, 12.536, 1.0] == of.iloc[0].values.tolist()

    # Assert specific order flow values at the end of the last row
    assert [
        4.0,
        3.0,
        -40.94800000000001,
        59.18200000000001,
        18.233999999999998,
        77.41600000000001,
        7.0,
    ] == of.iloc[-1].values.tolist()

    # --- Delta and Other Results ---

    # Assert delta value from the first row
    assert -50.519000000000005 == results["delta"]

    # Assert min and max delta values from the first row
    assert -79.469 == results["min_delta"]
    assert 17.298 == results["max_delta"]

    # Assert that stacked imbalances are NaN (not applicable in this test)
    assert np.isnan(results["stacked_imbalances_bid"])
    assert np.isnan(results["stacked_imbalances_ask"])

    # Repeat assertions for the third from last row
    results = df.iloc[-2]
    assert -20.86200000000008 == results["delta"]
    assert -54.55999999999999 == results["min_delta"]
    assert 82.842 == results["max_delta"]
    assert 234.99 == results["stacked_imbalances_bid"]
    assert 234.96 == results["stacked_imbalances_ask"]

    # Repeat assertions for the last row
    results = df.iloc[-1]
    assert -49.30200000000002 == results["delta"]
    assert -70.222 == results["min_delta"]
    assert 11.213000000000003 == results["max_delta"]
    assert np.isnan(results["stacked_imbalances_bid"])
    assert np.isnan(results["stacked_imbalances_ask"])


def test_public_trades_trades_mock_populate_dataframe_with_trades__check_trades(
    populate_dataframe_with_trades_dataframe, populate_dataframe_with_trades_trades
):
    """
    Tests the `populate_dataframe_with_trades` function's handling of trades,
    ensuring correct integration of trades data into the generated DataFrame.
    """

    # Create copies of the input data to avoid modifying the originals
    dataframe = populate_dataframe_with_trades_dataframe.copy()
    trades = populate_dataframe_with_trades_trades.copy()

    # --- Data Preparation ---

    # Convert the 'date' column to datetime format with milliseconds
    dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms")

    # Select the final row of the DataFrame
    dataframe = dataframe.tail().reset_index(drop=True)

    # Filter trades to those occurring after or at the same time as the first DataFrame date
    trades = trades.loc[trades.date >= dataframe.date[0]]
    trades.reset_index(inplace=True, drop=True)  # Reset index for clarity

    # Assert the first trade ID to ensure filtering worked correctly
    assert trades["id"][0] == "313881442"

    # --- Configuration and Function Call ---

    # Define configuration for order flow calculation (used for context)
    config = {
        "timeframe": "5m",
        "orderflow": {
            "scale": 0.5,
            "imbalance_volume": 0,
            "imbalance_ratio": 3,
            "stacked_imbalance_range": 3,
        },
    }

    # Populate the DataFrame with trades and order flow data
    df = populate_dataframe_with_trades(config, dataframe, trades)

    # --- DataFrame and Trade Data Validation ---

    row = df.iloc[0]  # Extract the first row for assertions

    # Assert DataFrame structure
    assert list(df.columns) == [
        # ... (list of expected column names)
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades",
        "orderflow",
        "bid",
        "ask",
        "delta",
        "min_delta",
        "max_delta",
        "total_trades",
        "stacked_imbalances_bid",
        "stacked_imbalances_ask",
    ]
    # Assert delta, bid, and ask values
    assert -50.519 == pytest.approx(row["delta"])
    assert 219.961 == row["bid"]
    assert 169.442 == row["ask"]

    # Assert the number of trades
    assert 151 == len(row.trades)

    # Assert specific details of the first trade
    t = row["trades"].iloc[0]
    assert trades["id"][0] == t["id"]
    assert int(trades["timestamp"][0]) == int(t["timestamp"])
    assert "sell" == t["side"]
    assert "313881442" == t["id"]
    assert 234.72 == t["price"]


def test_public_trades_put_volume_profile_into_ohlcv_candles(public_trades_list_simple, candles):
    """
    Tests the integration of volume profile data into OHLCV candles.

    This test verifies that
    the `trades_to_volumeprofile_with_total_delta_bid_ask`
    function correctly calculates the volume profile and that
    it correctly assigns the delta value from the volume profile to the
    corresponding candle in the `candles` DataFrame.
    """

    # Convert the trade list to a DataFrame
    df = trades_list_to_df(public_trades_list_simple[DEFAULT_TRADES_COLUMNS].values.tolist())

    # Generate the volume profile with the specified bin size
    df = trades_to_volumeprofile_with_total_delta_bid_ask(df, scale=BIN_SIZE_SCALE)

    # Initialize the 'vp' column in the candles DataFrame with NaNs
    candles["vp"] = np.nan

    # Select the second candle (index 1) and attempt to assign the volume profile data
    # (as a DataFrame) to the 'vp' element.
    candles.loc[candles.index == 1, ["vp"]] = candles.loc[candles.index == 1, ["vp"]].applymap(
        lambda x: pd.DataFrame(df.to_dict())
    )

    # Assert the delta value in the 'vp' element of the second candle
    assert 0.14 == candles["vp"][1].values.tolist()[1][2]

    # Alternative assertion using `.iat` accessor (assuming correct assignment logic)
    assert 0.14 == candles["vp"][1]["delta"].iat[1]


def test_public_trades_binned_big_sample_list(public_trades_list):
    """
    Tests the `trades_to_volumeprofile_with_total_delta_bid_ask` function
    with different bin sizes and verifies the generated DataFrame's structure and values.
    """

    # Define the bin size for the first test
    BIN_SIZE_SCALE = 0.05

    # Convert the trade list to a DataFrame
    trades = trades_list_to_df(public_trades_list[DEFAULT_TRADES_COLUMNS].values.tolist())

    # Generate the volume profile with the specified bin size
    df = trades_to_volumeprofile_with_total_delta_bid_ask(trades, scale=BIN_SIZE_SCALE)

    # Assert that the DataFrame has the expected columns
    assert df.columns.tolist() == [
        "bid",
        "ask",
        "delta",
        "bid_amount",
        "ask_amount",
        "total_volume",
        "total_trades",
    ]

    # Assert the number of rows in the DataFrame (expected 23 for this bin size)
    assert len(df) == 23

    # Assert that the index values are in ascending order and spaced correctly
    assert all(df.index[i] < df.index[i + 1] for i in range(len(df) - 1))
    assert df.index[0] + BIN_SIZE_SCALE == df.index[1]
    assert (trades["price"].min() - BIN_SIZE_SCALE) < df.index[0] < trades["price"].max()
    assert (df.index[0] + BIN_SIZE_SCALE) >= df.index[1]
    assert (trades["price"].max() - BIN_SIZE_SCALE) < df.index[-1] < trades["price"].max()

    # Assert specific values in the first and last rows of the DataFrame
    assert 32 == df["bid"].iloc[0]  # bid price
    assert 197.512 == df["bid_amount"].iloc[0]  # total bid amount
    assert 88.98 == df["ask_amount"].iloc[0]  # total ask amount
    assert 26 == df["ask"].iloc[0]  # ask price
    assert -108.532 == pytest.approx(df["delta"].iloc[0])  # delta (bid amount - ask amount)

    assert 3 == df["bid"].iloc[-1]  # bid price
    assert 50.659 == df["bid_amount"].iloc[-1]  # total bid amount
    assert 108.21 == df["ask_amount"].iloc[-1]  # total ask amount
    assert 44 == df["ask"].iloc[-1]  # ask price
    assert 57.551 == df["delta"].iloc[-1]  # delta (bid amount - ask amount)

    # Repeat the process with a larger bin size
    BIN_SIZE_SCALE = 1

    # Generate the volume profile with the larger bin size
    df = trades_to_volumeprofile_with_total_delta_bid_ask(trades, scale=BIN_SIZE_SCALE)

    # Assert the number of rows in the DataFrame (expected 2 for this bin size)
    assert len(df) == 2

    # Repeat similar assertions for index ordering and spacing
    assert all(df.index[i] < df.index[i + 1] for i in range(len(df) - 1))
    assert (trades["price"].min() - BIN_SIZE_SCALE) < df.index[0] < trades["price"].max()
    assert (df.index[0] + BIN_SIZE_SCALE) >= df.index[1]
    assert (trades["price"].max() - BIN_SIZE_SCALE) < df.index[-1] < trades["price"].max()

    # Assert the value in the last row of the DataFrame with the larger bin size
    assert 1667.0 == df.index[-1]
    assert 710.98 == df["bid_amount"].iat[0]
    assert 111 == df["bid"].iat[0]
    assert 52.7199999 == pytest.approx(df["delta"].iat[0])  # delta


def test_public_trades_testdata_sanity(
    candles,
    public_trades_list,
    public_trades_list_simple,
    populate_dataframe_with_trades_dataframe,
    populate_dataframe_with_trades_trades,
):
    assert 10999 == len(candles)
    assert 1000 == len(public_trades_list)
    assert 999 == len(populate_dataframe_with_trades_dataframe)
    assert 293532 == len(populate_dataframe_with_trades_trades)

    assert 7 == len(public_trades_list_simple)
    assert (
        5
        == public_trades_list_simple.loc[
            public_trades_list_simple["side"].str.contains("sell"), "id"
        ].count()
    )
    assert (
        2
        == public_trades_list_simple.loc[
            public_trades_list_simple["side"].str.contains("buy"), "id"
        ].count()
    )

    assert public_trades_list.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]

    assert public_trades_list.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]
    assert public_trades_list_simple.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert populate_dataframe_with_trades_trades.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]
