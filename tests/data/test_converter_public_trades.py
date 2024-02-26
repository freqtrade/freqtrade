
import numpy as np
import pandas as pd
import pytest

from freqtrade.constants import DEFAULT_TRADES_COLUMNS
from freqtrade.data.converter import populate_dataframe_with_trades
from freqtrade.data.converter.orderflow import trades_to_volumeprofile_with_total_delta_bid_ask
from freqtrade.data.converter.trade_converter import trades_list_to_df


BIN_SIZE_SCALE = 0.5


def read_csv(filename, converter_columns: list = ['side', 'type']):
    return pd.read_csv(filename, skipinitialspace=True, infer_datetime_format=True, index_col=0,
                       parse_dates=True, converters={col: str.strip for col in converter_columns})


@pytest.fixture
def populate_dataframe_with_trades_dataframe(testdatadir):
    return pd.read_feather(testdatadir / 'orderflow/populate_dataframe_with_trades_DF.feather')


@pytest.fixture
def populate_dataframe_with_trades_trades(testdatadir):
    return pd.read_feather(testdatadir / 'orderflow/populate_dataframe_with_trades_TRADES.feather')


@pytest.fixture
def candles(testdatadir):
    return pd.read_json(testdatadir / 'orderflow/candles.json').copy()


@pytest.fixture
def public_trades_list(testdatadir):
    return read_csv(testdatadir / 'orderflow/public_trades_list.csv').copy()


@pytest.fixture
def public_trades_list_simple(testdatadir):
    return read_csv(testdatadir / 'orderflow/public_trades_list_simple_example.csv').copy()


def test_public_trades_columns_before_change(
        populate_dataframe_with_trades_dataframe,
        populate_dataframe_with_trades_trades):
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == [
        'date', 'open', 'high', 'low', 'close', 'volume']
    assert populate_dataframe_with_trades_trades.columns.tolist() == [
        'timestamp', 'id', 'type', 'side', 'price',
        'amount', 'cost', 'date']


def test_public_trades_mock_populate_dataframe_with_trades__check_orderflow(
        populate_dataframe_with_trades_dataframe,
        populate_dataframe_with_trades_trades):
    """
    Tests the `populate_dataframe_with_trades` function's order flow calculation.

    This test checks the generated data frame and order flow for specific properties
    based on the provided configuration and sample data.
    """
    dataframe = populate_dataframe_with_trades_dataframe.copy()
    trades = populate_dataframe_with_trades_trades.copy()
    dataframe['date'] = pd.to_datetime(
        dataframe['date'], unit='ms')
    dataframe = dataframe.copy().tail().reset_index(drop=True)
    config = {'timeframe': '5m',
              'orderflow': {
                  'scale': 0.005,
                  'imbalance_volume': 0,
                  'imbalance_ratio': 300,
                  'stacked_imbalance_range': 3
              }}
    df = populate_dataframe_with_trades(config,
                                        dataframe, trades, pair='unitttest')
    results = df.iloc[0]
    t = results['trades']
    of = results['orderflow']
    assert 0 != len(results)  # 13 columns
    assert 151 == len(t)

    # orderflow/cluster/footprint
    assert 23 == len(of)
    assert [0.0, 1.0, 4.999, 0.0, 4.999, 4.999,
            1.0] == of.iloc[0].values.tolist()
    assert [0.0, 1.0, 0.103, 0.0, 0.103, 0.103,
            1.0] == of.iloc[-1].values.tolist()
    of = df.iloc[-1]['orderflow']
    assert 19 == len(of)
    assert [1.0, 0.0, -12.536, 12.536, 0.0,
            12.536, 1.0] == of.iloc[0].values.tolist()
    assert [4.0, 3.0, -40.94800000000001, 59.18200000000001,
            18.233999999999998, 77.41600000000001, 7.0] == of.iloc[-1].values.tolist()

    assert -50.519000000000005 == results['delta']
    assert -79.469 == results['min_delta']
    assert 17.298 == results['max_delta']
    assert np.isnan(results['stacked_imbalances_bid'])
    assert np.isnan(results['stacked_imbalances_ask'])

    results = df.iloc[-3]
    assert -112.71399999999994 == results['delta']
    assert -120.673 == results['min_delta']
    assert 11.664 == results['max_delta']
    assert np.isnan(results['stacked_imbalances_bid'])
    assert np.isnan(results['stacked_imbalances_ask'])

    results = df.iloc[-1]
    assert -49.30200000000002 == results['delta']
    assert -70.222 == results['min_delta']
    assert 11.213000000000003 == results['max_delta']
    assert np.isnan(results['stacked_imbalances_bid'])
    assert np.isnan(results['stacked_imbalances_ask'])


def test_public_trades_trades_mock_populate_dataframe_with_trades__check_trades(
        populate_dataframe_with_trades_dataframe, populate_dataframe_with_trades_trades):
    dataframe = populate_dataframe_with_trades_dataframe.copy()
    trades = populate_dataframe_with_trades_trades.copy()

    # slice of unnecessary trades
    dataframe['date'] = pd.to_datetime(
        dataframe['date'], unit='ms')
    dataframe = dataframe.copy().tail().reset_index(drop=True)
    trades = trades.copy().loc[trades.date >= dataframe.date[0]]
    trades.reset_index(inplace=True, drop=True)

    assert trades['id'][0] == '313881442'

    config = {
        'timeframe': '5m',
        'orderflow': {
            'scale': 0.5,
            'imbalance_volume': 0,
            'imbalance_ratio': 300,
            'stacked_imbalance_range': 3
        }
    }
    df = populate_dataframe_with_trades(config,
                                        dataframe, trades, pair='unitttest')
    row = df.iloc[0]
    assert list(df.columns) == ['date', 'open', 'high', 'low',
                                'close', 'volume', 'trades', 'orderflow',
                                'bid', 'ask', 'delta', 'min_delta',
                                'max_delta', 'total_trades',
                                'stacked_imbalances_bid',
                                'stacked_imbalances_ask']

    assert -50.519 == pytest.approx(row['delta'])
    assert 219.961 == row['bid']
    assert 169.442 == row['ask']

    assert 151 == len(row.trades)
    t = row['trades'].iloc[0]
    assert trades['id'][0] == t["id"]
    assert int(trades['timestamp'][0]) == int(t['timestamp'])
    assert 'sell' == t['side']
    assert '313881442' == t['id']
    assert 234.72 == t['price']


def test_public_trades_put_volume_profile_into_ohlcv_candles(public_trades_list_simple, candles):
    df = trades_list_to_df(
        public_trades_list_simple[DEFAULT_TRADES_COLUMNS].values.tolist())
    df = trades_to_volumeprofile_with_total_delta_bid_ask(
        df, scale=BIN_SIZE_SCALE)
    candles['vp'] = np.nan
    candles.loc[candles.index == 1, ['vp']] = candles.loc[candles.index == 1, [
        'vp']].applymap(lambda x: pd.DataFrame(df.to_dict()))
    assert 0.14 == candles['vp'][1].values.tolist()[1][2]  # delta
    assert 0.14 == candles['vp'][1]['delta'].iat[1]


def test_public_trades_binned_big_sample_list(public_trades_list):
    BIN_SIZE_SCALE = 0.05
    trades = trades_list_to_df(
        public_trades_list[DEFAULT_TRADES_COLUMNS].values.tolist())
    df = trades_to_volumeprofile_with_total_delta_bid_ask(
        trades, scale=BIN_SIZE_SCALE)
    assert df.columns.tolist() == ['bid', 'ask', 'delta',
                                   'bid_amount', 'ask_amount',
                                   'total_volume', 'total_trades']
    assert 23 == len(df)
    assert df.index[0] < df.index[1] < df.index[2]
    assert df.index[0] + BIN_SIZE_SCALE == df.index[1]
    assert (trades['price'].min() -
            BIN_SIZE_SCALE) < df.index[0] < trades['price'].max()
    assert (df.index[0] + BIN_SIZE_SCALE) >= df.index[1]
    assert (trades['price'].max() -
            BIN_SIZE_SCALE) < df.index[-1] < trades['price'].max()

    assert 32 == df['bid'].iat[0]  # bid
    assert 197.512 == df['bid_amount'].iat[0]  # bid
    assert 88.98 == df['ask_amount'].iat[0]  # ask
    assert 26 == df['ask'].iat[0]  # ask
    assert -108.532 == pytest.approx(df['delta'].iat[0])  # delta
    assert 3 == df['bid'].iat[-1]  # bid
    assert 50.659 == df['bid_amount'].iat[-1]  # bid
    assert 108.21 == df['ask_amount'].iat[-1]  # ask
    assert 44 == df['ask'].iat[-1]  # ask
    assert 57.551 == df['delta'].iat[-1]  # delta

    BIN_SIZE_SCALE = 1
    trades = trades_list_to_df(public_trades_list[DEFAULT_TRADES_COLUMNS].values.tolist())
    df = trades_to_volumeprofile_with_total_delta_bid_ask(
        trades, scale=BIN_SIZE_SCALE)
    assert 2 == len(df)
    assert df.index[0] < df.index[1]
    assert (trades['price'].min() -
            BIN_SIZE_SCALE) < df.index[0] < trades['price'].max()
    assert (df.index[0] + BIN_SIZE_SCALE) >= df.index[1]
    assert (trades['price'].max() -
            BIN_SIZE_SCALE) < df.index[-1] < trades['price'].max()

    assert 1667.0 == df.index[-1]

    # bid assert 763.7 == df['ask'].iat[0]  # ask
    assert 710.98 == df['bid_amount'].iat[0]
    assert 111 == df['bid'].iat[0]
    assert 52.7199999 == pytest.approx(df['delta'].iat[0])  # delta
    # assert 50.659 == df['bid'].iat[-1]  # bid
    # assert 108.21 == df['ask'].iat[-1]  # ask
    # assert 57.551 == df['delta'].iat[-1]  # delta
    #

# bidask


def test_public_trades_testdata_sanity(
        candles,
        public_trades_list,
        public_trades_list_simple,
        populate_dataframe_with_trades_dataframe,
        populate_dataframe_with_trades_trades):
    assert 10999 == len(candles)
    assert 1000 == len(public_trades_list)
    assert 999 == len(populate_dataframe_with_trades_dataframe)
    assert 293532 == len(populate_dataframe_with_trades_trades)

    assert 7 == len(public_trades_list_simple)
    assert 5 == public_trades_list_simple.loc[
        public_trades_list_simple['side'].str.contains(
            'sell'), 'id'].count()
    assert 2 == public_trades_list_simple.loc[
        public_trades_list_simple['side'].str.contains(
            'buy'), 'id'].count()

    assert public_trades_list.columns.tolist() == [
        'timestamp', 'id', 'type', 'side', 'price',
        'amount', 'cost', 'date']

    assert public_trades_list.columns.tolist() == [
        'timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost', 'date']
    assert public_trades_list_simple.columns.tolist() == [
        'timestamp', 'id', 'type', 'side', 'price',
        'amount', 'cost', 'date']
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == [
        'date', 'open', 'high', 'low', 'close', 'volume']
    assert populate_dataframe_with_trades_trades.columns.tolist() == [
        'timestamp', 'id', 'type', 'side', 'price',
        'amount', 'cost', 'date']
