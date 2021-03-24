from math import isclose

import numpy as np
import pandas as pd
import pytest

from freqtrade.strategy import merge_informative_pair, stoploss_from_open, timeframe_to_minutes


def generate_test_data(timeframe: str, size: int):
    np.random.seed(42)
    tf_mins = timeframe_to_minutes(timeframe)

    base = np.random.normal(20, 2, size=size)

    date = pd.period_range('2020-07-05', periods=size, freq=f'{tf_mins}min').to_timestamp()
    df = pd.DataFrame({
        'date': date,
        'open': base,
        'high': base + np.random.normal(2, 1, size=size),
        'low': base - np.random.normal(2, 1, size=size),
        'close': base + np.random.normal(0, 1, size=size),
        'volume': np.random.normal(200, size=size)
    }
    )
    df = df.dropna()
    return df


def test_merge_informative_pair():
    data = generate_test_data('15m', 40)
    informative = generate_test_data('1h', 40)

    result = merge_informative_pair(data, informative, '15m', '1h', ffill=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    assert 'date' in result.columns
    assert result['date'].equals(data['date'])
    assert 'date_1h' in result.columns

    assert 'open' in result.columns
    assert 'open_1h' in result.columns
    assert result['open'].equals(data['open'])

    assert 'close' in result.columns
    assert 'close_1h' in result.columns
    assert result['close'].equals(data['close'])

    assert 'volume' in result.columns
    assert 'volume_1h' in result.columns
    assert result['volume'].equals(data['volume'])

    # First 3 rows are empty
    assert result.iloc[0]['date_1h'] is pd.NaT
    assert result.iloc[1]['date_1h'] is pd.NaT
    assert result.iloc[2]['date_1h'] is pd.NaT
    # Next 4 rows contain the starting date (0:00)
    assert result.iloc[3]['date_1h'] == result.iloc[0]['date']
    assert result.iloc[4]['date_1h'] == result.iloc[0]['date']
    assert result.iloc[5]['date_1h'] == result.iloc[0]['date']
    assert result.iloc[6]['date_1h'] == result.iloc[0]['date']
    # Next 4 rows contain the next Hourly date original date row 4
    assert result.iloc[7]['date_1h'] == result.iloc[4]['date']
    assert result.iloc[8]['date_1h'] == result.iloc[4]['date']


def test_merge_informative_pair_same():
    data = generate_test_data('15m', 40)
    informative = generate_test_data('15m', 40)

    result = merge_informative_pair(data, informative, '15m', '15m', ffill=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    assert 'date' in result.columns
    assert result['date'].equals(data['date'])
    assert 'date_15m' in result.columns

    assert 'open' in result.columns
    assert 'open_15m' in result.columns
    assert result['open'].equals(data['open'])

    assert 'close' in result.columns
    assert 'close_15m' in result.columns
    assert result['close'].equals(data['close'])

    assert 'volume' in result.columns
    assert 'volume_15m' in result.columns
    assert result['volume'].equals(data['volume'])

    # Dates match 1:1
    assert result['date_15m'].equals(result['date'])


def test_merge_informative_pair_lower():
    data = generate_test_data('1h', 40)
    informative = generate_test_data('15m', 40)

    with pytest.raises(ValueError, match=r"Tried to merge a faster timeframe .*"):
        merge_informative_pair(data, informative, '1h', '15m', ffill=True)


def test_stoploss_from_open():
    open_price_ranges = [
        [0.01, 1.00, 30],
        [1, 100, 30],
        [100, 10000, 30],
    ]
    current_profit_range = [-0.99, 2, 30]
    desired_stop_range = [-0.50, 0.50, 30]

    for open_range in open_price_ranges:
        for open_price in np.linspace(*open_range):
            for desired_stop in np.linspace(*desired_stop_range):

                # -1 is not a valid current_profit, should return 1
                assert stoploss_from_open(desired_stop, -1) == 1

                for current_profit in np.linspace(*current_profit_range):
                    current_price = open_price * (1 + current_profit)
                    expected_stop_price = open_price * (1 + desired_stop)

                    stoploss = stoploss_from_open(desired_stop, current_profit)

                    assert stoploss >= 0
                    assert stoploss <= 1

                    stop_price = current_price * (1 - stoploss)

                    # there is no correct answer if the expected stop price is above
                    # the current price
                    if expected_stop_price > current_price:
                        assert stoploss == 0
                    else:
                        assert isclose(stop_price, expected_stop_price, rel_tol=0.00001)
