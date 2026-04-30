from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pandas import DataFrame, DateOffset, Timestamp, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.data.btanalysis import (
    load_backtest_data,
)
from freqtrade.data.history import load_data, load_pair_history
from freqtrade.data.metrics import (
    calculate_cagr,
    calculate_calmar,
    calculate_calmar_from_balance,
    calculate_csum,
    calculate_expectancy,
    calculate_market_change,
    calculate_max_drawdown,
    calculate_max_drawdown_from_balance,
    calculate_sharpe,
    calculate_sharpe_from_balance,
    calculate_sortino,
    calculate_sortino_from_balance,
    calculate_sqn,
    calculate_underwater,
    combine_dataframes_with_mean,
    combined_dataframes_with_rel_mean,
    create_cum_profit,
)
from freqtrade.util import dt_utc


def test_calculate_market_change(testdatadir):
    pairs = ["ETH/BTC", "ADA/BTC"]
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe="5m")
    result = calculate_market_change(data)
    assert isinstance(result, float)
    assert pytest.approx(result) == 0.01100002

    result = calculate_market_change(data, min_date=dt_utc(2018, 1, 20))
    assert isinstance(result, float)
    assert pytest.approx(result) == 0.0375149

    # Move min-date after the last date
    result = calculate_market_change(data, min_date=dt_utc(2018, 2, 20))
    assert pytest.approx(result) == 0.0


def test_combine_dataframes_with_mean(testdatadir):
    pairs = ["ETH/BTC", "ADA/BTC"]
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe="5m")
    df = combine_dataframes_with_mean(data)
    assert isinstance(df, DataFrame)
    assert "ETH/BTC" in df.columns
    assert "ADA/BTC" in df.columns
    assert "mean" in df.columns


def test_combined_dataframes_with_rel_mean(testdatadir):
    pairs = ["BTC/USDT", "XRP/USDT"]
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe="5m")
    df = combined_dataframes_with_rel_mean(
        data,
        fromdt=data["BTC/USDT"].at[0, "date"],
        todt=data["BTC/USDT"].at[data["BTC/USDT"].index[-1], "date"],
    )
    assert isinstance(df, DataFrame)
    assert "BTC/USDT" not in df.columns
    assert "XRP/USDT" not in df.columns
    assert "mean" in df.columns
    assert "rel_mean" in df.columns
    assert "count" in df.columns
    assert df.iloc[0]["count"] == 2
    assert df.iloc[-1]["count"] == 2
    assert len(df) < len(data["BTC/USDT"])
    assert df["rel_mean"].between(-0.5, 0.5).all()


def test_combine_dataframes_with_mean_no_data(testdatadir):
    pairs = ["ETH/BTC", "ADA/BTC"]
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe="6m")
    with pytest.raises(ValueError, match=r"No data provided\."):
        combine_dataframes_with_mean(data)


def test_create_cum_profit(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange("20180110-20180112")

    df = load_pair_history(pair="TRX/BTC", timeframe="5m", datadir=testdatadir, timerange=timerange)

    cum_profits = create_cum_profit(
        df.set_index("date"), bt_data[bt_data["pair"] == "TRX/BTC"], "cum_profits", timeframe="5m"
    )
    assert "cum_profits" in cum_profits.columns
    assert cum_profits.iloc[0]["cum_profits"] == 0
    assert pytest.approx(cum_profits.iloc[-1]["cum_profits"]) == 9.0225563e-05


def test_create_cum_profit1(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    # Move close-time to "off" the candle, to make sure the logic still works
    bt_data["close_date"] = bt_data.loc[:, "close_date"] + DateOffset(seconds=20)
    timerange = TimeRange.parse_timerange("20180110-20180112")

    df = load_pair_history(pair="TRX/BTC", timeframe="5m", datadir=testdatadir, timerange=timerange)

    cum_profits = create_cum_profit(
        df.set_index("date"), bt_data[bt_data["pair"] == "TRX/BTC"], "cum_profits", timeframe="5m"
    )
    assert "cum_profits" in cum_profits.columns
    assert cum_profits.iloc[0]["cum_profits"] == 0
    assert pytest.approx(cum_profits.iloc[-1]["cum_profits"]) == 9.0225563e-05

    with pytest.raises(ValueError, match=r"Trade dataframe empty\."):
        create_cum_profit(
            df.set_index("date"),
            bt_data[bt_data["pair"] == "NOTAPAIR"],
            "cum_profits",
            timeframe="5m",
        )


def test_calculate_max_drawdown(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    drawdown = calculate_max_drawdown(bt_data, value_col="profit_abs")
    assert isinstance(drawdown.relative_account_drawdown, float)
    assert pytest.approx(drawdown.relative_account_drawdown) == 0.29753914
    assert isinstance(drawdown.high_date, Timestamp)
    assert isinstance(drawdown.low_date, Timestamp)
    assert isinstance(drawdown.high_value, float)
    assert isinstance(drawdown.low_value, float)
    assert drawdown.high_date == Timestamp("2018-01-16 19:30:00", tz="UTC")
    assert drawdown.low_date == Timestamp("2018-01-16 22:25:00", tz="UTC")

    underwater = calculate_underwater(bt_data)
    assert isinstance(underwater, DataFrame)

    with pytest.raises(ValueError, match=r"Trade dataframe empty\."):
        calculate_max_drawdown(DataFrame())

    with pytest.raises(ValueError, match=r"Trade dataframe empty\."):
        calculate_underwater(DataFrame())


def test_calculate_max_drawdown_from_balance():
    balance_history = DataFrame(
        {
            "date": to_datetime(
                [
                    "2025-01-01 00:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 18:00:00+00:00",
                    "2025-01-04 00:00:00+00:00",
                ],
                utc=True,
            ),
            "total_quote": [100.0, 120.0, 80.0, 110.0],
        }
    )

    drawdown = calculate_max_drawdown_from_balance(balance_history)
    assert isinstance(drawdown.relative_account_drawdown, float)
    assert pytest.approx(drawdown.relative_account_drawdown) == 1 / 3
    assert pytest.approx(drawdown.drawdown_abs) == 40
    assert pytest.approx(drawdown.current_high_value) == 20
    assert pytest.approx(drawdown.low_value) == -20
    assert pytest.approx(drawdown.high_value) == 20

    assert drawdown.high_date == Timestamp("2025-01-01 12:00:00", tz="UTC")
    assert drawdown.low_date == Timestamp("2025-01-01 18:00:00", tz="UTC")


def test_calculate_max_drawdown_from_balance_empty_or_short():
    with pytest.raises(ValueError, match=r"Balance-history dataframe empty\."):
        calculate_max_drawdown_from_balance(DataFrame())

    one_point = DataFrame(
        {
            "date": to_datetime(["2025-01-01 00:00:00+00:00"], utc=True),
            "total_quote": [100.0],
        }
    )
    with pytest.raises(ValueError, match=r"Balance-history dataframe empty\."):
        calculate_max_drawdown_from_balance(one_point)


def test_calculate_csum(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    csum_min, csum_max = calculate_csum(bt_data)

    assert isinstance(csum_min, float)
    assert isinstance(csum_max, float)
    assert csum_min < csum_max
    assert csum_min < 0.0001
    assert csum_max > 0.0002
    csum_min1, csum_max1 = calculate_csum(bt_data, 5)

    assert csum_min1 == csum_min + 5
    assert csum_max1 == csum_max + 5

    with pytest.raises(ValueError, match=r"Trade dataframe empty\."):
        csum_min, csum_max = calculate_csum(DataFrame())


def test_calculate_expectancy(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)

    expectancy, expectancy_ratio = calculate_expectancy(DataFrame())
    assert expectancy == 0.0
    assert expectancy_ratio == 100

    expectancy, expectancy_ratio = calculate_expectancy(bt_data)
    assert isinstance(expectancy, float)
    assert isinstance(expectancy_ratio, float)
    assert pytest.approx(expectancy) == 5.820687070932315e-06
    assert pytest.approx(expectancy_ratio) == 0.07151374226574791

    data = {"profit_abs": [100, 200, 50, -150, 300, -100, 80, -30]}
    df = DataFrame(data)
    expectancy, expectancy_ratio = calculate_expectancy(df)

    assert pytest.approx(expectancy) == 56.25
    assert pytest.approx(expectancy_ratio) == 0.60267857


def test_calculate_sortino(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)

    sortino = calculate_sortino(DataFrame(), None, None, 0)
    assert sortino == 0.0

    sortino = calculate_sortino(
        bt_data,
        bt_data["open_date"].min(),
        bt_data["close_date"].max(),
        0.01,
    )
    assert isinstance(sortino, float)
    assert pytest.approx(sortino) == 35.17722


def test_calculate_sortino_from_balance():
    balance_history = DataFrame(
        {
            "date": to_datetime(
                [
                    "2025-01-01 00:00:00+00:00",
                    "2025-01-02 00:00:00+00:00",
                    "2025-01-03 00:00:00+00:00",
                    "2025-01-04 00:00:00+00:00",
                    "2025-01-05 00:00:00+00:00",
                ],
                utc=True,
            ),
            "total_quote": [100.0, 110.0, 104.5, 125.4, 112.86],
        }
    )

    sortino = calculate_sortino_from_balance(balance_history)
    expected_returns = np.array([0.1, -0.05, 0.2, -0.1])
    expected_sortino = expected_returns.mean() / np.std(expected_returns[expected_returns < 0])
    expected_sortino *= np.sqrt(365)

    assert isinstance(sortino, float)
    assert pytest.approx(sortino) == expected_sortino
    # Explicit assert
    assert pytest.approx(sortino) == 28.6574597


def test_calculate_sortino_from_balance_empty_or_no_downside():
    assert calculate_sortino_from_balance(DataFrame()) == 0.0

    positive_balance_history = DataFrame(
        {
            "date": to_datetime(
                [
                    "2025-01-01 00:00:00+00:00",
                    "2025-01-02 00:00:00+00:00",
                    "2025-01-03 00:00:00+00:00",
                ],
                utc=True,
            ),
            "total_quote": [100.0, 110.0, 121.0],
        }
    )
    assert calculate_sortino_from_balance(positive_balance_history) == -100


def test_calculate_sharpe(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)

    sharpe = calculate_sharpe(DataFrame(), None, None, 0)
    assert sharpe == 0.0

    sharpe = calculate_sharpe(
        bt_data,
        bt_data["open_date"].min(),
        bt_data["close_date"].max(),
        0.01,
    )
    assert isinstance(sharpe, float)
    assert pytest.approx(sharpe) == 44.5078669


def test_calculate_sharpe_from_balance():
    balance_history = DataFrame(
        {
            "date": to_datetime(
                [
                    "2025-01-01 00:00:00+00:00",
                    "2025-01-02 00:00:00+00:00",
                    "2025-01-03 00:00:00+00:00",
                    "2025-01-04 00:00:00+00:00",
                ],
                utc=True,
            ),
            "total_quote": [100.0, 110.0, 104.5, 125.4],
        }
    )

    sharpe = calculate_sharpe_from_balance(balance_history)
    expected_returns = np.array([0.1, -0.05, 0.2])
    expected_sharpe = expected_returns.mean() / expected_returns.std() * np.sqrt(365)

    assert isinstance(sharpe, float)
    assert pytest.approx(sharpe) == expected_sharpe


def test_calculate_sharpe_from_balance_empty_or_flat():
    assert calculate_sharpe_from_balance(DataFrame()) == 0.0

    flat_balance_history = DataFrame(
        {
            "date": to_datetime(
                ["2025-01-01 00:00:00+00:00", "2025-01-02 00:00:00+00:00"],
                utc=True,
            ),
            "total_quote": [100.0, 100.0],
        }
    )
    assert calculate_sharpe_from_balance(flat_balance_history) == -100


def test_calculate_calmar(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)

    calmar = calculate_calmar(DataFrame(), None, None, 0)
    assert calmar == 0.0

    calmar = calculate_calmar(
        bt_data,
        bt_data["open_date"].min(),
        bt_data["close_date"].max(),
        0.01,
    )
    assert isinstance(calmar, float)
    assert pytest.approx(calmar) == 559.040508


def test_calculate_calmar_from_balance():
    balance_history = DataFrame(
        {
            "date": to_datetime(
                [
                    "2025-01-01 00:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 18:00:00+00:00",
                    "2025-01-04 00:00:00+00:00",
                ],
                utc=True,
            ),
            "total_quote": [100.0, 120.0, 80.0, 110.0],
        }
    )

    calmar = calculate_calmar_from_balance(balance_history)
    expected_returns_mean = ((110.0 - 100.0) / 100.0) / 3 * 100
    expected_calmar = expected_returns_mean / (1 / 3) * np.sqrt(365)

    assert isinstance(calmar, float)
    assert pytest.approx(calmar) == expected_calmar


def test_calculate_calmar_from_balance_empty_or_flat():
    assert calculate_calmar_from_balance(DataFrame()) == 0.0

    flat_balance_history = DataFrame(
        {
            "date": to_datetime(
                ["2025-01-01 00:00:00+00:00", "2025-01-02 00:00:00+00:00"],
                utc=True,
            ),
            "total_quote": [100.0, 100.0],
        }
    )
    assert calculate_calmar_from_balance(flat_balance_history) == -100


def test_calculate_sqn(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)

    sqn = calculate_sqn(DataFrame(), 0)
    assert sqn == 0.0

    sqn = calculate_sqn(
        bt_data,
        0.01,
    )
    assert isinstance(sqn, float)
    assert pytest.approx(sqn) == 3.2991


@pytest.mark.parametrize(
    "profits,starting_balance,expected_sqn,description",
    [
        ([1.0, -0.5, 2.0, -1.0, 0.5, 1.5, -0.5, 1.0], 100, 1.3229, "Mixed profits/losses"),
        ([], 100, 0.0, "Empty dataframe"),
        ([1.0, 0.5, 2.0, 1.5, 0.8], 100, 4.3657, "All winning trades"),
        ([-1.0, -0.5, -2.0, -1.5, -0.8], 100, -4.3657, "All losing trades"),
        ([1.0], 100, -100, "Single trade"),
    ],
)
def test_calculate_sqn_cases(profits, starting_balance, expected_sqn, description):
    """
    Test SQN calculation with various scenarios:
    """
    trades = DataFrame({"profit_abs": profits})
    sqn = calculate_sqn(trades, starting_balance=starting_balance)

    assert isinstance(sqn, float)
    assert pytest.approx(sqn, rel=1e-4) == expected_sqn


@pytest.mark.parametrize(
    "start,end,days, expected",
    [
        (64900, 176000, 3 * 365, 0.3945),
        (64900, 176000, 365, 1.7119),
        (1000, 1000, 365, 0.0),
        (1000, 1500, 365, 0.5),
        (1000, 1500, 100, 3.3927),  # sub year
        (0.01000000, 0.01762792, 120, 4.6087),  # sub year BTC values
        (1000, 1010, 0, 0.0),  # zero days
        (-100, 100, 365, 0.0),  # negative starting balance
    ],
)
def test_calculate_cagr(start, end, days, expected):
    assert round(calculate_cagr(days, start, end), 4) == expected


def test_calculate_max_drawdown2():
    values = [
        0.011580,
        0.010048,
        0.011340,
        0.012161,
        0.010416,
        0.010009,
        0.020024,
        -0.024662,
        -0.022350,
        0.020496,
        -0.029859,
        -0.030511,
        0.010041,
        0.010872,
        -0.025782,
        0.010400,
        0.012374,
        0.012467,
        0.114741,
        0.010303,
        0.010088,
        -0.033961,
        0.010680,
        0.010886,
        -0.029274,
        0.011178,
        0.010693,
        0.010711,
    ]

    dates = [dt_utc(2020, 1, 1) + timedelta(days=i) for i in range(len(values))]
    df = DataFrame(zip(values, dates, strict=False), columns=["profit", "open_date"])
    # sort by profit and reset index
    df = df.sort_values("profit").reset_index(drop=True)
    df1 = df.copy()
    drawdown = calculate_max_drawdown(
        df, date_col="open_date", starting_balance=0.2, value_col="profit"
    )
    # Ensure df has not been altered.
    assert df.equals(df1)

    assert isinstance(drawdown.drawdown_abs, float)
    assert isinstance(drawdown.relative_account_drawdown, float)
    # High must be before low
    assert drawdown.high_date < drawdown.low_date
    # High value must be higher than low value
    assert drawdown.high_value > drawdown.low_value
    assert drawdown.drawdown_abs == 0.091755
    assert pytest.approx(drawdown.relative_account_drawdown) == 0.32129575

    df = DataFrame(zip(values[:5], dates[:5], strict=False), columns=["profit", "open_date"])
    # No losing trade ...
    drawdown = calculate_max_drawdown(df, date_col="open_date", value_col="profit")
    assert drawdown.drawdown_abs == 0.0
    assert drawdown.low_value == 0.0
    assert drawdown.current_high_value >= 0.0
    assert drawdown.current_drawdown_abs == 0.0

    df1 = DataFrame(zip(values[:5], dates[:5], strict=False), columns=["profit", "open_date"])
    df1.loc[:, "profit"] = df1["profit"] * -1
    # No winning trade ...
    drawdown = calculate_max_drawdown(df1, date_col="open_date", value_col="profit")
    assert drawdown.drawdown_abs == 0.055545
    assert drawdown.high_value == 0.0
    assert drawdown.current_high_value == 0.0
    assert drawdown.current_drawdown_abs == 0.055545


@pytest.mark.parametrize(
    "profits,relative,highd,lowdays,result,result_rel",
    [
        ([0.0, -500.0, 500.0, 10000.0, -1000.0], False, 3, 4, 1000.0, 0.090909),
        ([0.0, -500.0, 500.0, 10000.0, -1000.0], True, 0, 1, 500.0, 0.5),
    ],
)
def test_calculate_max_drawdown_abs(profits, relative, highd, lowdays, result, result_rel):
    """
    Test case from issue https://github.com/freqtrade/freqtrade/issues/6655
    [1000, 500,  1000, 11000, 10000] # absolute results
    [1000, 50%,  0%,   0%,       ~9%]   # Relative drawdowns
    """
    init_date = datetime(2020, 1, 1, tzinfo=UTC)
    dates = [init_date + timedelta(days=i) for i in range(len(profits))]
    df = DataFrame(zip(profits, dates, strict=False), columns=["profit_abs", "open_date"])
    # sort by profit and reset index
    df = df.sort_values("profit_abs").reset_index(drop=True)
    df1 = df.copy()
    drawdown = calculate_max_drawdown(
        df, date_col="open_date", starting_balance=1000, relative=relative
    )
    # Ensure df has not been altered.
    assert df.equals(df1)

    assert isinstance(drawdown.drawdown_abs, float)
    assert isinstance(drawdown.relative_account_drawdown, float)
    assert drawdown.high_date == init_date + timedelta(days=highd)
    assert drawdown.low_date == init_date + timedelta(days=lowdays)

    # High must be before low
    assert drawdown.high_date < drawdown.low_date
    # High value must be higher than low value
    assert drawdown.high_value > drawdown.low_value
    assert drawdown.drawdown_abs == result
    assert pytest.approx(drawdown.relative_account_drawdown) == result_rel
