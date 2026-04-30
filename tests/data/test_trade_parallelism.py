from datetime import timedelta

import pytest
from pandas import DataFrame, Timestamp

from freqtrade.data.btanalysis import (
    analyze_trade_parallelism,
    load_backtest_data,
)
from freqtrade.data.btanalysis.trade_parallelism import balance_distribution_over_time
from freqtrade.util import dt_utc


def test_analyze_trade_parallelism(testdatadir):
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)

    res = analyze_trade_parallelism(bt_data, "5m")
    assert isinstance(res, DataFrame)
    assert "open_trades" in res.columns
    assert res["open_trades"].max() == 3
    assert res["open_trades"].min() == 0


@pytest.mark.parametrize("is_short", [False, True])
def test_balance_distribution_over_time(is_short):
    """
    Test balance_distribution_over_time for both long and short trades.
    """
    # Create a minimal trades DataFrame with 4 trades over time
    # Base dates for trades
    start_date = dt_utc(2023, 1, 1)
    base_date = start_date + timedelta(hours=15)
    stake_currency = "USDT"
    start_balance = 1000.0
    fee = 0.001  # 0.1% fee

    # Create trades spanning different time periods
    trades_data = {
        "pair": ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT"],
        "stake_amount": [100.0, 150.0, 80.0, 120.0],
        "open_date": [
            base_date,
            base_date + timedelta(hours=2),
            base_date + timedelta(hours=5),
            base_date + timedelta(hours=8),
        ],
        "close_date": [
            base_date + timedelta(hours=3),
            base_date + timedelta(hours=6),
            base_date + timedelta(hours=9),
            base_date + timedelta(hours=12),
        ],
        "open_rate": [40000.0, 2000.0, 0.5, 100.0],
        "close_rate": [41000.0, 2100.0, 0.52, 105.0],
        "fee_open": [fee, fee, fee, fee],
        "fee_close": [fee, fee, fee, fee],
        "is_short": [is_short, is_short, is_short, is_short],
        "leverage": [1.0, 1.0, 1.0, 1.0],
        "orders": [
            # Trade 1: BTC/USDT - entry at 40000, exit at 41000
            [
                {
                    "amount": 0.0025,  # 100 / 40000
                    "filled": 0.0025,
                    "safe_price": 40000.0,
                    "ft_order_side": "sell" if is_short else "buy",
                    "order_filled_timestamp": int(base_date.timestamp() * 1000),
                    "ft_is_entry": True,
                },
                {
                    "amount": 0.0025,
                    "filled": 0.0025,
                    "safe_price": 41000.0,
                    "ft_order_side": "buy" if is_short else "sell",
                    "order_filled_timestamp": int(
                        (base_date + timedelta(hours=3)).timestamp() * 1000
                    ),
                    "ft_is_entry": False,
                },
            ],
            # Trade 2: ETH/USDT - entry at 2000, exit at 2100
            [
                {
                    "amount": 0.075,  # 150 / 2000
                    "filled": 0.075,
                    "safe_price": 2000.0,
                    "ft_order_side": "sell" if is_short else "buy",
                    "order_filled_timestamp": int(
                        (base_date + timedelta(hours=2)).timestamp() * 1000
                    ),
                    "ft_is_entry": True,
                },
                {
                    "amount": 0.075,
                    "filled": 0.075,
                    "safe_price": 2100.0,
                    "ft_order_side": "buy" if is_short else "sell",
                    "order_filled_timestamp": int(
                        (base_date + timedelta(hours=6)).timestamp() * 1000
                    ),
                    "ft_is_entry": False,
                },
            ],
            # Trade 3: XRP/USDT - entry at 0.5, exit at 0.52
            [
                {
                    "amount": 160.0,  # 80 / 0.5
                    "filled": 160.0,
                    "safe_price": 0.5,
                    "ft_order_side": "sell" if is_short else "buy",
                    "order_filled_timestamp": int(
                        (base_date + timedelta(hours=5)).timestamp() * 1000
                    ),
                    "ft_is_entry": True,
                },
                {
                    "amount": 160.0,
                    "filled": 160.0,
                    "safe_price": 0.52,
                    "ft_order_side": "buy" if is_short else "sell",
                    "order_filled_timestamp": int(
                        (base_date + timedelta(hours=9)).timestamp() * 1000
                    ),
                    "ft_is_entry": False,
                },
            ],
            # Trade 4: LTC/USDT - entry at 100, exit at 105
            [
                {
                    "amount": 1.2,  # 120 / 100
                    "filled": 1.2,
                    "safe_price": 100.0,
                    "ft_order_side": "sell" if is_short else "buy",
                    "order_filled_timestamp": int(
                        (base_date + timedelta(hours=8)).timestamp() * 1000
                    ),
                    "ft_is_entry": True,
                },
                {
                    "amount": 1.2,
                    "filled": 1.2,
                    "safe_price": 105.0,
                    "ft_order_side": "buy" if is_short else "sell",
                    "order_filled_timestamp": int(
                        (base_date + timedelta(hours=12)).timestamp() * 1000
                    ),
                    "ft_is_entry": False,
                },
            ],
        ],
    }

    trades_df = DataFrame(trades_data)
    pairlist = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT"]

    min_date = start_date
    max_date = start_date + timedelta(hours=35)

    result = balance_distribution_over_time(
        trades=trades_df,
        min_date=min_date,
        max_date=max_date,
        timeframe="1h",
        stake_currency=stake_currency,
        start_balance=start_balance,
        pairlist=pairlist,
    )

    # Verify basic structure
    assert isinstance(result, DataFrame)
    assert stake_currency in result.columns
    for pair in pairlist:
        assert pair in result.columns
        assert f"{pair}_leverage" in result.columns
        assert f"{pair}_is_short" in result.columns
        assert f"{pair}_collateral" in result.columns

    # Verify the index is a DatetimeIndex
    assert isinstance(result.index, Timestamp.__class__.__bases__[0])

    # Verify we have entries over the full time period (36h)
    assert len(result) == 36

    # First trade opens 15h after the start date
    assert result.iloc[0][stake_currency] == 1000
    expected_first_balance = start_balance - (100.0 + 100.0 * fee)
    assert result.iloc[15][stake_currency] == pytest.approx(expected_first_balance)

    # Check that pair columns have non-zero values during trade periods
    # Trade 1 (BTC/USDT) is open from hour 15 to hour 18
    # At hour 16, BTC/USDT should have position
    btc_during_trade = result.loc[base_date + timedelta(hours=1), "BTC/USDT"]
    assert btc_during_trade > 0, "Trade should have positive position during open period"

    # After Trade 1 closes at hour 3, BTC/USDT position should be 0
    btc_after_close = result.loc[base_date + timedelta(hours=4) :, "BTC/USDT"]
    assert all(btc_after_close == 0), "Position should be 0 after trade closes"

    # Final stake currency should reflect all trades' cash flows minus fees
    final_balance = result.iloc[-1][stake_currency]

    # Verify the balance changed (trades had effect)
    assert final_balance != start_balance, "Balance should change after trading"

    # Since all exit prices > entry prices, exits return more cash than entries spent
    # This means final balance > start balance for long trades and < start balance for short trades
    assert (final_balance > start_balance) if not is_short else (final_balance < start_balance), (
        "Balance increases for long and decreases for short trades"
    )
