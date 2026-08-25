# research/tests/trader_mining/test_metrics.py
import math
import statistics
from datetime import UTC, datetime, timedelta

import pytest

from research.models import ReconstructedTrade
from research.trader_mining.metrics import compute_metrics


def _trade(
    net_pnl,
    gross_pnl=None,
    fees=0.0,
    entry_price=100.0,
    quantity=1.0,
    direction="long",
    symbol="BTC/USDC:USDC",
    exit_ts=datetime(2026, 1, 1, tzinfo=UTC),
    holding_seconds=3600.0,
) -> ReconstructedTrade:
    return ReconstructedTrade(
        trader="0xAAA",
        symbol=symbol,
        direction=direction,
        entry_timestamp=exit_ts,
        entry_price=entry_price,
        exit_timestamp=exit_ts,
        exit_price=entry_price,
        quantity=quantity,
        gross_pnl=gross_pnl if gross_pnl is not None else net_pnl + fees,
        fees=fees,
        net_pnl=net_pnl,
        holding_time_seconds=holding_seconds,
        n_fills=2,
        is_truncated_start=False,
        was_liquidated=False,
    )


def test_zero_trades_returns_all_undefined():
    m = compute_metrics([])

    assert m.trade_count == 0
    assert m.total_volume == 0.0
    assert m.gross_pnl == 0.0
    assert m.net_pnl == 0.0
    assert m.win_rate is None
    assert m.avg_win is None
    assert m.avg_loss is None
    assert m.profit_factor is None
    assert m.expectancy is None
    assert m.payoff_ratio is None


def test_aggregate_metrics_two_wins_one_loss():
    trades = [
        _trade(net_pnl=100.0, fees=1.0, entry_price=100.0, quantity=1.0),  # volume 100
        _trade(net_pnl=50.0, fees=0.5, entry_price=200.0, quantity=1.0),  # volume 200
        _trade(net_pnl=-30.0, fees=0.3, entry_price=100.0, quantity=1.0),  # volume 100
    ]

    m = compute_metrics(trades)

    assert m.trade_count == 3
    assert m.total_volume == 400.0  # 100 + 200 + 100
    assert m.net_pnl == 120.0  # 100 + 50 - 30
    assert m.fees == 1.8
    assert m.gross_pnl == 121.8  # net + fees, by _trade's own default
    assert m.win_rate == 2 / 3
    assert m.avg_win == 75.0  # (100 + 50) / 2
    assert m.avg_loss == -30.0  # only one loss
    assert m.profit_factor == 150.0 / 30.0  # 5.0 -- sum(wins)/abs(sum(losses))
    assert m.expectancy == 40.0  # 120 / 3
    assert m.payoff_ratio == 75.0 / 30.0  # 2.5 -- avg_win / abs(avg_loss)


def test_all_winning_has_no_avg_loss_or_profit_factor():
    trades = [_trade(net_pnl=10.0), _trade(net_pnl=20.0)]

    m = compute_metrics(trades)

    assert m.win_rate == 1.0
    assert m.avg_loss is None
    assert m.profit_factor is None  # no losses to divide by -- not infinity, not 0
    assert m.payoff_ratio is None


def test_all_losing_has_no_avg_win_or_payoff_ratio_but_profit_factor_is_zero():
    """profit_factor = gross_win / gross_loss. Zero wins makes this 0.0 -- a
    well-defined, meaningful "worst possible" value, NOT an undefined case (unlike
    zero losses, which would be an undefined division by zero)."""
    trades = [_trade(net_pnl=-10.0), _trade(net_pnl=-20.0)]

    m = compute_metrics(trades)

    assert m.win_rate == 0.0
    assert m.avg_win is None
    assert m.profit_factor == 0.0
    assert m.payoff_ratio is None


def test_breakeven_trade_counts_toward_trades_but_is_not_a_win_or_loss():
    trades = [_trade(net_pnl=10.0), _trade(net_pnl=0.0), _trade(net_pnl=-10.0)]

    m = compute_metrics(trades)

    assert m.trade_count == 3
    assert m.win_rate == 1 / 3  # only the +10 trade counts as a win
    assert m.avg_win == 10.0
    assert m.avg_loss == -10.0


def test_distribution_metrics():
    trades = [
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0, direction="long", symbol="A"),
        _trade(net_pnl=20.0, entry_price=100.0, quantity=1.0, direction="short", symbol="A"),
        _trade(net_pnl=-5.0, entry_price=100.0, quantity=1.0, direction="long", symbol="B"),
        _trade(
            net_pnl=15.0,
            entry_price=100.0,
            quantity=1.0,
            direction="long",
            symbol="A",
            holding_seconds=7200.0,
        ),
    ]
    # per-trade returns (net_pnl / (entry_price*quantity)): 0.10, 0.20, -0.05, 0.15
    # median of [0.10, 0.20, -0.05, 0.15] sorted: [-0.05, 0.10, 0.15, 0.20] -> (0.10+0.15)/2

    m = compute_metrics(trades)

    assert m.median_trade_return == statistics.median([0.10, 0.20, -0.05, 0.15])
    assert m.avg_holding_period_seconds == (3600.0 * 3 + 7200.0) / 4
    assert m.median_holding_period_seconds == statistics.median([3600.0, 3600.0, 3600.0, 7200.0])
    assert m.long_count == 3
    assert m.short_count == 1
    assert m.long_pct == 0.75
    assert m.symbol_concentration == 0.75  # "A" appears in 3 of 4 trades


def test_max_drawdown_and_losing_streak_ordered_by_exit_timestamp():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        _trade(net_pnl=100.0, exit_ts=t0),  # cumulative 100, peak 100
        _trade(net_pnl=-30.0, exit_ts=t0 + timedelta(hours=1)),  # cumulative 70, dd 30
        _trade(net_pnl=-20.0, exit_ts=t0 + timedelta(hours=2)),  # cumulative 50, dd 50
        _trade(net_pnl=60.0, exit_ts=t0 + timedelta(hours=3)),  # cumulative 110, new peak
    ]

    m = compute_metrics(trades)

    assert m.max_drawdown == 50.0  # peak 100 down to a low of 50
    assert m.max_losing_streak == 2  # the two consecutive losses


def test_single_losing_trade_produces_nonzero_drawdown():
    """max_drawdown needs no 0/1-trade special case -- a lone losing trade is itself a
    real drawdown from the implicit starting peak of 0.0."""
    m = compute_metrics([_trade(net_pnl=-40.0)])

    assert m.max_drawdown == 40.0


def test_single_winning_trade_has_zero_drawdown():
    m = compute_metrics([_trade(net_pnl=40.0)])

    assert m.max_drawdown == 0.0


def test_pnl_concentration_top_5_sums_largest_winners_over_total_net_pnl():
    trades = [_trade(net_pnl=v) for v in [50.0, 40.0, 30.0, 20.0, 10.0, 5.0, -20.0]]
    # total_net_pnl = 50+40+30+20+10+5-20 = 135
    # top 5 winners: 50+40+30+20+10 = 150

    m = compute_metrics(trades)

    assert m.pnl_concentration_top_5 == 150.0 / 135.0


def test_pnl_concentration_top_5_is_none_when_total_net_pnl_is_zero():
    trades = [_trade(net_pnl=50.0), _trade(net_pnl=-50.0)]

    m = compute_metrics(trades)

    assert m.pnl_concentration_top_5 is None


def test_trade_consistency_score_is_none_for_zero_or_one_trade():
    assert compute_metrics([]).trade_consistency_score is None
    assert compute_metrics([_trade(net_pnl=10.0)]).trade_consistency_score is None


def test_trade_consistency_score_is_none_when_all_returns_identical():
    """Zero-variance guard -- every trade has the exact same return-on-notional. A real,
    if unusual, case (e.g. a market maker collecting an identical spread every trade), not
    just a theoretical one. Must not raise ZeroDivisionError/StatisticsError."""
    trades = [
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0),
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0),
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0),
    ]

    m = compute_metrics(trades)

    assert m.trade_consistency_score is None


def test_trade_consistency_score_hand_computed():
    # returns (net_pnl / (entry_price*quantity)): 0.10, 0.20, -0.05
    trades = [
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0),
        _trade(net_pnl=20.0, entry_price=100.0, quantity=1.0),
        _trade(net_pnl=-5.0, entry_price=100.0, quantity=1.0),
    ]
    returns = [0.10, 0.20, -0.05]
    expected_mean = statistics.mean(returns)
    expected_stdev = statistics.stdev(returns)  # sample stdev, N-1 denominator
    expected = math.sqrt(3) * expected_mean / expected_stdev

    m = compute_metrics(trades)

    assert m.trade_consistency_score == pytest.approx(expected)


def test_return_to_drawdown_ratio_is_none_when_no_drawdown():
    m = compute_metrics([_trade(net_pnl=10.0)])  # single winner -- max_drawdown is 0.0

    assert m.max_drawdown == 0.0
    assert m.return_to_drawdown_ratio is None


def test_return_to_drawdown_ratio_hand_computed():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        _trade(net_pnl=100.0, exit_ts=t0),
        _trade(net_pnl=-40.0, exit_ts=t0 + timedelta(hours=1)),
    ]
    # net_pnl total = 60.0, max_drawdown = 40.0 (peak 100 down to a low of 60)

    m = compute_metrics(trades)

    assert m.max_drawdown == 40.0
    assert m.return_to_drawdown_ratio == pytest.approx(60.0 / 40.0)
