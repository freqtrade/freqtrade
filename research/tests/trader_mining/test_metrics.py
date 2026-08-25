# research/tests/trader_mining/test_metrics.py
from datetime import UTC, datetime

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
