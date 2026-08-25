# research/trader_mining/metrics.py
"""Performance metrics for a wallet's reconstructed trades. compute_metrics is pure and
DB-free, mirroring research.trader_mining.engine.reconstruct_trades' own "pure function,
no DB access" precedent -- research/cli.py's trader-report subcommand is the only caller
that touches a database.

trade_consistency_score is NOT a Sharpe ratio and NOT risk-adjusted -- ReconstructedTrade
carries no per-trade stop-distance/initial-risk data, so a true R-multiple-based measure
can't be computed honestly here. It's an SQN-shaped statistic over return-on-notional; see
its own docstring and docs/superpowers/specs/2026-08-25-trader-mining-release-3-design.md.
sqrt(N) gives it a t-statistic-like shape but is NOT a real significance test -- trade
outcomes can be autocorrelated or regime-dependent; real out-of-sample validation is
Release 4's job (the TRAIN/VALIDATION/TEST/FORWARD framework), not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass

from research.models import ReconstructedTrade


@dataclass
class WalletMetrics:
    trade_count: int
    total_volume: float
    gross_pnl: float
    fees: float
    net_pnl: float
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    expectancy: float | None
    payoff_ratio: float | None
    median_trade_return: float | None
    avg_holding_period_seconds: float | None
    median_holding_period_seconds: float | None
    long_count: int
    short_count: int
    long_pct: float | None
    symbol_concentration: float | None
    max_drawdown: float
    max_losing_streak: int
    pnl_concentration_top_5: float | None
    trade_consistency_score: float | None
    return_to_drawdown_ratio: float | None


def compute_metrics(trades: list[ReconstructedTrade]) -> WalletMetrics:
    n = len(trades)
    if n == 0:
        return WalletMetrics(
            trade_count=0,
            total_volume=0.0,
            gross_pnl=0.0,
            fees=0.0,
            net_pnl=0.0,
            win_rate=None,
            avg_win=None,
            avg_loss=None,
            profit_factor=None,
            expectancy=None,
            payoff_ratio=None,
            median_trade_return=None,
            avg_holding_period_seconds=None,
            median_holding_period_seconds=None,
            long_count=0,
            short_count=0,
            long_pct=None,
            symbol_concentration=None,
            max_drawdown=0.0,
            max_losing_streak=0,
            pnl_concentration_top_5=None,
            trade_consistency_score=None,
            return_to_drawdown_ratio=None,
        )

    total_volume = sum(t.entry_price * t.quantity for t in trades)
    gross_pnl = sum(t.gross_pnl for t in trades)
    fees = sum(t.fees for t in trades)
    net_pnl = sum(t.net_pnl for t in trades)

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]

    win_rate = len(wins) / n
    avg_win = (sum(t.net_pnl for t in wins) / len(wins)) if wins else None
    avg_loss = (sum(t.net_pnl for t in losses) / len(losses)) if losses else None
    profit_factor = (
        sum(t.net_pnl for t in wins) / abs(sum(t.net_pnl for t in losses)) if losses else None
    )
    expectancy = net_pnl / n
    payoff_ratio = (
        avg_win / abs(avg_loss) if (avg_win is not None and avg_loss is not None) else None
    )

    return WalletMetrics(
        trade_count=n,
        total_volume=total_volume,
        gross_pnl=gross_pnl,
        fees=fees,
        net_pnl=net_pnl,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        payoff_ratio=payoff_ratio,
        median_trade_return=None,
        avg_holding_period_seconds=None,
        median_holding_period_seconds=None,
        long_count=0,
        short_count=0,
        long_pct=None,
        symbol_concentration=None,
        max_drawdown=0.0,
        max_losing_streak=0,
        pnl_concentration_top_5=None,
        trade_consistency_score=None,
        return_to_drawdown_ratio=None,
    )
