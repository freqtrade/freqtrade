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

import math
import statistics
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

    returns = [t.net_pnl / (t.entry_price * t.quantity) for t in trades]
    median_trade_return = statistics.median(returns)

    holding_periods = [t.holding_time_seconds for t in trades]
    avg_holding_period_seconds = sum(holding_periods) / n
    median_holding_period_seconds = statistics.median(holding_periods)

    long_count = sum(1 for t in trades if t.direction == "long")
    short_count = n - long_count
    long_pct = long_count / n

    symbol_counts: dict[str, int] = {}
    for t in trades:
        symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1
    symbol_concentration = max(symbol_counts.values()) / n

    # Sequence-dependent metrics: ordered by exit_timestamp, walked once together --
    # both need the same chronological cumulative-P&L walk.
    ordered_by_exit = sorted(trades, key=lambda t: t.exit_timestamp)
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    losing_streak = 0
    longest_losing_streak = 0
    for t in ordered_by_exit:
        cumulative += t.net_pnl
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
        if t.net_pnl < 0:
            losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, losing_streak)
        else:
            losing_streak = 0

    winners_sorted = sorted((t.net_pnl for t in trades if t.net_pnl > 0), reverse=True)
    pnl_concentration_top_5 = sum(winners_sorted[:5]) / net_pnl if net_pnl != 0 else None

    if n < 2:
        trade_consistency_score = None
    else:
        stdev_r = statistics.stdev(returns)
        trade_consistency_score = (
            None if stdev_r == 0 else math.sqrt(n) * statistics.mean(returns) / stdev_r
        )

    return_to_drawdown_ratio = net_pnl / drawdown if drawdown > 0 else None

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
        median_trade_return=median_trade_return,
        avg_holding_period_seconds=avg_holding_period_seconds,
        median_holding_period_seconds=median_holding_period_seconds,
        long_count=long_count,
        short_count=short_count,
        long_pct=long_pct,
        symbol_concentration=symbol_concentration,
        max_drawdown=drawdown,
        max_losing_streak=longest_losing_streak,
        pnl_concentration_top_5=pnl_concentration_top_5,
        trade_consistency_score=trade_consistency_score,
        return_to_drawdown_ratio=return_to_drawdown_ratio,
    )


def _fmt(value: float | None, spec: str = "{:.4f}") -> str:
    return "n/a" if value is None else spec.format(value)


def format_report(metrics: WalletMetrics, trader: str) -> str:
    """Plain console/Markdown text, per the proposal review notes' explicit
    simplification (no separate reports.py module, no templating engine)."""
    lines = [
        f"## Wallet Report: {trader}",
        "",
        f"- Trades: {metrics.trade_count}",
        f"- Total volume: {_fmt(metrics.total_volume, '{:.2f}')}",
        f"- Gross P&L: {_fmt(metrics.gross_pnl, '{:.2f}')}",
        f"- Fees: {_fmt(metrics.fees, '{:.2f}')}",
        f"- Net P&L: {_fmt(metrics.net_pnl, '{:.2f}')}",
        f"- Win rate: {_fmt(metrics.win_rate, '{:.1%}')}",
        f"- Avg win: {_fmt(metrics.avg_win, '{:.2f}')}",
        f"- Avg loss: {_fmt(metrics.avg_loss, '{:.2f}')}",
        f"- Profit factor: {_fmt(metrics.profit_factor)}",
        f"- Expectancy: {_fmt(metrics.expectancy, '{:.2f}')}",
        f"- Payoff ratio: {_fmt(metrics.payoff_ratio)}",
        f"- Median trade return: {_fmt(metrics.median_trade_return, '{:.2%}')}",
        f"- Avg holding period: {_fmt(metrics.avg_holding_period_seconds, '{:.0f}')}s",
        f"- Median holding period: {_fmt(metrics.median_holding_period_seconds, '{:.0f}')}s",
        f"- Long/short: {metrics.long_count}/{metrics.short_count}",
        f"- Symbol concentration: {_fmt(metrics.symbol_concentration, '{:.1%}')}",
        f"- Max losing streak: {metrics.max_losing_streak}",
        f"- P&L concentration (top 5 winners): {_fmt(metrics.pnl_concentration_top_5, '{:.1%}')}",
        (
            f"- Closed-trade P&L drawdown: {_fmt(metrics.max_drawdown, '{:.2f}')} "
            "(absolute $, NOT mark-to-market portfolio drawdown -- overlapping/open "
            "positions aren't captured)"
        ),
        (
            f"- Return/drawdown ratio: {_fmt(metrics.return_to_drawdown_ratio)} "
            "(not annualized -- not Calmar)"
        ),
        (
            f"- Trade consistency score: {_fmt(metrics.trade_consistency_score)} "
            "(NOT Sharpe/risk-adjusted -- sqrt(N)*mean(r)/stdev(r) over "
            "return-on-notional; not a real significance test)"
        ),
    ]
    return "\n".join(lines)
