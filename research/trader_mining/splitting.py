"""Chronological TRAIN/VALIDATION/TEST/FORWARD split for a single wallet's reconstructed
trades -- Phase 6 of TRADER_WALLET_MINING_PROPOSAL.md, "the most important research
requirement." Pure and DB-free, mirroring research.trader_mining.metrics.compute_metrics'
own "pure function, no DB access" precedent.

A trade's period is a function of its own entry_timestamp and the configured
PeriodBoundaries alone -- never a statistic computed across the full trade set. See
research/regime.py's documented anti-lookahead trap for why: that classifier ranks each
window against a full-sample median computed across windows including future ones, safe
only because it's a post-hoc whole-run report. This module must not repeat that pattern.

Boundaries are compared timezone-naive throughout: a naive datetime in this codebase always
means "already UTC" (the ingestion invariant -- see research/trader_mining/ingestion.py),
which is also the only convention ReconstructedTrade timestamps are guaranteed to carry once
read back from SQLite (SQLite silently drops tzinfo on a fresh query even though fills are
written tz-aware UTC at ingestion time -- verified empirically, see the design doc). Tz-aware
input is converted to UTC before stripping, not just discarded, so a non-UTC offset still
normalizes correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from research.models import ReconstructedTrade


PERIODS: tuple[str, str, str, str] = ("TRAIN", "VALIDATION", "TEST", "FORWARD")


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    return dt.replace(tzinfo=None)


@dataclass(frozen=True)
class PeriodBoundaries:
    """The three cut points partitioning a wallet's trade history into four periods:
    TRAIN (< train_end), VALIDATION ([train_end, validation_end)), TEST
    ([validation_end, test_end)), FORWARD (>= test_end, open-ended). Normalized to naive
    UTC at construction; rejects non-strictly-increasing dates."""

    train_end: datetime
    validation_end: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_end", _to_naive_utc(self.train_end))
        object.__setattr__(self, "validation_end", _to_naive_utc(self.validation_end))
        object.__setattr__(self, "test_end", _to_naive_utc(self.test_end))
        if not (self.train_end < self.validation_end < self.test_end):
            raise ValueError(
                "PeriodBoundaries requires train_end < validation_end < test_end, got "
                f"train_end={self.train_end}, validation_end={self.validation_end}, "
                f"test_end={self.test_end}"
            )


def _period_of(ts: datetime, boundaries: PeriodBoundaries) -> str:
    ts = _to_naive_utc(ts)
    if ts < boundaries.train_end:
        return "TRAIN"
    if ts < boundaries.validation_end:
        return "VALIDATION"
    if ts < boundaries.test_end:
        return "TEST"
    return "FORWARD"


def assign_period(trade: ReconstructedTrade, boundaries: PeriodBoundaries) -> str:
    """A trade's period is decided by its entry_timestamp alone -- never exit_timestamp,
    which would let the (only-knowable-at-exit) outcome influence which research phase the
    trade is scored in."""
    return _period_of(trade.entry_timestamp, boundaries)


def straddles_boundary(trade: ReconstructedTrade, boundaries: PeriodBoundaries) -> bool:
    """True when a trade's entry and exit fall in different periods. Diagnostic only --
    never changes assign_period's result or splits the trade itself."""
    return _period_of(trade.entry_timestamp, boundaries) != _period_of(
        trade.exit_timestamp, boundaries
    )


@dataclass(frozen=True)
class SplitTrades:
    train: list[ReconstructedTrade]
    validation: list[ReconstructedTrade]
    test: list[ReconstructedTrade]
    forward: list[ReconstructedTrade]
    n_straddling: int


def split_trades(trades: list[ReconstructedTrade], boundaries: PeriodBoundaries) -> SplitTrades:
    buckets: dict[str, list[ReconstructedTrade]] = {p: [] for p in PERIODS}
    n_straddling = 0
    for t in trades:
        buckets[assign_period(t, boundaries)].append(t)
        if straddles_boundary(t, boundaries):
            n_straddling += 1
    return SplitTrades(
        train=buckets["TRAIN"],
        validation=buckets["VALIDATION"],
        test=buckets["TEST"],
        forward=buckets["FORWARD"],
        n_straddling=n_straddling,
    )
