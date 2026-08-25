from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CandidateResult(Base):
    """One row per (research run, strategy, parameter-set) — survivors AND losers
    both get logged, so `family_trial_count` reflects the real search history."""

    __tablename__ = "candidate_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_stamp: Mapped[datetime] = mapped_column(DateTime)
    strategy_id: Mapped[str] = mapped_column(String(120))
    strategy_family: Mapped[str] = mapped_column(String(120), index=True)
    params_json: Mapped[str] = mapped_column(String)
    universe: Mapped[str] = mapped_column(String(200))
    timeframe: Mapped[str] = mapped_column(String(10))
    discovery_start: Mapped[str] = mapped_column(String(20))
    discovery_end: Mapped[str] = mapped_column(String(20))
    validation_start: Mapped[str | None] = mapped_column(String(20), nullable=True)
    validation_end: Mapped[str | None] = mapped_column(String(20), nullable=True)
    oos_start: Mapped[str | None] = mapped_column(String(20), nullable=True)
    oos_end: Mapped[str | None] = mapped_column(String(20), nullable=True)
    n_trials_this_run: Mapped[int] = mapped_column(Integer)
    is_sharpe: Mapped[float] = mapped_column(Float)
    oos_sharpe: Mapped[float] = mapped_column(Float)
    deflated_sharpe: Mapped[float] = mapped_column(Float)
    permutation_p: Mapped[float] = mapped_column(Float)
    pbo: Mapped[float] = mapped_column(Float)
    survived: Mapped[bool] = mapped_column(Boolean)
    evidence_json: Mapped[str] = mapped_column(String, default="{}")


class PromotionRecord(Base):
    """One row per promotion attempt for a specific passing CandidateResult -- tracks
    the Paper-Trading -> Live-eligibility lifecycle. A candidate can have many
    CandidateResult rows (re-runs, parameter sweeps); only specific passing runs ever
    get a PromotionRecord, and most never do."""

    __tablename__ = "promotion_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_result_id: Mapped[int] = mapped_column(Integer, index=True)
    state: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    paper_trading_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    paper_trading_db_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolution_reason: Mapped[str | None] = mapped_column(String, nullable=True)


class HealthCheck(Base):
    """One row per live-health evaluation of a PromotionRecord already in LIVE state.
    The record's CURRENT health state is simply the latest row for its
    promotion_record_id, ordered by evaluated_at -- deliberately not a mutable field on
    PromotionRecord, to avoid a second source of truth that could drift out of sync."""

    __tablename__ = "health_checks"
    __table_args__ = (
        Index(
            "ix_health_checks_promotion_id_evaluated_at",
            "promotion_record_id",
            "evaluated_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    promotion_record_id: Mapped[int] = mapped_column(Integer, index=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime)
    state: Mapped[str] = mapped_column(String(20))
    enough_evidence: Mapped[bool] = mapped_column(Boolean)
    n_trades: Mapped[int] = mapped_column(Integer)
    live_sharpe: Mapped[float] = mapped_column(Float)
    degradation_ratio: Mapped[float] = mapped_column(Float)
    win_rate: Mapped[float] = mapped_column(Float)
    max_drawdown: Mapped[float] = mapped_column(Float)
    reasons_json: Mapped[str] = mapped_column(String, default="[]")
