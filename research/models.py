from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
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
