from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Index, Integer, String
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


class RawFill(Base):
    """One row per raw fill exactly as a provider returned it -- payload_json preserves
    the full original payload verbatim (research.trader_mining.provider's
    trade["info"]), so normalization bugs in NormalizedFill can be investigated against
    real captured data later, per the proposal's explicit requirement."""

    __tablename__ = "raw_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(40))
    trader: Mapped[str] = mapped_column(String(120), index=True)
    # BigInteger: real Hyperliquid tid values are 15-16 digits, beyond 32-bit range --
    # only worked as plain Integer because SQLite's INTEGER affinity is dynamically
    # sized; would silently break on a fixed-width backend.
    tid: Mapped[int] = mapped_column(BigInteger, unique=True)
    payload_json: Mapped[str] = mapped_column(String)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)


class NormalizedFill(Base):
    """One row per fill, mapped to a stable internal shape independent of the upstream
    provider. tid matches a RawFill.tid -- indexed, not a formal ForeignKey, matching
    this file's existing PromotionRecord/HealthCheck convention of a plain indexed
    column rather than a hard constraint."""

    __tablename__ = "normalized_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trader: Mapped[str] = mapped_column(String(120), index=True)
    # unique=True already creates its own index -- no separate index=True needed.
    tid: Mapped[int] = mapped_column(BigInteger, unique=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    symbol: Mapped[str] = mapped_column(String(80))
    side: Mapped[str] = mapped_column(String(10))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    notional: Mapped[float] = mapped_column(Float)
    position: Mapped[float] = mapped_column(Float)
    closed_pnl: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(40))
    crossed: Mapped[bool] = mapped_column(Boolean)
    fee: Mapped[float] = mapped_column(Float)
    fee_currency: Mapped[str] = mapped_column(String(20))
    order_id: Mapped[str] = mapped_column(String(60))


class RawLedgerEvent(Base):
    """One row per ccxt fetch_ledger entry, near-verbatim -- info_json preserves the
    raw payload (matches RawFill's existing precedent), since real captured data shows
    ccxt's own unified top-level fields (amount/currency) are unreliable across event
    types (deposit/spotTransfer/spotGenesis/cStakingTransfer/send all shape their real
    delta differently -- see docs/superpowers/specs/2026-08-25-trader-mining-ledger-
    reconciliation-design.md's event type survey) and info["delta"] is the only field
    actually used at reconciliation time.

    event_id is NOT the dedup key -- Hyperliquid's cStakingTransfer events don't get a
    real transaction hash; ccxt's `id` (and info["hash"]) is an identical zero-sentinel
    for EVERY such event (confirmed against real data via code review: one wallet had 7
    cStakingTransfer events, all sharing one id). Trusting event_id alone as a unique key
    silently dropped 6 of those 7 real events on ingest, and being globally unique made a
    SECOND wallet's cStakingTransfer event crash the whole trader-import run with an
    IntegrityError. dedup_key -- derived from (trader, event_id, timestamp, event_type,
    a hash of info_json) via research.trader_mining.ingestion._dedup_key -- is what
    actually gets deduped on; it disambiguates same-event_id-but-different-content
    events since their timestamp/payload differ, while still correctly deduping a true
    re-fetch of the exact same event (identical inputs -> identical dedup_key)."""

    __tablename__ = "raw_ledger_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trader: Mapped[str] = mapped_column(String(120), index=True)
    event_id: Mapped[str] = mapped_column(String(80))
    dedup_key: Mapped[str] = mapped_column(String(64), unique=True)
    event_type: Mapped[str] = mapped_column(String(40))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    info_json: Mapped[str] = mapped_column(String)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)


class ReconstructedTrade(Base):
    """One row per logical trade, grouped from NormalizedFill rows by
    research.trader_mining.engine.reconstruct_trades -- zero-to-zero position spans, not
    an imposed lot-accounting convention. Recomputed from scratch on every
    reconstruct_and_persist_trades run, not incrementally patched."""

    __tablename__ = "reconstructed_trades"
    __table_args__ = (Index("ix_reconstructed_trades_trader_symbol", "trader", "symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trader: Mapped[str] = mapped_column(String(120))
    symbol: Mapped[str] = mapped_column(String(80))
    direction: Mapped[str] = mapped_column(String(10))
    entry_timestamp: Mapped[datetime] = mapped_column(DateTime)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_timestamp: Mapped[datetime] = mapped_column(DateTime)
    exit_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    gross_pnl: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float)
    net_pnl: Mapped[float] = mapped_column(Float)
    holding_time_seconds: Mapped[float] = mapped_column(Float)
    n_fills: Mapped[int] = mapped_column(Integer)
    is_truncated_start: Mapped[bool] = mapped_column(Boolean)
    was_liquidated: Mapped[bool] = mapped_column(Boolean)
