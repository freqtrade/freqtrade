"""Platform persistence models for SQLite metadata tables."""

from __future__ import annotations

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade_platform.storage.database import PlatformBase


class PlatformProfileRecord(PlatformBase):
    """Platform profile metadata persistence record."""

    __tablename__ = "platform_profiles"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    universe_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    symbol_scope: Mapped[str | None] = mapped_column(String(512), nullable=True)
    primary_timeframe: Mapped[str | None] = mapped_column(String(32), nullable=True)
    informative_timeframes: Mapped[str | None] = mapped_column(String(512), nullable=True)
    assigned_strategies: Mapped[str | None] = mapped_column(String(512), nullable=True)
    regime_policy: Mapped[str | None] = mapped_column(String(128), nullable=True)
    risk_configuration: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    execution_configuration: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    capital_allocation: Mapped[float | None] = mapped_column(Float, nullable=True)


class PlatformUniverseRecord(PlatformBase):
    """Platform-owned universe metadata record."""

    __tablename__ = "platform_universes"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    universe_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    include_symbols: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    exclude_symbols: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    max_symbols: Mapped[int | None] = mapped_column(nullable=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    metadata_json: Mapped[str | None] = mapped_column(String(2048), nullable=True)


class PlatformStrategyRecord(PlatformBase):
    """Platform strategy metadata persistence record."""

    __tablename__ = "platform_strategies"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True)


class PlatformStrategySourceRecord(PlatformBase):
    """Platform strategy source code and metadata persistence record."""

    __tablename__ = "platform_strategy_sources"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_code: Mapped[str] = mapped_column(String, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    lifecycle_state: Mapped[str] = mapped_column(String(32), default="REGISTERED")
    metadata_json: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[str | None] = mapped_column(String(64), nullable=True)


class PlatformRuntimeRecord(PlatformBase):
    """Platform runtime instance persistence record."""

    __tablename__ = "platform_runtimes"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    runtime_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy_source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    workspace_path: Mapped[str] = mapped_column(String(512), nullable=False)
    process_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stopped_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(2048), nullable=True)


class StrategyAssignmentRecord(PlatformBase):
    """Assignment between a profile and a strategy."""

    __tablename__ = "strategy_assignments"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True)


class StrategyPerformanceRecord(PlatformBase):
    """Strategy performance metadata storage record."""

    __tablename__ = "strategy_performance"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)


class AccountSnapshotRecord(PlatformBase):
    """Real account snapshot persistence record."""

    __tablename__ = "account_snapshots"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    available_balance: Mapped[float] = mapped_column(Float, default=0.0)
    total_balance: Mapped[float] = mapped_column(Float, default=0.0)
    equity: Mapped[float] = mapped_column(Float, default=0.0)


class SimulationAccountRecord(PlatformBase):
    """Simulation-only account record."""

    __tablename__ = "simulation_accounts"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    starting_balance: Mapped[float] = mapped_column(Float, default=0.0)
    available_balance: Mapped[float] = mapped_column(Float, default=0.0)
    total_balance: Mapped[float] = mapped_column(Float, default=0.0)
    equity: Mapped[float] = mapped_column(Float, default=0.0)


class SimulationBootstrapRecord(PlatformBase):
    """Bootstrap for simulation initialization state."""

    __tablename__ = "simulation_bootstraps"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    starting_balance: Mapped[float] = mapped_column(Float, default=0.0)


class CapitalAllocationRecord(PlatformBase):
    """Capital allocation persistence record."""

    __tablename__ = "capital_allocations"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    allocation_percent: Mapped[float] = mapped_column(Float, nullable=False)
