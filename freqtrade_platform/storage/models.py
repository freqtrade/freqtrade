"""Platform persistence models for SQLite metadata tables."""

from __future__ import annotations

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade_platform.storage.database import PlatformBase


class PlatformProfileRecord(PlatformBase):
    """Platform profile metadata persistence record."""

    __tablename__ = "platform_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    universe_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    symbol_scope: Mapped[str | None] = mapped_column(String(512), nullable=True)
    capital_allocation: Mapped[float | None] = mapped_column(Float, nullable=True)


class PlatformStrategyRecord(PlatformBase):
    """Platform strategy metadata persistence record."""

    __tablename__ = "platform_strategies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True)


class StrategyAssignmentRecord(PlatformBase):
    """Assignment between a profile and a strategy."""

    __tablename__ = "strategy_assignments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True)


class StrategyPerformanceRecord(PlatformBase):
    """Strategy performance metadata storage record."""

    __tablename__ = "strategy_performance"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)


class AccountSnapshotRecord(PlatformBase):
    """Real account snapshot persistence record."""

    __tablename__ = "account_snapshots"

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

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    starting_balance: Mapped[float] = mapped_column(Float, default=0.0)


class CapitalAllocationRecord(PlatformBase):
    """Capital allocation persistence record."""

    __tablename__ = "capital_allocations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    allocation_percent: Mapped[float] = mapped_column(Float, nullable=False)
