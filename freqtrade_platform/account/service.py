"""Service boundary for explicit account state composition."""

from __future__ import annotations

from typing import Any

from freqtrade_platform.account.models import RealAccountSnapshot, SimulationAccount, SimulationBootstrap


class AccountService:
    """Provides explicit real and simulation account APIs without exchange synchronization."""

    def __init__(self, adapter: Any | None = None) -> None:
        self._adapter = adapter

    def create_real_snapshot(
        self,
        *,
        timestamp: str,
        exchange: str,
        market_type: str,
        available_balance: float,
        total_balance: float,
        equity: float,
        positions: dict[str, float] | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> RealAccountSnapshot:
        return RealAccountSnapshot(
            timestamp=timestamp,
            exchange=exchange,
            market_type=market_type,
            available_balance=available_balance,
            total_balance=total_balance,
            equity=equity,
            positions=positions or {},
            raw_source_metadata=source_metadata or {},
        )

    def create_simulation_bootstrap(
        self,
        *,
        timestamp: str,
        exchange: str,
        market_type: str,
        starting_balance: float,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationBootstrap:
        return SimulationBootstrap(
            timestamp=timestamp,
            exchange=exchange,
            market_type=market_type,
            starting_balance=starting_balance,
            metadata=metadata or {},
        )

    def create_simulation_account(
        self,
        *,
        timestamp: str,
        exchange: str,
        market_type: str,
        starting_balance: float,
        available_balance: float,
        total_balance: float,
        equity: float,
        positions: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        bootstrap: SimulationBootstrap | None = None,
    ) -> SimulationAccount:
        return SimulationAccount(
            timestamp=timestamp,
            exchange=exchange,
            market_type=market_type,
            starting_balance=starting_balance,
            available_balance=available_balance,
            total_balance=total_balance,
            equity=equity,
            positions=positions or {},
            metadata=metadata or {},
            bootstrap=bootstrap,
        )
