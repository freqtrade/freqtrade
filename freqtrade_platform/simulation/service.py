"""Simulation service boundary for later dry-run or backtest orchestration."""

from __future__ import annotations

from typing import Any

from freqtrade_platform.simulation.models import SimulationContext, SimulationRun


class SimulationService:
    """Initial simulation boundary; no synchronization mechanism is implemented yet."""

    def __init__(self, adapter: Any | None = None) -> None:
        self._adapter = adapter

    def create_context(self, *, initial_equity: float, mode: str = "dry-run", **kwargs: Any) -> SimulationContext:
        return SimulationContext(
            base_account_snapshot=kwargs.get("base_account_snapshot"),
            initial_equity=initial_equity,
            mode=mode,
            metadata=kwargs.get("metadata", {}),
        )

    def create_run(self, simulation_id: str, mode: str, initial_equity: float, **kwargs: Any) -> SimulationRun:
        return SimulationRun(
            simulation_id=simulation_id,
            mode=mode,
            initial_equity=initial_equity,
            metadata=kwargs.get("metadata", {}),
        )
