"""Simulation domain models and boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SimulationContext:
    """Defines the boundary between real account state and simulated equity."""

    base_account_snapshot: Any | None = None
    initial_equity: float | None = None
    mode: str = "dry-run"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationRun:
    """A future simulation run linked to later Freqtrade backtesting or dry run execution."""

    simulation_id: str
    mode: str
    initial_equity: float
    metadata: dict[str, object] = field(default_factory=dict)
