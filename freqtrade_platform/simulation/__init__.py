"""Simulation namespace for real-account and dry-run separation."""

from freqtrade_platform.simulation.models import SimulationContext, SimulationRun
from freqtrade_platform.simulation.service import SimulationService

__all__ = ["SimulationContext", "SimulationRun", "SimulationService"]
