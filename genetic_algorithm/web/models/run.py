"""
Pydantic models for evolution run data.

Used in REST API responses for listing, inspecting, and controlling runs.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationStatsModel(BaseModel):
    """Per-generation summary statistics."""

    generation: int
    size: int = 0
    best_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    worst_fitness: Optional[float] = None
    median_fitness: Optional[float] = None
    best_raw_fitness: Optional[float] = None
    avg_raw_fitness: Optional[float] = None
    genetic_diversity: Optional[float] = None
    holdout_avg_degradation: Optional[float] = None
    holdout_best_degradation: Optional[float] = None
    holdout_num_evaluated: Optional[int] = None
    holdout_num_profitable: Optional[int] = None

    # Extras (from evolution loop)
    mutation_rate: Optional[float] = None
    holdout_penalties_applied: Optional[int] = None
    avg_holdout_penalty: Optional[float] = None
    avg_unused_indicators: Optional[float] = None
    eval_seconds: Optional[float] = None


class RunSummary(BaseModel):
    """Compact run info for list views."""

    run_id: str
    status: RunStatus
    config_name: str = ""
    current_generation: int = 0
    total_generations: int = 0
    best_fitness: Optional[float] = None
    best_profit: Optional[float] = None
    population_size: int = 0
    started_at: Optional[float] = None  # epoch
    elapsed_seconds: Optional[float] = None
    pairs: List[str] = Field(default_factory=list)


class RunDetail(RunSummary):
    """Full run detail with generation history."""

    config: Dict[str, Any] = Field(default_factory=dict)
    generation_stats: List[GenerationStatsModel] = Field(default_factory=list)
    best_individual_id: Optional[str] = None
    mode: str = "single_objective"


class StartRunRequest(BaseModel):
    """Request body for POST /api/runs — start a new evolution."""

    config: Dict[str, Any]  # Full GA config dict
    run_id: Optional[str] = None  # Optional custom ID
    resume_from: Optional[str] = None  # Checkpoint path to resume from


class InjectStrategyRequest(BaseModel):
    """Request body for POST /api/runs/{id}/inject."""

    strategy_gene: Dict[str, Any]
    source_description: Optional[str] = None  # e.g. "Hall of Fame #3"
