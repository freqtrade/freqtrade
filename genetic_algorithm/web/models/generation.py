"""
Pydantic models for generation-level data.

Used when drilling down into a specific generation to see all individuals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IndividualSummary(BaseModel):
    """Compact individual info for generation tables."""

    id: str
    fitness: Optional[float] = None
    raw_fitness: Optional[float] = None
    rank: int = 0
    crowding_distance: float = 0.0
    evaluated: bool = False
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Convenience fields (extracted from metrics for quick display)
    profit: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    num_trades: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    complexity: Optional[int] = None
    indicators: List[str] = Field(default_factory=list)

    @classmethod
    def from_individual_dict(cls, d: dict) -> "IndividualSummary":
        """Create from an Individual.to_dict() output."""
        m = d.get("metrics", {})
        # Extract indicator names from strategy gene if available
        gene = d.get("strategy_gene", d.get("strategy_gene_dict", {}))
        indicator_names: List[str] = []
        if isinstance(gene, dict):
            for ind in gene.get("indicators", []):
                if isinstance(ind, dict):
                    # Use instance_id (e.g. "RSI_0") or type (e.g. "RSI")
                    name = ind.get("instance_id") or ind.get("type") or ind.get("name")
                    if name:
                        indicator_names.append(name)
        return cls(
            id=d.get("id", ""),
            fitness=d.get("fitness"),
            raw_fitness=d.get("raw_fitness"),
            rank=d.get("rank", 0),
            crowding_distance=d.get("crowding_distance", 0.0),
            evaluated=d.get("evaluated", False),
            metrics=m,
            profit=m.get("profit"),
            sharpe_ratio=m.get("sharpe_ratio"),
            sortino_ratio=m.get("sortino_ratio"),
            win_rate=m.get("win_rate"),
            num_trades=m.get("num_trades"),
            max_drawdown=m.get("max_drawdown"),
            profit_factor=m.get("profit_factor"),
            complexity=int(m["complexity"]) if "complexity" in m else None,
            indicators=indicator_names,
        )


class GenerationDetail(BaseModel):
    """Full generation data with all individuals."""

    run_id: str
    generation: int
    individuals: List[IndividualSummary] = Field(default_factory=list)
    stats: Optional[Dict[str, Any]] = None
