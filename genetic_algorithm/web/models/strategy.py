"""
Pydantic models for detailed strategy inspection.

Used when clicking into a specific strategy to see its gene, metrics,
quality assessment, and lineage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IndicatorModel(BaseModel):
    type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
    instance_id: Optional[str] = None
    timeframe: Optional[str] = None


class ConditionModel(BaseModel):
    indicator: str
    operator: str
    threshold: Any = None
    logic: str = "AND"
    threshold_upper: Any = None
    lookback: Optional[int] = None


class StrategyGeneModel(BaseModel):
    """Full strategy gene representation."""

    generation: int = 0
    individual_id: int = 0
    indicators: List[IndicatorModel] = Field(default_factory=list)
    entry_conditions: List[ConditionModel] = Field(default_factory=list)
    exit_conditions: List[ConditionModel] = Field(default_factory=list)
    timeframe: str = "5m"
    stoploss: float = -0.1
    minimal_roi: Dict[str, float] = Field(default_factory=dict)
    max_open_trades: int = 3
    informative_timeframes: List[str] = Field(default_factory=list)
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: Optional[float] = None
    can_short: bool = False


class QualityAssessment(BaseModel):
    """Overfitting / robustness assessment for a strategy."""

    holdout_degradation: Optional[float] = None
    holdout_label: str = "UNKNOWN"
    wf_gap: Optional[float] = None
    wf_label: str = "UNKNOWN"
    mc_robustness: Optional[float] = None
    mc_label: str = "UNKNOWN"
    composite_score: Optional[float] = None
    overall_label: str = "UNKNOWN"


class StrategyDetail(BaseModel):
    """Full strategy inspection view — gene + metrics + quality."""

    id: str
    run_id: str
    generation: int = 0
    fitness: Optional[float] = None
    raw_fitness: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    gene: Optional[StrategyGeneModel] = None
    quality: Optional[QualityAssessment] = None

    # Lineage
    parent_ids: List[str] = Field(default_factory=list)
    mutations: List[str] = Field(default_factory=list)

    # Walk-forward per-window breakdown (if available)
    walk_forward_windows: Optional[List[Dict[str, Any]]] = None

    # Monte Carlo results (if available)
    monte_carlo: Optional[Dict[str, Any]] = None


class BacktestRequest(BaseModel):
    """Request body for on-demand backtest."""

    strategy_gene: Dict[str, Any]
    timerange: str  # e.g. "20250101-20250301"
    pairs: List[str] = Field(default_factory=list)
    timeframe: str = "5m"
    stake_amount: float = 100.0
    exchange: str = "binance"


class BacktestResultModel(BaseModel):
    """Result of an on-demand backtest."""

    backtest_id: str
    status: str = "pending"  # pending / running / completed / failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LineageNode(BaseModel):
    """A node in a strategy's ancestral chain."""

    id: str
    generation: int
    fitness: Optional[float] = None
    raw_fitness: Optional[float] = None
    profit: Optional[float] = None
    parent_ids: List[str] = Field(default_factory=list)
    mutations: List[Any] = Field(default_factory=list)


class LineageResponse(BaseModel):
    """Response for strategy lineage tracing."""

    strategy_id: str
    run_id: str
    chain: List[LineageNode] = Field(default_factory=list)


class DryRunRequest(BaseModel):
    """Request body for launching a dry-run trading session."""

    strategy_gene: Dict[str, Any]
    exchange: str = "binance"
    pairs: List[str] = Field(default_factory=list)
    stake_amount: float = 100.0
    timeframe: str = "5m"


class DryRunStatus(BaseModel):
    """Status of a dry-run trading session."""

    dry_run_id: str
    status: str = "starting"  # starting / running / stopped / failed
    strategy_name: str = ""
    pid: Optional[int] = None
    started_at: Optional[float] = None
    error: Optional[str] = None
    log_tail: List[str] = Field(default_factory=list)
