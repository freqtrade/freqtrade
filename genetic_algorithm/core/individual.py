"""
Individual Class

Represents a single individual in the population.
Wraps a StrategyGene with fitness and metadata.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from genetic_algorithm.core.strategy_gene import StrategyGene


@dataclass
class Individual:
    """
    An individual in the genetic algorithm population.
    
    Wraps a StrategyGene with fitness score and metadata for tracking
    performance and evolution history.
    
    Supports both single-objective (fitness) and multi-objective (objectives) modes.
    """
    
    strategy_gene: StrategyGene
    fitness: Optional[float] = None  # This is the shared_fitness used for selection
    raw_fitness: Optional[float] = None  # Original fitness before fitness sharing
    
    # Multi-objective support (NSGA-II)
    objectives: Optional[List[float]] = None  # Vector of objective values (e.g., [profit, -drawdown, sharpe])
    rank: int = 0  # Pareto front rank (1 = best front, 2 = second front, etc.)
    crowding_distance: float = 0.0  # Crowding distance for diversity preservation
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    evaluated: bool = False
    
    # Evolution history
    parent_ids: list = field(default_factory=list)
    mutations: list = field(default_factory=list)
    
    def __lt__(self, other: 'Individual') -> bool:
        """Enable sorting by fitness (higher is better)."""
        if self.fitness is None and other.fitness is None:
            return False
        if self.fitness is None:
            return True  # Unevaluated individuals go last
        if other.fitness is None:
            return False
        return self.fitness < other.fitness
    
    def __eq__(self, other: 'Individual') -> bool:
        """Check equality based on fitness."""
        if self.fitness is None or other.fitness is None:
            return False
        return abs(self.fitness - other.fitness) < 1e-6
    
    @property
    def id(self) -> str:
        """Unique identifier for this individual."""
        return f"Gen{self.strategy_gene.generation}_Ind{self.strategy_gene.individual_id}"
    
    def set_fitness(self, fitness: float, metrics: Dict[str, float]):
        """
        Set fitness and metrics for this individual.
        
        Args:
            fitness: Overall fitness score (raw fitness before sharing)
            metrics: Dictionary of performance metrics
        """
        self.raw_fitness = fitness
        self.fitness = fitness  # Initially same as raw_fitness, may be adjusted by fitness sharing
        # Preserve metadata keys set before evaluation
        _PRESERVE_KEYS = ('origin', 'llm_provider', 'island_name')
        preserved = {k: v for k, v in self.metrics.items() if k in _PRESERVE_KEYS}
        self.metrics = metrics
        self.metrics.update(preserved)
        self.evaluated = True
    
    def set_shared_fitness(self, shared_fitness: float):
        """
        Set shared fitness (after fitness sharing applied).
        
        Args:
            shared_fitness: Fitness after diversity-based sharing adjustment
        """
        self.fitness = shared_fitness
    
    def set_objectives(self, objectives: List[float], metrics: Dict[str, float]):
        """
        Set objectives for multi-objective optimization (NSGA-II).
        
        Args:
            objectives: List of objective values (all to be maximized)
            metrics: Dictionary of performance metrics
        """
        self.objectives = objectives
        # Preserve metadata keys set before evaluation
        _PRESERVE_KEYS = ('origin', 'llm_provider', 'island_name')
        preserved = {k: v for k, v in self.metrics.items() if k in _PRESERVE_KEYS}
        self.metrics = metrics
        self.metrics.update(preserved)
        self.evaluated = True
        # Also set fitness to first objective for backwards compatibility
        if objectives:
            self.fitness = objectives[0]
            self.raw_fitness = objectives[0]
    
    def nsga2_compare(self, other: 'Individual') -> int:
        """
        NSGA-II comparison: prefer lower rank, then higher crowding distance.
        
        Returns:
            1 if self is better, -1 if other is better, 0 if equal
        """
        # Lower rank is better
        if self.rank < other.rank:
            return 1
        if self.rank > other.rank:
            return -1
        # Same rank: higher crowding distance is better (more diverse)
        if self.crowding_distance > other.crowding_distance:
            return 1
        if self.crowding_distance < other.crowding_distance:
            return -1
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to dictionary for storage."""
        return {
            'id': self.id,
            'strategy_gene': self.strategy_gene.to_dict(),
            'fitness': self.fitness,
            'raw_fitness': self.raw_fitness,
            'objectives': self.objectives,
            'rank': self.rank,
            'crowding_distance': self.crowding_distance,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat(),
            'evaluated': self.evaluated,
            'parent_ids': self.parent_ids,
            'mutations': self.mutations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """Create individual from dictionary."""
        strategy_gene = StrategyGene.from_dict(data['strategy_gene'])
        
        individual = cls(
            strategy_gene=strategy_gene,
            fitness=data.get('fitness'),
            raw_fitness=data.get('raw_fitness', data.get('fitness')),  # Fall back to fitness if raw_fitness not available
            objectives=data.get('objectives'),
            rank=data.get('rank', 0),
            crowding_distance=data.get('crowding_distance', 0.0),
            metrics=data.get('metrics', {}),
            evaluated=data.get('evaluated', False),
            parent_ids=data.get('parent_ids', []),
            mutations=data.get('mutations', []),
        )
        
        if 'created_at' in data:
            individual.created_at = datetime.fromisoformat(data['created_at'])
        
        return individual
