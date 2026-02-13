"""
Individual Class

Represents a single individual in the population.
Wraps a StrategyGene with fitness and metadata.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from genetic_algorithm.core.strategy_gene import StrategyGene


@dataclass
class Individual:
    """
    An individual in the genetic algorithm population.
    
    Wraps a StrategyGene with fitness score and metadata for tracking
    performance and evolution history.
    """
    
    strategy_gene: StrategyGene
    fitness: Optional[float] = None
    
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
            fitness: Overall fitness score
            metrics: Dictionary of performance metrics
        """
        self.fitness = fitness
        self.metrics = metrics
        self.evaluated = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to dictionary for storage."""
        return {
            'id': self.id,
            'strategy_gene': self.strategy_gene.to_dict(),
            'fitness': self.fitness,
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
            metrics=data.get('metrics', {}),
            evaluated=data.get('evaluated', False),
            parent_ids=data.get('parent_ids', []),
            mutations=data.get('mutations', []),
        )
        
        if 'created_at' in data:
            individual.created_at = datetime.fromisoformat(data['created_at'])
        
        return individual
