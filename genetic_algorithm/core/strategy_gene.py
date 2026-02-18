"""
Strategy Gene Representation

This module defines how strategies are represented genetically.
Each strategy is encoded as a set of genes that can be mutated and crossed over.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random


@dataclass
class IndicatorGene:
    """Represents a single technical indicator with its parameters."""
    
    type: str  # e.g., 'RSI', 'MACD', 'BBANDS'
    parameters: Dict[str, Any]  # indicator-specific parameters
    weight: float = 1.0  # importance weight
    
    def mutate(self, mutation_rate: float, param_ranges: Dict[str, tuple]) -> 'IndicatorGene':
        """
        Mutate indicator parameters.
        
        Args:
            mutation_rate: Probability of mutation
            param_ranges: Valid parameter ranges for this indicator type
            
        Returns:
            Mutated indicator gene
        """
        # TODO: Implement parameter mutation
        pass
    
    def to_code(self) -> str:
        """Generate Python code for this indicator."""
        # TODO: Convert to FreqTrade indicator code
        pass


@dataclass
class ConditionGene:
    """Represents an entry/exit condition."""
    
    indicator: str  # Which indicator to use
    operator: str  # Comparison operator: '<', '>', 'cross_above', 'cross_below'
    threshold: float  # Threshold value
    logic: str = 'AND'  # Logic operator: 'AND', 'OR'
    
    def to_code(self) -> str:
        """Generate Python code for this condition."""
        # TODO: Convert to FreqTrade condition code
        pass


@dataclass
class StrategyGene:
    """
    Complete genetic representation of a trading strategy.
    
    This class encodes all aspects of a strategy that can be evolved:
    - Indicators used
    - Entry conditions
    - Exit conditions
    - Risk management parameters
    """
    
    # Identifiers
    generation: int
    individual_id: int
    
    # Strategy components
    indicators: List[IndicatorGene] = field(default_factory=list)
    entry_conditions: List[ConditionGene] = field(default_factory=list)
    exit_conditions: List[ConditionGene] = field(default_factory=list)
    
    # Risk management
    timeframe: str = '5m'
    stoploss: float = -0.10
    minimal_roi: Dict[str, float] = field(default_factory=lambda: {"0": 0.04, "30": 0.02, "60": 0.01})
    
    # Optional parameters
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: Optional[float] = None
    
    def __post_init__(self):
        """Validate strategy gene after initialization."""
        if not self.indicators:
            raise ValueError("Strategy must have at least one indicator")
        if not self.entry_conditions:
            raise ValueError("Strategy must have at least one entry condition")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy gene to dictionary for storage."""
        return {
            'generation': self.generation,
            'individual_id': self.individual_id,
            'indicators': [
                {'type': ind.type, 'parameters': ind.parameters, 'weight': ind.weight}
                for ind in self.indicators
            ],
            'entry_conditions': [
                {
                    'indicator': cond.indicator,
                    'operator': cond.operator,
                    'threshold': cond.threshold,
                    'logic': cond.logic
                }
                for cond in self.entry_conditions
            ],
            'exit_conditions': [
                {
                    'indicator': cond.indicator,
                    'operator': cond.operator,
                    'threshold': cond.threshold,
                    'logic': cond.logic
                }
                for cond in self.exit_conditions
            ],
            'timeframe': self.timeframe,
            'stoploss': self.stoploss,
            'minimal_roi': self.minimal_roi,
            'trailing_stop': self.trailing_stop,
            'trailing_stop_positive': self.trailing_stop_positive,
            'trailing_stop_positive_offset': self.trailing_stop_positive_offset,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyGene':
        """Create strategy gene from dictionary."""
        indicators = [
            IndicatorGene(
                type=ind['type'],
                parameters=ind['parameters'],
                weight=ind.get('weight', 1.0)
            )
            for ind in data['indicators']
        ]
        
        entry_conditions = [
            ConditionGene(
                indicator=cond['indicator'],
                operator=cond['operator'],
                threshold=cond['threshold'],
                logic=cond.get('logic', 'AND')
            )
            for cond in data['entry_conditions']
        ]
        
        exit_conditions = [
            ConditionGene(
                indicator=cond['indicator'],
                operator=cond['operator'],
                threshold=cond['threshold'],
                logic=cond.get('logic', 'AND')
            )
            for cond in data.get('exit_conditions', [])
        ]
        
        return cls(
            generation=data['generation'],
            individual_id=data['individual_id'],
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            timeframe=data.get('timeframe', '5m'),
            stoploss=data.get('stoploss', -0.10),
            minimal_roi=data.get('minimal_roi', {"0": 0.04, "30": 0.02, "60": 0.01}),
            trailing_stop=data.get('trailing_stop', False),
            trailing_stop_positive=data.get('trailing_stop_positive'),
            trailing_stop_positive_offset=data.get('trailing_stop_positive_offset'),
        )
    
    def copy(self) -> 'StrategyGene':
        """Create a deep copy of this strategy gene."""
        return StrategyGene.from_dict(self.to_dict())
    
    def get_missing_indicators(self) -> List[str]:
        """
        Find indicator types referenced in conditions but not in indicators list.
        
        Returns:
            List of missing indicator types
        """
        # Get all indicator types present
        present_types = {ind.type for ind in self.indicators}
        
        # Get all indicator types referenced in conditions
        referenced_types = set()
        for cond in self.entry_conditions + self.exit_conditions:
            referenced_types.add(cond.indicator)
        
        # Find missing types
        missing_types = referenced_types - present_types
        return list(missing_types)
    
    def ensure_indicators_for_conditions(self, indicator_config: Dict[str, Any]) -> None:
        """
        Ensure all indicators referenced in conditions are present in indicators list.
        Adds missing indicators with default parameters.
        
        Args:
            indicator_config: Configuration with indicator parameters
        """
        from genetic_algorithm.utils.indicator_factory import create_random_indicator
        
        missing_types = self.get_missing_indicators()
        
        for ind_type in missing_types:
            # Create a new indicator of this type
            new_indicator = create_random_indicator(ind_type, indicator_config)
            self.indicators.append(new_indicator)
    
    def calculate_complexity(self) -> int:
        """
        Calculate the complexity of this strategy.
        
        Complexity is measured as the sum of:
        - Number of indicators
        - Number of entry conditions
        - Number of exit conditions
        
        Returns:
            Total complexity score (higher = more complex)
        """
        return (
            len(self.indicators) +
            len(self.entry_conditions) +
            len(self.exit_conditions)
        )
