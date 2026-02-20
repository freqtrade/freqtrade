"""
Strategy Gene Representation

This module defines how strategies are represented genetically.
Each strategy is encoded as a set of genes that can be mutated and crossed over.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random


# Timeframe ordering for comparison (lower index = shorter timeframe)
TIMEFRAME_ORDER = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']


def timeframe_to_minutes(tf: str) -> int:
    """Convert a timeframe string to minutes for comparison."""
    _tf_map = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
        '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200,
    }
    return _tf_map.get(tf, 0)


def is_higher_timeframe(candidate: str, base: str) -> bool:
    """Check if candidate timeframe is strictly higher than base timeframe."""
    return timeframe_to_minutes(candidate) > timeframe_to_minutes(base)


@dataclass
class IndicatorGene:
    """Represents a single technical indicator with its parameters."""
    
    type: str  # e.g., 'RSI', 'MACD', 'BBANDS'
    parameters: Dict[str, Any]  # indicator-specific parameters
    weight: float = 1.0  # importance weight
    instance_id: Optional[str] = None  # Unique instance identifier (e.g., 'RSI_0', 'RSI_1')
    timeframe: Optional[str] = None  # None = base timeframe, e.g. '1h', '4h' for informative


@dataclass
class ConditionGene:
    """Represents an entry/exit condition."""
    
    indicator: str  # Which indicator to use (can be instance_id like 'RSI_0' or type like 'RSI')
    operator: str  # Comparison operator: '<', '>', 'cross_above', 'cross_below'
    threshold: float  # Threshold value
    logic: str = 'AND'  # Logic operator: 'AND', 'OR'


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
    max_open_trades: int = 3  # Maximum number of concurrent open trades
    
    # Multi-timeframe
    informative_timeframes: List[str] = field(default_factory=list)  # e.g. ['1h', '4h']
    
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
                {'type': ind.type, 'parameters': dict(ind.parameters), 'weight': ind.weight,
                 'instance_id': ind.instance_id, 'timeframe': ind.timeframe}
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
            'informative_timeframes': self.informative_timeframes,
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
                weight=ind.get('weight', 1.0),
                instance_id=ind.get('instance_id'),
                timeframe=ind.get('timeframe')
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
            informative_timeframes=data.get('informative_timeframes', []),
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
        Find indicator references in conditions that are not in indicators list.
        Now handles both instance_ids (e.g., 'RSI_0') and type names (e.g., 'RSI').
        
        Returns:
            List of missing indicator references
        """
        # Get all indicator instance_ids and types present
        present_refs = set()
        for ind in self.indicators:
            if ind.instance_id:
                present_refs.add(ind.instance_id)
            present_refs.add(ind.type)
        
        # Get all indicator references in conditions
        referenced_refs = set()
        for cond in self.entry_conditions + self.exit_conditions:
            referenced_refs.add(cond.indicator)
        
        # Find missing references
        missing_refs = referenced_refs - present_refs
        return list(missing_refs)
    
    def ensure_indicators_for_conditions(self, indicator_config: Dict[str, Any]) -> None:
        """
        Ensure all indicators referenced in conditions are present in indicators list.
        Adds missing indicators with default parameters.
        
        Args:
            indicator_config: Configuration with indicator parameters
        """
        from genetic_algorithm.utils.indicator_factory import create_random_indicator
        
        missing_types = self.get_missing_indicators()
        
        for ind_ref in missing_types:
            # Extract base type from instance_id format (e.g., 'RSI_0' -> 'RSI')
            base_type = ind_ref.split('_')[0] if '_' in ind_ref else ind_ref
            new_indicator = create_random_indicator(base_type, indicator_config)
            self.indicators.append(new_indicator)
    
    def assign_instance_ids(self) -> None:
        """
        Assign unique instance IDs to all indicators.
        
        Creates IDs in the format: {type}_{index} for base timeframe indicators
        or {type}_{timeframe}_{index} for informative timeframe indicators.
        E.g., RSI_0, RSI_1, RSI_1h_0, EMA_4h_0, etc.
        
        Also updates condition references if they currently use type names
        to use the new instance IDs.
        """
        # Count instances of each (type, timeframe) combination
        type_tf_counts: Dict[str, int] = {}
        
        def _make_key(ind_type: str, tf: Optional[str]) -> str:
            return f"{ind_type}_{tf}" if tf else ind_type
        
        # Assign instance IDs to indicators
        for ind in self.indicators:
            key = _make_key(ind.type, ind.timeframe)
            if key not in type_tf_counts:
                type_tf_counts[key] = 0
            
            # Assign instance ID if not already set
            if not ind.instance_id:
                if ind.timeframe:
                    ind.instance_id = f"{ind.type}_{ind.timeframe}_{type_tf_counts[key]}"
                else:
                    ind.instance_id = f"{ind.type}_{type_tf_counts[key]}"
                type_tf_counts[key] += 1
            else:
                type_tf_counts[key] += 1
        
        # Create mapping from type to instance IDs
        type_to_instances: Dict[str, List[str]] = {}
        for ind in self.indicators:
            if ind.type not in type_to_instances:
                type_to_instances[ind.type] = []
            type_to_instances[ind.type].append(ind.instance_id)
        
        # Update condition references: if a condition references a type name
        # and there's only one instance of that type, update it to use the instance_id
        for cond in self.entry_conditions + self.exit_conditions:
            # If condition references a type name directly
            if cond.indicator in type_to_instances:
                instances = type_to_instances[cond.indicator]
                # If there's only one instance, use it; otherwise keep the type reference
                if len(instances) == 1:
                    cond.indicator = instances[0]
                # If there are multiple instances and we're using type reference,
                # default to the first instance for backward compatibility.
                # Note: This is a pragmatic choice. In the future, we could:
                # - Keep type references as-is and let strategy code handle them
                # - Use heuristics to pick the "best" instance based on parameters
                # - Require explicit instance references in all conditions
                elif len(instances) > 1:
                    cond.indicator = instances[0]
    
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
    
    def get_base_indicators(self) -> List['IndicatorGene']:
        """Return indicators on the base timeframe (timeframe is None)."""
        return [ind for ind in self.indicators if ind.timeframe is None]
    
    def get_informative_indicators(self) -> List['IndicatorGene']:
        """Return indicators on informative (higher) timeframes."""
        return [ind for ind in self.indicators if ind.timeframe is not None]
    
    def get_indicators_by_timeframe(self, tf: Optional[str] = None) -> List['IndicatorGene']:
        """Return indicators for a specific timeframe (None = base)."""
        return [ind for ind in self.indicators if ind.timeframe == tf]
