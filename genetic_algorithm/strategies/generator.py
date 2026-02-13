"""
Strategy Generator

Generates random trading strategies and converts genetic
representations to FreqTrade strategy code.
"""

import random
from typing import Dict, Any, List

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


class StrategyGenerator:
    """
    Generates trading strategies for the genetic algorithm.
    
    Responsible for:
    - Creating random strategies
    - Converting genetic representation to Python code
    - Validating strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.indicator_config = config.get('indicators', {})
        self.strategy_constraints = config.get('strategy_constraints', {})
        
        # Available components
        self.available_indicators = self.indicator_config.get('available', [])
        self.available_timeframes = self.strategy_constraints.get('timeframes', ['5m', '15m', '1h'])
    
    def generate_random_strategy(self, generation: int, individual_id: int) -> StrategyGene:
        """
        Generate a random trading strategy.
        
        Args:
            generation: Generation number
            individual_id: Individual ID
            
        Returns:
            Random StrategyGene
        """
        # Determine number of indicators
        min_indicators = self.indicator_config.get('min_per_strategy', 2)
        max_indicators = self.indicator_config.get('max_per_strategy', 5)
        num_indicators = random.randint(min_indicators, max_indicators)
        
        # Generate random indicators
        indicators = []
        selected_types = random.sample(self.available_indicators, num_indicators)
        
        for ind_type in selected_types:
            indicator = self._generate_random_indicator(ind_type)
            indicators.append(indicator)
        
        # Generate entry conditions
        entry_conditions = self._generate_random_conditions(indicators, is_entry=True)
        
        # Generate exit conditions
        exit_conditions = self._generate_random_conditions(indicators, is_entry=False)
        
        # Generate risk parameters
        stoploss_range = self.strategy_constraints.get('stoploss_range', [-0.20, -0.05])
        stoploss = random.uniform(*stoploss_range)
        
        # Generate ROI
        roi_range = self.strategy_constraints.get('roi_range', [0.01, 0.10])
        minimal_roi = {
            0: random.uniform(roi_range[0] * 2, roi_range[1]),
            30: random.uniform(roi_range[0] * 1.5, roi_range[1] * 0.7),
            60: random.uniform(roi_range[0], roi_range[1] * 0.5),
        }
        
        # Random timeframe
        timeframe = random.choice(self.available_timeframes)
        
        return StrategyGene(
            generation=generation,
            individual_id=individual_id,
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            timeframe=timeframe,
            stoploss=stoploss,
            minimal_roi=minimal_roi,
            trailing_stop=random.choice([True, False]),
        )
    
    def _generate_random_indicator(self, indicator_type: str) -> IndicatorGene:
        """Generate a random indicator with appropriate parameters."""
        # Get parameter ranges for this indicator type
        ind_config = self.indicator_config.get(indicator_type, {})
        
        parameters = {}
        
        # Generate parameters based on indicator type
        if indicator_type == 'RSI':
            period_range = ind_config.get('period', [7, 21])
            parameters['period'] = random.randint(*period_range)
        
        elif indicator_type == 'MACD':
            parameters['fast_period'] = random.randint(*ind_config.get('fast_period', [8, 21]))
            parameters['slow_period'] = random.randint(*ind_config.get('slow_period', [21, 50]))
            parameters['signal_period'] = random.randint(*ind_config.get('signal_period', [5, 14]))
        
        elif indicator_type == 'BBANDS':
            parameters['period'] = random.randint(*ind_config.get('period', [15, 30]))
            parameters['std_dev'] = random.uniform(*ind_config.get('std_dev', [1.5, 3.0]))
        
        elif indicator_type in ['EMA', 'SMA']:
            parameters['period'] = random.randint(*ind_config.get('period', [10, 50]))
        
        elif indicator_type == 'STOCH':
            parameters['k_period'] = random.randint(*ind_config.get('k_period', [5, 21]))
            parameters['d_period'] = random.randint(*ind_config.get('d_period', [3, 14]))
        
        elif indicator_type in ['ATR', 'ADX', 'CCI']:
            parameters['period'] = random.randint(*ind_config.get('period', [10, 20]))
        
        return IndicatorGene(
            type=indicator_type,
            parameters=parameters,
            weight=random.uniform(0.3, 1.0)
        )
    
    def _generate_random_conditions(self, indicators: List[IndicatorGene], 
                                   is_entry: bool) -> List[ConditionGene]:
        """Generate random entry or exit conditions."""
        conditions = []
        
        # Generate 1-3 conditions
        num_conditions = random.randint(1, min(3, len(indicators)))
        
        for _ in range(num_conditions):
            # Pick a random indicator
            indicator = random.choice(indicators)
            
            # Generate condition based on indicator type
            condition = self._generate_condition_for_indicator(indicator, is_entry)
            if condition:
                conditions.append(condition)
        
        return conditions
    
    def _generate_condition_for_indicator(self, indicator: IndicatorGene, 
                                         is_entry: bool) -> ConditionGene:
        """Generate a condition for a specific indicator."""
        ind_config = self.indicator_config.get(indicator.type, {})
        
        if indicator.type == 'RSI':
            if is_entry:
                threshold_range = ind_config.get('buy_threshold', [20, 40])
                operator = 'cross_below'
            else:
                threshold_range = ind_config.get('sell_threshold', [60, 80])
                operator = 'cross_above'
            
            return ConditionGene(
                indicator='RSI',
                operator=operator,
                threshold=random.randint(*threshold_range),
                logic=random.choice(['AND', 'OR'])
            )
        
        elif indicator.type == 'MACD':
            return ConditionGene(
                indicator='MACD',
                operator='cross_above' if is_entry else 'cross_below',
                threshold=0,
                logic=random.choice(['AND', 'OR'])
            )
        
        # TODO: Add conditions for other indicators
        
        return None
    
    def generate_strategy_code(self, strategy_gene: StrategyGene) -> str:
        """
        Convert a StrategyGene to FreqTrade Python code.
        
        Args:
            strategy_gene: Strategy genetic representation
            
        Returns:
            Python code as string
        """
        # TODO: Implement code generation
        # This will convert the genetic representation to a valid FreqTrade strategy file
        
        strategy_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
        
        code = f'''"""
Auto-generated strategy by Genetic Algorithm
Generation: {strategy_gene.generation}
Individual: {strategy_gene.individual_id}
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class {strategy_name}(IStrategy):
    """Auto-generated GA strategy"""
    
    INTERFACE_VERSION = 3
    
    # Strategy parameters
    timeframe = '{strategy_gene.timeframe}'
    stoploss = {strategy_gene.stoploss}
    minimal_roi = {strategy_gene.minimal_roi}
    trailing_stop = {strategy_gene.trailing_stop}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators"""
        # TODO: Generate indicator code
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals"""
        # TODO: Generate entry condition code
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals"""
        # TODO: Generate exit condition code
        return dataframe
'''
        
        return code
