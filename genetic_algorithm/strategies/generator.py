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
        
        # Generate ROI (keys must be strings for FreqTrade config validation)
        roi_range = self.strategy_constraints.get('roi_range', [0.01, 0.10])
        minimal_roi = {
            "0": random.uniform(roi_range[0] * 2, roi_range[1]),
            "30": random.uniform(roi_range[0] * 1.5, roi_range[1] * 0.7),
            "60": random.uniform(roi_range[0], roi_range[1] * 0.5),
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
        
        # Filter indicators that can generate conditions
        valid_indicators = [ind for ind in indicators 
                          if ind.type in ['RSI', 'MACD', 'STOCH', 'CCI', 'ADX']]
        
        if not valid_indicators:
            # If no valid indicators, use first available indicator and create a basic condition
            indicator = indicators[0]
            conditions.append(ConditionGene(
                indicator=indicator.type,
                operator='>' if is_entry else '<',
                threshold=50,
                logic='AND'
            ))
            return conditions
        
        # Generate 1-3 conditions
        num_conditions = random.randint(1, min(3, len(valid_indicators)))
        
        for _ in range(num_conditions):
            # Pick a random indicator
            indicator = random.choice(valid_indicators)
            
            # Generate condition based on indicator type
            condition = self._generate_condition_for_indicator(indicator, is_entry)
            if condition:
                conditions.append(condition)
        
        # Ensure at least one condition
        if not conditions and valid_indicators:
            indicator = valid_indicators[0]
            condition = self._generate_condition_for_indicator(indicator, is_entry)
            if condition:
                conditions.append(condition)
        
        return conditions if conditions else [ConditionGene(
            indicator=indicators[0].type,
            operator='>' if is_entry else '<',
            threshold=50,
            logic='AND'
        )]
    
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
        
        elif indicator.type == 'STOCH':
            if is_entry:
                threshold_range = ind_config.get('k_threshold', [20, 40])
                operator = '<'
            else:
                threshold_range = ind_config.get('d_threshold', [60, 80])
                operator = '>'
            
            return ConditionGene(
                indicator='STOCH',
                operator=operator,
                threshold=random.randint(*threshold_range),
                logic=random.choice(['AND', 'OR'])
            )
        
        elif indicator.type == 'CCI':
            if is_entry:
                threshold_range = ind_config.get('buy_threshold', [-200, -100])
                operator = '<'
            else:
                threshold_range = ind_config.get('sell_threshold', [100, 200])
                operator = '>'
            
            return ConditionGene(
                indicator='CCI',
                operator=operator,
                threshold=random.randint(*threshold_range),
                logic=random.choice(['AND', 'OR'])
            )
        
        elif indicator.type == 'ADX':
            threshold_range = ind_config.get('threshold', [20, 40])
            return ConditionGene(
                indicator='ADX',
                operator='>',
                threshold=random.randint(*threshold_range),
                logic=random.choice(['AND', 'OR'])
            )
        
        # For other indicators, return a generic condition
        return ConditionGene(
            indicator=indicator.type,
            operator='>' if is_entry else '<',
            threshold=50,
            logic='AND'
        )
    
    def generate_strategy_code(self, strategy_gene: StrategyGene) -> str:
        """
        Convert a StrategyGene to FreqTrade Python code.
        
        Args:
            strategy_gene: Strategy genetic representation
            
        Returns:
            Python code as string
        """
        strategy_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
        
        # Generate indicator code
        indicator_code = self._generate_indicator_code(strategy_gene.indicators)
        
        # Generate entry condition code
        entry_code = self._generate_condition_code(strategy_gene.entry_conditions, is_entry=True)
        
        # Generate exit condition code
        exit_code = self._generate_condition_code(strategy_gene.exit_conditions, is_entry=False)
        
        # Generate trailing stop parameters
        trailing_stop_params = ""
        if strategy_gene.trailing_stop:
            if strategy_gene.trailing_stop_positive is not None:
                trailing_stop_params = f"""
    trailing_stop_positive = {strategy_gene.trailing_stop_positive}
    trailing_stop_positive_offset = {strategy_gene.trailing_stop_positive_offset}"""
        
        code = f'''"""
Auto-generated strategy by Genetic Algorithm
Generation: {strategy_gene.generation}
Individual: {strategy_gene.individual_id}
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class {strategy_name}(IStrategy):
    """Auto-generated GA strategy"""
    
    INTERFACE_VERSION = 3
    
    # Strategy parameters
    timeframe = '{strategy_gene.timeframe}'
    stoploss = {strategy_gene.stoploss}
    minimal_roi = {strategy_gene.minimal_roi}
    trailing_stop = {strategy_gene.trailing_stop}{trailing_stop_params}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators"""
{indicator_code}
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals"""
{entry_code}
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals"""
{exit_code}
        return dataframe
'''
        
        return code
    
    def _generate_indicator_code(self, indicators: List[IndicatorGene]) -> str:
        """Generate Python code for indicators."""
        lines = []
        
        for ind in indicators:
            if ind.type == 'RSI':
                period = ind.parameters.get('period', 14)
                lines.append(f"        dataframe['rsi_{period}'] = ta.RSI(dataframe, timeperiod={period})")
            
            elif ind.type == 'MACD':
                fast = ind.parameters.get('fast_period', 12)
                slow = ind.parameters.get('slow_period', 26)
                signal = ind.parameters.get('signal_period', 9)
                lines.append(f"        macd = ta.MACD(dataframe, fastperiod={fast}, slowperiod={slow}, signalperiod={signal})")
                lines.append(f"        dataframe['macd'] = macd['macd']")
                lines.append(f"        dataframe['macdsignal'] = macd['macdsignal']")
                lines.append(f"        dataframe['macdhist'] = macd['macdhist']")
            
            elif ind.type == 'BBANDS':
                period = ind.parameters.get('period', 20)
                std_dev = ind.parameters.get('std_dev', 2.0)
                lines.append(f"        bollinger = ta.BBANDS(dataframe, timeperiod={period}, nbdevup={std_dev}, nbdevdn={std_dev})")
                lines.append(f"        dataframe['bb_upperband'] = bollinger['upperband']")
                lines.append(f"        dataframe['bb_middleband'] = bollinger['middleband']")
                lines.append(f"        dataframe['bb_lowerband'] = bollinger['lowerband']")
            
            elif ind.type == 'EMA':
                period = ind.parameters.get('period', 20)
                lines.append(f"        dataframe['ema_{period}'] = ta.EMA(dataframe, timeperiod={period})")
            
            elif ind.type == 'SMA':
                period = ind.parameters.get('period', 20)
                lines.append(f"        dataframe['sma_{period}'] = ta.SMA(dataframe, timeperiod={period})")
            
            elif ind.type == 'STOCH':
                k_period = ind.parameters.get('k_period', 14)
                d_period = ind.parameters.get('d_period', 3)
                lines.append(f"        stoch = ta.STOCH(dataframe, fastk_period={k_period}, slowk_period={d_period}, slowd_period={d_period})")
                lines.append(f"        dataframe['slowk'] = stoch['slowk']")
                lines.append(f"        dataframe['slowd'] = stoch['slowd']")
            
            elif ind.type == 'ATR':
                period = ind.parameters.get('period', 14)
                lines.append(f"        dataframe['atr_{period}'] = ta.ATR(dataframe, timeperiod={period})")
            
            elif ind.type == 'ADX':
                period = ind.parameters.get('period', 14)
                lines.append(f"        dataframe['adx_{period}'] = ta.ADX(dataframe, timeperiod={period})")
            
            elif ind.type == 'CCI':
                period = ind.parameters.get('period', 20)
                lines.append(f"        dataframe['cci_{period}'] = ta.CCI(dataframe, timeperiod={period})")
        
        return '\n'.join(lines) if lines else "        # No indicators"
    
    def _generate_condition_code(self, conditions: List[ConditionGene], is_entry: bool) -> str:
        """Generate Python code for entry/exit conditions."""
        if not conditions:
            signal_col = 'enter_long' if is_entry else 'exit_long'
            return f"        dataframe['{signal_col}'] = 0\n"
        
        signal_col = 'enter_long' if is_entry else 'exit_long'
        
        # Build condition expressions
        condition_exprs = []
        
        for i, cond in enumerate(conditions):
            expr = self._generate_single_condition(cond)
            if expr:
                condition_exprs.append(expr)
        
        if not condition_exprs:
            return f"        dataframe['{signal_col}'] = 0\n"
        
        # Combine conditions based on logic operators
        # For simplicity, we'll combine with AND by default
        # In a more advanced version, we'd parse the logic field properly
        combined_condition = ' &\n            '.join(f"({expr})" for expr in condition_exprs)
        
        code = f"""        conditions = (
            {combined_condition}
        )
        dataframe.loc[conditions, '{signal_col}'] = 1
"""
        
        return code
    
    def _generate_single_condition(self, condition: ConditionGene) -> str:
        """Generate a single condition expression."""
        if condition.indicator == 'RSI':
            # Find the RSI column (may have different periods)
            if condition.operator == 'cross_below':
                return f"(dataframe['rsi_14'] < {condition.threshold})"
            elif condition.operator == 'cross_above':
                return f"(dataframe['rsi_14'] > {condition.threshold})"
            elif condition.operator == '<':
                return f"(dataframe['rsi_14'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['rsi_14'] > {condition.threshold})"
        
        elif condition.indicator == 'MACD':
            if condition.operator == 'cross_above':
                return "(dataframe['macd'] > dataframe['macdsignal'])"
            elif condition.operator == 'cross_below':
                return "(dataframe['macd'] < dataframe['macdsignal'])"
        
        elif condition.indicator == 'STOCH':
            if condition.operator == '<':
                return f"(dataframe['slowk'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['slowk'] > {condition.threshold})"
            elif condition.operator == 'cross_above':
                return f"(dataframe['slowk'] > dataframe['slowd'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['slowk'] < dataframe['slowd'])"
        
        elif condition.indicator == 'CCI':
            if condition.operator == '<':
                return f"(dataframe['cci_20'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['cci_20'] > {condition.threshold})"
        
        elif condition.indicator == 'ADX':
            if condition.operator == '>':
                return f"(dataframe['adx_14'] > {condition.threshold})"
            elif condition.operator == '<':
                return f"(dataframe['adx_14'] < {condition.threshold})"
        
        # Default fallback
        return "True"
