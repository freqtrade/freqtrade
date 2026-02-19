"""
Strategy Generator

Generates random trading strategies and converts genetic
representations to FreqTrade strategy code.
"""

import random
from typing import Dict, Any, List

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.utils.indicator_factory import create_random_indicator


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
        # Guard against num_indicators > len(available_indicators)
        num_indicators = min(num_indicators, len(self.available_indicators))
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
        
        strategy = StrategyGene(
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
        
        # Assign unique instance IDs to all indicators
        strategy.assign_instance_ids()
        
        return strategy
    
    def _generate_random_indicator(self, indicator_type: str) -> IndicatorGene:
        """Generate a random indicator with appropriate parameters."""
        return create_random_indicator(indicator_type, self.indicator_config)
    
    def _generate_random_conditions(self, indicators: List[IndicatorGene], 
                                   is_entry: bool) -> List[ConditionGene]:
        """Generate random entry or exit conditions."""
        conditions = []
        
        # Filter indicators that can generate conditions
        valid_indicators = [ind for ind in indicators 
                          if ind.type in ['RSI', 'MACD', 'STOCH', 'CCI', 'ADX', 'BBANDS', 'EMA', 'SMA']]
        
        if not valid_indicators:
            # If no valid indicators, use first available indicator and create a basic condition
            indicator = indicators[0]
            conditions.append(ConditionGene(
                indicator=indicator.type,
                operator='>' if is_entry else '<',
                threshold=50,
                logic='OR'  # Use OR for more lenient conditions
            ))
            return conditions
        
        # Generate 1-2 conditions (reduced from 1-3 to make strategies less restrictive)
        num_conditions = random.randint(1, min(2, len(valid_indicators)))
        
        # Use OR logic more often to make strategies less restrictive
        primary_logic = random.choice(['OR', 'OR', 'AND'])  # 2/3 chance of OR
        
        for _ in range(num_conditions):
            # Pick a random indicator
            indicator = random.choice(valid_indicators)
            
            # Generate condition based on indicator type
            condition = self._generate_condition_for_indicator(indicator, is_entry)
            if condition:
                # Override logic with primary logic for consistency
                condition.logic = primary_logic
                conditions.append(condition)
        
        # Ensure at least one condition
        if not conditions and valid_indicators:
            indicator = valid_indicators[0]
            condition = self._generate_condition_for_indicator(indicator, is_entry)
            if condition:
                condition.logic = 'OR'
                conditions.append(condition)
        
        return conditions if conditions else [ConditionGene(
            indicator=indicators[0].type,
            operator='>' if is_entry else '<',
            threshold=50,
            logic='OR'
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
        
        elif indicator.type == 'BBANDS':
            # Bollinger Bands entry/exit conditions
            if is_entry:
                # Buy when price crosses below lower band
                operator = 'cross_below'
            else:
                # Sell when price crosses above upper band
                operator = 'cross_above'
            
            return ConditionGene(
                indicator='BBANDS',
                operator=operator,
                threshold=0,  # Not used for BBANDS
                logic=random.choice(['AND', 'OR'])
            )
        
        elif indicator.type in ['EMA', 'SMA']:
            # Moving average crossover conditions
            if is_entry:
                operator = 'cross_above'  # Price crosses above MA (bullish)
            else:
                operator = 'cross_below'  # Price crosses below MA (bearish)
            
            return ConditionGene(
                indicator=indicator.type,
                operator=operator,
                threshold=0,  # Not used for MA crossovers
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
        entry_code = self._generate_condition_code(strategy_gene.entry_conditions, strategy_gene.indicators, is_entry=True)
        
        # Generate exit condition code
        exit_code = self._generate_condition_code(strategy_gene.exit_conditions, strategy_gene.indicators, is_entry=False)
        
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
    
    def _generate_condition_code(self, conditions: List[ConditionGene], indicators: List[IndicatorGene], is_entry: bool) -> str:
        """Generate Python code for entry/exit conditions."""
        if not conditions:
            signal_col = 'enter_long' if is_entry else 'exit_long'
            return f"        dataframe['{signal_col}'] = 0\n"
        
        signal_col = 'enter_long' if is_entry else 'exit_long'
        
        # Build condition expressions, filtering out conditions that reference non-existent indicators
        condition_exprs = []
        valid_conditions = []
        
        for i, cond in enumerate(conditions):
            # Validate that the condition's indicator exists in the strategy
            if self._condition_has_valid_indicator(cond, indicators):
                expr = self._generate_single_condition(cond, indicators)
                if expr:
                    condition_exprs.append(expr)
                    valid_conditions.append(cond)
        
        # If no valid conditions, create a default safe condition
        if not condition_exprs:
            # Use a simple volume-based condition as fallback
            return f"        dataframe.loc[dataframe['volume'] > 0, '{signal_col}'] = 1\n"
        
        # Combine conditions based on logic operators
        # Check if all valid conditions use the same logic
        logics = [cond.logic for cond in valid_conditions if hasattr(cond, 'logic')]
        use_or = logics and logics[0] == 'OR'
        
        # Combine with OR or AND
        if use_or:
            combined_condition = ' |\n            '.join(f"({expr})" for expr in condition_exprs)
        else:
            combined_condition = ' &\n            '.join(f"({expr})" for expr in condition_exprs)
        
        code = f"""        conditions = (
            {combined_condition}
        )
        dataframe.loc[conditions, '{signal_col}'] = 1
"""
        
        return code
    
    def _condition_has_valid_indicator(self, condition: ConditionGene, indicators: List[IndicatorGene]) -> bool:
        """
        Check if a condition references an indicator that exists in the strategy.
        
        Args:
            condition: Condition to validate
            indicators: List of indicators in the strategy
            
        Returns:
            True if the condition's indicator exists, False otherwise
        """
        # Extract indicator type from condition reference
        indicator_ref = condition.indicator
        indicator_type = indicator_ref.split('_')[0] if '_' in indicator_ref else indicator_ref
        
        # Check if any indicator in the list matches this type
        for ind in indicators:
            # Match by instance_id if available, otherwise by type
            if ind.instance_id and ind.instance_id == indicator_ref:
                return True
            elif ind.type == indicator_type:
                return True
        
        return False
    
    def _generate_single_condition(self, condition: ConditionGene, indicators: List[IndicatorGene]) -> str:
        """Generate a single condition expression.
        
        Handles both type-based references (e.g., 'RSI') and instance-based references (e.g., 'RSI_0').
        """
        # Extract indicator type from condition reference
        # Handle both 'RSI' and 'RSI_0' formats
        indicator_ref = condition.indicator
        indicator_type = indicator_ref.split('_')[0] if '_' in indicator_ref else indicator_ref
        
        # Find the specific indicator instance or use first matching type
        target_indicator = None
        for ind in indicators:
            # Match by instance_id if available, otherwise by type
            if ind.instance_id and ind.instance_id == indicator_ref:
                target_indicator = ind
                break
            elif ind.type == indicator_type and not target_indicator:
                target_indicator = ind
        
        # Build mapping of indicator types/instances to their parameters
        indicator_periods = {}
        for ind in indicators:
            # Map both type and instance_id to parameters
            if ind.type in ['RSI', 'EMA', 'SMA', 'ATR', 'ADX', 'CCI', 'BBANDS']:
                period = ind.parameters.get('period', 14 if ind.type != 'BBANDS' else 20)
                indicator_periods[ind.type] = period
                if ind.instance_id:
                    indicator_periods[ind.instance_id] = period
            elif ind.type == 'STOCH':
                k_period = ind.parameters.get('k_period', 14)
                indicator_periods[ind.type] = k_period
                if ind.instance_id:
                    indicator_periods[ind.instance_id] = k_period
        
        # Use the specific indicator's parameters if found
        if target_indicator:
            if target_indicator.type in ['RSI', 'EMA', 'SMA', 'ATR', 'ADX', 'CCI', 'BBANDS']:
                period = target_indicator.parameters.get('period', 14 if target_indicator.type != 'BBANDS' else 20)
                indicator_periods[indicator_ref] = period
            elif target_indicator.type == 'STOCH':
                k_period = target_indicator.parameters.get('k_period', 14)
                indicator_periods[indicator_ref] = k_period
        
        if indicator_type == 'RSI':
            # Use actual RSI period if available (try instance_id first, then type)
            period = indicator_periods.get(indicator_ref, indicator_periods.get('RSI', 14))
            if condition.operator == 'cross_below':
                return f"(dataframe['rsi_{period}'] < {condition.threshold})"
            elif condition.operator == 'cross_above':
                return f"(dataframe['rsi_{period}'] > {condition.threshold})"
            elif condition.operator == '<':
                return f"(dataframe['rsi_{period}'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['rsi_{period}'] > {condition.threshold})"
        
        elif indicator_type == 'MACD':
            if condition.operator == 'cross_above':
                return "(dataframe['macd'] > dataframe['macdsignal'])"
            elif condition.operator == 'cross_below':
                return "(dataframe['macd'] < dataframe['macdsignal'])"
        
        elif indicator_type == 'STOCH':
            if condition.operator == '<':
                return f"(dataframe['slowk'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['slowk'] > {condition.threshold})"
            elif condition.operator == 'cross_above':
                return f"(dataframe['slowk'] > dataframe['slowd'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['slowk'] < dataframe['slowd'])"
        
        elif indicator_type == 'CCI':
            # Use actual CCI period if available (try instance_id first, then type)
            period = indicator_periods.get(indicator_ref, indicator_periods.get('CCI', 20))
            if condition.operator == '<':
                return f"(dataframe['cci_{period}'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['cci_{period}'] > {condition.threshold})"
        
        elif indicator_type == 'ADX':
            # Use actual ADX period if available (try instance_id first, then type)
            period = indicator_periods.get(indicator_ref, indicator_periods.get('ADX', 14))
            if condition.operator == '>':
                return f"(dataframe['adx_{period}'] > {condition.threshold})"
            elif condition.operator == '<':
                return f"(dataframe['adx_{period}'] < {condition.threshold})"
        
        elif indicator_type == 'BBANDS':
            # Bollinger Bands conditions
            if condition.operator == 'cross_below':
                return "(dataframe['close'] < dataframe['bb_lowerband'])"
            elif condition.operator == 'cross_above':
                return "(dataframe['close'] > dataframe['bb_upperband'])"
            elif condition.operator == '<':
                return "(dataframe['close'] < dataframe['bb_middleband'])"
            elif condition.operator == '>':
                return "(dataframe['close'] > dataframe['bb_middleband'])"
        
        elif indicator_type in ['EMA', 'SMA']:
            # Moving average conditions
            # Use actual period from indicator_periods (try instance_id first, then type)
            default_ma_period = 20
            period = indicator_periods.get(indicator_ref, indicator_periods.get(indicator_type, default_ma_period))
            col_name = f"{indicator_type.lower()}_{period}"
            if condition.operator == 'cross_above':
                return f"(dataframe['close'] > dataframe['{col_name}'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['close'] < dataframe['{col_name}'])"
            elif condition.operator == '>':
                return f"(dataframe['close'] > dataframe['{col_name}'])"
            elif condition.operator == '<':
                return f"(dataframe['close'] < dataframe['{col_name}'])"
        
        # Default fallback - use vectorized condition, not scalar
        return "(dataframe['volume'] > 0)"
