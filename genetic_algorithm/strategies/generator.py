"""
Strategy Generator

Generates random trading strategies and converts genetic
representations to FreqTrade strategy code.
"""

import random
import logging
from typing import Dict, Any, List

from genetic_algorithm.core.strategy_gene import (
    StrategyGene, IndicatorGene, ConditionGene, is_higher_timeframe
)
from genetic_algorithm.utils.indicator_factory import create_random_indicator

logger = logging.getLogger(__name__)


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
        
        # Multi-timeframe config
        self.multi_tf_config = config.get('multi_timeframe', {})
    
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
        
        # Random max_open_trades
        max_open_trades_range = self.strategy_constraints.get('max_open_trades_range', [1, 10])
        max_open_trades = random.randint(*max_open_trades_range)
        
        # Multi-timeframe: optionally add informative timeframes and indicators
        informative_timeframes = []
        if self.multi_tf_config.get('enabled', False):
            available_itfs = self.multi_tf_config.get('available', [])
            max_tfs = self.multi_tf_config.get('max_timeframes', 2)
            # Filter to only higher TFs than base
            valid_itfs = [tf for tf in available_itfs if is_higher_timeframe(tf, timeframe)]
            if valid_itfs and random.random() < 0.7:  # 70% chance to add informative TFs
                num_itfs = random.randint(1, min(max_tfs, len(valid_itfs)))
                informative_timeframes = random.sample(valid_itfs, num_itfs)
                # Add informative indicators
                htf_pref = self.multi_tf_config.get('higher_timeframe_preference', [])
                for itf in informative_timeframes:
                    itf_indicator = self._generate_informative_indicator(itf, htf_pref)
                    indicators.append(itf_indicator)
                    # Add a condition using this informative indicator
                    itf_cond = self._generate_condition_for_indicator(itf_indicator, is_entry=True)
                    if itf_cond:
                        itf_cond.logic = 'AND'  # Higher TF acts as a filter
                        entry_conditions.append(itf_cond)
        
        strategy = StrategyGene(
            generation=generation,
            individual_id=individual_id,
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            timeframe=timeframe,
            stoploss=stoploss,
            minimal_roi=minimal_roi,
            max_open_trades=max_open_trades,
            informative_timeframes=informative_timeframes,
            trailing_stop=random.choice([True, False]),
        )
        
        # Assign unique instance IDs to all indicators
        strategy.assign_instance_ids()
        
        return strategy
    
    def _generate_informative_indicator(self, timeframe: str, 
                                        preferred_types: List[str] = None) -> IndicatorGene:
        """Generate a random indicator for an informative (higher) timeframe."""
        if preferred_types:
            # Prefer trend/volatility indicators for higher TFs
            candidates = [t for t in preferred_types if t in self.available_indicators]
            if candidates:
                ind_type = random.choice(candidates)
            else:
                ind_type = random.choice(self.available_indicators)
        else:
            ind_type = random.choice(self.available_indicators)
        
        indicator = create_random_indicator(ind_type, self.indicator_config)
        indicator.timeframe = timeframe
        return indicator
    
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
        
        # ALWAYS use OR logic to avoid contradictory conditions that produce 0 trades
        # AND logic often creates impossible combinations (e.g., price > SMA AND price < lower BB)
        primary_logic = 'OR'
        
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
        # CRITICAL FIX: Ensure all indicators referenced in conditions actually exist
        # This is a safety net in case mutation/crossover created mismatches
        strategy_gene.ensure_indicators_for_conditions(self.indicator_config)
        
        # Re-assign instance IDs after ensuring indicators exist
        strategy_gene.assign_instance_ids()
        
        strategy_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
        
        # Separate base and informative indicators
        base_indicators = strategy_gene.get_base_indicators()
        informative_indicators = strategy_gene.get_informative_indicators()
        
        # Generate indicator code for base timeframe
        indicator_code = self._generate_indicator_code(base_indicators)
        
        # Generate informative pairs code
        informative_pairs_code = self._generate_informative_pairs_code(strategy_gene)
        informative_indicator_code = self._generate_informative_indicator_code(
            informative_indicators, strategy_gene.timeframe
        )
        
        # Generate entry condition code
        entry_code = self._generate_condition_code(
            strategy_gene.entry_conditions, strategy_gene.indicators, is_entry=True
        )
        
        # Generate exit condition code
        exit_code = self._generate_condition_code(
            strategy_gene.exit_conditions, strategy_gene.indicators, is_entry=False
        )
        
        # Generate trailing stop parameters
        trailing_stop_params = ""
        if strategy_gene.trailing_stop:
            if strategy_gene.trailing_stop_positive is not None:
                trailing_stop_params = f"""
    trailing_stop_positive = {strategy_gene.trailing_stop_positive}
    trailing_stop_positive_offset = {strategy_gene.trailing_stop_positive_offset}"""
        
        # Build informative_pairs method body
        if informative_indicators:
            inf_tfs = sorted(set(ind.timeframe for ind in informative_indicators))
            informative_pairs_method = f"""
    def informative_pairs(self):
        \"\"\"Define additional informative pair/interval combinations.\"\"\"
        pairs = self.dp.current_whitelist()
        informative = []
        for pair in pairs:
            for tf in {inf_tfs!r}:
                informative.append((pair, tf))
        return informative"""
        else:
            informative_pairs_method = """
    def informative_pairs(self):
        \"\"\"Define additional informative pair/interval combinations.\"\"\"
        return []"""
        
        # Build populate_indicators body
        if informative_indicator_code:
            populate_indicators_body = f"""{indicator_code}
        
        # --- Informative timeframe indicators ---
{informative_indicator_code}"""
        else:
            populate_indicators_body = indicator_code
        
        code = f'''"""
Auto-generated strategy by Genetic Algorithm
Generation: {strategy_gene.generation}
Individual: {strategy_gene.individual_id}
"""

from freqtrade.strategy import IStrategy, merge_informative_pair
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
    max_open_trades = {strategy_gene.max_open_trades}
{informative_pairs_method}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators"""
{populate_indicators_body}
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
    
    def _generate_informative_pairs_code(self, strategy_gene: StrategyGene) -> str:
        """Generate the informative_pairs() return value."""
        inf_indicators = strategy_gene.get_informative_indicators()
        if not inf_indicators:
            return ""
        tfs = sorted(set(ind.timeframe for ind in inf_indicators))
        return repr(tfs)
    
    def _generate_informative_indicator_code(self, informative_indicators: List[IndicatorGene],
                                              base_timeframe: str) -> str:
        """Generate code that fetches informative data and merges it into the base dataframe."""
        if not informative_indicators:
            return ""
        
        # Group indicators by timeframe
        by_tf: Dict[str, List[IndicatorGene]] = {}
        for ind in informative_indicators:
            tf = ind.timeframe
            if tf not in by_tf:
                by_tf[tf] = []
            by_tf[tf].append(ind)
        
        lines = []
        for tf in sorted(by_tf.keys()):
            inds = by_tf[tf]
            lines.append(f"        # Informative indicators for {tf}")
            lines.append(f"        if self.dp:")
            lines.append(f"            inf_tf = '{tf}'")
            lines.append(f"            pair = metadata['pair']")
            lines.append(f"            informative = self.dp.get_pair_dataframe(pair=pair, timeframe=inf_tf)")
            lines.append(f"            if informative is not None and len(informative) > 0:")
            for ind in inds:
                ind_lines = self._generate_single_indicator_code(ind, prefix="                ")
                lines.append(ind_lines)
            lines.append(f"                dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)")
        
        return '\n'.join(lines)
    
    def _generate_single_indicator_code(self, ind: IndicatorGene, prefix: str = "        ") -> str:
        """Generate code for a single indicator calculation, used for informative indicators."""
        if ind.type == 'RSI':
            period = ind.parameters.get('period', 14)
            return f"{prefix}informative['rsi_{period}'] = ta.RSI(informative, timeperiod={period})"
        elif ind.type == 'MACD':
            fast = ind.parameters.get('fast_period', 12)
            slow = ind.parameters.get('slow_period', 26)
            signal = ind.parameters.get('signal_period', 9)
            return (f"{prefix}macd = ta.MACD(informative, fastperiod={fast}, slowperiod={slow}, signalperiod={signal})\n"
                    f"{prefix}informative['macd'] = macd['macd']\n"
                    f"{prefix}informative['macdsignal'] = macd['macdsignal']\n"
                    f"{prefix}informative['macdhist'] = macd['macdhist']")
        elif ind.type == 'BBANDS':
            period = ind.parameters.get('period', 20)
            std_dev = ind.parameters.get('std_dev', 2.0)
            return (f"{prefix}bollinger = ta.BBANDS(informative, timeperiod={period}, nbdevup={std_dev}, nbdevdn={std_dev})\n"
                    f"{prefix}informative['bb_upperband'] = bollinger['upperband']\n"
                    f"{prefix}informative['bb_middleband'] = bollinger['middleband']\n"
                    f"{prefix}informative['bb_lowerband'] = bollinger['lowerband']")
        elif ind.type in ['EMA', 'SMA']:
            period = ind.parameters.get('period', 20)
            return f"{prefix}informative['{ind.type.lower()}_{period}'] = ta.{ind.type}(informative, timeperiod={period})"
        elif ind.type == 'STOCH':
            k_period = ind.parameters.get('k_period', 14)
            d_period = ind.parameters.get('d_period', 3)
            return (f"{prefix}stoch = ta.STOCH(informative, fastk_period={k_period}, slowk_period={d_period}, slowd_period={d_period})\n"
                    f"{prefix}informative['slowk'] = stoch['slowk']\n"
                    f"{prefix}informative['slowd'] = stoch['slowd']")
        elif ind.type == 'ATR':
            period = ind.parameters.get('period', 14)
            return f"{prefix}informative['atr_{period}'] = ta.ATR(informative, timeperiod={period})"
        elif ind.type == 'ADX':
            period = ind.parameters.get('period', 14)
            return f"{prefix}informative['adx_{period}'] = ta.ADX(informative, timeperiod={period})"
        elif ind.type == 'CCI':
            period = ind.parameters.get('period', 20)
            return f"{prefix}informative['cci_{period}'] = ta.CCI(informative, timeperiod={period})"
        return f"{prefix}pass  # Unsupported indicator: {ind.type}"
    
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
        filtered_conditions = []
        
        for i, cond in enumerate(conditions):
            # Validate that the condition's indicator exists in the strategy
            if self._condition_has_valid_indicator(cond, indicators):
                expr = self._generate_single_condition(cond, indicators)
                if expr:
                    condition_exprs.append(expr)
                    valid_conditions.append(cond)
            else:
                # Log filtered condition for debugging
                filtered_conditions.append(cond.indicator)
                logger.debug(f"Filtered out condition referencing non-existent indicator: {cond.indicator}")
        
        # Log summary if conditions were filtered
        if filtered_conditions:
            indicator_types = [ind.type for ind in indicators]
            logger.warning(f"Filtered {len(filtered_conditions)} condition(s) referencing missing indicators: {filtered_conditions}. "
                         f"Available indicators: {indicator_types}")
        
        # If no valid conditions, create a default safe condition
        if not condition_exprs:
            signal_type = 'entry' if is_entry else 'exit'
            logger.warning(f"No valid {signal_type} conditions found. Using fallback volume-based condition.")
            # Use a volume-above-average condition as fallback to avoid always-true signal
            return f"""        # Fallback condition: volume above 20-period average
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe.loc[dataframe['volume'] > dataframe['volume_sma'], '{signal_col}'] = 1
"""
        
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
        For informative timeframe indicators, appends the TF suffix (e.g., rsi_14_1h).
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
        
        # Determine TF suffix for informative indicators
        tf_suffix = ""
        if target_indicator and target_indicator.timeframe:
            tf_suffix = f"_{target_indicator.timeframe}"
        
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
            col = f"rsi_{period}{tf_suffix}"
            if condition.operator == 'cross_below':
                return f"(dataframe['{col}'] < {condition.threshold})"
            elif condition.operator == 'cross_above':
                return f"(dataframe['{col}'] > {condition.threshold})"
            elif condition.operator == '<':
                return f"(dataframe['{col}'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['{col}'] > {condition.threshold})"
        
        elif indicator_type == 'MACD':
            macd_col = f"macd{tf_suffix}"
            signal_col = f"macdsignal{tf_suffix}"
            if condition.operator == 'cross_above':
                return f"(dataframe['{macd_col}'] > dataframe['{signal_col}'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['{macd_col}'] < dataframe['{signal_col}'])"
        
        elif indicator_type == 'STOCH':
            slowk_col = f"slowk{tf_suffix}"
            slowd_col = f"slowd{tf_suffix}"
            if condition.operator == '<':
                return f"(dataframe['{slowk_col}'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['{slowk_col}'] > {condition.threshold})"
            elif condition.operator == 'cross_above':
                return f"(dataframe['{slowk_col}'] > dataframe['{slowd_col}'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['{slowk_col}'] < dataframe['{slowd_col}'])"
        
        elif indicator_type == 'CCI':
            # Use actual CCI period if available (try instance_id first, then type)
            period = indicator_periods.get(indicator_ref, indicator_periods.get('CCI', 20))
            col = f"cci_{period}{tf_suffix}"
            if condition.operator == '<':
                return f"(dataframe['{col}'] < {condition.threshold})"
            elif condition.operator == '>':
                return f"(dataframe['{col}'] > {condition.threshold})"
        
        elif indicator_type == 'ADX':
            # Use actual ADX period if available (try instance_id first, then type)
            period = indicator_periods.get(indicator_ref, indicator_periods.get('ADX', 14))
            col = f"adx_{period}{tf_suffix}"
            if condition.operator == '>':
                return f"(dataframe['{col}'] > {condition.threshold})"
            elif condition.operator == '<':
                return f"(dataframe['{col}'] < {condition.threshold})"
        
        elif indicator_type == 'BBANDS':
            # Bollinger Bands conditions
            upper = f"bb_upperband{tf_suffix}"
            middle = f"bb_middleband{tf_suffix}"
            lower = f"bb_lowerband{tf_suffix}"
            close = f"close{tf_suffix}" if tf_suffix else "close"
            if condition.operator == 'cross_below':
                return f"(dataframe['{close}'] < dataframe['{lower}'])"
            elif condition.operator == 'cross_above':
                return f"(dataframe['{close}'] > dataframe['{upper}'])"
            elif condition.operator == '<':
                return f"(dataframe['{close}'] < dataframe['{middle}'])"
            elif condition.operator == '>':
                return f"(dataframe['{close}'] > dataframe['{middle}'])"
        
        elif indicator_type in ['EMA', 'SMA']:
            # Moving average conditions
            # Use actual period from indicator_periods (try instance_id first, then type)
            default_ma_period = 20
            period = indicator_periods.get(indicator_ref, indicator_periods.get(indicator_type, default_ma_period))
            col_name = f"{indicator_type.lower()}_{period}{tf_suffix}"
            close = f"close{tf_suffix}" if tf_suffix else "close"
            if condition.operator == 'cross_above':
                return f"(dataframe['{close}'] > dataframe['{col_name}'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['{close}'] < dataframe['{col_name}'])"
            elif condition.operator == '>':
                return f"(dataframe['{close}'] > dataframe['{col_name}'])"
            elif condition.operator == '<':
                return f"(dataframe['{close}'] < dataframe['{col_name}'])"
        
        # Default fallback - use vectorized condition, not scalar
        return "(dataframe['volume'] > 0)"
