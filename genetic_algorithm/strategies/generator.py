"""
Strategy Generator

Generates random trading strategies and converts genetic
representations to FreqTrade strategy code.
"""

import random
import logging
from typing import Dict, Any, List

from genetic_algorithm.core.strategy_gene import (
    StrategyGene, IndicatorGene, ConditionGene, RegimeGene, is_higher_timeframe
)
from genetic_algorithm.utils.indicator_factory import create_random_indicator
from genetic_algorithm.strategies.operator_registry import (
    is_valid_operator, resolve_indicator_type, get_valid_operators,
)

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
        
        # Short selling config
        self.short_selling_config = config.get('short_selling', {})
    
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
            can_short=self.short_selling_config.get('enabled', False) and random.random() < self.short_selling_config.get('probability', 0.5),
        )
        
        # Generate independent short conditions when configured
        if strategy.can_short and self.short_selling_config.get('independent_conditions', False):
            num_short_entry = random.randint(1, max(1, len(entry_conditions)))
            num_short_exit = random.randint(1, max(1, len(exit_conditions)))
            short_entry_conds = []
            short_exit_conds = []
            for _ in range(num_short_entry):
                ind = random.choice(strategy.indicators)
                cond = self._generate_condition_for_indicator(ind, is_entry=True)
                if cond:
                    short_entry_conds.append(cond)
            for _ in range(num_short_exit):
                ind = random.choice(strategy.indicators)
                cond = self._generate_condition_for_indicator(ind, is_entry=False)
                if cond:
                    short_exit_conds.append(cond)
            strategy.short_entry_conditions = short_entry_conds
            strategy.short_exit_conditions = short_exit_conds
        
        # Assign unique instance IDs to all indicators
        strategy.assign_instance_ids()
        
        # Enforce min_entry_conditions — random generation can under-produce
        min_entry = self.indicator_config.get('min_entry_conditions', 2)
        while len(strategy.entry_conditions) < min_entry:
            valid_inds = [ind for ind in strategy.indicators
                         if ind.type in ['RSI', 'MACD', 'STOCH', 'CCI', 'ADX', 'BBANDS', 'EMA', 'SMA',
                                          'SUPERTREND', 'ICHIMOKU', 'DONCHIAN', 'VWAP', 'PSAR', 'CMF', 'VROC',
                                          'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_DOJI', 'CDL_MORNINGSTAR',
                                          'CDL_EVENINGSTAR', 'CDL_SHOOTINGSTAR', 'CDL_HARAMI']]
            if not valid_inds:
                break
            ind = random.choice(valid_inds)
            cond = self._generate_condition_for_indicator(ind, is_entry=True)
            if cond:
                # Avoid duplicates
                existing = {(c.indicator, c.operator, str(c.threshold)) for c in strategy.entry_conditions}
                key = (cond.indicator, cond.operator, str(cond.threshold))
                if key not in existing:
                    cond.logic = 'AND'
                    strategy.entry_conditions.append(cond)
                else:
                    break  # Can't add more unique conditions
        
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
        # Includes candlestick patterns
        PATTERN_TYPES = ['CDL_ENGULFING', 'CDL_HAMMER', 'CDL_DOJI', 'CDL_MORNINGSTAR', 'CDL_EVENINGSTAR',
                        'CDL_SHOOTINGSTAR', 'CDL_HARAMI', 'CDL_PIERCING', 'CDL_DARKCLOUD', 
                        'CDL_3WHITESOLDIERS', 'CDL_3BLACKCROWS']
        valid_indicators = [ind for ind in indicators 
                          if ind.type in ['RSI', 'MACD', 'STOCH', 'CCI', 'ADX', 'BBANDS', 'EMA', 'SMA',
                                          'SUPERTREND', 'ICHIMOKU', 'DONCHIAN', 'VWAP', 'PSAR', 'CMF', 'VROC', 'ATR'] + PATTERN_TYPES]
        
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
        
        # Generate 2-4 conditions for richer signal construction
        # Min 2 prevents single-condition overfitting (configurable via indicators.min_entry_conditions)
        min_conds = self.indicator_config.get('min_entry_conditions', 2) if is_entry else self.indicator_config.get('min_exit_conditions', 1)
        max_conds = self.indicator_config.get('max_entry_conditions', 4) if is_entry else 3
        num_conditions = random.randint(min(min_conds, len(valid_indicators)), min(max_conds, len(valid_indicators)))
        
        # Use AND logic by default to create more selective (higher-quality) entry signals.
        # The fitness function's trade count penalty handles zero-trade strategies.
        # OR logic made entries too permissive, generating many low-quality trades.
        primary_logic = 'AND'
        
        for _ in range(num_conditions):
            # Pick a random indicator
            indicator = random.choice(valid_indicators)
            
            # Generate condition based on indicator type
            condition = self._generate_condition_for_indicator(indicator, is_entry)
            if condition:
                # Dedup: skip if same indicator + operator already present
                key = (condition.indicator, condition.operator)
                if any((c.indicator, c.operator) == key for c in conditions):
                    continue
                # Override logic with primary logic for consistency
                condition.logic = primary_logic
                conditions.append(condition)
        
        # Ensure at least one condition
        if not conditions and valid_indicators:
            indicator = valid_indicators[0]
            condition = self._generate_condition_for_indicator(indicator, is_entry)
            if condition:
                condition.logic = 'AND'
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
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'MACD':
            return ConditionGene(
                indicator='MACD',
                operator='cross_above' if is_entry else 'cross_below',
                threshold=0,
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
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
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
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
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'ADX':
            threshold_range = ind_config.get('threshold', [20, 40])
            return ConditionGene(
                indicator='ADX',
                operator='>',
                threshold=random.randint(*threshold_range),
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'ATR':
            # ATR: volatility filter — compare ATR to a fraction of close price
            # ATR values are typically 0.1%-5% of close, so threshold is a ratio
            atr_threshold_range = ind_config.get('threshold_pct', [0.005, 0.03])
            threshold = random.uniform(*atr_threshold_range)
            return ConditionGene(
                indicator='ATR',
                operator='>' if is_entry else '<',
                threshold=round(threshold, 4),
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
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
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
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
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        # === NEW INDICATOR CONDITIONS ===
        
        elif indicator.type == 'SUPERTREND':
            # SuperTrend: trend direction changes
            return ConditionGene(
                indicator='SUPERTREND',
                operator='cross_above' if is_entry else 'cross_below',
                threshold=0,  # Trend flip indicator
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'ICHIMOKU':
            # Ichimoku: Tenkan/Kijun crossover
            return ConditionGene(
                indicator='ICHIMOKU',
                operator='cross_above' if is_entry else 'cross_below',
                threshold=0,  # TK crossover
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'DONCHIAN':
            # Donchian: breakout conditions
            return ConditionGene(
                indicator='DONCHIAN',
                operator='cross_above' if is_entry else 'cross_below',
                threshold=0,  # Upper/lower channel breakout
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'VWAP':
            # VWAP: price vs VWAP
            return ConditionGene(
                indicator='VWAP',
                operator='cross_above' if is_entry else 'cross_below',
                threshold=0,
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'PSAR':
            # Parabolic SAR: trend flip
            return ConditionGene(
                indicator='PSAR',
                operator='cross_above' if is_entry else 'cross_below',
                threshold=0,
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'CMF':
            # Chaikin Money Flow: above/below threshold
            if is_entry:
                threshold_range = ind_config.get('buy_threshold', [0.05, 0.2])
                operator = '>'
            else:
                threshold_range = ind_config.get('sell_threshold', [-0.2, -0.05])
                operator = '<'
            
            return ConditionGene(
                indicator='CMF',
                operator=operator,
                threshold=random.uniform(*threshold_range),
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type == 'VROC':
            # Volume Rate of Change: volume spike
            threshold_range = ind_config.get('threshold', [50, 200])
            return ConditionGene(
                indicator='VROC',
                operator='>' if is_entry else '<',
                threshold=random.uniform(*threshold_range),
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        # === CANDLESTICK PATTERN CONDITIONS ===
        # Patterns that work for entry signals
        elif indicator.type in ['CDL_ENGULFING', 'CDL_HARAMI']:
            # Bidirectional patterns: >0 bullish, <0 bearish
            return ConditionGene(
                indicator=indicator.type,
                operator='>' if is_entry else '<',  # Bullish for entry, bearish for exit
                threshold=0,
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        elif indicator.type in ['CDL_HAMMER', 'CDL_MORNINGSTAR', 'CDL_PIERCING', 'CDL_3WHITESOLDIERS']:
            # Bullish-only patterns: good for entry signals
            if is_entry:
                return ConditionGene(
                    indicator=indicator.type,
                    operator='>',  # Pattern detected (non-zero)
                    threshold=0,
                    logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
                )
            # Not suitable for exit, return None and let fallback handle it
            return None
        
        elif indicator.type in ['CDL_EVENINGSTAR', 'CDL_SHOOTINGSTAR', 'CDL_DARKCLOUD', 'CDL_3BLACKCROWS']:
            # Bearish-only patterns: good for exit signals
            if not is_entry:
                return ConditionGene(
                    indicator=indicator.type,
                    operator='<',  # Pattern detected (non-zero, negative for bearish)
                    threshold=0,
                    logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
                )
            # Not suitable for entry, return None
            return None
        
        elif indicator.type == 'CDL_DOJI':
            # Doji indicates indecision, can be used as a filter
            return ConditionGene(
                indicator=indicator.type,
                operator='>',  # Doji detected (value > 0)
                threshold=0,
                logic=random.choices(['AND', 'OR'], weights=[0.75, 0.25])[0]
            )
        
        # For other indicators, return a generic condition
        return ConditionGene(
            indicator=indicator.type,
            operator='>' if is_entry else '<',
            threshold=50,
            logic='AND'
        )
    
    def _compute_startup_candle_count(self, strategy_gene: StrategyGene) -> int:
        """Compute startup_candle_count from the maximum indicator lookback period.
        
        Accounts for informative timeframe ratios: if a 200-period SMA runs on 4h
        and the base timeframe is 15m, we need 200 * (240/15) = 3200 base candles.
        """
        from genetic_algorithm.core.strategy_gene import timeframe_to_minutes
        base_minutes = timeframe_to_minutes(strategy_gene.timeframe) or 15
        max_lookback = 0

        for ind in strategy_gene.indicators:
            # Determine the raw period for this indicator
            period = 0
            params = ind.parameters or {}
            if ind.type in ('RSI', 'ATR', 'ADX', 'CCI', 'MFI', 'WILLR', 'ROC',
                            'EMA', 'SMA', 'TEMA', 'KAMA', 'DONCHIAN'):
                period = params.get('period', 20)
            elif ind.type == 'MACD':
                period = params.get('slow_period', 26) + params.get('signal_period', 9)
            elif ind.type == 'BBANDS':
                period = params.get('period', 20)
            elif ind.type == 'STOCH':
                period = params.get('k_period', 14) + params.get('d_period', 3)
            elif ind.type == 'SUPERTREND':
                period = params.get('period', 10)
            elif ind.type == 'ICHIMOKU':
                period = max(params.get('tenkan_period', 9),
                             params.get('kijun_period', 26),
                             params.get('senkou_b_period', 52)) + params.get('kijun_period', 26)
            elif ind.type == 'VWAP':
                period = params.get('period', 20)
            elif ind.type == 'CMF':
                period = params.get('period', 20)
            elif ind.type == 'VROC':
                period = params.get('period', 12)
            elif ind.type == 'AROON':
                period = params.get('period', 14)
            else:
                period = max(params.get('period', 0), 5)  # CDL patterns need ~5

            # Scale by timeframe ratio for informative indicators
            if ind.timeframe:
                ind_minutes = timeframe_to_minutes(ind.timeframe) or base_minutes
                tf_ratio = max(1, ind_minutes // base_minutes)
                period = period * tf_ratio

            max_lookback = max(max_lookback, period)

        # Add a safety buffer (10%) and floor at 30
        return max(30, int(max_lookback * 1.1) + 5)

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
        
        # Compute startup candle count from indicator lookback periods
        startup_candle_count = self._compute_startup_candle_count(strategy_gene)
        
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
        
        # Pre-validate: count conditions that will survive the indicator-existence filter.
        # If too few survive, add replacement conditions from available indicators to meet minimums.
        min_entry = self.indicator_config.get('min_entry_conditions', 2)
        min_exit = self.indicator_config.get('min_exit_conditions', 1)
        
        valid_entry_count = sum(
            1 for c in strategy_gene.entry_conditions
            if self._condition_has_valid_indicator(c, strategy_gene.indicators)
        )
        valid_exit_count = sum(
            1 for c in strategy_gene.exit_conditions
            if self._condition_has_valid_indicator(c, strategy_gene.indicators)
        )
        
        # Top up entry conditions if too few will survive filtering
        if valid_entry_count < min_entry:
            needed = min_entry - valid_entry_count
            logger.info(f"Pre-code-gen fix: only {valid_entry_count} valid entry conditions, "
                       f"adding {needed} to reach min {min_entry}")
            self._add_replacement_conditions(strategy_gene, needed, is_entry=True)
        
        # Top up exit conditions if too few will survive filtering
        if valid_exit_count < min_exit:
            needed = min_exit - valid_exit_count
            logger.info(f"Pre-code-gen fix: only {valid_exit_count} valid exit conditions, "
                       f"adding {needed} to reach min {min_exit}")
            self._add_replacement_conditions(strategy_gene, needed, is_entry=False)
        
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
        
        # --- Regime awareness code injection ---
        regime_gene = strategy_gene.regime_gene
        regime_indicator_code = ""
        regime_entry_filter = ""
        regime_exit_filter = ""
        regime_inf_tfs = []
        
        if regime_gene and regime_gene.enabled:
            regime_inf_tfs = regime_gene.regime_timeframes or ['4h', '1d']
            regime_indicator_code = self._generate_regime_indicator_code(
                regime_gene, strategy_gene.timeframe
            )
            regime_entry_filter = self._generate_regime_entry_filter(regime_gene)
            regime_exit_filter = self._generate_regime_exit_filter(regime_gene)
            
            # Merge regime timeframes into informative_pairs
            all_inf_tfs = sorted(set(
                [ind.timeframe for ind in informative_indicators if ind.timeframe]
                + regime_inf_tfs
            ))
            if all_inf_tfs:
                informative_pairs_method = f"""
    def informative_pairs(self):
        \"\"\"Define additional informative pair/interval combinations.\"\"\"
        pairs = self.dp.current_whitelist()
        informative = []
        for pair in pairs:
            for tf in {all_inf_tfs!r}:
                informative.append((pair, tf))
        return informative"""
        
        # Build populate_indicators body
        if informative_indicator_code:
            populate_indicators_body = f"""{indicator_code}
        
        # --- Informative timeframe indicators ---
{informative_indicator_code}"""
        else:
            populate_indicators_body = indicator_code
        
        # Append regime indicator code if enabled
        if regime_indicator_code:
            populate_indicators_body = f"""{populate_indicators_body}
        
        # --- Regime awareness indicators ---
{regime_indicator_code}"""
        
        # Generate short entry/exit code when can_short is enabled
        can_short_attr = ""
        short_entry_code = ""
        short_exit_code = ""
        if strategy_gene.can_short:
            can_short_attr = "\n    can_short = True"
            if strategy_gene.short_entry_conditions:
                # Use independent short conditions
                short_entry_code = "\n        # Short entry signals (independent conditions)\n" + self._generate_condition_code(
                    strategy_gene.short_entry_conditions, strategy_gene.indicators, is_entry=True, signal_col_override='enter_short'
                )
                short_exit_code = "\n        # Short exit signals (independent conditions)\n" + self._generate_condition_code(
                    strategy_gene.short_exit_conditions or strategy_gene.entry_conditions,
                    strategy_gene.indicators, is_entry=False, signal_col_override='exit_short'
                )
            else:
                # Fallback: Short entry uses inverted exit conditions (exit long = enter short logic)
                short_entry_code = "\n        # Short entry signals (inverted long exit logic)\n" + self._generate_condition_code(
                    strategy_gene.exit_conditions, strategy_gene.indicators, is_entry=True, signal_col_override='enter_short'
                )
                short_exit_code = "\n        # Short exit signals (inverted long entry logic)\n" + self._generate_condition_code(
                    strategy_gene.entry_conditions, strategy_gene.indicators, is_entry=False, signal_col_override='exit_short'
                )
        
        code = f'''"""
Auto-generated strategy by Genetic Algorithm
Generation: {strategy_gene.generation}
Individual: {strategy_gene.individual_id}
"""

from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import pandas as pd
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib

class {strategy_name}(IStrategy):
    """Auto-generated GA strategy"""
    
    INTERFACE_VERSION = 3
    
    # Strategy parameters
    timeframe = '{strategy_gene.timeframe}'
    stoploss = {strategy_gene.stoploss}
    minimal_roi = {strategy_gene.minimal_roi}
    trailing_stop = {strategy_gene.trailing_stop}{trailing_stop_params}
    max_open_trades = {strategy_gene.max_open_trades}
    startup_candle_count = {startup_candle_count}{can_short_attr}
{informative_pairs_method}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators"""
{populate_indicators_body}
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals"""
{entry_code}{regime_entry_filter}{short_entry_code}
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals"""
{exit_code}{regime_exit_filter}{short_exit_code}
        return dataframe
'''
        
        # Pre-flight compile check — catch syntax errors before backtest submission
        try:
            compile(code, f"<GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}>", "exec")
        except SyntaxError as e:
            logger.error(
                "Generated strategy code has syntax error at line %d: %s (gen=%d, ind=%d)",
                e.lineno or 0, e.msg, strategy_gene.generation, strategy_gene.individual_id,
            )
            # Return a minimal valid strategy so the individual gets zero fitness
            # instead of crashing the evaluator
            code = self._generate_fallback_strategy(strategy_gene)
        
        return code
    
    def _generate_fallback_strategy(self, strategy_gene: StrategyGene) -> str:
        """Return a syntactically valid but non-trading strategy (produces zero trades → zero fitness)."""
        name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
        return f'''"""
Auto-generated FALLBACK strategy (original had syntax error)
Generation: {strategy_gene.generation}  Individual: {strategy_gene.individual_id}
"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class {name}(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '{strategy_gene.timeframe}'
    stoploss = -0.99
    minimal_roi = {{"0": 100}}
    max_open_trades = 0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
'''

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
    
    def _generate_regime_indicator_code(
        self, regime_gene: 'RegimeGene', base_timeframe: str
    ) -> str:
        """
        Generate code that computes regime trend_score and volatility_score
        from higher-timeframe ADX/DI indicators, merges them into the base
        dataframe, and produces a composite ``regime_trend_score`` column.

        The generated code in ``populate_indicators()`` will:
        1. For each regime timeframe, fetch the informative dataframe.
        2. Compute ADX, +DI, -DI on that timeframe.
        3. Derive a per-TF trend_score: (plus_di - minus_di) / (plus_di + minus_di) * (adx / 50).
        4. Derive a per-TF volatility_score from rolling vol percentile.
        5. Merge the scores into the base dataframe via merge_informative_pair.
        6. Combine per-TF scores into a composite using weighted average.
        """
        if not regime_gene or not regime_gene.enabled:
            return ''

        regime_tfs = regime_gene.regime_timeframes or ['4h', '1d']
        combination = regime_gene.combination or 'weighted_voting'

        # Default weights: higher TF gets higher weight
        tf_weights = {'30m': 0.5, '1h': 1.0, '4h': 2.0, '1d': 3.0}

        lines = []
        lines.append("        # --- Regime detection: compute trend_score per timeframe ---")

        for tf in regime_tfs:
            w = tf_weights.get(tf, 1.0)
            safe_tf = tf.replace('m', 'min').replace('h', 'hr').replace('d', 'day')
            lines.append(f"        # Regime indicators for {tf}")
            lines.append(f"        if self.dp:")
            lines.append(f"            _regime_inf_{safe_tf} = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='{tf}')")
            lines.append(f"            if _regime_inf_{safe_tf} is not None and len(_regime_inf_{safe_tf}) > 0:")
            lines.append(f"                _ri = _regime_inf_{safe_tf}")
            # Compute ADX and DI
            lines.append(f"                _plus_dm = _ri['high'].diff()")
            lines.append(f"                _minus_dm = -_ri['low'].diff()")
            lines.append(f"                _plus_dm = _plus_dm.where((_plus_dm > _minus_dm) & (_plus_dm > 0), 0)")
            lines.append(f"                _minus_dm = _minus_dm.where((_minus_dm > _plus_dm) & (_minus_dm > 0), 0)")
            lines.append(f"                _tr = pd.concat([_ri['high'] - _ri['low'], abs(_ri['high'] - _ri['close'].shift(1)), abs(_ri['low'] - _ri['close'].shift(1))], axis=1).max(axis=1)")
            lines.append(f"                _atr = _tr.ewm(alpha=1/14, adjust=False).mean()")
            lines.append(f"                _pdi = 100 * _plus_dm.ewm(alpha=1/14, adjust=False).mean() / _atr")
            lines.append(f"                _mdi = 100 * _minus_dm.ewm(alpha=1/14, adjust=False).mean() / _atr")
            lines.append(f"                _di_sum = _pdi + _mdi")
            lines.append(f"                _di_sum = _di_sum.replace(0, np.nan)")
            lines.append(f"                _dx = 100 * abs(_pdi - _mdi) / _di_sum")
            lines.append(f"                _adx = _dx.ewm(alpha=1/14, adjust=False).mean()")
            # Compute trend_score
            lines.append(f"                _direction = (_pdi - _mdi) / _di_sum")
            lines.append(f"                _strength = (_adx / 50.0).clip(0, 1)")
            lines.append(f"                _ri['regime_trend_{tf}'] = (_direction * _strength).clip(-1, 1)")
            # Compute volatility_score
            lines.append(f"                _rets = _ri['close'].pct_change()")
            lines.append(f"                _vol = _rets.ewm(span=20, adjust=False).std()")
            lines.append(f"                _vol_min = _vol.rolling(window=60, min_periods=30).min()")
            lines.append(f"                _vol_max = _vol.rolling(window=60, min_periods=30).max()")
            lines.append(f"                _vol_range = _vol_max - _vol_min")
            lines.append(f"                _ri['regime_vol_{tf}'] = ((_vol - _vol_min) / _vol_range.replace(0, np.nan)).clip(0, 1)")
            # Merge into base dataframe
            lines.append(f"                dataframe = merge_informative_pair(dataframe, _ri[['date', 'regime_trend_{tf}', 'regime_vol_{tf}']].copy(), self.timeframe, '{tf}', ffill=True)")
            lines.append(f"            else:")
            lines.append(f"                dataframe['regime_trend_{tf}_{tf}'] = 0.0")
            lines.append(f"                dataframe['regime_vol_{tf}_{tf}'] = 0.5")

        # Composite score: weighted average of per-TF scores
        lines.append("")
        lines.append("        # --- Composite regime score ---")
        weight_parts = []
        total_w = 0.0
        for tf in regime_tfs:
            w = tf_weights.get(tf, 1.0)
            total_w += w
            # After merge_informative_pair, columns are suffixed with _{tf}
            weight_parts.append(f"dataframe['regime_trend_{tf}_{tf}'].fillna(0) * {w}")

        if weight_parts:
            composite_expr = " + ".join(weight_parts)
            lines.append(f"        dataframe['regime_trend_score'] = ({composite_expr}) / {total_w}")
            lines.append(f"        dataframe['regime_trend_score'] = dataframe['regime_trend_score'].clip(-1, 1)")
        else:
            lines.append(f"        dataframe['regime_trend_score'] = 0.0")

        # Composite volatility score
        vol_parts = []
        for tf in regime_tfs:
            w = tf_weights.get(tf, 1.0)
            vol_parts.append(f"dataframe['regime_vol_{tf}_{tf}'].fillna(0.5) * {w}")

        if vol_parts:
            vol_expr = " + ".join(vol_parts)
            lines.append(f"        dataframe['regime_vol_score'] = ({vol_expr}) / {total_w}")
            lines.append(f"        dataframe['regime_vol_score'] = dataframe['regime_vol_score'].clip(0, 1)")

        return '\n'.join(lines)

    def _generate_regime_entry_filter(self, regime_gene: 'RegimeGene') -> str:
        """
        Generate entry condition code that filters based on regime_trend_score.

        Only adds a filter if the RegimeGene's entry bounds are not the full
        [-1, 1] range (i.e., there's actually a restriction).
        """
        if not regime_gene or not regime_gene.enabled:
            return ""

        min_t = regime_gene.entry_trend_min
        max_t = regime_gene.entry_trend_max

        # If effectively no filter, skip
        if min_t <= -0.99 and max_t >= 0.99:
            return ""

        lines = []
        lines.append("")
        lines.append("        # Regime entry filter: only enter when trend_score is in range")
        lines.append(f"        regime_filter = (")
        if min_t > -0.99:
            lines.append(f"            (dataframe['regime_trend_score'] >= {min_t:.4f}) &")
        if max_t < 0.99:
            lines.append(f"            (dataframe['regime_trend_score'] <= {max_t:.4f}) &")
        lines.append(f"            (dataframe['regime_trend_score'].notna())")
        lines.append(f"        )")
        lines.append(f"        dataframe.loc[~regime_filter, 'enter_long'] = 0")

        return '\n'.join(lines)

    def _generate_regime_exit_filter(self, regime_gene: 'RegimeGene') -> str:
        """
        Generate exit condition code based on regime changes.

        When ``exit_on_regime_change`` is True, generates code that triggers
        an exit signal when the trend_score crosses zero against the long
        position (i.e., turns bearish).
        """
        if not regime_gene or not regime_gene.enabled:
            return ""

        if not regime_gene.exit_on_regime_change:
            return ""

        lines = []
        lines.append("")
        lines.append("        # Regime exit: exit long when trend turns bearish")
        lines.append("        regime_exit = (")
        lines.append("            (dataframe['regime_trend_score'] < -0.1) &")
        lines.append("            (dataframe['regime_trend_score'].shift(1) >= -0.1)")
        lines.append("        )")
        lines.append("        dataframe.loc[regime_exit, 'exit_long'] = 1")

        return '\n'.join(lines)
    
    def _generate_single_indicator_code(self, ind: IndicatorGene, prefix: str = "        ", 
                                        df_var: str = "informative") -> str:
        """Generate code for a single indicator calculation.
        
        This is the canonical per-indicator code generator used by both base
        timeframe and informative timeframe code paths.
        
        Args:
            ind: Indicator gene to generate code for
            prefix: Indentation prefix  
            df_var: Variable name for the dataframe (e.g. 'informative', 'dataframe')
        """
        d = df_var  # Shorthand for template readability
        
        # Normalize CDL types: strip cascading _0 suffixes from stale HOF data
        _ind_type = ind.type
        if _ind_type.startswith('CDL_'):
            from genetic_algorithm.core.strategy_gene import StrategyGene
            _ind_type = StrategyGene._strip_cdl_suffixes(_ind_type)
        
        # === STANDARD TA-LIB INDICATORS ===
        
        if _ind_type == 'RSI':
            period = ind.parameters.get('period', 14)
            return f"{prefix}{d}['rsi_{period}'] = ta.RSI({d}, timeperiod={period})"
        
        elif _ind_type == 'MACD':
            fast = ind.parameters.get('fast_period', 12)
            slow = ind.parameters.get('slow_period', 26)
            signal = ind.parameters.get('signal_period', 9)
            return (f"{prefix}macd = ta.MACD({d}, fastperiod={fast}, slowperiod={slow}, signalperiod={signal})\n"
                    f"{prefix}{d}['macd'] = macd['macd']\n"
                    f"{prefix}{d}['macdsignal'] = macd['macdsignal']\n"
                    f"{prefix}{d}['macdhist'] = macd['macdhist']")
        
        elif _ind_type == 'BBANDS':
            period = ind.parameters.get('period', 20)
            std_dev = ind.parameters.get('std_dev', 2.0)
            return (f"{prefix}bollinger = ta.BBANDS({d}, timeperiod={period}, nbdevup={std_dev}, nbdevdn={std_dev})\n"
                    f"{prefix}{d}['bb_upperband'] = bollinger['upperband']\n"
                    f"{prefix}{d}['bb_middleband'] = bollinger['middleband']\n"
                    f"{prefix}{d}['bb_lowerband'] = bollinger['lowerband']")
        
        elif _ind_type in ('EMA', 'SMA'):
            period = ind.parameters.get('period', 20)
            return f"{prefix}{d}['{_ind_type.lower()}_{period}'] = ta.{_ind_type}({d}, timeperiod={period})"
        
        elif _ind_type == 'STOCH':
            k_period = ind.parameters.get('k_period', 14)
            d_period = ind.parameters.get('d_period', 3)
            return (f"{prefix}stoch = ta.STOCH({d}, fastk_period={k_period}, slowk_period={d_period}, slowd_period={d_period})\n"
                    f"{prefix}{d}['slowk'] = stoch['slowk']\n"
                    f"{prefix}{d}['slowd'] = stoch['slowd']")
        
        elif _ind_type == 'ATR':
            period = ind.parameters.get('period', 14)
            return f"{prefix}{d}['atr_{period}'] = ta.ATR({d}, timeperiod={period})"
        
        elif _ind_type == 'ADX':
            period = ind.parameters.get('period', 14)
            return f"{prefix}{d}['adx_{period}'] = ta.ADX({d}, timeperiod={period})"
        
        elif _ind_type == 'CCI':
            period = ind.parameters.get('period', 20)
            return f"{prefix}{d}['cci_{period}'] = ta.CCI({d}, timeperiod={period})"
        
        elif _ind_type == 'MFI':
            period = ind.parameters.get('period', 14)
            return f"{prefix}{d}['mfi_{period}'] = ta.MFI({d}, timeperiod={period})"
        
        elif _ind_type == 'OBV':
            return f"{prefix}{d}['obv'] = ta.OBV({d})"
        
        elif _ind_type == 'WILLR':
            period = ind.parameters.get('period', 14)
            return f"{prefix}{d}['willr_{period}'] = ta.WILLR({d}, timeperiod={period})"
        
        elif _ind_type == 'ROC':
            period = ind.parameters.get('period', 10)
            return f"{prefix}{d}['roc_{period}'] = ta.ROC({d}, timeperiod={period})"
        
        elif _ind_type == 'TEMA':
            period = ind.parameters.get('period', 30)
            return f"{prefix}{d}['tema_{period}'] = ta.TEMA({d}, timeperiod={period})"
        
        elif _ind_type == 'KAMA':
            period = ind.parameters.get('period', 30)
            return f"{prefix}{d}['kama_{period}'] = ta.KAMA({d}, timeperiod={period})"
        
        elif _ind_type == 'AROON':
            period = ind.parameters.get('period', 14)
            return (f"{prefix}aroon = ta.AROON({d}, timeperiod={period})\n"
                    f"{prefix}{d}['aroon_up'] = aroon['aroonup']\n"
                    f"{prefix}{d}['aroon_down'] = aroon['aroondown']")
        
        elif _ind_type == 'PSAR':
            acceleration = ind.parameters.get('acceleration', 0.02)
            maximum = ind.parameters.get('maximum', 0.2)
            return f"{prefix}{d}['psar'] = ta.SAR({d}, acceleration={acceleration}, maximum={maximum})"
        
        # === COMPUTED INDICATORS (non-talib) ===
        
        elif _ind_type == 'SUPERTREND':
            period = ind.parameters.get('period', 10)
            multiplier = ind.parameters.get('multiplier', 3.0)
            # Vectorized SuperTrend using numpy arrays instead of slow Python for-loop.
            # The direction state requires a forward pass, but operating on numpy
            # arrays with direct indexing is orders of magnitude faster than
            # pandas .iloc[] inside a Python for-loop.
            lines = [
                f"{prefix}# SuperTrend ({period}, {multiplier}) — vectorized",
                f"{prefix}import numpy as np",
                f"{prefix}_hl2 = ({d}['high'] + {d}['low']) / 2",
                f"{prefix}_atr_st = ta.ATR({d}, timeperiod={period})",
                f"{prefix}_st_upper = (_hl2 + ({multiplier} * _atr_st)).values.copy()",
                f"{prefix}_st_lower = (_hl2 - ({multiplier} * _atr_st)).values.copy()",
                f"{prefix}_close = {d}['close'].values",
                f"{prefix}_st_dir = np.ones(len({d}), dtype=np.float64)",
                f"{prefix}for i in range(1, len(_close)):",
                f"{prefix}    if _st_lower[i] < _st_lower[i-1] and _close[i-1] >= _st_lower[i-1]:",
                f"{prefix}        _st_lower[i] = _st_lower[i-1]",
                f"{prefix}    if _st_upper[i] > _st_upper[i-1] and _close[i-1] <= _st_upper[i-1]:",
                f"{prefix}        _st_upper[i] = _st_upper[i-1]",
                f"{prefix}    if _st_dir[i-1] == 1.0:",
                f"{prefix}        _st_dir[i] = -1.0 if _close[i] < _st_lower[i] else 1.0",
                f"{prefix}    else:",
                f"{prefix}        _st_dir[i] = 1.0 if _close[i] > _st_upper[i] else -1.0",
                f"{prefix}{d}['supertrend_upper'] = _st_upper",
                f"{prefix}{d}['supertrend_lower'] = _st_lower",
                f"{prefix}{d}['supertrend_direction'] = _st_dir",
                f"{prefix}{d}['supertrend'] = (_st_dir == 1)",
            ]
            return '\n'.join(lines)
        
        elif _ind_type == 'ICHIMOKU':
            tenkan = ind.parameters.get('tenkan_period', 9)
            kijun = ind.parameters.get('kijun_period', 26)
            senkou_b = ind.parameters.get('senkou_b_period', 52)
            lines = [
                f"{prefix}# Ichimoku Cloud",
                f"{prefix}_hi_t = {d}['high'].rolling(window={tenkan}).max()",
                f"{prefix}_lo_t = {d}['low'].rolling(window={tenkan}).min()",
                f"{prefix}{d}['tenkan_sen'] = (_hi_t + _lo_t) / 2",
                f"{prefix}_hi_k = {d}['high'].rolling(window={kijun}).max()",
                f"{prefix}_lo_k = {d}['low'].rolling(window={kijun}).min()",
                f"{prefix}{d}['kijun_sen'] = (_hi_k + _lo_k) / 2",
                f"{prefix}{d}['senkou_span_a'] = (({d}['tenkan_sen'] + {d}['kijun_sen']) / 2).shift({kijun})",
                f"{prefix}_hi_sb = {d}['high'].rolling(window={senkou_b}).max()",
                f"{prefix}_lo_sb = {d}['low'].rolling(window={senkou_b}).min()",
                f"{prefix}{d}['senkou_span_b'] = ((_hi_sb + _lo_sb) / 2).shift({kijun})",
                f"{prefix}{d}['cloud_green'] = {d}['senkou_span_a'] > {d}['senkou_span_b']",
            ]
            return '\n'.join(lines)
        
        elif _ind_type == 'DONCHIAN':
            period = ind.parameters.get('period', 20)
            lines = [
                f"{prefix}# Donchian Channels",
                f"{prefix}{d}['donchian_upper'] = {d}['high'].rolling({period}).max()",
                f"{prefix}{d}['donchian_lower'] = {d}['low'].rolling({period}).min()",
                f"{prefix}{d}['donchian_mid'] = ({d}['donchian_upper'] + {d}['donchian_lower']) / 2",
            ]
            return '\n'.join(lines)
        
        elif _ind_type == 'VWAP':
            vwap_period = ind.parameters.get('period', 20)
            lines = [
                f"{prefix}# VWAP (rolling {vwap_period}-period)",
                f"{prefix}_tp = ({d}['high'] + {d}['low'] + {d}['close']) / 3",
                f"{prefix}{d}['vwap'] = (_tp * {d}['volume']).rolling({vwap_period}).sum() / {d}['volume'].rolling({vwap_period}).sum()",
            ]
            return '\n'.join(lines)
        
        elif _ind_type == 'CMF':
            period = ind.parameters.get('period', 20)
            lines = [
                f"{prefix}# Chaikin Money Flow",
                f"{prefix}_mfv = (({d}['close'] - {d}['low']) - ({d}['high'] - {d}['close'])) / ({d}['high'] - {d}['low'])",
                f"{prefix}_mfv = _mfv.fillna(0) * {d}['volume']",
                f"{prefix}{d}['cmf'] = _mfv.rolling({period}).sum() / {d}['volume'].rolling({period}).sum()",
            ]
            return '\n'.join(lines)
        
        elif _ind_type == 'VROC':
            period = ind.parameters.get('period', 12)
            return (f"{prefix}# Volume Rate of Change\n"
                    f"{prefix}{d}['vroc'] = (({d}['volume'] - {d}['volume'].shift({period})) / {d}['volume'].shift({period})) * 100")
        
        # === CANDLESTICK PATTERNS ===
        
        elif _ind_type == 'CDL_ENGULFING':
            return f"{prefix}{d}['cdl_engulfing'] = ta.CDLENGULFING({d})"
        
        elif _ind_type == 'CDL_HAMMER':
            return f"{prefix}{d}['cdl_hammer'] = ta.CDLHAMMER({d})"
        
        elif _ind_type == 'CDL_DOJI':
            return f"{prefix}{d}['cdl_doji'] = ta.CDLDOJI({d})"
        
        elif _ind_type == 'CDL_MORNINGSTAR':
            penetration = ind.parameters.get('penetration', 0.0)
            return f"{prefix}{d}['cdl_morningstar'] = ta.CDLMORNINGSTAR({d}, penetration={penetration})"
        
        elif _ind_type == 'CDL_EVENINGSTAR':
            penetration = ind.parameters.get('penetration', 0.0)
            return f"{prefix}{d}['cdl_eveningstar'] = ta.CDLEVENINGSTAR({d}, penetration={penetration})"
        
        elif _ind_type == 'CDL_SHOOTINGSTAR':
            return f"{prefix}{d}['cdl_shootingstar'] = ta.CDLSHOOTINGSTAR({d})"
        
        elif _ind_type == 'CDL_HARAMI':
            return f"{prefix}{d}['cdl_harami'] = ta.CDLHARAMI({d})"
        
        elif _ind_type == 'CDL_PIERCING':
            return f"{prefix}{d}['cdl_piercing'] = ta.CDLPIERCING({d})"
        
        elif _ind_type == 'CDL_DARKCLOUD':
            return f"{prefix}{d}['cdl_darkcloud'] = ta.CDLDARKCLOUDCOVER({d})"
        
        elif _ind_type == 'CDL_3WHITESOLDIERS':
            return f"{prefix}{d}['cdl_3whitesoldiers'] = ta.CDL3WHITESOLDIERS({d})"
        
        elif _ind_type == 'CDL_3BLACKCROWS':
            return f"{prefix}{d}['cdl_3blackcrows'] = ta.CDL3BLACKCROWS({d})"
        
        # Fallback: log warning for truly unknown indicator types
        logger.warning(f"Unknown indicator type '{_ind_type}' — no code generated (instance_id={ind.instance_id})")
        return f"{prefix}pass  # Unknown indicator: {_ind_type}"
    
    def _generate_indicator_code(self, indicators: List[IndicatorGene]) -> str:
        """Generate Python code for base-timeframe indicators.
        
        Deduplicates identical indicators (same type + params) to avoid
        redundant calculations (e.g., multiple CDL_DOJI with no parameters).
        Delegates per-indicator code generation to _generate_single_indicator_code().
        """
        lines = []
        
        # Deduplicate: track (type, frozen_params) to skip identical indicators
        seen_indicators = set()
        unique_indicators = []
        for ind in indicators:
            # Build a hashable key from type + sorted params
            # Normalize CDL types for dedup so CDL_HAMMER_0 and CDL_HAMMER aren't treated as different
            _dedup_type = ind.type
            if _dedup_type.startswith('CDL_'):
                from genetic_algorithm.core.strategy_gene import StrategyGene
                _dedup_type = StrategyGene._strip_cdl_suffixes(_dedup_type)
            params_key = tuple(sorted(ind.parameters.items())) if ind.parameters else ()
            dedup_key = (_dedup_type, params_key)
            if dedup_key in seen_indicators:
                logger.debug(f"Skipping duplicate indicator: {ind.type} (instance_id={ind.instance_id})")
                continue
            seen_indicators.add(dedup_key)
            unique_indicators.append(ind)
        
        if len(unique_indicators) < len(indicators):
            logger.info(f"[DEDUP] Removed {len(indicators) - len(unique_indicators)} duplicate indicator(s)")
        
        for ind in unique_indicators:
            ind_code = self._generate_single_indicator_code(ind, prefix="        ", df_var="dataframe")
            lines.append(ind_code)
        
        return '\n'.join(lines) if lines else "        # No indicators"
    
    def _add_replacement_conditions(self, strategy_gene: StrategyGene, count: int, is_entry: bool) -> None:
        """
        Add replacement conditions from the strategy's existing indicators.
        
        Called when pre-code-gen validation detects that too few conditions
        will survive the indicator-existence filter.
        
        Args:
            strategy_gene: Strategy gene to modify in-place
            count: Number of conditions to add
            is_entry: True for entry, False for exit
        """
        import random
        
        existing = strategy_gene.entry_conditions if is_entry else strategy_gene.exit_conditions
        existing_keys = {(c.indicator, c.operator, str(c.threshold)) for c in existing}
        
        added = 0
        attempts = 0
        max_attempts = count * 10
        
        while added < count and attempts < max_attempts:
            attempts += 1
            # Pick from indicators actually present in strategy
            if not strategy_gene.indicators:
                break
            ind = random.choice(strategy_gene.indicators)
            cond = self._generate_condition_for_indicator(ind, is_entry)
            if cond:
                key = (cond.indicator, cond.operator, str(cond.threshold))
                if key not in existing_keys:
                    cond.logic = 'AND'
                    existing.append(cond)
                    existing_keys.add(key)
                    added += 1
                    logger.debug(f"Added replacement {'entry' if is_entry else 'exit'} condition: "
                               f"{cond.indicator} {cond.operator} {cond.threshold}")
        
        if added < count:
            logger.warning(f"Could only add {added}/{count} replacement {'entry' if is_entry else 'exit'} conditions")
    
    def _generate_condition_code(self, conditions: List[ConditionGene], indicators: List[IndicatorGene], is_entry: bool, signal_col_override: str = None) -> str:
        """Generate Python code for entry/exit conditions.
        
        Args:
            conditions: List of condition genes
            indicators: List of indicator genes
            is_entry: True for entry signals, False for exit
            signal_col_override: Override signal column name (e.g. 'enter_short', 'exit_short')
        """
        signal_col = signal_col_override or ('enter_long' if is_entry else 'exit_long')
        if not conditions:
            return f"        dataframe['{signal_col}'] = 0\n"
        
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
        
        # Combine conditions based on per-condition logic operators
        # Group conditions: OR conditions are grouped together, then ANDed with AND conditions
        # This creates: (AND_cond1) & (AND_cond2) & (OR_cond1 | OR_cond2 | OR_cond3)
        # This way, all AND conditions must be true, plus at least one OR condition must be true
        
        and_exprs = []
        or_exprs = []
        
        for cond, expr in zip(valid_conditions, condition_exprs):
            logic = getattr(cond, 'logic', 'AND')
            if logic == 'OR':
                or_exprs.append(expr)
            else:
                and_exprs.append(expr)
        
        # Build the final combined expression
        parts = []
        
        # Each AND condition is its own required term
        for expr in and_exprs:
            parts.append(f"({expr})")
        
        # OR conditions are grouped together as a single term
        if or_exprs:
            if len(or_exprs) == 1:
                parts.append(f"({or_exprs[0]})")
            else:
                or_combined = ' |\n                '.join(f"({expr})" for expr in or_exprs)
                parts.append(f"({or_combined})")
        
        # If we only have OR conditions and no AND, at least one must fire
        # If we only have AND conditions, all must fire (original behavior)
        # If we have both, all AND + at least one OR must fire
        
        # For entry signals, always require volume > 0 to avoid trading on bad data
        if is_entry:
            parts.append("(dataframe['volume'] > 0)")
        
        combined_condition = ' &\n            '.join(parts)
        
        code = f"""        conditions = (
            {combined_condition}
        )
        dataframe.loc[conditions, '{signal_col}'] = 1
"""
        
        return code
    
    def _condition_has_valid_indicator(self, condition: ConditionGene, indicators: List[IndicatorGene]) -> bool:
        """
        Check if a condition references a valid indicator and has a valid operator.
        
        Validates both:
        1. The condition's indicator exists in the strategy
        2. The condition's operator is valid for that indicator type
        
        Args:
            condition: Condition to validate
            indicators: List of indicators in the strategy
            
        Returns:
            True if the condition is valid, False otherwise
        """
        # Extract indicator type from condition reference
        indicator_ref = condition.indicator
        indicator_type = resolve_indicator_type(indicator_ref)
        
        # Check if any indicator in the list matches this type
        indicator_exists = False
        actual_type = indicator_type  # May be refined by matching indicator
        for ind in indicators:
            # Match by instance_id if available, otherwise by type
            if ind.instance_id and ind.instance_id == indicator_ref:
                indicator_exists = True
                actual_type = ind.type
                break
            elif ind.type == indicator_type:
                indicator_exists = True
                actual_type = ind.type
                break
        
        if not indicator_exists:
            return False
        
        # Validate operator is supported for this indicator type
        if not is_valid_operator(actual_type, condition.operator):
            return False
        
        return True
    
    def _generate_single_condition(self, condition: ConditionGene, indicators: List[IndicatorGene]) -> str:
        """Generate a single condition expression.
        
        Handles both type-based references (e.g., 'RSI') and instance-based references (e.g., 'RSI_0').
        For informative timeframe indicators, appends the TF suffix (e.g., rsi_14_1h).
        
        Supported operators:
        - '<', '>', 'cross_above', 'cross_below': Standard comparisons
        - 'increasing': Value rising over lookback bars (slope > 0)
        - 'decreasing': Value falling over lookback bars (slope < 0)
        - 'between': Value between threshold (lower) and threshold_upper
        - 'value_above_ago': Current value > value from lookback bars ago
        """
        # Extract indicator type from condition reference
        # Handle both 'RSI' and 'RSI_0' formats, but preserve full type for CDL_* patterns
        indicator_ref = condition.indicator
        
        # CDL_* patterns have underscore in type name, not instance ID
        # Instance IDs look like 'RSI_0', 'EMA_1' (type + number)
        # Patterns look like 'CDL_HAMMER', 'CDL_ENGULFING' (CDL + descriptor)
        # BUT patterns can also have instance IDs like 'CDL_HAMMER_0'
        if indicator_ref.startswith('CDL_'):
            # Strip ALL trailing numeric suffixes to handle cascaded names
            # CDL_HAMMER_0 -> CDL_HAMMER, CDL_ENGULFING_0_0_0 -> CDL_ENGULFING
            indicator_type = indicator_ref
            while '_' in indicator_type:
                parts = indicator_type.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    indicator_type = parts[0]
                else:
                    break
            # Safety: never strip below CDL_ base
            if not (indicator_type.startswith('CDL_') and len(indicator_type) > 4):
                indicator_type = indicator_ref  # revert if stripping went too far
        elif '_' in indicator_ref:
            # Check if this looks like an instance ID (ends with digit)
            parts = indicator_ref.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                indicator_type = parts[0]  # e.g., 'RSI_0' -> 'RSI'
            else:
                indicator_type = indicator_ref  # Unknown format, use as-is
        else:
            indicator_type = indicator_ref
        
        # Find the specific indicator instance or use first matching type
        target_indicator = None
        for ind in indicators:
            # Match by instance_id if available, otherwise by type
            if ind.instance_id and ind.instance_id == indicator_ref:
                target_indicator = ind
                break
            elif ind.type == indicator_type and not target_indicator:
                target_indicator = ind
        
        # If we found the target indicator, use its canonical type.
        # This handles instance_ids that embed a timeframe component
        # (e.g., 'EMA_1h_0' → parsed as indicator_type='EMA_1h', but
        # the actual type is 'EMA').
        if target_indicator:
            indicator_type = target_indicator.type
        
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
        
        # --- Handle advanced operators generically for all indicators ---
        # These operators (increasing, decreasing, between, value_above_ago) work on any
        # numeric column, so we resolve the primary column name and delegate.
        if condition.operator in ('increasing', 'decreasing', 'between', 'value_above_ago'):
            primary_col = self._resolve_primary_column(indicator_type, indicator_ref,
                                                        indicator_periods, tf_suffix, target_indicator)
            if primary_col:
                result = self._generate_advanced_operator_condition(primary_col, condition)
                if result:
                    return result
        
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
            elif condition.operator == '>':
                return f"(dataframe['{macd_col}'] > {condition.threshold})"
            elif condition.operator == '<':
                return f"(dataframe['{macd_col}'] < {condition.threshold})"
        
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
            elif condition.operator == 'cross_above':
                return f"(qtpylib.crossed_above(dataframe['{col}'], {condition.threshold}))"
            elif condition.operator == 'cross_below':
                return f"(qtpylib.crossed_below(dataframe['{col}'], {condition.threshold}))"
        
        elif indicator_type == 'ADX':
            # Use actual ADX period if available (try instance_id first, then type)
            period = indicator_periods.get(indicator_ref, indicator_periods.get('ADX', 14))
            col = f"adx_{period}{tf_suffix}"
            if condition.operator == '>':
                return f"(dataframe['{col}'] > {condition.threshold})"
            elif condition.operator == '<':
                return f"(dataframe['{col}'] < {condition.threshold})"
            elif condition.operator == 'cross_above':
                return f"(qtpylib.crossed_above(dataframe['{col}'], {condition.threshold}))"
            elif condition.operator == 'cross_below':
                return f"(qtpylib.crossed_below(dataframe['{col}'], {condition.threshold}))"
        
        elif indicator_type == 'ATR':
            # ATR conditions — compare ATR value against threshold
            # ATR threshold is a ratio of close price for portability
            period = indicator_periods.get(indicator_ref, indicator_periods.get('ATR', 14))
            col = f"atr_{period}{tf_suffix}"
            close = f"close{tf_suffix}" if tf_suffix else "close"
            threshold = condition.threshold if condition.threshold not in (None, 0) else 0.01
            if condition.operator == '>':
                return f"(dataframe['{col}'] > dataframe['{close}'] * {threshold})"
            elif condition.operator == '<':
                return f"(dataframe['{col}'] < dataframe['{close}'] * {threshold})"
            elif condition.operator == 'cross_above':
                return f"(qtpylib.crossed_above(dataframe['{col}'], dataframe['{close}'] * {threshold}))"
            elif condition.operator == 'cross_below':
                return f"(qtpylib.crossed_below(dataframe['{col}'], dataframe['{close}'] * {threshold}))"
        
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
        
        # === NEW INDICATOR CONDITIONS ===
        
        elif indicator_type == 'SUPERTREND':
            close = f"close{tf_suffix}" if tf_suffix else "close"
            if condition.operator == 'cross_above':
                return f"(dataframe['supertrend'] == True)"
            elif condition.operator == 'cross_below':
                return f"(dataframe['supertrend'] == False)"
            elif condition.operator == '>':
                return f"(dataframe['{close}'] > dataframe['supertrend_lower'])"
            elif condition.operator == '<':
                return f"(dataframe['{close}'] < dataframe['supertrend_upper'])"
        
        elif indicator_type == 'ICHIMOKU':
            if condition.operator == 'cross_above':
                return f"(dataframe['tenkan_sen'] > dataframe['kijun_sen'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['tenkan_sen'] < dataframe['kijun_sen'])"
            elif condition.operator == '>':
                return f"(dataframe['cloud_green'] == True)"
            elif condition.operator == '<':
                return f"(dataframe['cloud_green'] == False)"
        
        elif indicator_type == 'DONCHIAN':
            close = f"close{tf_suffix}" if tf_suffix else "close"
            if condition.operator == 'cross_above':
                return f"(dataframe['{close}'] > dataframe['donchian_upper'].shift(1))"
            elif condition.operator == 'cross_below':
                return f"(dataframe['{close}'] < dataframe['donchian_lower'].shift(1))"
            elif condition.operator == '>':
                return f"(dataframe['{close}'] > dataframe['donchian_mid'])"
            elif condition.operator == '<':
                return f"(dataframe['{close}'] < dataframe['donchian_mid'])"
        
        elif indicator_type == 'VWAP':
            close = f"close{tf_suffix}" if tf_suffix else "close"
            if condition.operator == 'cross_above':
                return f"(dataframe['{close}'] > dataframe['vwap'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['{close}'] < dataframe['vwap'])"
            elif condition.operator == '>':
                return f"(dataframe['{close}'] > dataframe['vwap'])"
            elif condition.operator == '<':
                return f"(dataframe['{close}'] < dataframe['vwap'])"
        
        elif indicator_type == 'PSAR':
            close = f"close{tf_suffix}" if tf_suffix else "close"
            if condition.operator == 'cross_above':
                return f"(dataframe['{close}'] > dataframe['psar'])"
            elif condition.operator == 'cross_below':
                return f"(dataframe['{close}'] < dataframe['psar'])"
            elif condition.operator == '>':
                return f"(dataframe['{close}'] > dataframe['psar'])"
            elif condition.operator == '<':
                return f"(dataframe['{close}'] < dataframe['psar'])"
        
        elif indicator_type == 'CMF':
            threshold = condition.threshold if condition.threshold is not None else 0.1
            if condition.operator in ['>', 'cross_above']:
                return f"(dataframe['cmf'] > {threshold})"
            elif condition.operator in ['<', 'cross_below']:
                return f"(dataframe['cmf'] < {threshold})"
        
        elif indicator_type == 'VROC':
            threshold = condition.threshold if condition.threshold is not None else 100
            if condition.operator in ['>', 'cross_above']:
                return f"(dataframe['vroc'] > {threshold})"
            elif condition.operator in ['<', 'cross_below']:
                return f"(dataframe['vroc'] < -{abs(threshold)})"
        
        # === CANDLESTICK PATTERN CONDITIONS ===
        # Patterns return: >0 bullish, <0 bearish, ==0 no pattern
        
        elif indicator_type == 'CDL_ENGULFING':
            if condition.operator == '>':
                return "(dataframe['cdl_engulfing'] > 0)"  # Bullish engulfing
            elif condition.operator == '<':
                return "(dataframe['cdl_engulfing'] < 0)"  # Bearish engulfing
            else:
                return "(dataframe['cdl_engulfing'] != 0)"  # Any engulfing
        
        elif indicator_type == 'CDL_HARAMI':
            if condition.operator == '>':
                return "(dataframe['cdl_harami'] > 0)"  # Bullish harami
            elif condition.operator == '<':
                return "(dataframe['cdl_harami'] < 0)"  # Bearish harami
            else:
                return "(dataframe['cdl_harami'] != 0)"  # Any harami
        
        elif indicator_type == 'CDL_HAMMER':
            return "(dataframe['cdl_hammer'] != 0)"  # Hammer detected
        
        elif indicator_type == 'CDL_MORNINGSTAR':
            return "(dataframe['cdl_morningstar'] != 0)"  # Morning star detected
        
        elif indicator_type == 'CDL_EVENINGSTAR':
            return "(dataframe['cdl_eveningstar'] != 0)"  # Evening star detected
        
        elif indicator_type == 'CDL_SHOOTINGSTAR':
            return "(dataframe['cdl_shootingstar'] != 0)"  # Shooting star detected
        
        elif indicator_type == 'CDL_DOJI':
            if condition.operator == '>':
                return "(dataframe['cdl_doji'] > 0)"  # Doji detected (bullish signal)
            elif condition.operator == '<':
                return "(dataframe['cdl_doji'] < 0)"  # Doji detected (bearish signal)
            else:
                return "(dataframe['cdl_doji'] != 0)"  # Any doji detected
        
        elif indicator_type == 'CDL_PIERCING':
            return "(dataframe['cdl_piercing'] != 0)"  # Piercing line detected
        
        elif indicator_type == 'CDL_DARKCLOUD':
            return "(dataframe['cdl_darkcloud'] != 0)"  # Dark cloud cover detected
        
        elif indicator_type == 'CDL_3WHITESOLDIERS':
            return "(dataframe['cdl_3whitesoldiers'] != 0)"  # Three white soldiers detected
        
        elif indicator_type == 'CDL_3BLACKCROWS':
            return "(dataframe['cdl_3blackcrows'] != 0)"  # Three black crows detected
        
        # Default fallback - return None and log warning instead of always-true condition
        logger.warning(f"No condition handler for indicator type '{indicator_type}' "
                       f"with operator '{condition.operator}'. Skipping condition.")
        return None
    
    def _resolve_primary_column(self, indicator_type: str, indicator_ref: str,
                                 indicator_periods: dict, tf_suffix: str,
                                 target_indicator) -> str:
        """Resolve the primary dataframe column for an indicator type.
        
        Used by advanced operators (increasing, decreasing, between, value_above_ago)
        to determine which column to operate on.
        
        Returns:
            Column name string, or empty string if unknown.
        """
        if indicator_type == 'RSI':
            period = indicator_periods.get(indicator_ref, indicator_periods.get('RSI', 14))
            return f"rsi_{period}{tf_suffix}"
        elif indicator_type == 'MACD':
            return f"macd{tf_suffix}"
        elif indicator_type == 'STOCH':
            return f"slowk{tf_suffix}"
        elif indicator_type == 'CCI':
            period = indicator_periods.get(indicator_ref, indicator_periods.get('CCI', 20))
            return f"cci_{period}{tf_suffix}"
        elif indicator_type == 'ADX':
            period = indicator_periods.get(indicator_ref, indicator_periods.get('ADX', 14))
            return f"adx_{period}{tf_suffix}"
        elif indicator_type in ('EMA', 'SMA'):
            period = indicator_periods.get(indicator_ref, indicator_periods.get(indicator_type, 20))
            return f"{indicator_type.lower()}_{period}{tf_suffix}"
        elif indicator_type == 'BBANDS':
            return f"bb_middleband{tf_suffix}"
        elif indicator_type == 'CMF':
            return "cmf"
        elif indicator_type == 'VROC':
            return "vroc"
        elif indicator_type == 'ATR':
            period = indicator_periods.get(indicator_ref, indicator_periods.get('ATR', 14))
            return f"atr_{period}{tf_suffix}"
        elif indicator_type == 'PSAR':
            return "psar"
        elif indicator_type == 'SUPERTREND':
            return "supertrend_lower"
        elif indicator_type == 'ICHIMOKU':
            return "tenkan_sen"
        elif indicator_type == 'DONCHIAN':
            return "donchian_mid"
        elif indicator_type == 'VWAP':
            return "vwap"
        return ""
    
    def _generate_advanced_operator_condition(self, col: str, condition: ConditionGene) -> str:
        """Generate code for advanced operators (increasing, decreasing, between, value_above_ago).
        
        These operators work on any numeric column and provide richer signal logic than
        simple threshold comparisons.
        
        Args:
            col: The dataframe column name (e.g., 'rsi_14', 'macd')
            condition: The ConditionGene with operator, threshold, lookback, etc.
            
        Returns:
            Python expression string for the condition, or empty string if not applicable.
        """
        lookback = getattr(condition, 'lookback', 3)
        lookback = max(2, lookback)  # Minimum 2 bars for slope/comparison
        
        if condition.operator == 'increasing':
            # Value is rising: current value > value N bars ago
            return f"(dataframe['{col}'] > dataframe['{col}'].shift({lookback}))"
        
        elif condition.operator == 'decreasing':
            # Value is falling: current value < value N bars ago
            return f"(dataframe['{col}'] < dataframe['{col}'].shift({lookback}))"
        
        elif condition.operator == 'between':
            # Value is between lower and upper thresholds
            lower = min(condition.threshold, getattr(condition, 'threshold_upper', condition.threshold + 10))
            upper = max(condition.threshold, getattr(condition, 'threshold_upper', condition.threshold + 10))
            return f"((dataframe['{col}'] > {lower}) & (dataframe['{col}'] < {upper}))"
        
        elif condition.operator == 'value_above_ago':
            # Current value exceeds its value from N bars ago by threshold amount
            return f"(dataframe['{col}'] - dataframe['{col}'].shift({lookback}) > {condition.threshold})"
        
        return ""
