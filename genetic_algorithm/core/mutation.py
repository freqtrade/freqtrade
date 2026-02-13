"""
Mutation Operators

Implements various mutation strategies for introducing
variation into strategies.
"""

import random
from typing import Dict, Any, Optional

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.utils.indicator_factory import create_random_indicator


def mutate_parameters(individual: Individual, mutation_rate: float,
                     config: Dict[str, Any]) -> Individual:
    """
    Mutate numeric parameters of indicators and conditions.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each parameter
        config: Configuration with parameter ranges
        
    Returns:
        Mutated individual
    """
    mutated_gene = individual.strategy_gene.copy()
    indicator_config = config.get('indicators', {})
    strategy_constraints = config.get('strategy_constraints', {})
    
    mutations_applied = []
    
    # Mutate indicator parameters
    for i, indicator in enumerate(mutated_gene.indicators):
        if random.random() < mutation_rate:
            ind_config = indicator_config.get(indicator.type, {})
            
            # Mutate based on indicator type
            if indicator.type == 'RSI' and 'period' in indicator.parameters:
                period_range = ind_config.get('period', [7, 21])
                indicator.parameters['period'] = random.randint(*period_range)
                mutations_applied.append(f"RSI_period_{i}")
            
            elif indicator.type == 'MACD':
                if 'fast_period' in indicator.parameters and random.random() < 0.5:
                    indicator.parameters['fast_period'] = random.randint(
                        *ind_config.get('fast_period', [8, 21]))
                    mutations_applied.append(f"MACD_fast_{i}")
                if 'slow_period' in indicator.parameters and random.random() < 0.5:
                    indicator.parameters['slow_period'] = random.randint(
                        *ind_config.get('slow_period', [21, 50]))
                    mutations_applied.append(f"MACD_slow_{i}")
                if 'signal_period' in indicator.parameters and random.random() < 0.5:
                    indicator.parameters['signal_period'] = random.randint(
                        *ind_config.get('signal_period', [5, 14]))
                    mutations_applied.append(f"MACD_signal_{i}")
            
            elif indicator.type == 'BBANDS':
                if 'period' in indicator.parameters and random.random() < 0.5:
                    indicator.parameters['period'] = random.randint(
                        *ind_config.get('period', [15, 30]))
                    mutations_applied.append(f"BBANDS_period_{i}")
                if 'std_dev' in indicator.parameters and random.random() < 0.5:
                    indicator.parameters['std_dev'] = random.uniform(
                        *ind_config.get('std_dev', [1.5, 3.0]))
                    mutations_applied.append(f"BBANDS_std_{i}")
            
            elif indicator.type in ['EMA', 'SMA'] and 'period' in indicator.parameters:
                period_range = ind_config.get('period', [10, 50])
                indicator.parameters['period'] = random.randint(*period_range)
                mutations_applied.append(f"{indicator.type}_period_{i}")
            
            elif indicator.type == 'STOCH':
                if 'k_period' in indicator.parameters and random.random() < 0.5:
                    indicator.parameters['k_period'] = random.randint(
                        *ind_config.get('k_period', [5, 21]))
                    mutations_applied.append(f"STOCH_k_{i}")
                if 'd_period' in indicator.parameters and random.random() < 0.5:
                    indicator.parameters['d_period'] = random.randint(
                        *ind_config.get('d_period', [3, 14]))
                    mutations_applied.append(f"STOCH_d_{i}")
            
            elif indicator.type in ['ATR', 'ADX', 'CCI'] and 'period' in indicator.parameters:
                period_range = ind_config.get('period', [10, 20])
                indicator.parameters['period'] = random.randint(*period_range)
                mutations_applied.append(f"{indicator.type}_period_{i}")
            
            # Mutate weight
            if random.random() < 0.3:
                indicator.weight = random.uniform(0.3, 1.0)
                mutations_applied.append(f"weight_{i}")
    
    # Mutate condition thresholds
    for i, condition in enumerate(mutated_gene.entry_conditions):
        if random.random() < mutation_rate:
            ind_config = indicator_config.get(condition.indicator, {})
            
            if condition.indicator == 'RSI':
                threshold_range = ind_config.get('buy_threshold', [20, 40])
                condition.threshold = random.randint(*threshold_range)
                mutations_applied.append(f"entry_RSI_threshold_{i}")
            elif condition.indicator == 'CCI':
                threshold_range = ind_config.get('buy_threshold', [-200, -100])
                condition.threshold = random.randint(*threshold_range)
                mutations_applied.append(f"entry_CCI_threshold_{i}")
    
    for i, condition in enumerate(mutated_gene.exit_conditions):
        if random.random() < mutation_rate:
            ind_config = indicator_config.get(condition.indicator, {})
            
            if condition.indicator == 'RSI':
                threshold_range = ind_config.get('sell_threshold', [60, 80])
                condition.threshold = random.randint(*threshold_range)
                mutations_applied.append(f"exit_RSI_threshold_{i}")
            elif condition.indicator == 'CCI':
                threshold_range = ind_config.get('sell_threshold', [100, 200])
                condition.threshold = random.randint(*threshold_range)
                mutations_applied.append(f"exit_CCI_threshold_{i}")
    
    # Mutate stoploss
    if random.random() < mutation_rate:
        stoploss_range = strategy_constraints.get('stoploss_range', [-0.20, -0.05])
        mutated_gene.stoploss = random.uniform(*stoploss_range)
        mutations_applied.append("stoploss")
    
    # Mutate ROI values
    if random.random() < mutation_rate:
        roi_range = strategy_constraints.get('roi_range', [0.01, 0.10])
        mutated_gene.minimal_roi = {
            0: random.uniform(roi_range[0] * 2, roi_range[1]),
            30: random.uniform(roi_range[0] * 1.5, roi_range[1] * 0.7),
            60: random.uniform(roi_range[0], roi_range[1] * 0.5),
        }
        mutations_applied.append("roi")
    
    # Create new individual with mutation record
    new_individual = Individual(strategy_gene=mutated_gene, parent_ids=[individual.id])
    new_individual.mutations = individual.mutations + [{
        'type': 'parameter',
        'rate': mutation_rate,
        'applied': mutations_applied
    }]
    
    return new_individual


def mutate_indicators(individual: Individual, mutation_rate: float,
                     config: Dict[str, Any]) -> Individual:
    """
    Mutate indicator set (add, remove, or replace indicators).
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation
        config: Configuration with available indicators
        
    Returns:
        Mutated individual
    """
    mutated_gene = individual.strategy_gene.copy()
    indicator_config = config.get('indicators', {})
    available_indicators = indicator_config.get('available', [])
    max_indicators = indicator_config.get('max_per_strategy', 5)
    min_indicators = indicator_config.get('min_per_strategy', 2)
    
    mutations_applied = []
    
    # Choose mutation operation
    operations = []
    if len(mutated_gene.indicators) < max_indicators:
        operations.append('add')
    if len(mutated_gene.indicators) > min_indicators:
        operations.append('remove')
    operations.append('replace')
    
    operation = random.choice(operations)
    
    if operation == 'add':
        # Add a new random indicator
        existing_types = {ind.type for ind in mutated_gene.indicators}
        available_new = [t for t in available_indicators if t not in existing_types]
        
        if available_new:
            new_type = random.choice(available_new)
            new_indicator = _create_random_indicator(new_type, indicator_config)
            mutated_gene.indicators.append(new_indicator)
            mutations_applied.append(f"add_{new_type}")
    
    elif operation == 'remove':
        # Remove a random indicator
        if len(mutated_gene.indicators) > min_indicators:
            removed = random.choice(mutated_gene.indicators)
            mutated_gene.indicators.remove(removed)
            mutations_applied.append(f"remove_{removed.type}")
            
            # Clean up conditions that reference removed indicator
            mutated_gene.entry_conditions = [
                c for c in mutated_gene.entry_conditions 
                if c.indicator != removed.type
            ]
            mutated_gene.exit_conditions = [
                c for c in mutated_gene.exit_conditions 
                if c.indicator != removed.type
            ]
    
    elif operation == 'replace':
        # Replace an indicator with a different type
        if mutated_gene.indicators:
            idx = random.randrange(len(mutated_gene.indicators))
            old_type = mutated_gene.indicators[idx].type
            
            # Choose a different indicator type
            available_new = [t for t in available_indicators if t != old_type]
            if available_new:
                new_type = random.choice(available_new)
                new_indicator = _create_random_indicator(new_type, indicator_config)
                mutated_gene.indicators[idx] = new_indicator
                mutations_applied.append(f"replace_{old_type}_with_{new_type}")
                
                # Update conditions that referenced the old indicator
                for condition in mutated_gene.entry_conditions:
                    if condition.indicator == old_type:
                        condition.indicator = new_type
                for condition in mutated_gene.exit_conditions:
                    if condition.indicator == old_type:
                        condition.indicator = new_type
    
    # Create new individual with mutation record
    new_individual = Individual(strategy_gene=mutated_gene, parent_ids=[individual.id])
    new_individual.mutations = individual.mutations + [{
        'type': 'indicator',
        'operation': operation,
        'applied': mutations_applied
    }]
    
    return new_individual


def _create_random_indicator(indicator_type: str, indicator_config: Dict[str, Any]) -> IndicatorGene:
    """Helper function to create a random indicator of given type."""
    return create_random_indicator(indicator_type, indicator_config)


def mutate_conditions(individual: Individual, mutation_rate: float,
                     config: Dict[str, Any]) -> Individual:
    """
    Mutate entry/exit conditions.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation
        config: Configuration with condition constraints
        
    Returns:
        Mutated individual
    """
    mutated_gene = individual.strategy_gene.copy()
    indicator_config = config.get('indicators', {})
    
    mutations_applied = []
    
    # Mutate entry conditions
    if mutated_gene.entry_conditions and random.random() < mutation_rate:
        idx = random.randrange(len(mutated_gene.entry_conditions))
        condition = mutated_gene.entry_conditions[idx]
        
        # Choose what to mutate
        mutation_type = random.choice(['operator', 'logic', 'threshold'])
        
        if mutation_type == 'operator':
            # Change comparison operator
            operators = ['<', '>', 'cross_above', 'cross_below']
            condition.operator = random.choice(operators)
            mutations_applied.append(f"entry_operator_{idx}")
        
        elif mutation_type == 'logic':
            # Toggle logic operator
            condition.logic = 'OR' if condition.logic == 'AND' else 'AND'
            mutations_applied.append(f"entry_logic_{idx}")
        
        elif mutation_type == 'threshold':
            # Mutate threshold based on indicator type
            ind_config = indicator_config.get(condition.indicator, {})
            if condition.indicator == 'RSI':
                threshold_range = ind_config.get('buy_threshold', [20, 40])
                condition.threshold = random.randint(*threshold_range)
            elif condition.indicator == 'STOCH':
                threshold_range = ind_config.get('k_threshold', [20, 40])
                condition.threshold = random.randint(*threshold_range)
            elif condition.indicator == 'CCI':
                threshold_range = ind_config.get('buy_threshold', [-200, -100])
                condition.threshold = random.randint(*threshold_range)
            mutations_applied.append(f"entry_threshold_{idx}")
    
    # Mutate exit conditions
    if mutated_gene.exit_conditions and random.random() < mutation_rate:
        idx = random.randrange(len(mutated_gene.exit_conditions))
        condition = mutated_gene.exit_conditions[idx]
        
        # Choose what to mutate
        mutation_type = random.choice(['operator', 'logic', 'threshold'])
        
        if mutation_type == 'operator':
            operators = ['<', '>', 'cross_above', 'cross_below']
            condition.operator = random.choice(operators)
            mutations_applied.append(f"exit_operator_{idx}")
        
        elif mutation_type == 'logic':
            condition.logic = 'OR' if condition.logic == 'AND' else 'AND'
            mutations_applied.append(f"exit_logic_{idx}")
        
        elif mutation_type == 'threshold':
            ind_config = indicator_config.get(condition.indicator, {})
            if condition.indicator == 'RSI':
                threshold_range = ind_config.get('sell_threshold', [60, 80])
                condition.threshold = random.randint(*threshold_range)
            elif condition.indicator == 'STOCH':
                threshold_range = ind_config.get('d_threshold', [60, 80])
                condition.threshold = random.randint(*threshold_range)
            elif condition.indicator == 'CCI':
                threshold_range = ind_config.get('sell_threshold', [100, 200])
                condition.threshold = random.randint(*threshold_range)
            mutations_applied.append(f"exit_threshold_{idx}")
    
    # Possibly add a new condition
    if random.random() < mutation_rate * 0.5:
        available_indicators = [ind.type for ind in mutated_gene.indicators]
        if available_indicators:
            # Add new entry condition
            if len(mutated_gene.entry_conditions) < 3:
                indicator = random.choice(available_indicators)
                new_condition = _create_random_condition(indicator, True, indicator_config)
                if new_condition:
                    mutated_gene.entry_conditions.append(new_condition)
                    mutations_applied.append(f"add_entry_condition_{indicator}")
            
            # Add new exit condition
            if len(mutated_gene.exit_conditions) < 3:
                indicator = random.choice(available_indicators)
                new_condition = _create_random_condition(indicator, False, indicator_config)
                if new_condition:
                    mutated_gene.exit_conditions.append(new_condition)
                    mutations_applied.append(f"add_exit_condition_{indicator}")
    
    # Possibly remove a condition
    if random.random() < mutation_rate * 0.3:
        if len(mutated_gene.entry_conditions) > 1:
            removed = mutated_gene.entry_conditions.pop(random.randrange(len(mutated_gene.entry_conditions)))
            mutations_applied.append(f"remove_entry_condition_{removed.indicator}")
        
        if len(mutated_gene.exit_conditions) > 0:
            removed = mutated_gene.exit_conditions.pop(random.randrange(len(mutated_gene.exit_conditions)))
            mutations_applied.append(f"remove_exit_condition_{removed.indicator}")
    
    # Create new individual with mutation record
    new_individual = Individual(strategy_gene=mutated_gene, parent_ids=[individual.id])
    new_individual.mutations = individual.mutations + [{
        'type': 'condition',
        'applied': mutations_applied
    }]
    
    return new_individual


def _create_random_condition(indicator_type: str, is_entry: bool, 
                            indicator_config: Dict[str, Any]) -> Optional[ConditionGene]:
    """Helper function to create a random condition for an indicator."""
    ind_config = indicator_config.get(indicator_type, {})
    
    if indicator_type == 'RSI':
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
    
    elif indicator_type == 'MACD':
        return ConditionGene(
            indicator='MACD',
            operator='cross_above' if is_entry else 'cross_below',
            threshold=0,
            logic=random.choice(['AND', 'OR'])
        )
    
    elif indicator_type == 'STOCH':
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
    
    elif indicator_type == 'CCI':
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
    
    return None


def mutate_structure(individual: Individual, mutation_rate: float,
                    config: Dict[str, Any]) -> Individual:
    """
    Mutate structural parameters (timeframe, stoploss, roi).
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation
        config: Configuration with structural constraints
        
    Returns:
        Mutated individual
    """
    mutated_gene = individual.strategy_gene.copy()
    strategy_constraints = config.get('strategy_constraints', {})
    
    mutations_applied = []
    
    # Mutate timeframe
    if random.random() < mutation_rate:
        available_timeframes = strategy_constraints.get('timeframes', ['5m', '15m', '1h'])
        mutated_gene.timeframe = random.choice(available_timeframes)
        mutations_applied.append(f"timeframe_{mutated_gene.timeframe}")
    
    # Mutate stoploss
    if random.random() < mutation_rate:
        stoploss_range = strategy_constraints.get('stoploss_range', [-0.20, -0.05])
        mutated_gene.stoploss = random.uniform(*stoploss_range)
        mutations_applied.append("stoploss")
    
    # Mutate ROI
    if random.random() < mutation_rate:
        roi_range = strategy_constraints.get('roi_range', [0.01, 0.10])
        mutated_gene.minimal_roi = {
            0: random.uniform(roi_range[0] * 2, roi_range[1]),
            30: random.uniform(roi_range[0] * 1.5, roi_range[1] * 0.7),
            60: random.uniform(roi_range[0], roi_range[1] * 0.5),
        }
        mutations_applied.append("roi")
    
    # Mutate trailing stop
    if random.random() < mutation_rate:
        mutated_gene.trailing_stop = not mutated_gene.trailing_stop
        mutations_applied.append(f"trailing_stop_{mutated_gene.trailing_stop}")
        
        # If enabling trailing stop, set appropriate parameters
        if mutated_gene.trailing_stop:
            mutated_gene.trailing_stop_positive = random.uniform(0.01, 0.03)
            mutated_gene.trailing_stop_positive_offset = random.uniform(0.02, 0.05)
            mutations_applied.append("trailing_stop_params")
        else:
            mutated_gene.trailing_stop_positive = None
            mutated_gene.trailing_stop_positive_offset = None
    
    # Create new individual with mutation record
    new_individual = Individual(strategy_gene=mutated_gene, parent_ids=[individual.id])
    new_individual.mutations = individual.mutations + [{
        'type': 'structure',
        'applied': mutations_applied
    }]
    
    return new_individual


def mutate(individual: Individual, mutation_rate: float,
          config: Dict[str, Any],
          methods: list = None) -> Individual:
    """
    Apply multiple mutation operators.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Base probability of mutation
        config: Configuration with mutation parameters
        methods: List of mutation methods to apply
        
    Returns:
        Mutated individual
    """
    if methods is None:
        methods = ['parameters', 'indicators', 'conditions', 'structure']
    
    mutated = individual
    
    for method in methods:
        if random.random() < mutation_rate:
            if method == 'parameters':
                mutated = mutate_parameters(mutated, mutation_rate, config)
            elif method == 'indicators':
                mutated = mutate_indicators(mutated, mutation_rate, config)
            elif method == 'conditions':
                mutated = mutate_conditions(mutated, mutation_rate, config)
            elif method == 'structure':
                mutated = mutate_structure(mutated, mutation_rate, config)
    
    return mutated
