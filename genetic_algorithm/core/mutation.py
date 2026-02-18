"""
Mutation Operators

Implements various mutation strategies for introducing
variation into strategies.
"""

import random
import logging
from typing import Dict, Any, Optional

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.utils.indicator_factory import create_random_indicator

# Set up logger for mutation operations
logger = logging.getLogger(__name__)


def _mutate_indicator_params(indicator, ind_config, i, mutations_applied):
    """Helper to mutate indicator parameters based on type."""
    # Period-based indicators (simple case)
    if 'period' in indicator.parameters and indicator.type in ['RSI', 'EMA', 'SMA', 'ATR', 'ADX', 'CCI']:
        period_range = ind_config.get('period', [7, 21] if indicator.type == 'RSI' else [10, 50])
        indicator.parameters['period'] = random.randint(*period_range)
        mutations_applied.append(f"{indicator.type}_period_{i}")
    
    # Multi-parameter indicators
    elif indicator.type == 'MACD':
        for param, default in [('fast_period', [8, 21]), ('slow_period', [21, 50]), ('signal_period', [5, 14])]:
            if param in indicator.parameters and random.random() < 0.5:
                indicator.parameters[param] = random.randint(*ind_config.get(param, default))
                mutations_applied.append(f"MACD_{param.split('_')[0]}_{i}")
    
    elif indicator.type == 'BBANDS':
        if 'period' in indicator.parameters and random.random() < 0.5:
            indicator.parameters['period'] = random.randint(*ind_config.get('period', [15, 30]))
            mutations_applied.append(f"BBANDS_period_{i}")
        if 'std_dev' in indicator.parameters and random.random() < 0.5:
            indicator.parameters['std_dev'] = random.uniform(*ind_config.get('std_dev', [1.5, 3.0]))
            mutations_applied.append(f"BBANDS_std_{i}")
    
    elif indicator.type == 'STOCH':
        for param, default in [('k_period', [5, 21]), ('d_period', [3, 14])]:
            if param in indicator.parameters and random.random() < 0.5:
                indicator.parameters[param] = random.randint(*ind_config.get(param, default))
                mutations_applied.append(f"STOCH_{param[0]}_{i}")
    
    # Mutate weight
    if random.random() < 0.3:
        indicator.weight = random.uniform(0.3, 1.0)
        mutations_applied.append(f"weight_{i}")


def _mutate_condition_threshold(condition, ind_config, is_entry, i, mutations_applied):
    """Helper to mutate condition thresholds."""
    threshold_key = 'buy_threshold' if is_entry else 'sell_threshold'
    if condition.indicator == 'RSI':
        threshold_range = ind_config.get(threshold_key, [20, 40] if is_entry else [60, 80])
    elif condition.indicator == 'CCI':
        threshold_range = ind_config.get(threshold_key, [-200, -100] if is_entry else [100, 200])
    else:
        return
    
    condition.threshold = random.randint(*threshold_range)
    mutations_applied.append(f"{'entry' if is_entry else 'exit'}_{condition.indicator}_threshold_{i}")


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
            _mutate_indicator_params(indicator, ind_config, i, mutations_applied)
    
    # Mutate condition thresholds
    for i, condition in enumerate(mutated_gene.entry_conditions):
        if random.random() < mutation_rate:
            ind_config = indicator_config.get(condition.indicator, {})
            _mutate_condition_threshold(condition, ind_config, True, i, mutations_applied)
    
    for i, condition in enumerate(mutated_gene.exit_conditions):
        if random.random() < mutation_rate:
            ind_config = indicator_config.get(condition.indicator, {})
            _mutate_condition_threshold(condition, ind_config, False, i, mutations_applied)
    
    # Mutate stoploss
    if random.random() < mutation_rate:
        mutated_gene.stoploss = random.uniform(*strategy_constraints.get('stoploss_range', [-0.20, -0.05]))
        mutations_applied.append("stoploss")
    
    # Mutate ROI values
    if random.random() < mutation_rate:
        roi_range = strategy_constraints.get('roi_range', [0.01, 0.10])
        mutated_gene.minimal_roi = {
            "0": random.uniform(roi_range[0] * 2, roi_range[1]),
            "30": random.uniform(roi_range[0] * 1.5, roi_range[1] * 0.7),
            "60": random.uniform(roi_range[0], roi_range[1] * 0.5),
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
            
            # Ensure at least one entry condition remains
            if not mutated_gene.entry_conditions:
                # Add a condition using one of the remaining indicators
                available_indicators = [ind.type for ind in mutated_gene.indicators]
                if available_indicators:
                    # Try to create a condition, try multiple indicators if needed
                    condition_created = False
                    for indicator in available_indicators:
                        new_condition = _create_random_condition(indicator, True, indicator_config)
                        if new_condition:
                            mutated_gene.entry_conditions.append(new_condition)
                            mutations_applied.append(f"add_entry_condition_{indicator}")
                            condition_created = True
                            break
                    
                    # If still no entry condition, this mutation failed validation
                    # The try-catch in the mutate() function will catch this
                    if not condition_created:
                        logger.warning("Failed to create entry condition - all available indicators failed to generate valid conditions")
                        raise ValueError("Failed to create entry condition - all available indicators failed to generate valid conditions")

    
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
    
    # Ensure all indicators referenced in conditions are calculated
    indicator_config = config.get('indicators', {})
    new_individual.strategy_gene.ensure_indicators_for_conditions(indicator_config)
    
    # Reassign instance IDs after mutation to maintain unique IDs
    new_individual.strategy_gene.assign_instance_ids()
    
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
    
    # Helper to mutate a single condition list
    def mutate_condition_list(conditions, is_entry, label):
        if not conditions or random.random() >= mutation_rate:
            return
        
        idx = random.randrange(len(conditions))
        condition = conditions[idx]
        mutation_type = random.choice(['operator', 'logic', 'threshold'])
        
        if mutation_type == 'operator':
            condition.operator = random.choice(['<', '>', 'cross_above', 'cross_below'])
            mutations_applied.append(f"{label}_operator_{idx}")
        elif mutation_type == 'logic':
            condition.logic = 'OR' if condition.logic == 'AND' else 'AND'
            mutations_applied.append(f"{label}_logic_{idx}")
        elif mutation_type == 'threshold':
            ind_config = indicator_config.get(condition.indicator, {})
            threshold_key = 'buy_threshold' if is_entry else 'sell_threshold'
            
            # Set default ranges based on indicator type
            defaults = {
                'RSI': ([20, 40], [60, 80]),
                'STOCH': ([20, 40], [60, 80]),
                'CCI': ([-200, -100], [100, 200])
            }
            
            if condition.indicator in defaults:
                default_range = defaults[condition.indicator][0 if is_entry else 1]
                threshold_range = ind_config.get(threshold_key, default_range)
                condition.threshold = random.randint(*threshold_range)
                mutations_applied.append(f"{label}_threshold_{idx}")
    
    # Mutate entry and exit conditions
    mutate_condition_list(mutated_gene.entry_conditions, True, 'entry')
    mutate_condition_list(mutated_gene.exit_conditions, False, 'exit')
    
    # Possibly add new conditions
    if random.random() < mutation_rate * 0.5:
        available_indicators = [ind.type for ind in mutated_gene.indicators]
        if available_indicators:
            # Add entry condition if needed
            if len(mutated_gene.entry_conditions) < 3:
                indicator = random.choice(available_indicators)
                new_condition = _create_random_condition(indicator, True, indicator_config)
                if new_condition:
                    mutated_gene.entry_conditions.append(new_condition)
                    mutations_applied.append(f"add_entry_condition_{indicator}")
            
            # Add exit condition if needed
            if len(mutated_gene.exit_conditions) < 3:
                indicator = random.choice(available_indicators)
                new_condition = _create_random_condition(indicator, False, indicator_config)
                if new_condition:
                    mutated_gene.exit_conditions.append(new_condition)
                    mutations_applied.append(f"add_exit_condition_{indicator}")
    
    # Possibly remove conditions
    # IMPORTANT: Must maintain at least 1 entry condition to satisfy validation
    if random.random() < mutation_rate * 0.3:
        if len(mutated_gene.entry_conditions) > 1:
            removed = mutated_gene.entry_conditions.pop(random.randrange(len(mutated_gene.entry_conditions)))
            mutations_applied.append(f"remove_entry_condition_{removed.indicator}")
        
        # Exit conditions can be empty, so we can remove them freely
        if len(mutated_gene.exit_conditions) > 0:
            removed = mutated_gene.exit_conditions.pop(random.randrange(len(mutated_gene.exit_conditions)))
            mutations_applied.append(f"remove_exit_condition_{removed.indicator}")
    
    # Create new individual with mutation record
    new_individual = Individual(strategy_gene=mutated_gene, parent_ids=[individual.id])
    new_individual.mutations = individual.mutations + [{
        'type': 'condition',
        'applied': mutations_applied
    }]
    
    # Ensure all indicators referenced in conditions are calculated
    indicator_config = config.get('indicators', {})
    new_individual.strategy_gene.ensure_indicators_for_conditions(indicator_config)
    
    # Reassign instance IDs to maintain unique references
    new_individual.strategy_gene.assign_instance_ids()
    
    return new_individual


def _create_random_condition(indicator_type: str, is_entry: bool, 
                            indicator_config: Dict[str, Any]) -> Optional[ConditionGene]:
    """Helper function to create a random condition for an indicator."""
    ind_config = indicator_config.get(indicator_type, {})
    
    # Configuration map: indicator -> (entry_op, exit_op, entry_threshold_key, exit_threshold_key, default_entry, default_exit)
    config_map = {
        'RSI': ('cross_below', 'cross_above', 'buy_threshold', 'sell_threshold', [20, 40], [60, 80]),
        'MACD': ('cross_above', 'cross_below', None, None, None, None),
        'STOCH': ('<', '>', 'k_threshold', 'd_threshold', [20, 40], [60, 80]),
        'CCI': ('<', '>', 'buy_threshold', 'sell_threshold', [-200, -100], [100, 200]),
        'ADX': ('>', '>', 'threshold', 'threshold', [20, 40], [20, 40]),
        'BBANDS': ('cross_below', 'cross_above', None, None, None, None),
        'EMA': ('cross_above', 'cross_below', None, None, None, None),
        'SMA': ('cross_above', 'cross_below', None, None, None, None),
        'ATR': ('>', '<', None, None, None, None),
    }
    
    if indicator_type not in config_map:
        return None
    
    entry_op, exit_op, entry_key, exit_key, entry_default, exit_default = config_map[indicator_type]
    operator = entry_op if is_entry else exit_op
    
    # MACD, BBANDS, EMA, SMA, ATR use threshold 0 (not used in comparison)
    # ATR doesn't use threshold because it's compared against price or other values
    if indicator_type in ['MACD', 'BBANDS', 'EMA', 'SMA', 'ATR']:
        threshold = 0
    else:
        threshold_key = entry_key if is_entry else exit_key
        default_range = entry_default if is_entry else exit_default
        threshold_range = ind_config.get(threshold_key, default_range)
        threshold = random.randint(*threshold_range)
    
    return ConditionGene(
        indicator=indicator_type,
        operator=operator,
        threshold=threshold,
        logic=random.choice(['AND', 'OR'])
    )


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
            "0": random.uniform(roi_range[0] * 2, roi_range[1]),
            "30": random.uniform(roi_range[0] * 1.5, roi_range[1] * 0.7),
            "60": random.uniform(roi_range[0], roi_range[1] * 0.5),
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


def mutate_gaussian(individual: Individual, mutation_rate: float,
                   config: Dict[str, Any], sigma: float = 0.1) -> Individual:
    """
    Gaussian mutation - adds normally distributed noise to numeric parameters.
    
    This provides smooth, incremental adjustments rather than discrete jumps,
    allowing fine-tuning of promising strategies.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation
        config: Configuration with parameter ranges
        sigma: Standard deviation of Gaussian noise (relative to parameter range)
        
    Returns:
        Mutated individual
    """
    mutated_gene = individual.strategy_gene.copy()
    indicator_config = config.get('indicators', {})
    strategy_constraints = config.get('strategy_constraints', {})
    mutations_applied = []
    
    # Gaussian mutation for indicator parameters
    for i, indicator in enumerate(mutated_gene.indicators):
        if random.random() < mutation_rate:
            ind_config = indicator_config.get(indicator.type, {})
            
            # Period-based indicators
            if 'period' in indicator.parameters:
                period_range = ind_config.get('period', [7, 50])
                current = indicator.parameters['period']
                range_size = period_range[1] - period_range[0]
                noise = random.gauss(0, sigma * range_size)
                new_period = int(max(period_range[0], min(period_range[1], current + noise)))
                indicator.parameters['period'] = new_period
                mutations_applied.append(f"gaussian_{indicator.type}_period_{i}")
            
            # Continuous parameters (e.g., BBANDS std_dev)
            if 'std_dev' in indicator.parameters and indicator.type == 'BBANDS':
                std_range = ind_config.get('std_dev', [1.5, 3.0])
                current = indicator.parameters['std_dev']
                range_size = std_range[1] - std_range[0]
                noise = random.gauss(0, sigma * range_size)
                new_std = max(std_range[0], min(std_range[1], current + noise))
                indicator.parameters['std_dev'] = new_std
                mutations_applied.append(f"gaussian_BBANDS_std_{i}")
    
    # Gaussian mutation for stoploss
    if random.random() < mutation_rate:
        stoploss_range = strategy_constraints.get('stoploss_range', [-0.20, -0.05])
        current = mutated_gene.stoploss
        range_size = stoploss_range[1] - stoploss_range[0]
        noise = random.gauss(0, sigma * range_size)
        mutated_gene.stoploss = max(stoploss_range[0], min(stoploss_range[1], current + noise))
        mutations_applied.append("gaussian_stoploss")
    
    # Gaussian mutation for ROI values
    if random.random() < mutation_rate:
        roi_range = strategy_constraints.get('roi_range', [0.01, 0.10])
        range_size = roi_range[1] - roi_range[0]
        
        # Mutate each ROI level
        new_roi = {}
        for time_key, current_val in mutated_gene.minimal_roi.items():
            noise = random.gauss(0, sigma * range_size)
            new_val = max(roi_range[0], min(roi_range[1], current_val + noise))
            new_roi[time_key] = new_val
        
        # Ensure ROI decreases over time
        sorted_keys = sorted([int(k) for k in new_roi.keys()])
        for i in range(len(sorted_keys) - 1):
            if new_roi[str(sorted_keys[i])] < new_roi[str(sorted_keys[i + 1])]:
                new_roi[str(sorted_keys[i + 1])] = new_roi[str(sorted_keys[i])] * 0.9
        
        mutated_gene.minimal_roi = {str(k): new_roi[str(k)] for k in sorted_keys}
        mutations_applied.append("gaussian_roi")
    
    # Create new individual with mutation record
    new_individual = Individual(strategy_gene=mutated_gene, parent_ids=[individual.id])
    new_individual.mutations = individual.mutations + [{
        'type': 'gaussian',
        'sigma': sigma,
        'applied': mutations_applied
    }]
    
    return new_individual


def mutate_swap(individual: Individual, mutation_rate: float,
               config: Dict[str, Any]) -> Individual:
    """
    Swap mutation - swaps positions of indicators or conditions.
    
    Can discover better orderings and combinations by rearranging
    existing components rather than modifying them.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation
        config: Configuration
        
    Returns:
        Mutated individual
    """
    mutated_gene = individual.strategy_gene.copy()
    mutations_applied = []
    
    # Swap indicators
    if len(mutated_gene.indicators) >= 2 and random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated_gene.indicators)), 2)
        mutated_gene.indicators[i], mutated_gene.indicators[j] = \
            mutated_gene.indicators[j], mutated_gene.indicators[i]
        mutations_applied.append(f"swap_indicators_{i}_{j}")
    
    # Swap entry conditions
    if len(mutated_gene.entry_conditions) >= 2 and random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated_gene.entry_conditions)), 2)
        mutated_gene.entry_conditions[i], mutated_gene.entry_conditions[j] = \
            mutated_gene.entry_conditions[j], mutated_gene.entry_conditions[i]
        mutations_applied.append(f"swap_entry_conditions_{i}_{j}")
    
    # Swap exit conditions
    if len(mutated_gene.exit_conditions) >= 2 and random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated_gene.exit_conditions)), 2)
        mutated_gene.exit_conditions[i], mutated_gene.exit_conditions[j] = \
            mutated_gene.exit_conditions[j], mutated_gene.exit_conditions[i]
        mutations_applied.append(f"swap_exit_conditions_{i}_{j}")
    
    # Create new individual with mutation record
    new_individual = Individual(strategy_gene=mutated_gene, parent_ids=[individual.id])
    new_individual.mutations = individual.mutations + [{
        'type': 'swap',
        'applied': mutations_applied
    }]
    
    return new_individual


def mutate_adaptive_per_gene(individual: Individual, base_mutation_rate: float,
                             config: Dict[str, Any]) -> Individual:
    """
    Adaptive per-gene mutation - adjusts mutation rate based on gene fitness history.
    
    Genes that contributed to high fitness mutate less (exploitation),
    while poorly performing genes mutate more (exploration).
    
    Args:
        individual: Individual to mutate
        base_mutation_rate: Base mutation rate
        config: Configuration
        
    Returns:
        Mutated individual
    """
    mutated_gene = individual.strategy_gene.copy()
    mutations_applied = []
    
    # Calculate adaptive rates for different gene components
    # If individual has high fitness, reduce mutation of good components
    # Handle None fitness (unevaluated individuals)
    if individual.fitness is None or individual.fitness <= 0:
        fitness_factor = 1.0
    else:
        fitness_factor = min(1.0, individual.fitness)
    
    # Indicators: lower mutation rate for high fitness (preserve good indicators)
    indicator_rate = base_mutation_rate * (1.5 - fitness_factor)
    
    # Conditions: moderate mutation rate
    condition_rate = base_mutation_rate * (1.3 - 0.5 * fitness_factor)
    
    # Structure: higher mutation for low fitness (try different approaches)
    structure_rate = base_mutation_rate * (1.0 + 0.5 * (1.0 - fitness_factor))
    
    # Apply mutations with adaptive rates
    if random.random() < indicator_rate:
        mutated = mutate_indicators(individual, indicator_rate, config)
        mutations_applied.append(f"adaptive_indicators_{indicator_rate:.3f}")
        individual = mutated
    
    if random.random() < condition_rate:
        mutated = mutate_conditions(individual, condition_rate, config)
        mutations_applied.append(f"adaptive_conditions_{condition_rate:.3f}")
        individual = mutated
    
    if random.random() < structure_rate:
        mutated = mutate_structure(individual, structure_rate, config)
        mutations_applied.append(f"adaptive_structure_{structure_rate:.3f}")
        individual = mutated
    
    # Record adaptive mutation
    individual.mutations = individual.mutations + [{
        'type': 'adaptive_per_gene',
        'base_rate': base_mutation_rate,
        'fitness_factor': fitness_factor,
        'applied': mutations_applied
    }]
    
    return individual


def mutate(individual: Individual, mutation_rate: float,
          config: Dict[str, Any],
          methods: list = None) -> Individual:
    """
    Apply multiple mutation operators.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Base probability of mutation
        config: Configuration with mutation parameters
        methods: List of mutation methods to apply (if None, uses default set)
        
    Returns:
        Mutated individual (original if mutation fails)
    """
    if methods is None:
        # Default: include new advanced mutation operators with lower probability
        methods = ['parameters', 'indicators', 'conditions', 'structure']
        
        # Add advanced operators based on random selection
        if random.random() < 0.2:  # 20% chance to use Gaussian mutation
            methods.append('gaussian')
        if random.random() < 0.1:  # 10% chance to use swap mutation
            methods.append('swap')
        if random.random() < 0.15:  # 15% chance to use adaptive mutation
            methods.append('adaptive')
    
    mutated = individual
    
    for method in methods:
        if random.random() < mutation_rate:
            try:
                if method == 'parameters':
                    mutated = mutate_parameters(mutated, mutation_rate, config)
                elif method == 'indicators':
                    mutated = mutate_indicators(mutated, mutation_rate, config)
                elif method == 'conditions':
                    mutated = mutate_conditions(mutated, mutation_rate, config)
                elif method == 'structure':
                    mutated = mutate_structure(mutated, mutation_rate, config)
                elif method == 'gaussian':
                    mutated = mutate_gaussian(mutated, mutation_rate, config, sigma=0.1)
                elif method == 'swap':
                    mutated = mutate_swap(mutated, mutation_rate, config)
                elif method == 'adaptive':
                    mutated = mutate_adaptive_per_gene(mutated, mutation_rate, config)
            except (ValueError, KeyError, AttributeError, TypeError) as e:
                # Log the error but continue with the current mutated state
                # This ensures that a failed mutation doesn't crash the evolution
                logger.warning(f"Mutation method '{method}' failed: {e}. Continuing with current state.")
                # If this is the first mutation attempt, return the original individual
                if mutated is individual:
                    logger.debug(f"Returning original individual due to failed mutation")
    
    return mutated
