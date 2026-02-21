"""
Crossover Operators

Implements various crossover strategies for combining two parent
strategies to create offspring.
"""

import copy
import random
from typing import Tuple

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene


def single_point_crossover(parent1: Individual, parent2: Individual, 
                          generation: int, ind_id: int,
                          config: dict = None) -> Tuple[Individual, Individual]:
    """
    Single-point crossover.
    
    Split both parents at a random point and swap the second parts.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Crossover indicators
    if len(parent1.strategy_gene.indicators) > 1 and len(parent2.strategy_gene.indicators) > 1:
        point = random.randint(1, min(len(parent1.strategy_gene.indicators),
                                     len(parent2.strategy_gene.indicators)) - 1)
        child1_gene.indicators = ([copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators[:point]] + 
                                 [copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators[point:]])
        child2_gene.indicators = ([copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators[:point]] + 
                                 [copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators[point:]])
    
    # Crossover entry conditions
    if len(parent1.strategy_gene.entry_conditions) > 1 and len(parent2.strategy_gene.entry_conditions) > 1:
        point = random.randint(1, min(len(parent1.strategy_gene.entry_conditions),
                                     len(parent2.strategy_gene.entry_conditions)) - 1)
        child1_gene.entry_conditions = ([copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions[:point]] + 
                                       [copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions[point:]])
        child2_gene.entry_conditions = ([copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions[:point]] + 
                                       [copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions[point:]])
    
    # Crossover exit conditions
    if (len(parent1.strategy_gene.exit_conditions) > 1 and 
        len(parent2.strategy_gene.exit_conditions) > 1):
        point = random.randint(1, min(len(parent1.strategy_gene.exit_conditions),
                                     len(parent2.strategy_gene.exit_conditions)) - 1)
        child1_gene.exit_conditions = ([copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions[:point]] + 
                                      [copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions[point:]])
        child2_gene.exit_conditions = ([copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions[:point]] + 
                                      [copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions[point:]])
    
    # Randomly inherit scalar parameters
    for attr in ['timeframe', 'stoploss']:
        if random.random() < 0.5:
            setattr(child1_gene, attr, getattr(parent2.strategy_gene, attr))
            setattr(child2_gene, attr, getattr(parent1.strategy_gene, attr))
    
    # Swap informative_timeframes along with timeframe for consistency
    if random.random() < 0.5:
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if random.random() < 0.5:
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Ensure all indicators referenced in conditions are calculated
    if config:
        indicator_config = config.get('indicators', {})
        child1_gene.ensure_indicators_for_conditions(indicator_config)
        child2_gene.ensure_indicators_for_conditions(indicator_config)
    
    # Reassign instance IDs after crossover to avoid ID conflicts
    child1_gene.assign_instance_ids()
    child2_gene.assign_instance_ids()
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def _uniform_crossover_lists(parent1_list, parent2_list, swap_prob):
    """Helper for uniform crossover on lists."""
    child1_list, child2_list = [], []
    max_len = max(len(parent1_list), len(parent2_list))
    
    for i in range(max_len):
        if random.random() < swap_prob:
            if i < len(parent2_list):
                child1_list.append(copy.deepcopy(parent2_list[i]))
            if i < len(parent1_list):
                child2_list.append(copy.deepcopy(parent1_list[i]))
        else:
            if i < len(parent1_list):
                child1_list.append(copy.deepcopy(parent1_list[i]))
            if i < len(parent2_list):
                child2_list.append(copy.deepcopy(parent2_list[i]))
    
    return child1_list, child2_list


def uniform_crossover(parent1: Individual, parent2: Individual,
                     generation: int, ind_id: int,
                     swap_prob: float = 0.5,
                     config: dict = None) -> Tuple[Individual, Individual]:
    """
    Uniform crossover.
    
    Each component is randomly inherited from either parent.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual  
        generation: Generation number for offspring
        ind_id: Starting individual ID
        swap_prob: Probability of swapping each component
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Uniform crossover for indicators
    child1_indicators, child2_indicators = _uniform_crossover_lists(
        parent1.strategy_gene.indicators, parent2.strategy_gene.indicators, swap_prob)
    
    # Ensure at least one indicator in each child
    child1_gene.indicators = child1_indicators if child1_indicators else [parent1.strategy_gene.indicators[0]]
    child2_gene.indicators = child2_indicators if child2_indicators else [parent2.strategy_gene.indicators[0]]
    
    # Uniform crossover for entry conditions
    child1_entry, child2_entry = _uniform_crossover_lists(
        parent1.strategy_gene.entry_conditions, parent2.strategy_gene.entry_conditions, swap_prob)
    
    # Ensure at least one entry condition in each child
    child1_gene.entry_conditions = child1_entry if child1_entry else [parent1.strategy_gene.entry_conditions[0]]
    child2_gene.entry_conditions = child2_entry if child2_entry else [parent2.strategy_gene.entry_conditions[0]]
    
    # Uniform crossover for exit conditions
    child1_gene.exit_conditions, child2_gene.exit_conditions = _uniform_crossover_lists(
        parent1.strategy_gene.exit_conditions, parent2.strategy_gene.exit_conditions, swap_prob)
    
    # Randomly inherit scalar parameters
    for attr in ['timeframe', 'stoploss']:
        if random.random() < swap_prob:
            val1 = getattr(parent2.strategy_gene, attr)
            val2 = getattr(parent1.strategy_gene, attr)
            setattr(child1_gene, attr, val1)
            setattr(child2_gene, attr, val2)
    
    if random.random() < swap_prob:
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if random.random() < swap_prob:
        child1_gene.trailing_stop = parent2.strategy_gene.trailing_stop
        child2_gene.trailing_stop = parent1.strategy_gene.trailing_stop
    
    if random.random() < swap_prob:
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Ensure all indicators referenced in conditions are calculated
    if config:
        indicator_config = config.get('indicators', {})
        child1_gene.ensure_indicators_for_conditions(indicator_config)
        child2_gene.ensure_indicators_for_conditions(indicator_config)
    
    # Reassign instance IDs after crossover to avoid ID conflicts
    child1_gene.assign_instance_ids()
    child2_gene.assign_instance_ids()
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def component_crossover(parent1: Individual, parent2: Individual,
                       generation: int, ind_id: int,
                       config: dict = None) -> Tuple[Individual, Individual]:
    """
    Component-based crossover.
    
    Exchange entire indicator sets or rule sets between parents.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Swap components based on random decisions
    swaps = {
        'indicators': random.random() < 0.5,
        'entry': random.random() < 0.5,
        'exit': random.random() < 0.5,
        'risk': random.random() < 0.5,
    }
    
    if swaps['indicators']:
        child1_gene.indicators, child2_gene.indicators = [copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators], [copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators]
        # Swap informative_timeframes with indicators for consistency
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if swaps['entry']:
        child1_gene.entry_conditions, child2_gene.entry_conditions = [copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions], [copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions]
    
    if swaps['exit']:
        child1_gene.exit_conditions, child2_gene.exit_conditions = [copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions], [copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions]
    
    if swaps['risk']:
        # Swap all risk parameters (stoploss, ROI, trailing stop)
        child1_gene.stoploss = parent2.strategy_gene.stoploss
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child1_gene.trailing_stop = parent2.strategy_gene.trailing_stop
        
        child2_gene.stoploss = parent1.strategy_gene.stoploss
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
        child2_gene.trailing_stop = parent1.strategy_gene.trailing_stop
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Ensure all indicators referenced in conditions are calculated
    if config:
        indicator_config = config.get('indicators', {})
        child1_gene.ensure_indicators_for_conditions(indicator_config)
        child2_gene.ensure_indicators_for_conditions(indicator_config)
    
    # Reassign instance IDs after crossover to avoid ID conflicts
    child1_gene.assign_instance_ids()
    child2_gene.assign_instance_ids()
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def crossover(parent1: Individual, parent2: Individual,
             generation: int, ind_id: int,
             method: str = 'single_point',
             config: dict = None,
             **kwargs) -> Tuple[Individual, Individual]:
    """
    Perform crossover using specified method.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        method: Crossover method ('single_point', 'uniform', 'component')
        config: Configuration dictionary with indicator settings
        **kwargs: Additional arguments for crossover method
        
    Returns:
        Tuple of two offspring individuals
    """
    crossover_methods = {
        'single_point': single_point_crossover,
        'uniform': uniform_crossover,
        'component': component_crossover,
    }
    
    if method not in crossover_methods:
        raise ValueError(f"Unknown crossover method: {method}")
    
    # Pass config to crossover methods
    kwargs['config'] = config
    return crossover_methods[method](parent1, parent2, generation, ind_id, **kwargs)
