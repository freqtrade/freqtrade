"""
Crossover Operators

Implements various crossover strategies for combining two parent
strategies to create offspring.
"""

import random
from typing import Tuple

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene


def single_point_crossover(parent1: Individual, parent2: Individual, 
                          generation: int, ind_id: int) -> Tuple[Individual, Individual]:
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
    
    # Crossover indicators: split at random point
    if len(parent1.strategy_gene.indicators) > 1 and len(parent2.strategy_gene.indicators) > 1:
        crossover_point = random.randint(1, min(len(parent1.strategy_gene.indicators),
                                                len(parent2.strategy_gene.indicators)) - 1)
        
        # Swap indicator tails
        child1_gene.indicators = (parent1.strategy_gene.indicators[:crossover_point] + 
                                 parent2.strategy_gene.indicators[crossover_point:])
        child2_gene.indicators = (parent2.strategy_gene.indicators[:crossover_point] + 
                                 parent1.strategy_gene.indicators[crossover_point:])
    
    # Crossover entry conditions
    if len(parent1.strategy_gene.entry_conditions) > 1 and len(parent2.strategy_gene.entry_conditions) > 1:
        crossover_point = random.randint(1, min(len(parent1.strategy_gene.entry_conditions),
                                                len(parent2.strategy_gene.entry_conditions)) - 1)
        
        # Swap condition tails
        child1_gene.entry_conditions = (parent1.strategy_gene.entry_conditions[:crossover_point] + 
                                       parent2.strategy_gene.entry_conditions[crossover_point:])
        child2_gene.entry_conditions = (parent2.strategy_gene.entry_conditions[:crossover_point] + 
                                       parent1.strategy_gene.entry_conditions[crossover_point:])
    
    # Crossover exit conditions
    if (len(parent1.strategy_gene.exit_conditions) > 1 and 
        len(parent2.strategy_gene.exit_conditions) > 1):
        crossover_point = random.randint(1, min(len(parent1.strategy_gene.exit_conditions),
                                                len(parent2.strategy_gene.exit_conditions)) - 1)
        
        # Swap exit condition tails
        child1_gene.exit_conditions = (parent1.strategy_gene.exit_conditions[:crossover_point] + 
                                      parent2.strategy_gene.exit_conditions[crossover_point:])
        child2_gene.exit_conditions = (parent2.strategy_gene.exit_conditions[:crossover_point] + 
                                      parent1.strategy_gene.exit_conditions[crossover_point:])
    
    # Randomly inherit other parameters
    if random.random() < 0.5:
        child1_gene.timeframe = parent2.strategy_gene.timeframe
        child2_gene.timeframe = parent1.strategy_gene.timeframe
    
    if random.random() < 0.5:
        child1_gene.stoploss = parent2.strategy_gene.stoploss
        child2_gene.stoploss = parent1.strategy_gene.stoploss
    
    if random.random() < 0.5:
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    child1 = Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id])
    child2 = Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id])
    
    return child1, child2


def uniform_crossover(parent1: Individual, parent2: Individual,
                     generation: int, ind_id: int,
                     swap_prob: float = 0.5) -> Tuple[Individual, Individual]:
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
    child1_indicators = []
    child2_indicators = []
    max_len = max(len(parent1.strategy_gene.indicators), len(parent2.strategy_gene.indicators))
    
    for i in range(max_len):
        if random.random() < swap_prob:
            # Swap: child1 gets from parent2, child2 gets from parent1
            if i < len(parent2.strategy_gene.indicators):
                child1_indicators.append(parent2.strategy_gene.indicators[i])
            if i < len(parent1.strategy_gene.indicators):
                child2_indicators.append(parent1.strategy_gene.indicators[i])
        else:
            # No swap: child1 gets from parent1, child2 gets from parent2
            if i < len(parent1.strategy_gene.indicators):
                child1_indicators.append(parent1.strategy_gene.indicators[i])
            if i < len(parent2.strategy_gene.indicators):
                child2_indicators.append(parent2.strategy_gene.indicators[i])
    
    # Ensure at least one indicator in each child
    if not child1_indicators:
        child1_indicators = [parent1.strategy_gene.indicators[0]]
    if not child2_indicators:
        child2_indicators = [parent2.strategy_gene.indicators[0]]
    
    child1_gene.indicators = child1_indicators
    child2_gene.indicators = child2_indicators
    
    # Uniform crossover for entry conditions
    child1_entry = []
    child2_entry = []
    max_len = max(len(parent1.strategy_gene.entry_conditions), len(parent2.strategy_gene.entry_conditions))
    
    for i in range(max_len):
        if random.random() < swap_prob:
            if i < len(parent2.strategy_gene.entry_conditions):
                child1_entry.append(parent2.strategy_gene.entry_conditions[i])
            if i < len(parent1.strategy_gene.entry_conditions):
                child2_entry.append(parent1.strategy_gene.entry_conditions[i])
        else:
            if i < len(parent1.strategy_gene.entry_conditions):
                child1_entry.append(parent1.strategy_gene.entry_conditions[i])
            if i < len(parent2.strategy_gene.entry_conditions):
                child2_entry.append(parent2.strategy_gene.entry_conditions[i])
    
    # Ensure at least one entry condition in each child
    if not child1_entry:
        child1_entry = [parent1.strategy_gene.entry_conditions[0]]
    if not child2_entry:
        child2_entry = [parent2.strategy_gene.entry_conditions[0]]
    
    child1_gene.entry_conditions = child1_entry
    child2_gene.entry_conditions = child2_entry
    
    # Uniform crossover for exit conditions
    child1_exit = []
    child2_exit = []
    max_len = max(len(parent1.strategy_gene.exit_conditions), len(parent2.strategy_gene.exit_conditions))
    
    for i in range(max_len):
        if random.random() < swap_prob:
            if i < len(parent2.strategy_gene.exit_conditions):
                child1_exit.append(parent2.strategy_gene.exit_conditions[i])
            if i < len(parent1.strategy_gene.exit_conditions):
                child2_exit.append(parent1.strategy_gene.exit_conditions[i])
        else:
            if i < len(parent1.strategy_gene.exit_conditions):
                child1_exit.append(parent1.strategy_gene.exit_conditions[i])
            if i < len(parent2.strategy_gene.exit_conditions):
                child2_exit.append(parent2.strategy_gene.exit_conditions[i])
    
    child1_gene.exit_conditions = child1_exit
    child2_gene.exit_conditions = child2_exit
    
    # Randomly inherit scalar parameters
    if random.random() < swap_prob:
        child1_gene.timeframe = parent2.strategy_gene.timeframe
        child2_gene.timeframe = parent1.strategy_gene.timeframe
    
    if random.random() < swap_prob:
        child1_gene.stoploss = parent2.strategy_gene.stoploss
        child2_gene.stoploss = parent1.strategy_gene.stoploss
    
    if random.random() < swap_prob:
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
    
    if random.random() < swap_prob:
        child1_gene.trailing_stop = parent2.strategy_gene.trailing_stop
        child2_gene.trailing_stop = parent1.strategy_gene.trailing_stop
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    child1 = Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id])
    child2 = Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id])
    
    return child1, child2


def component_crossover(parent1: Individual, parent2: Individual,
                       generation: int, ind_id: int) -> Tuple[Individual, Individual]:
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
    
    # Randomly decide which components to swap
    swap_indicators = random.random() < 0.5
    swap_entry = random.random() < 0.5
    swap_exit = random.random() < 0.5
    swap_risk = random.random() < 0.5
    
    # Swap indicator sets
    if swap_indicators:
        child1_gene.indicators = parent2.strategy_gene.indicators[:]
        child2_gene.indicators = parent1.strategy_gene.indicators[:]
    
    # Swap entry condition sets
    if swap_entry:
        child1_gene.entry_conditions = parent2.strategy_gene.entry_conditions[:]
        child2_gene.entry_conditions = parent1.strategy_gene.entry_conditions[:]
    
    # Swap exit condition sets
    if swap_exit:
        child1_gene.exit_conditions = parent2.strategy_gene.exit_conditions[:]
        child2_gene.exit_conditions = parent1.strategy_gene.exit_conditions[:]
    
    # Swap risk management parameters
    if swap_risk:
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
    
    child1 = Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id])
    child2 = Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id])
    
    return child1, child2


def crossover(parent1: Individual, parent2: Individual,
             generation: int, ind_id: int,
             method: str = 'single_point',
             **kwargs) -> Tuple[Individual, Individual]:
    """
    Perform crossover using specified method.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        method: Crossover method ('single_point', 'uniform', 'component')
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
    
    return crossover_methods[method](parent1, parent2, generation, ind_id, **kwargs)
