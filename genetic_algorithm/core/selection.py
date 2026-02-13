"""
Selection Operators

Implements various selection strategies for choosing parents
for the next generation.
"""

import random
from typing import List, Callable

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population


def tournament_selection(population: Population, tournament_size: int = 3) -> Individual:
    """
    Tournament selection.
    
    Randomly select tournament_size individuals and return the best one.
    
    Args:
        population: Population to select from
        tournament_size: Number of individuals in tournament
        
    Returns:
        Selected individual
    """
    # TODO: Implement tournament selection
    tournament = random.sample(population.individuals, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))


def roulette_wheel_selection(population: Population) -> Individual:
    """
    Roulette wheel (fitness-proportionate) selection.
    
    Probability of selection is proportional to fitness.
    
    Args:
        population: Population to select from
        
    Returns:
        Selected individual
    """
    # TODO: Implement roulette wheel selection
    pass


def rank_based_selection(population: Population) -> Individual:
    """
    Rank-based selection.
    
    Selection probability based on rank rather than absolute fitness.
    
    Args:
        population: Population to select from
        
    Returns:
        Selected individual
    """
    # TODO: Implement rank-based selection
    pass


def select_parents(population: Population, 
                   num_parents: int,
                   method: str = 'tournament',
                   **kwargs) -> List[Individual]:
    """
    Select multiple parents using specified method.
    
    Args:
        population: Population to select from
        num_parents: Number of parents to select
        method: Selection method ('tournament', 'roulette', 'rank')
        **kwargs: Additional arguments for selection method
        
    Returns:
        List of selected individuals
    """
    selection_methods = {
        'tournament': tournament_selection,
        'roulette': roulette_wheel_selection,
        'rank': rank_based_selection,
    }
    
    if method not in selection_methods:
        raise ValueError(f"Unknown selection method: {method}")
    
    selector = selection_methods[method]
    parents = []
    
    for _ in range(num_parents):
        if method == 'tournament':
            parent = selector(population, kwargs.get('tournament_size', 3))
        else:
            parent = selector(population)
        parents.append(parent)
    
    return parents
