"""
Selection Operators

Implements various selection strategies for choosing parents
for the next generation, including NSGA-II multi-objective selection.
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
    evaluated = [ind for ind in population.individuals if ind.fitness is not None]
    
    if not evaluated:
        raise ValueError("No evaluated individuals in population")
    
    # Adjust fitness to be non-negative for proportionate selection
    min_fitness = min(ind.fitness for ind in evaluated)
    if min_fitness < 0:
        adjusted_fitnesses = [ind.fitness - min_fitness + 1 for ind in evaluated]
    else:
        adjusted_fitnesses = [ind.fitness for ind in evaluated]
    
    total_fitness = sum(adjusted_fitnesses)
    
    if total_fitness == 0:
        return random.choice(evaluated)
    
    # Select based on fitness proportion
    spin = random.uniform(0, total_fitness)
    cumulative = 0
    for ind, fitness in zip(evaluated, adjusted_fitnesses):
        cumulative += fitness
        if cumulative >= spin:
            return ind
    
    return evaluated[-1]


def rank_based_selection(population: Population) -> Individual:
    """
    Rank-based selection.
    
    Selection probability based on rank rather than absolute fitness.
    
    Args:
        population: Population to select from
        
    Returns:
        Selected individual
    """
    # Get all evaluated individuals
    evaluated = [ind for ind in population.individuals if ind.fitness is not None]
    
    if not evaluated:
        raise ValueError("No evaluated individuals in population")
    
    # Sort by fitness (worst to best) and select based on linear rank
    sorted_individuals = sorted(evaluated, key=lambda x: x.fitness)
    n = len(sorted_individuals)
    total_rank = n * (n + 1) // 2
    spin = random.uniform(0, total_rank)
    
    cumulative = 0
    for rank, ind in enumerate(sorted_individuals, start=1):
        cumulative += rank
        if cumulative >= spin:
            return ind
    
    return sorted_individuals[-1]


def nsga2_tournament_selection(population: Population, tournament_size: int = 2) -> Individual:
    """
    NSGA-II binary tournament selection.
    
    Selects winner based on:
    1. Lower rank (better Pareto front)
    2. If same rank, higher crowding distance (more diverse)
    
    Args:
        population: Population to select from
        tournament_size: Number of individuals in tournament (default: 2 for NSGA-II)
        
    Returns:
        Selected individual
    """
    evaluated = [ind for ind in population.individuals if ind.objectives is not None]
    
    if not evaluated:
        # Fallback to fitness-based tournament
        return tournament_selection(population, tournament_size)
    
    tournament = random.sample(evaluated, min(tournament_size, len(evaluated)))
    
    # Select using NSGA-II comparison
    best = tournament[0]
    for ind in tournament[1:]:
        comparison = ind.nsga2_compare(best)
        if comparison > 0:  # ind is better
            best = ind
    
    return best


def select_parents(population: Population, 
                   num_parents: int,
                   method: str = 'tournament',
                   allow_duplicates: bool = True,
                   **kwargs) -> List[Individual]:
    """
    Select multiple parents using specified method.
    
    Args:
        population: Population to select from
        num_parents: Number of parents to select
        method: Selection method ('tournament', 'roulette', 'rank', 'nsga2')
        allow_duplicates: Whether to allow selecting the same parent multiple times
        **kwargs: Additional arguments for selection method
        
    Returns:
        List of selected individuals
    """
    selection_methods = {
        'tournament': tournament_selection,
        'roulette': roulette_wheel_selection,
        'rank': rank_based_selection,
        'nsga2': nsga2_tournament_selection,
    }
    
    if method not in selection_methods:
        raise ValueError(f"Unknown selection method: {method}")
    
    selector = selection_methods[method]
    parents = []
    
    for _ in range(num_parents):
        max_attempts = 100  # Prevent infinite loop
        attempt = 0
        
        while attempt < max_attempts:
            if method == 'tournament':
                parent = selector(population, kwargs.get('tournament_size', 3))
            elif method == 'nsga2':
                parent = selector(population, kwargs.get('tournament_size', 2))
            else:
                parent = selector(population)
            
            # Check if we should accept this parent
            if allow_duplicates or parent not in parents:
                parents.append(parent)
                break
            
            attempt += 1
        
        # If we couldn't find a unique parent after max attempts, accept duplicate
        if attempt == max_attempts and len(parents) < num_parents:
            parents.append(parent)
    
    return parents
