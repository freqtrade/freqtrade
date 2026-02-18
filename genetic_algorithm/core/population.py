"""
Population Management

Manages a collection of individuals representing trading strategies.
"""

from typing import List, Optional
import random
from dataclasses import dataclass, field
import math

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene


@dataclass
class PopulationStats:
    """Statistics for a population at a given generation."""
    
    generation: int
    size: int
    best_fitness: Optional[float] = None
    worst_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    median_fitness: Optional[float] = None
    diversity_score: Optional[float] = None
    genetic_diversity: Optional[float] = None  # New: structural diversity


def calculate_strategy_distance(ind1: Individual, ind2: Individual) -> float:
    """
    Calculate genetic distance between two strategies.
    
    Measures how different two strategies are based on:
    - Indicator types and parameters
    - Condition types and thresholds
    - Risk parameters
    
    Args:
        ind1: First individual
        ind2: Second individual
        
    Returns:
        Distance score (0 = identical, higher = more different)
    """
    gene1 = ind1.strategy_gene
    gene2 = ind2.strategy_gene
    
    distance = 0.0
    
    # Indicator type difference
    types1 = set(ind.type for ind in gene1.indicators)
    types2 = set(ind.type for ind in gene2.indicators)
    indicator_diff = len(types1.symmetric_difference(types2)) / max(len(types1), len(types2), 1)
    distance += indicator_diff * 0.3
    
    # Condition count difference
    entry_diff = abs(len(gene1.entry_conditions) - len(gene2.entry_conditions))
    exit_diff = abs(len(gene1.exit_conditions) - len(gene2.exit_conditions))
    condition_diff = (entry_diff + exit_diff) / 10.0  # Normalize
    distance += min(condition_diff, 1.0) * 0.2
    
    # Timeframe difference
    if gene1.timeframe != gene2.timeframe:
        distance += 0.2
    
    # Stoploss difference (normalized)
    stoploss_diff = abs(gene1.stoploss - gene2.stoploss) / 0.20  # Max range ~0.20
    distance += min(stoploss_diff, 1.0) * 0.15
    
    # Trailing stop difference
    if gene1.trailing_stop != gene2.trailing_stop:
        distance += 0.15
    
    return distance


def apply_fitness_sharing(population: 'Population', sigma_share: float = 0.3, 
                         alpha: float = 1.0) -> None:
    """
    Apply fitness sharing to preserve diversity.
    
    Reduces fitness of individuals in crowded regions of the solution space,
    encouraging exploration of diverse strategies.
    
    Args:
        population: Population to apply sharing to
        sigma_share: Sharing radius (how similar strategies must be to share)
        alpha: Sharing function exponent
    """
    individuals = list(population.individuals)
    n = len(individuals)
    
    if n < 2:
        return
    
    # Calculate pairwise distances
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_strategy_distance(individuals[i], individuals[j])
            distances[i][j] = dist
            distances[j][i] = dist
    
    # Calculate niche counts for each individual
    for i, individual in enumerate(individuals):
        if individual.fitness is None or individual.fitness <= 0:
            continue
        
        # Calculate sharing function
        niche_count = 0.0
        for j in range(n):
            dist = distances[i][j]
            if dist < sigma_share:
                # Sharing function: sh(d) = 1 - (d/sigma)^alpha
                sharing = 1.0 - (dist / sigma_share) ** alpha
                niche_count += sharing
        
        # Adjust fitness by niche count (shared fitness)
        if niche_count > 0:
            individual.fitness = individual.fitness / niche_count


def calculate_genetic_diversity(population: 'Population') -> float:
    """
    Calculate overall genetic diversity of the population.
    
    Higher diversity means more varied strategies, which is good for exploration.
    
    Args:
        population: Population to measure
        
    Returns:
        Diversity score (0 = no diversity, 1 = maximum diversity)
    """
    individuals = list(population.individuals)
    n = len(individuals)
    
    if n < 2:
        return 0.0
    
    # Calculate average pairwise distance
    total_distance = 0.0
    num_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += calculate_strategy_distance(individuals[i], individuals[j])
            num_pairs += 1
    
    if num_pairs == 0:
        return 0.0
    
    avg_distance = total_distance / num_pairs
    
    # Normalize to 0-1 range (assuming max distance ~2.0)
    diversity = min(1.0, avg_distance / 1.0)
    
    return diversity


class Population:
    """
    Manages a population of strategy individuals.
    
    Provides methods for initialization, sorting, selection,
    and tracking population statistics.
    """
    
    def __init__(self, size: int, generation: int = 0):
        """
        Initialize a population.
        
        Args:
            size: Number of individuals in the population
            generation: Current generation number
        """
        self.size = size
        self.generation = generation
        self.individuals: List[Individual] = []
    
    def add_individual(self, individual: Individual):
        """Add an individual to the population."""
        self.individuals.append(individual)
    
    def remove_individual(self, individual: Individual):
        """Remove an individual from the population."""
        self.individuals.remove(individual)
    
    def sort_by_fitness(self, reverse: bool = True):
        """
        Sort population by fitness.
        
        Args:
            reverse: If True, sort descending (best first)
        """
        self.individuals.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
                            reverse=reverse)
    
    def get_best(self, n: int = 1) -> List[Individual]:
        """
        Get the top N individuals.
        
        Args:
            n: Number of top individuals to return
            
        Returns:
            List of top N individuals
        """
        self.sort_by_fitness(reverse=True)
        return self.individuals[:n]
    
    def get_worst(self, n: int = 1) -> List[Individual]:
        """
        Get the bottom N individuals.
        
        Args:
            n: Number of bottom individuals to return
            
        Returns:
            List of bottom N individuals
        """
        self.sort_by_fitness(reverse=False)
        return self.individuals[:n]
    
    def get_stats(self) -> PopulationStats:
        """
        Calculate population statistics.
        
        Returns:
            PopulationStats object with current statistics
        """
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        
        if not evaluated:
            return PopulationStats(generation=self.generation, size=len(self.individuals))
        
        # Create sorted copy of fitnesses for median/min/max calculations
        fitnesses = [ind.fitness for ind in evaluated]
        fitnesses_sorted = sorted(fitnesses)
        n = len(fitnesses_sorted)
        
        avg_fitness = sum(fitnesses) / n
        stats = PopulationStats(
            generation=self.generation,
            size=len(self.individuals),
            best_fitness=fitnesses_sorted[-1],
            worst_fitness=fitnesses_sorted[0],
            avg_fitness=avg_fitness,
            median_fitness=fitnesses_sorted[n // 2],
        )
        
        # Calculate diversity (standard deviation of fitness)
        if n > 1:
            variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / n
            stats.diversity_score = variance ** 0.5
        
        # Calculate genetic diversity (structural differences)
        stats.genetic_diversity = calculate_genetic_diversity(self)
        
        return stats
    
    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return len(self.individuals)
    
    def __iter__(self):
        """Iterate over individuals."""
        return iter(self.individuals)
    
    def __getitem__(self, index: int) -> Individual:
        """Get individual by index."""
        return self.individuals[index]
